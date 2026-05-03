#!/usr/bin/env bash
# Run `experiment=self_forced_npfm_h100_4` with automatic batch-size fallback
# on CUDA OOM. The first attempt starts from a pretrained (non-self-forced)
# Generator checkpoint via `action=finetune`. Every subsequent attempt resumes
# from the latest self-forced `epoch_last.ckpt` saved by Lightning, so
# completed epochs are not redone. When the run dies and the per-attempt log
# contains an OOM marker, `data.train_batch_size` is decremented by `OOM_STEP`
# (default 2) and the run restarts; non-OOM failures bubble up immediately.
#
# This is the 4xH100 sibling of `self_forced_h100_6_with_oom_retry.sh`; only
# the GPU count, default `EXPERIMENT`, default `TASK_NAME` and CUDA device
# list differ. Per-GPU memory ceiling is unchanged (same H100 80GB hardware),
# so `INITIAL_BS` defaults to the same conservative value.
#
# Required:
#   PRETRAIN_CKPT  Path to the 2s-horizon pretrained Generator ckpt (the same
#                  ckpt you would have passed to the bare `ckpt_path=` argument).
#
# Optional knobs (env vars; sensible defaults shown):
#   INITIAL_BS=36        Initial `data.train_batch_size`. Defaults to the
#                        preset's conservative random-terminal setting.
#   OOM_STEP=2           Decrement applied to `data.train_batch_size` per OOM.
#   MIN_BS=2             Stop trying when `bs` would fall below this value.
#   TASK_NAME=flow_semi_continuous_self_forced_h1004
#   CACHE_ROOT=/mnt/nuplan/womd_v1_3/SMART_cache
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#   EXPERIMENT=self_forced_npfm_h100_4
#   RANDOM_TERMINAL_SCOPE=         Optional override: global_batch.
#   RANDOM_TERMINAL_POLICY=        Optional override: paper_uniform.
#   EMA_WEIGHT=                    Optional Generator EMA decay override.
#   EMA_START_STEP=                Optional Generator EMA start update override.
#   CLEAN_DMD_NORMALIZER_EPS=      Optional Clean-DMD normalizer epsilon override.
#   CLEAN_DMD_TAU_LOW=             Optional Clean-DMD guidance tau lower bound.
#   CLEAN_DMD_TAU_HIGH=            Optional Clean-DMD guidance tau upper bound.
#
# Usage example:
#   PRETRAIN_CKPT=/mnt/nuplan/projects/catk/downloads/wandb_ckpts/.../epoch_last.ckpt \
#   bash scripts/self_forced_h100_4_with_oom_retry.sh
#
# Re-running this script on the same `TASK_NAME` after a successful or
# interrupted run will automatically detect the latest self-forced
# checkpoint under `logs/<TASK_NAME>/runs/*/checkpoints/epoch_last.ckpt`
# and resume with `action=fit`.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="${EXPERIMENT:-self_forced_npfm_h100_4}"
TASK_NAME="${TASK_NAME:-flow_semi_continuous_self_forced_h1004}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nuplan/womd_v1_3/SMART_cache}"
INITIAL_BS="${INITIAL_BS:-36}"
OOM_STEP="${OOM_STEP:-4}"
MIN_BS="${MIN_BS:-2}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:?must set PRETRAIN_CKPT to the 2s-horizon pretrained Generator ckpt}"

EXTRA_OVERRIDES=()
if [[ -n "${RANDOM_TERMINAL_SCOPE:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.sampling.random_terminal_step.scope=${RANDOM_TERMINAL_SCOPE}")
fi
if [[ -n "${RANDOM_TERMINAL_POLICY:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.sampling.random_terminal_step.policy=${RANDOM_TERMINAL_POLICY}")
fi
if [[ -n "${EMA_WEIGHT:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.ema_weight=${EMA_WEIGHT}")
fi
if [[ -n "${EMA_START_STEP:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.ema_start_step=${EMA_START_STEP}")
fi
if [[ -n "${CLEAN_DMD_NORMALIZER_EPS:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.clean_dmd_normalizer_eps=${CLEAN_DMD_NORMALIZER_EPS}")
fi
if [[ -n "${CLEAN_DMD_TAU_LOW:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.clean_dmd_tau_low=${CLEAN_DMD_TAU_LOW}")
fi
if [[ -n "${CLEAN_DMD_TAU_HIGH:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.clean_dmd_tau_high=${CLEAN_DMD_TAU_HIGH}")
fi

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: PRETRAIN_CKPT does not exist: $PRETRAIN_CKPT" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/logs/_self_forced_oom_retry/${TASK_NAME}"
mkdir -p "$LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

find_latest_self_forced_ckpt() {
  # Latest `epoch_last.ckpt` under any run directory for this task.
  # Returns empty string when none exists yet.
  ls -t "${REPO_ROOT}/logs/${TASK_NAME}/runs"/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1
}

bs="$INITIAL_BS"
attempt=0
while (( bs >= MIN_BS )); do
  attempt=$(( attempt + 1 ))
  attempt_log="${LOG_DIR}/attempt_$(printf '%03d' "$attempt")_bs${bs}.log"

  latest_ckpt="$(find_latest_self_forced_ckpt)"
  if [[ -n "$latest_ckpt" ]]; then
    action="fit"
    ckpt_path="$latest_ckpt"
  else
    action="finetune"
    ckpt_path="$PRETRAIN_CKPT"
  fi

  log "Attempt #${attempt}: bs=${bs} action=${action} ckpt=${ckpt_path}"
  log "  per-attempt log -> ${attempt_log}"
  if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    log "  extra overrides -> ${EXTRA_OVERRIDES[*]}"
  fi

  # `tee` so the user can watch tqdm + Lightning logs live in this pane while
  # we still keep a per-attempt log file for OOM detection / post-hoc analysis.
  # `PYTHONUNBUFFERED=1` keeps Python's stdout/stderr line-flushed so tee
  # sees output immediately rather than after a 4 KiB buffer fills.
  # `${PIPESTATUS[0]}` recovers torchrun's real exit code (tee's own exit
  # would otherwise mask it as 0).
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m src.run \
    experiment="$EXPERIMENT" \
    action="$action" \
    paths.cache_root="$CACHE_ROOT" \
    task_name="$TASK_NAME" \
    ckpt_path="$ckpt_path" \
    data.train_batch_size="$bs" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "$attempt_log"
  exit_code=${PIPESTATUS[0]}

  if (( exit_code == 0 )); then
    log "Training completed successfully (attempt #${attempt}, bs=${bs})."
    exit 0
  fi

  if grep -Eq "$OOM_REGEX" "$attempt_log"; then
    new_bs=$(( bs - OOM_STEP ))
    log "OOM detected at bs=${bs} (exit=${exit_code}). Lowering to bs=${new_bs}."
    bs="$new_bs"
    continue
  fi

  log "Non-OOM failure (exit=${exit_code}). See ${attempt_log}. Aborting retry loop."
  exit "$exit_code"
done

log "Reached MIN_BS=${MIN_BS} without a successful run. Aborting."
exit 1
