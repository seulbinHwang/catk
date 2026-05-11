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
#   INITIAL_BS=28        Initial `data.train_batch_size`. Defaults to the
#                        preset's conservative random-terminal setting.
#   OOM_STEP=2           Decrement applied to `data.train_batch_size` per OOM.
#   MIN_BS=2             Stop trying when `bs` would fall below this value.
#   TASK_NAME=flow_semi_continuous_self_forced_h1004
#   CACHE_ROOT=/mnt/nuplan/womd_v1_3/SMART_cache
#   CATK_LOG_DIR=<repo>/logs
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#   EXPERIMENT=self_forced_npfm_h100_4
#   CATK_LR=                     Optional Generator learning-rate override.
#   FLOW_WINDOW_STEPS=           Optional decoder horizon override.
#   SCORER_SCENE_NUM=            Optional fit-time closed-loop scorer scene target.
#   DISTRIBUTION_MATCHING_OBJECTIVE=
#                                  Optional self-forced objective override: dmd or sid.
#   DETACH_BLOCK_TRANSITION=      Optional rollout block detach override: true or false.
#   USE_ANCHOR_FLOW_MATCHING_LOSS=
#                                  Optional anchor FM loss gate override: true or false.
#   ANCHOR_WEIGHT=                Optional anchor FM loss weight override.
#   ESTIMATOR_WARMUP_EPOCHS=     Optional self-forced warmup override.
#   SELF_FORCED_USE_STOP_MOTION= Optional training rollout stop-motion gate.
#   DECODER_USE_STOP_MOTION=     Optional validation/test inference gate.
#   LIMIT_TRAIN_BATCHES=         Optional Trainer limit_train_batches override.
#   LIMIT_VAL_BATCHES=           Optional Trainer limit_val_batches override.
#   MAX_EPOCHS=                  Optional Trainer max_epochs override.
#   CHECK_VAL_EVERY_N_EPOCH=     Optional Trainer check_val_every_n_epoch override.
#   RANDOM_TERMINAL_SCOPE=         Optional override: global_batch.
#   RANDOM_TERMINAL_POLICY=        Optional override: paper_uniform.
#   BACKPROP_LAST_K=              Optional policy=all gradient-step override.
#   UNFROZEN_RANGE=               Optional trainable Generator range override.
#   EMA_WEIGHT=                    Optional Generator EMA decay override.
#   EMA_START_STEP=                Optional Generator EMA start update override.
#   CLEAN_DMD_NORMALIZER_EPS=      Optional Clean-DMD normalizer epsilon override.
#   CLEAN_DMD_TAU_LOW=             Optional Clean-DMD guidance tau lower bound.
#   CLEAN_DMD_TAU_HIGH=            Optional Clean-DMD guidance tau upper bound.
#   CATK_EXTRA_OVERRIDES=          Optional whitespace-separated Hydra overrides.
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
CATK_LOG_DIR="${CATK_LOG_DIR:-${REPO_ROOT}/logs}"
INITIAL_BS="${INITIAL_BS:-28}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-2}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:?must set PRETRAIN_CKPT to the 2s-horizon pretrained Generator ckpt}"

EXTRA_OVERRIDES=()
if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
  EXTRA_OVERRIDES+=("data.val_batch_size=${VAL_BATCH_SIZE}")
fi
if [[ -n "${TEST_BATCH_SIZE:-}" ]]; then
  EXTRA_OVERRIDES+=("data.test_batch_size=${TEST_BATCH_SIZE}")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  EXTRA_OVERRIDES+=("trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  EXTRA_OVERRIDES+=("trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
fi
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  EXTRA_OVERRIDES+=("trainer.max_epochs=${MAX_EPOCHS}")
fi
if [[ -n "${CHECK_VAL_EVERY_N_EPOCH:-}" ]]; then
  EXTRA_OVERRIDES+=("trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}")
fi
if [[ -n "${CATK_LR:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.lr=${CATK_LR}")
fi
if [[ -n "${FLOW_WINDOW_STEPS:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.decoder.flow_window_steps=${FLOW_WINDOW_STEPS}")
fi
if [[ -n "${SCORER_SCENE_NUM:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.scorer_scene_num=${SCORER_SCENE_NUM}")
fi
if [[ -n "${DISTRIBUTION_MATCHING_OBJECTIVE:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.distribution_matching_objective=${DISTRIBUTION_MATCHING_OBJECTIVE}")
fi
if [[ -n "${DETACH_BLOCK_TRANSITION:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.detach_block_transition=${DETACH_BLOCK_TRANSITION}")
fi
if [[ -n "${USE_ANCHOR_FLOW_MATCHING_LOSS:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.use_anchor_flow_matching_loss=${USE_ANCHOR_FLOW_MATCHING_LOSS}")
fi
if [[ -n "${ANCHOR_WEIGHT:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.anchor_weight=${ANCHOR_WEIGHT}")
fi
if [[ -n "${ESTIMATOR_WARMUP_EPOCHS:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.estimator_warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}")
fi
if [[ -n "${SELF_FORCED_USE_STOP_MOTION:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.use_stop_motion=${SELF_FORCED_USE_STOP_MOTION}")
fi
if [[ -n "${DECODER_USE_STOP_MOTION:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.decoder.use_stop_motion=${DECODER_USE_STOP_MOTION}")
fi
if [[ -n "${UNFROZEN_RANGE:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.unfrozen_range=${UNFROZEN_RANGE}")
fi
if [[ -n "${RANDOM_TERMINAL_SCOPE:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.sampling.random_terminal_step.scope=${RANDOM_TERMINAL_SCOPE}")
fi
if [[ -n "${RANDOM_TERMINAL_POLICY:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.sampling.random_terminal_step.policy=${RANDOM_TERMINAL_POLICY}")
fi
if [[ -n "${BACKPROP_LAST_K:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.sampling.backprop_last_k=${BACKPROP_LAST_K}")
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
if [[ -n "${CATK_EXTRA_OVERRIDES:-}" ]]; then
  read -r -a EXTRA_FROM_ENV <<< "$CATK_EXTRA_OVERRIDES"
  EXTRA_OVERRIDES+=("${EXTRA_FROM_ENV[@]}")
fi

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: PRETRAIN_CKPT does not exist: $PRETRAIN_CKPT" >&2
  exit 1
fi

LOG_DIR="${CATK_LOG_DIR}/_self_forced_oom_retry/${TASK_NAME}"
mkdir -p "$LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

find_latest_self_forced_ckpt() {
  # Latest `epoch_last.ckpt` under any run directory for this task.
  # Returns empty string when none exists yet.
  ls -t "${CATK_LOG_DIR}/${TASK_NAME}/runs"/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1
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
    paths.log_dir="$CATK_LOG_DIR" \
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
