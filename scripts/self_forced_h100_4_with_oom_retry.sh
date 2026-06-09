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
#   CATK_GENERATED_ESTIMATOR_LR= Optional generated-estimator learning-rate
#                                override. Defaults to CATK_EXTRA_OVERRIDES
#                                generated_estimator_lr, then CATK_LR.
#   ESTIMATOR_WARMUP_EPOCHS=     Optional self-forced warmup override.
#   SELF_FORCED_USE_STOP_MOTION= Optional training rollout stop-motion gate.
#   DECODER_USE_STOP_MOTION=     Optional validation/test inference gate.
#   LIMIT_TRAIN_BATCHES=         Optional Trainer limit_train_batches override.
#   LIMIT_VAL_BATCHES=           Optional Trainer limit_val_batches override.
#   MAX_EPOCHS=                  Optional Trainer max_epochs override.
#   CHECK_VAL_EVERY_N_EPOCH=     Optional Trainer check_val_every_n_epoch override.
#   TRAIN_EPOCH_SAMPLE_FRACTION= Optional train dataset fraction override.
#   TRAIN_MEMORY_BALANCED_BATCHES=Optional train memory-balanced batching override.
#   RANDOM_TERMINAL_SCOPE=         Optional override: global_batch.
#   RANDOM_TERMINAL_POLICY=        Optional override: paper_uniform.
#   BACKPROP_LAST_K=              Optional policy=all gradient-step override.
#   UNFROZEN_RANGE=               Optional trainable Generator range override.
#   EMA_WEIGHT=                    Optional Generator EMA decay override.
#   EMA_START_STEP=                Optional Generator EMA start update override.
#   CLEAN_DMD_NORMALIZER_EPS=      Optional Clean-DMD stable scale floor override.
#   CLEAN_DMD_TAU_LOW=             Optional Clean-DMD guidance tau lower bound.
#   CLEAN_DMD_TAU_HIGH=            Optional Clean-DMD guidance tau upper bound.
#   ESTIMATOR_WARMUP_BANK_ENABLED=true
#                                  true이면 fresh finetune 시작 전에 W&B
#                                  generated-estimator bank에서
#                                  (warmup_epochs, generated_estimator_lr)
#                                  entry를 찾아 warmup을 건너뜁니다.
#   ESTIMATOR_WARMUP_BANK_ARTIFACT=generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6:latest
#                                  W&B bank artifact name/ref. 예:
#                                  generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6:latest
#   ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME=generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6
#                                  warmup을 새로 돌린 뒤 upsert할 artifact name.
#   ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS=true
#                                  bank hit 시 max_epochs에서 warmup epoch 수를
#                                  빼서 DMD 학습 epoch budget을 유지합니다.
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
DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME="generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6"
DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT="${DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME}:latest"
ESTIMATOR_WARMUP_BANK_ENABLED="${ESTIMATOR_WARMUP_BANK_ENABLED:-true}"
ESTIMATOR_WARMUP_BANK_ARTIFACT="${ESTIMATOR_WARMUP_BANK_ARTIFACT:-$DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT}"
ESTIMATOR_WARMUP_BANK_ENTITY="${ESTIMATOR_WARMUP_BANK_ENTITY:-${WANDB_ENTITY:-jksg01019-naver-labs}}"
ESTIMATOR_WARMUP_BANK_PROJECT="${ESTIMATOR_WARMUP_BANK_PROJECT:-${WANDB_PROJECT:-SMART-FLOW}}"
ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS="${ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS:-true}"
ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME="${ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME:-$DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME}"
ESTIMATOR_WARMUP_BANK_INIT_PATH=""
ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH=""
ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP="${ESTIMATOR_WARMUP_EPOCHS:-}"
ESTIMATOR_WARMUP_BANK_LOADED_WARMUP=0
ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="${ESTIMATOR_WARMUP_EPOCHS:-0}"
ESTIMATOR_WARMUP_BANK_LR=""

if [[ -z "$ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME" && -n "$ESTIMATOR_WARMUP_BANK_ARTIFACT" ]]; then
  ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME="${ESTIMATOR_WARMUP_BANK_ARTIFACT##*/}"
  ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME="${ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME%%:*}"
fi

has_latest_self_forced_ckpt() {
  ls -t "${CATK_LOG_DIR}/${TASK_NAME}/runs"/*/checkpoints/epoch_last.ckpt >/dev/null 2>&1
}

strip_shell_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s\n' "$value"
}

resolve_generated_estimator_lr_config() {
  if [[ -n "${CATK_GENERATED_ESTIMATOR_LR:-}" ]]; then
    strip_shell_quotes "$CATK_GENERATED_ESTIMATOR_LR"
    return 0
  fi

  local value=""
  local token
  if [[ -n "${CATK_EXTRA_OVERRIDES:-}" ]]; then
    for token in $CATK_EXTRA_OVERRIDES; do
      case "$token" in
        model.model_config.self_forced.generated_estimator_lr=*|+model.model_config.self_forced.generated_estimator_lr=*)
          value="${token#*=}"
          ;;
      esac
    done
  fi

  if [[ -n "$value" ]]; then
    strip_shell_quotes "$value"
    return 0
  fi

  strip_shell_quotes "${CATK_LR:-}"
}

apply_estimator_warmup_bank_progress() {
  local loaded_warmup="$1"
  local remaining_warmup="$2"
  local context="$3"

  ESTIMATOR_WARMUP_BANK_LOADED_WARMUP="$loaded_warmup"
  ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="$remaining_warmup"
  if [[ "$ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS" == "true" && "${MAX_EPOCHS:-}" =~ ^[0-9]+$ ]]; then
    local adjusted_epochs=$(( MAX_EPOCHS - ESTIMATOR_WARMUP_BANK_LOADED_WARMUP ))
    if (( adjusted_epochs < 1 )); then
      adjusted_epochs=1
    fi
    log "Adjusting MAX_EPOCHS ${MAX_EPOCHS} -> ${adjusted_epochs} after estimator-bank ${context}."
    MAX_EPOCHS="$adjusted_epochs"
  fi
  ESTIMATOR_WARMUP_EPOCHS="$ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP"
}

raise_max_epochs_to_checkpoint_floor() {
  local ckpt_path="$1"
  if [[ -z "${MAX_EPOCHS:-}" || ! "${MAX_EPOCHS:-}" =~ ^[0-9]+$ || ! -f "$ckpt_path" ]]; then
    return 0
  fi
  local checkpoint_floor
  checkpoint_floor="$(
    python - "$ckpt_path" <<'PY' 2>/dev/null
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
if epoch is not None:
    print(int(epoch) + 1)
PY
  )"
  if [[ "$checkpoint_floor" =~ ^[0-9]+$ && "$MAX_EPOCHS" -lt "$checkpoint_floor" ]]; then
    log "Raising MAX_EPOCHS ${MAX_EPOCHS} -> ${checkpoint_floor} so resume checkpoint epoch is not past the trainer budget."
    MAX_EPOCHS="$checkpoint_floor"
  fi
}

maybe_prepare_estimator_warmup_bank() {
  if [[ "$ESTIMATOR_WARMUP_BANK_ENABLED" != "true" ]]; then
    return 0
  fi
  if [[ -z "$ESTIMATOR_WARMUP_BANK_ARTIFACT" ]]; then
    log "Estimator warmup bank enabled but ESTIMATOR_WARMUP_BANK_ARTIFACT is empty; running warmup normally."
    return 0
  fi
  if [[ -z "${ESTIMATOR_WARMUP_EPOCHS:-}" || "${ESTIMATOR_WARMUP_EPOCHS}" == "0" ]]; then
    return 0
  fi
  ESTIMATOR_WARMUP_BANK_LR="$(resolve_generated_estimator_lr_config)"
  if [[ -z "${ESTIMATOR_WARMUP_BANK_LR:-}" ]]; then
    log "Estimator warmup bank enabled but generated estimator lr is empty; running warmup normally."
    return 0
  fi
  local bank_root="${CATK_LOG_DIR}/_self_forced_estimator_bank/${TASK_NAME}"
  mkdir -p "$bank_root"
  local requested_warmup="${ESTIMATOR_WARMUP_EPOCHS}"
  local resolved_env="${bank_root}/resolved_warmup_${requested_warmup}_lr_${ESTIMATOR_WARMUP_BANK_LR}.env"
  local latest_existing_ckpt=""
  latest_existing_ckpt="$(ls -t "${CATK_LOG_DIR}/${TASK_NAME}/runs"/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1 || true)"
  if [[ -n "$latest_existing_ckpt" ]]; then
    ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH="${bank_root}/snapshot_warmup_${requested_warmup}_lr_${ESTIMATOR_WARMUP_BANK_LR}_generated_estimator.pt"
    if [[ -f "$resolved_env" ]]; then
      # shellcheck disable=SC1090
      source "$resolved_env"
      apply_estimator_warmup_bank_progress \
        "${ESTIMATOR_WARMUP_BANK_RESOLVED_WARMUP:-0}" \
        "${ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP:-0}" \
        "resume"
      if [[ "${ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP:-0}" == "0" ]]; then
        ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH=""
      fi
      log "Existing self-forced checkpoint found; restoring estimator-bank resume state: loaded_warmup=${ESTIMATOR_WARMUP_BANK_LOADED_WARMUP} remaining_warmup=${ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP}."
    else
      ESTIMATOR_WARMUP_BANK_LOADED_WARMUP=0
      ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="$requested_warmup"
      log "Existing self-forced checkpoint found; no estimator-bank resolve state for this task, so resume keeps requested_warmup=${requested_warmup}."
    fi
    raise_max_epochs_to_checkpoint_floor "$latest_existing_ckpt"
    return 0
  fi

  ESTIMATOR_WARMUP_BANK_INIT_PATH="${bank_root}/resolved_for_warmup_${requested_warmup}_lr_${ESTIMATOR_WARMUP_BANK_LR}_generated_estimator.pt"
  ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH="${bank_root}/snapshot_warmup_${requested_warmup}_lr_${ESTIMATOR_WARMUP_BANK_LR}_generated_estimator.pt"

  log "Checking estimator warmup bank: artifact=${ESTIMATOR_WARMUP_BANK_ARTIFACT} requested_warmup=${requested_warmup} generated_estimator_lr=${ESTIMATOR_WARMUP_BANK_LR}"
  if python scripts/self_forced_estimator_bank.py resolve \
      --artifact "$ESTIMATOR_WARMUP_BANK_ARTIFACT" \
      --warmup-epochs "$requested_warmup" \
      --lr "$ESTIMATOR_WARMUP_BANK_LR" \
      --output "$ESTIMATOR_WARMUP_BANK_INIT_PATH" \
      --env-output "$resolved_env" \
      --entity "$ESTIMATOR_WARMUP_BANK_ENTITY" \
      --project "$ESTIMATOR_WARMUP_BANK_PROJECT"; then
    # shellcheck disable=SC1090
    source "$resolved_env"
    local loaded_warmup="${ESTIMATOR_WARMUP_BANK_RESOLVED_WARMUP:-0}"
    local remaining_warmup="${ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP:-0}"
    log "Estimator bank hit: loaded_warmup=${loaded_warmup} requested_warmup=${requested_warmup} remaining_warmup=${remaining_warmup}"
    apply_estimator_warmup_bank_progress "$loaded_warmup" "$remaining_warmup" "hit"
  else
    log "Estimator bank miss; warmup will run and snapshot will be saved to $ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH"
    ESTIMATOR_WARMUP_BANK_LOADED_WARMUP=0
    ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="$requested_warmup"
  fi
}

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

maybe_prepare_estimator_warmup_bank

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
if [[ -n "${TRAIN_EPOCH_SAMPLE_FRACTION:-}" ]]; then
  EXTRA_OVERRIDES+=("data.train_epoch_sample_fraction=${TRAIN_EPOCH_SAMPLE_FRACTION}")
fi
if [[ -n "${TRAIN_MEMORY_BALANCED_BATCHES:-}" ]]; then
  EXTRA_OVERRIDES+=("data.train_memory_balanced_batches=${TRAIN_MEMORY_BALANCED_BATCHES}")
fi
if [[ -n "${CATK_LR:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.lr=${CATK_LR}")
fi
if [[ -n "${ESTIMATOR_WARMUP_EPOCHS:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.estimator_warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}")
fi
if [[ -n "${ESTIMATOR_WARMUP_BANK_INIT_PATH:-}" && -f "$ESTIMATOR_WARMUP_BANK_INIT_PATH" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_init_path=${ESTIMATOR_WARMUP_BANK_INIT_PATH}")
  if [[ "${ESTIMATOR_WARMUP_EPOCHS:-0}" == "0" ]]; then
    EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_skip_warmup_on_load=true")
  else
    EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_skip_warmup_on_load=false")
    EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_bank_snapshot_path=${ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH}")
  fi
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_init_strict=true")
elif [[ -n "${ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH:-}" && "$ESTIMATOR_WARMUP_BANK_ENABLED" == "true" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_bank_snapshot_path=${ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH}")
fi
if [[ -n "${ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP:-}" && "$ESTIMATOR_WARMUP_BANK_ENABLED" == "true" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_bank_target_warmup_epochs=${ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP}")
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_bank_loaded_warmup_epochs=${ESTIMATOR_WARMUP_BANK_LOADED_WARMUP}")
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
if [[ -n "${CATK_GENERATED_ESTIMATOR_LR:-}" ]]; then
  EXTRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_lr=$(strip_shell_quotes "$CATK_GENERATED_ESTIMATOR_LR")")
fi

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: PRETRAIN_CKPT does not exist: $PRETRAIN_CKPT" >&2
  exit 1
fi

LOG_DIR="${CATK_LOG_DIR}/_self_forced_oom_retry/${TASK_NAME}"
mkdir -p "$LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

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
    if [[ "$ESTIMATOR_WARMUP_BANK_ENABLED" == "true" \
        && -n "${ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME:-}" \
        && -n "${ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH:-}" \
        && -f "$ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH" \
        && -n "${ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP:-}" \
        && -n "${ESTIMATOR_WARMUP_BANK_LR:-}" ]]; then
      log "Uploading generated-estimator warmup snapshot to W&B bank: ${ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME} generated_estimator_lr=${ESTIMATOR_WARMUP_BANK_LR}"
      python scripts/self_forced_estimator_bank.py upsert \
        --artifact-name "$ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME" \
        --entry "${ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP}:${ESTIMATOR_WARMUP_BANK_LR}:${ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH}" \
        --entity "$ESTIMATOR_WARMUP_BANK_ENTITY" \
        --project "$ESTIMATOR_WARMUP_BANK_PROJECT" \
        --run-name "${TASK_NAME}_generated_estimator_bank" \
        --alias latest \
        --alias "pretrain_x5f9g0ce_v57" || true
    fi
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
