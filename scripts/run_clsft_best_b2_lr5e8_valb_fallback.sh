#!/usr/bin/env bash
# Reproduce the best CLSFT/DMD setting from testsv with validation-batch fallback.
#
# This wrapper calls scripts/train_self_forced_npfm_pareto.sh with:
#   train_b=2, lr=5e-8, estimator_lr=5e-8, beta=1.0, anchor FM off.
# It first tries VAL_B=16, then falls back to 8 and 4 if the run exits with
# an OOM/traceback/runtime error. Each attempt starts from CKPT_PATH.
#
# Required on a new server:
#   - Run from the catk repo root.
#   - Put the fake warmup checkpoint at CKPT_PATH, or override CKPT_PATH.
#   - Ensure CACHE_ROOT points to the WOMD SMART cache.
#
# Example:
#   CKPT_PATH=/path/to/fake_warmup_epoch0.ckpt \
#   CACHE_ROOT=/workspace/womd_v1_3/SMART_cache \
#   WANDB_ENTITY=se99an WANDB_PROJECT=clsft-catk \
#   bash scripts/run_clsft_best_b2_lr5e8_valb_fallback.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

timestamp() { TZ=Asia/Seoul date +%F_%T_KST; }
log() { printf '[clsft-best] %s %s\n' "$(timestamp)" "$*"; }

DEFAULT_CKPT_PATH="logs/pareto_mapencfreeze_lr1e7_beta1_step200_skipfake_clsft_pareto_clsft_v100x4_0529_162458_fake1ep_trainb4_valb4/runs/2026-05-29_16-52-48/checkpoints/fake_warmup_epoch0.ckpt"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT_PATH}}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  cat >&2 <<EOF
ERROR: CKPT_PATH not found: ${CKPT_PATH}

Set CKPT_PATH to fake_warmup_epoch0.ckpt before running this script.
EOF
  exit 1
fi

VAL_B_CANDIDATES="${VAL_B_CANDIDATES:-16 8 4}"
TASK_PREFIX="${TASK_PREFIX:-auto_clsft_best_b2_lr5e8_valbfallback_$(TZ=Asia/Seoul date +%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/test_runs}"
mkdir -p "${LOG_ROOT}"

SUPERVISOR_LOG="${LOG_ROOT}/${TASK_PREFIX}.supervisor.log"
FALLBACK_REGEX="${FALLBACK_REGEX:-CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|Traceback|RuntimeError}"
CLEAR_WANDB_API_KEY="${CLEAR_WANDB_API_KEY:-false}"

log "repo=${REPO_ROOT}" | tee -a "${SUPERVISOR_LOG}"
log "task_prefix=${TASK_PREFIX}" | tee -a "${SUPERVISOR_LOG}"
log "ckpt_path=${CKPT_PATH}" | tee -a "${SUPERVISOR_LOG}"
log "val_b_candidates=${VAL_B_CANDIDATES}" | tee -a "${SUPERVISOR_LOG}"

for val_b in ${VAL_B_CANDIDATES}; do
  task="${TASK_PREFIX}_trainb2_valb${val_b}"
  log_path="${LOG_ROOT}/${task}.log"

  log "launching task=${task}" | tee -a "${SUPERVISOR_LOG}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  export NUM_NODES="${NUM_NODES:-1}"
  export ACTION="${ACTION:-finetune}"
  export MY_TASK_NAME="${task}"
  export CKPT_PATH

  export WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
  export WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_OFFLINE="${WANDB_OFFLINE:-false}"
  export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-all}"

  export MAX_EPOCHS="${MAX_EPOCHS:-16}"
  export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
  export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"
  export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
  export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
  export PRECISION="${PRECISION:-32-true}"
  export TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp_find_unused_parameters_true}"
  export NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
  export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"

  export TRAIN_B="${TRAIN_B:-2}"
  export VAL_B="${val_b}"
  export TEST_B="${val_b}"
  export NUM_WORKERS="${NUM_WORKERS:-8}"
  export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
  export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
  export PIN_MEMORY="${PIN_MEMORY:-true}"
  export TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.5}"
  export TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-true}"

  export LR="${LR:-5.0e-8}"
  export ESTIMATOR_LR="${ESTIMATOR_LR:-5.0e-8}"
  export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
  export LR_MIN_RATIO="${LR_MIN_RATIO:-1.0}"

  export DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"
  export DMD_BETA="${DMD_BETA:-1.0}"
  export SF_ENABLED="${SF_ENABLED:-true}"
  export SF_START_EPOCH="${SF_START_EPOCH:-0}"
  export SF_WEIGHT="${SF_WEIGHT:-1.0}"
  export SF_PATH_STEP_SIZE="${SF_PATH_STEP_SIZE:-0.05}"
  export USE_ANCHOR_FM="${USE_ANCHOR_FM:-false}"
  export ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.1}"
  export ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-3}"
  export SF_N_ROLLOUTS="${SF_N_ROLLOUTS:-1}"
  export SF_N_ANCHORS="${SF_N_ANCHORS:-1}"
  export SF_ANCHOR_STRIDE="${SF_ANCHOR_STRIDE:-1}"
  export ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"
  export ESTIMATOR_WARMUP_STEPS="${ESTIMATOR_WARMUP_STEPS:-0}"
  export SF_INIT_AUX_FROM_GEN="${SF_INIT_AUX_FROM_GEN:-false}"
  export SF_UNFROZEN_RANGE="${SF_UNFROZEN_RANGE:-except_map_encoder}"
  export SF_EMA_WEIGHT="${SF_EMA_WEIGHT:-0.0}"
  export SF_EMA_START_STEP="${SF_EMA_START_STEP:-1000000000}"
  export SF_GRAD_CLIP="${SF_GRAD_CLIP:-1.0}"

  export SAMPLING_SAMPLE_STEPS="${SAMPLING_SAMPLE_STEPS:-16}"
  export SAMPLING_SAMPLE_METHOD="${SAMPLING_SAMPLE_METHOD:-euler}"
  export SAMPLING_NOISE_SCALE="${SAMPLING_NOISE_SCALE:-1.0}"
  export SAMPLING_RTS_ENABLED="${SAMPLING_RTS_ENABLED:-true}"
  export SAMPLING_RTS_POLICY="${SAMPLING_RTS_POLICY:-all}"
  export SAMPLING_RTS_MIN_EXECUTED_STEPS="${SAMPLING_RTS_MIN_EXECUTED_STEPS:-16}"
  export SAMPLING_RTS_BACKPROP_LAST_K="${SAMPLING_RTS_BACKPROP_LAST_K:-8}"
  export SAMPLING_RTS_SCOPE="${SAMPLING_RTS_SCOPE:-global_batch}"

  export VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-16}"
  export VAL_SAMPLE_METHOD="${VAL_SAMPLE_METHOD:-euler}"
  export VAL_NOISE_SCALE="${VAL_NOISE_SCALE:-1.0}"
  export N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
  export N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-100000}"
  export SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1728}"
  export SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-8}"
  export CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS="${CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS:-50000}"
  export CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS="${CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS:-50000}"
  export VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-true}"
  export VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"
  export N_VIS_BATCH="${N_VIS_BATCH:-0}"
  export N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
  export N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
  export DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-true}"
  export CLOSED_LOOP_ROLLOUT_MODE="${CLOSED_LOOP_ROLLOUT_MODE:-raw_fm}"
  export DECODER_USE_LQR="${DECODER_USE_LQR:-false}"
  export WOSAC_CPD_REFERENCE="${WOSAC_CPD_REFERENCE:-null}"
  export CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
  export CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
  export CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"

  export EXTRA_ARGS="${EXTRA_ARGS:-+model.model_config.self_forced.allow_auxiliary_finetune=true +callbacks.self_forced_warmup_validation_gate._target_=src.utils.self_forced_warmup_validation_gate.SelfForcedWarmupValidationGateCallback +callbacks.self_forced_warmup_validation_gate.verbose=false}"

  if [[ "${CLEAR_WANDB_API_KEY}" == "true" ]]; then
    env -u WANDB_API_KEY bash scripts/train_self_forced_npfm_pareto.sh 2>&1 | tee "${log_path}"
    status=${PIPESTATUS[0]}
  else
    bash scripts/train_self_forced_npfm_pareto.sh 2>&1 | tee "${log_path}"
    status=${PIPESTATUS[0]}
  fi

  log "task=${task} exited status=${status}" | tee -a "${SUPERVISOR_LOG}"
  if [[ "${status}" -eq 0 ]]; then
    log "success at VAL_B=${val_b}; stopping fallback sequence" | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi

  if grep -Eq "${FALLBACK_REGEX}" "${log_path}"; then
    log "fallback signal detected at VAL_B=${val_b}; trying next candidate" | tee -a "${SUPERVISOR_LOG}"
    sleep "${FALLBACK_SLEEP_SECONDS:-20}"
    continue
  fi

  log "non-fallback failure at VAL_B=${val_b}; see ${log_path}" | tee -a "${SUPERVISOR_LOG}"
  exit "${status}"
done

log "exhausted VAL_B candidates: ${VAL_B_CANDIDATES}" | tee -a "${SUPERVISOR_LOG}"
exit 1
