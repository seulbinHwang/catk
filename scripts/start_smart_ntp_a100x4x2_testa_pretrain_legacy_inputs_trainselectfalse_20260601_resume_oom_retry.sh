#!/usr/bin/env bash
# Resume the interrupted 2026-06-01 SMART NTP A100x4x2 pretrain run.
#
# The checkpoint for this run was produced from main@4b65c615, where the model
# still used decoder.num_freq_bands=88.  Keep GIT_REF pinned to that commit so a
# later main checkout with a different model shape cannot be used by accident.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PODS="${PODS:-testa testaa}"
export BRANCH="${BRANCH:-main}"
export GIT_REF="${GIT_REF:-4b65c615dbd0f33a626203352ae913ed7747c12d}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_latest_main_fresh}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_20260601}"
export SESSION="${SESSION:-catk-smart-ntp-a100x4x2-fresh-main-resume-20260601}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export INITIAL_BS="${INITIAL_BS:-13}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export FRESH_START="${FRESH_START:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-2}"

WANDB_RUN_ID="${WANDB_RUN_ID:-1iapr5ed}"

RESUME_OVERRIDES=("data.train_use_eval_agent_selection=false")
if [[ -n "${WANDB_RUN_ID}" ]]; then
  RESUME_OVERRIDES+=("logger.wandb.id=${WANDB_RUN_ID}")
fi

if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES} ${RESUME_OVERRIDES[*]}"
else
  export EXTRA_HYDRA_OVERRIDES="${RESUME_OVERRIDES[*]}"
fi

exec bash "${SCRIPT_DIR}/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh" "$@"
