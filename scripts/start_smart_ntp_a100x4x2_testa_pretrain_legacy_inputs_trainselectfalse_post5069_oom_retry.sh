#!/usr/bin/env bash
# Successor run for
# smart_ntp_pretrain_a100x4x2_bs14_oom_retry_main_original_legacy_inputs_trainselectfalse_20260528.
#
# This wrapper keeps the original legacy-inputs/trainselectfalse A100x4x2
# comparison recipe, but pins the training checkout to main@5069a44, i.e. with
# the 3406b070..5069a44 changes applied:
#   - pre_bc_a100x4x2
#   - testa + testaa
#   - CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
#   - INITIAL_BS=13, OOM_STEP=1, MIN_BS=8
#   - VAL_BATCH_SIZE=12, TEST_BATCH_SIZE=12
#   - data.train_use_eval_agent_selection=false
#   - num_freq_bands=88
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PODS="${PODS:-testa testaa}"
export BRANCH="${BRANCH:-main}"
export GIT_REF="${GIT_REF:-5069a44}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_legacy_inputs_trainselectfalse_post5069_main}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_post5069_20260531}"
export SESSION="${SESSION:-catk-smart-ntp-a100x4x2-legacy-inputs-post5069}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export INITIAL_BS="${INITIAL_BS:-13}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

LEGACY_TARGET_SELECTION_OVERRIDE="data.train_use_eval_agent_selection=false"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES} ${LEGACY_TARGET_SELECTION_OVERRIDE}"
else
  export EXTRA_HYDRA_OVERRIDES="${LEGACY_TARGET_SELECTION_OVERRIDE}"
fi

exec bash "${SCRIPT_DIR}/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh" "$@"
