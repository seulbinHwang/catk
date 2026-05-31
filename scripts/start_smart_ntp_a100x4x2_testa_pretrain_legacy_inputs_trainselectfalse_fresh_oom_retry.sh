#!/usr/bin/env bash
# Start a fresh SMART NTP A100x4x2 pretrain from the latest main checkout.
#
# This is the safe fresh-run variant for testa + testaa:
#   - no fixed GIT_REF; pod checkouts follow origin/main
#   - CACHE_ROOT is explicitly fixed to /workspace/womd_v1_3/SMART_cache
#   - first attempt starts at train batch size 13
#   - the first attempt ignores old task-local checkpoints
#   - if the new run OOMs, later attempts resume from that new run's checkpoint
#   - data.train_use_eval_agent_selection=false
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export PODS="${PODS:-testa testaa}"
export BRANCH="${BRANCH:-main}"
export GIT_REF="${GIT_REF:-}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_latest_main_fresh}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_${RUN_STAMP}}"
export SESSION="${SESSION:-catk-smart-ntp-a100x4x2-fresh-main}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export INITIAL_BS="${INITIAL_BS:-13}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export FRESH_START="${FRESH_START:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

LEGACY_TARGET_SELECTION_OVERRIDE="data.train_use_eval_agent_selection=false"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES} ${LEGACY_TARGET_SELECTION_OVERRIDE}"
else
  export EXTRA_HYDRA_OVERRIDES="${LEGACY_TARGET_SELECTION_OVERRIDE}"
fi

exec bash "${SCRIPT_DIR}/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh" "$@"
