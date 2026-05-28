#!/usr/bin/env bash
# Re-run the measured SMART NTP A100x4x2 OOM-retry pretrain after the
# 5a31008 legacy-input revert.
#
# This intentionally keeps the same runtime recipe as
# smart_ntp_pretrain_a100x4x2_bs16_oom_retry_main_20260523, except that this
# wrapper starts from train batch size 14, records "original" in the task, and
# uses the legacy SMART training target selection:
#   - testa + testaa
#   - pre_bc_a100x4x2
#   - INITIAL_BS=14, OOM_STEP=1, MIN_BS=8
#   - VAL_BATCH_SIZE=12, TEST_BATCH_SIZE=12
#   - no gradient accumulation
#   - data.train_use_eval_agent_selection=false
#
# The intended model/data-path differences are the current main code that
# includes 5a31008, which folds SMART inputs back to the older main feature set,
# and the legacy train target selection used by d238-era SMART pretraining.
set -Eeuo pipefail

export PODS="${PODS:-testa testaa}"
export BRANCH="${BRANCH:-main}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_legacy_inputs_oom_retry_main}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs14_oom_retry_main_original_legacy_inputs}"
export SESSION="${SESSION:-catk-smart-ntp-a100x4x2-legacy-inputs}"
export INITIAL_BS="${INITIAL_BS:-14}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"

LEGACY_TARGET_SELECTION_OVERRIDE="data.train_use_eval_agent_selection=false"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES} ${LEGACY_TARGET_SELECTION_OVERRIDE}"
else
  export EXTRA_HYDRA_OVERRIDES="${LEGACY_TARGET_SELECTION_OVERRIDE}"
fi

exec bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh "$@"
