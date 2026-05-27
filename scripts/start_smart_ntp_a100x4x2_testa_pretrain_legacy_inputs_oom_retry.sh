#!/usr/bin/env bash
# Re-run the measured SMART NTP A100x4x2 OOM-retry pretrain after the
# 5a31008 legacy-input revert.
#
# This intentionally keeps the same training recipe as
# smart_ntp_pretrain_a100x4x2_bs16_oom_retry_main_20260523, except that this
# wrapper starts from train batch size 14 and records "original" in the task:
#   - testa + testaa
#   - pre_bc_a100x4x2
#   - INITIAL_BS=14, OOM_STEP=1, MIN_BS=8
#   - VAL_BATCH_SIZE=12, TEST_BATCH_SIZE=12
#   - no gradient accumulation
#
# The only intended model/data-path difference is the current main code that
# includes 5a31008, which folds SMART inputs back to the older main feature set.
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

exec bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh "$@"
