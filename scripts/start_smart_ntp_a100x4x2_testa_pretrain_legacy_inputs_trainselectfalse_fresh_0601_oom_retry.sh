#!/usr/bin/env bash
# Start a fresh SMART NTP A100x4x2 pretrain from latest origin/main with a
# stable task name ending in "_0601".
#
# This is a thin wrapper over
# start_smart_ntp_a100x4x2_testa_pretrain_legacy_inputs_trainselectfalse_fresh_oom_retry.sh.
# It keeps the same training recipe but fixes the task/run suffix so the W&B
# experiment name is easy to identify.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_STAMP="${RUN_STAMP:-0601}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_0601}"
export SESSION="${SESSION:-catk-smart-ntp-a100x4x2-fresh-main-0601}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_latest_main_fresh_0601}"

exec bash "${SCRIPT_DIR}/start_smart_ntp_a100x4x2_testa_pretrain_legacy_inputs_trainselectfalse_fresh_oom_retry.sh" "$@"
