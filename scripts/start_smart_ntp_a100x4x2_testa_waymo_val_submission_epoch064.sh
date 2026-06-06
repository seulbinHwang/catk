#!/usr/bin/env bash
set -Eeuo pipefail

# Epoch-64 preset for the SMART NTP A100x4x2 full validation-set Waymo
# submission. User-facing epoch 64 is zero-based epoch 63 in Lightning logs.

TASK_SOURCE="${TASK_SOURCE:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_20260601}"
RUN_ID_SOURCE="${RUN_ID_SOURCE:-2026-06-05_08-20-25}"

export CKPT_PATH="${CKPT_PATH:-/mnt/nuplan/projects/catk/logs/${TASK_SOURCE}/runs/${RUN_ID_SOURCE}/checkpoints/epoch_last.ckpt}"
export TASK_NAME="${TASK_NAME:-smart_ntp_waymo_val_epoch064_a100x4x2_main_original_legacy_inputs_trainselectfalse}"
export SESSION="${SESSION:-catk-smart-ntp-waymo-val-submission-a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_latest_main_fresh}"
export METHOD_NAME="${METHOD_NAME:-SMART_7M}"
export SUBMISSION_DESCRIPTION="${SUBMISSION_DESCRIPTION:-smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_20260601}"
export WAYMO_STORAGE_STATE_PATH="${WAYMO_STORAGE_STATE_PATH:-/tmp/catk_smart_ntp_a100x4x2_legacy_inputs_oom_retry_main/secrets/waymo/waymo_storage_state.json}"
export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-model.model_config.decoder.num_freq_bands=88}"

exec bash scripts/start_smart_ntp_a100x4x2_testa_waymo_val_submission.sh "$@"
