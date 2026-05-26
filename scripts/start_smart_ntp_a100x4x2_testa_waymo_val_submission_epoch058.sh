#!/usr/bin/env bash
set -Eeuo pipefail

# Generate the full Waymo Sim Agents validation submission with the epoch-58
# SMART NTP checkpoint, then upload it through the repository's Waymo uploader.
#
# This script only runs commands inside existing pods. It does not create,
# delete, or restart pods.

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_oom_retry_main_20260523}"
CKPT_PATH="${CKPT_PATH:-/mnt/nuplan/projects/catk/checkpoints/smart_ntp_rmm_sweep_rj5nc4v1/epoch_057.ckpt}"
TASK_NAME="${TASK_NAME:-smart_ntp_waymo_val_epoch058_a100x4x2_main}"
SESSION="${SESSION:-catk-smart-ntp-waymo-val-submission-a100x4x2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
POLL_SUBMISSION_STATUS="${POLL_SUBMISSION_STATUS:-false}"
GIT_REF="${GIT_REF:-origin/main}"
NO_PULL="${NO_PULL:-false}"

extra_overrides=(
  "waymo_submission.enabled=true"
  "waymo_submission.submit_validate=true"
  "waymo_submission.submit_test=false"
  "waymo_submission.poll_submission_status=${POLL_SUBMISSION_STATUS}"
  "logger.wandb.job_type=waymo_validation_submission"
)

if [[ -n "${WAYMO_STORAGE_STATE_PATH:-}" ]]; then
  extra_overrides+=("waymo_submission.storage_state_path=${WAYMO_STORAGE_STATE_PATH}")
fi

printf -v extra_hydra_string '%q ' "${extra_overrides[@]}"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  extra_hydra_string+="${EXTRA_HYDRA_OVERRIDES}"
fi

launcher_args=(
  --replace \
  --project-root "${PROJECT_ROOT}" \
  --action validate \
  --experiment wosac_sub \
  --task-name "${TASK_NAME}" \
  --session "${SESSION}" \
  --ckpt-path "${CKPT_PATH}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --test-batch-size "${TEST_BATCH_SIZE}" \
  --limit-val-batches "${LIMIT_VAL_BATCHES}" \
  --extra-hydra-overrides "${extra_hydra_string}"
)

if [[ "${NO_PULL}" == "true" || "${NO_PULL}" == "1" ]]; then
  launcher_args+=(--no-pull)
else
  launcher_args+=(--git-ref "${GIT_REF}")
fi

python scripts/launch_smart_ntp_a100x4x2_testa.py "${launcher_args[@]}" "$@"
