#!/usr/bin/env bash
# Generate and optionally upload a full Waymo Sim Agents 2025 validation
# submission on the existing testa/testaa A100 x4 pods.
#
# Required:
#   CKPT_PATH=/path/to/checkpoint.ckpt
#
# This script only uses existing pods. It never creates, deletes, or restarts pods.
set -Eeuo pipefail

if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "ERROR: CKPT_PATH is required." >&2
  echo "Example:" >&2
  echo "  CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/run/epoch_last.ckpt \\" >&2
  echo "  TASK_NAME=smart_waymo_val_epoch063_a100x4x2 \\" >&2
  echo "  bash scripts/start_smart_a100x4x2_testa_waymo_val_submission.sh" >&2
  exit 2
fi

export ACTION="${ACTION:-validate}"
export EXPERIMENT="${EXPERIMENT:-sim_agents_sub}"
export TASK_NAME="${TASK_NAME:-smart_waymo_val_a100x4x2_$(date +%Y%m%d_%H%M%S)}"
export SESSION="${SESSION:-catk-smart-waymo-val-submission-a100x4x2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
export WANDB_MODE="${WANDB_MODE:-online}"

export CATK_SUBMISSION_STREAM_SHARDS="${CATK_SUBMISSION_STREAM_SHARDS:-1}"
export CATK_SUBMISSION_SHARD_STREAM_PORT="${CATK_SUBMISSION_SHARD_STREAM_PORT:-29631}"
export CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS="${CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS:-16}"
export CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL="${CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL:-1}"

POLL_SUBMISSION_STATUS="${POLL_SUBMISSION_STATUS:-false}"
WAYMO_UPLOAD_TIMEOUT_MS="${WAYMO_UPLOAD_TIMEOUT_MS:-7200000}"

extra_overrides=(
  "waymo_submission.enabled=true"
  "waymo_submission.submit_validate=true"
  "waymo_submission.submit_test=false"
  "waymo_submission.evaluation_set=validation"
  "waymo_submission.poll_submission_status=${POLL_SUBMISSION_STATUS}"
  "waymo_submission.upload_timeout_ms=${WAYMO_UPLOAD_TIMEOUT_MS}"
  "logger.wandb.job_type=waymo_validation_submission"
)

if [[ -n "${WAYMO_STORAGE_STATE_PATH:-}" ]]; then
  extra_overrides+=("waymo_submission.storage_state_path=${WAYMO_STORAGE_STATE_PATH}")
fi

printf -v extra_hydra_string '%q ' "${extra_overrides[@]}"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  extra_hydra_string+="${EXTRA_HYDRA_OVERRIDES}"
fi
export EXTRA_HYDRA_OVERRIDES="${extra_hydra_string}"

exec bash scripts/start_smart_a100x4x2_testa_pretrain.sh "$@"
