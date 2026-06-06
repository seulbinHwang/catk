#!/usr/bin/env bash
set -Eeuo pipefail

# Generate a full Waymo Sim Agents validation submission on the existing
# testa/testaa A100x4x2 pods, then upload it through the repository's Waymo
# uploader.
#
# This script only runs commands inside existing pods. It does not create,
# delete, or restart pods.
#
# Required:
#   CKPT_PATH=/path/to/checkpoint.ckpt
#
# Recommended:
#   TASK_NAME=unique_task_name_for_this_submission

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_oom_retry_main_20260523}"
CKPT_PATH="${CKPT_PATH:-}"
TASK_NAME="${TASK_NAME:-smart_ntp_waymo_val_a100x4x2_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-catk-smart-ntp-waymo-val-submission-a100x4x2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
POLL_SUBMISSION_STATUS="${POLL_SUBMISSION_STATUS:-false}"
WAYMO_UPLOAD_TIMEOUT_MS="${WAYMO_UPLOAD_TIMEOUT_MS:-7200000}"
GIT_REF="${GIT_REF:-origin/main}"
NO_PULL="${NO_PULL:-false}"
METHOD_NAME="${METHOD_NAME:-}"
SUBMISSION_DESCRIPTION="${SUBMISSION_DESCRIPTION:-}"
SUBMISSION_ACCOUNT_NAME="${SUBMISSION_ACCOUNT_NAME:-}"
SUBMISSION_AUTHORS="${SUBMISSION_AUTHORS:-}"
SUBMISSION_AFFILIATION="${SUBMISSION_AFFILIATION:-}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: CKPT_PATH is required." >&2
  echo "Example:" >&2
  echo "  CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/run/epoch_057.ckpt \\" >&2
  echo "  TASK_NAME=smart_ntp_waymo_val_epoch058_a100x4x2 \\" >&2
  echo "  bash scripts/start_smart_ntp_a100x4x2_testa_waymo_val_submission.sh" >&2
  exit 2
fi

# testa/testaa use the same path strings but not the same filesystem. Stream
# non-master-node submission shards to rank 0 before creating the Waymo archive.
export CATK_SUBMISSION_STREAM_SHARDS="${CATK_SUBMISSION_STREAM_SHARDS:-1}"
export CATK_SUBMISSION_SHARD_STREAM_PORT="${CATK_SUBMISSION_SHARD_STREAM_PORT:-29631}"
export CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS="${CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS:-16}"
export CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL="${CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL:-1}"

extra_overrides=(
  "waymo_submission.enabled=true"
  "waymo_submission.submit_validate=true"
  "waymo_submission.submit_test=false"
  "waymo_submission.poll_submission_status=${POLL_SUBMISSION_STATUS}"
  "waymo_submission.upload_timeout_ms=${WAYMO_UPLOAD_TIMEOUT_MS}"
  "logger.wandb.job_type=waymo_validation_submission"
)

if [[ -n "${WAYMO_STORAGE_STATE_PATH:-}" ]]; then
  extra_overrides+=("waymo_submission.storage_state_path=${WAYMO_STORAGE_STATE_PATH}")
fi
if [[ -n "${METHOD_NAME}" ]]; then
  extra_overrides+=("model.model_config.sim_agents_submission.method_name=${METHOD_NAME}")
fi
if [[ -n "${SUBMISSION_DESCRIPTION}" ]]; then
  extra_overrides+=("model.model_config.sim_agents_submission.description=${SUBMISSION_DESCRIPTION}")
  extra_overrides+=("submission.description=${SUBMISSION_DESCRIPTION}")
fi
if [[ -n "${SUBMISSION_ACCOUNT_NAME}" ]]; then
  extra_overrides+=("model.model_config.sim_agents_submission.account_name=${SUBMISSION_ACCOUNT_NAME}")
fi
if [[ -n "${SUBMISSION_AUTHORS}" ]]; then
  extra_overrides+=("model.model_config.sim_agents_submission.authors=${SUBMISSION_AUTHORS}")
fi
if [[ -n "${SUBMISSION_AFFILIATION}" ]]; then
  extra_overrides+=("model.model_config.sim_agents_submission.affiliation=${SUBMISSION_AFFILIATION}")
fi

printf -v extra_hydra_string '%q ' "${extra_overrides[@]}"
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  extra_hydra_string+="${EXTRA_HYDRA_OVERRIDES}"
fi

launcher_args=(
  --replace
  --project-root "${PROJECT_ROOT}"
  --action validate
  --experiment wosac_sub
  --task-name "${TASK_NAME}"
  --session "${SESSION}"
  --ckpt-path "${CKPT_PATH}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --test-batch-size "${TEST_BATCH_SIZE}"
  --limit-val-batches "${LIMIT_VAL_BATCHES}"
  --extra-hydra-overrides "${extra_hydra_string}"
)

if [[ -n "${RUN_ID:-}" ]]; then
  launcher_args+=(--run-id "${RUN_ID}")
elif [[ -n "${CATK_RUN_ID:-}" ]]; then
  launcher_args+=(--run-id "${CATK_RUN_ID}")
fi

if [[ "${NO_PULL}" == "true" || "${NO_PULL}" == "1" ]]; then
  launcher_args+=(--no-pull)
else
  launcher_args+=(--git-ref "${GIT_REF}")
fi

python scripts/launch_smart_ntp_a100x4x2_testa.py "${launcher_args[@]}" "$@"
