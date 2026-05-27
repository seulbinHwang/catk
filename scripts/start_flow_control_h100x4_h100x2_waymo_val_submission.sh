#!/usr/bin/env bash
set -Eeuo pipefail

# Generate and upload a full Waymo Sim Agents validation submission on the
# existing hsb-npc-training / wo-pvc-2 H100 4+2 pods.
#
# This script only runs commands inside existing pods. It does not create,
# delete, restart, or otherwise manage pods.
#
# Required:
#   CKPT_PATH=/path/to/epoch_last.ckpt
#
# Recommended:
#   TASK_NAME=unique_task_name_for_this_submission

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
CKPT_PATH="${CKPT_PATH:-}"
TASK_NAME="${TASK_NAME:-flow_control_waymo_val_h100x4_h100x2_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-catk-flow-waymo-val-submission-h100x4-h100x2}"
RUN_ID="${RUN_ID:-${CATK_RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
LOG_DIR="${LOG_DIR:-/workspace/exp_logs}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
WAYMO_FLOW_SAMPLE_STEPS="${WAYMO_FLOW_SAMPLE_STEPS:-16}"
POLL_SUBMISSION_STATUS="${POLL_SUBMISSION_STATUS:-false}"
WAYMO_UPLOAD_TIMEOUT_MS="${WAYMO_UPLOAD_TIMEOUT_MS:-7200000}"
GIT_REF="${GIT_REF:-origin/semi_control_stable}"
NO_PULL="${NO_PULL:-false}"
MASTER_PORT="${MASTER_PORT:-29651}"
CHECKPOINT_SYNC_PORT="${CHECKPOINT_SYNC_PORT:-29652}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: CKPT_PATH is required." >&2
  echo "Example:" >&2
  echo "  CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/flow_run/epoch_061/epoch_last.ckpt \\" >&2
  echo "  TASK_NAME=flow_control_waymo_val_epoch061_h100x4_h100x2 \\" >&2
  echo "  bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission.sh" >&2
  exit 2
fi

# hsb-npc-training and wo-pvc-2 use pod-local filesystems. Stream non-master
# submission shards to rank 0 before creating the Waymo tar.gz archive.
export CATK_SUBMISSION_STREAM_SHARDS="${CATK_SUBMISSION_STREAM_SHARDS:-1}"
export CATK_SUBMISSION_SHARD_STREAM_PORT="${CATK_SUBMISSION_SHARD_STREAM_PORT:-29653}"
export CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS="${CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS:-16}"
export CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL="${CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL:-1}"

extra_overrides=(
  "++trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy"
  "++trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment"
  "waymo_submission.enabled=true"
  "waymo_submission.submit_validate=true"
  "waymo_submission.submit_test=false"
  "waymo_submission.poll_submission_status=${POLL_SUBMISSION_STATUS}"
  "waymo_submission.upload_timeout_ms=${WAYMO_UPLOAD_TIMEOUT_MS}"
  "model.model_config.validation_rollout_sampling.sample_steps=${WAYMO_FLOW_SAMPLE_STEPS}"
  "logger.wandb.job_type=waymo_validation_submission"
)

if [[ -n "${WAYMO_STORAGE_STATE_PATH:-}" ]]; then
  extra_overrides+=("waymo_submission.storage_state_path=${WAYMO_STORAGE_STATE_PATH}")
fi

if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides+=(${EXTRA_HYDRA_OVERRIDES})
fi

printf -v extra_hydra_string '%q ' "${extra_overrides[@]}"

launcher_args=(
  --replace
  --project-root "${PROJECT_ROOT}"
  --branch semi_control_stable
  --pods hsb-npc-training wo-pvc-2
  --action validate
  --experiment sim_agents_sub_flow
  --task-name "${TASK_NAME}"
  --session "${SESSION}"
  --run-id "${RUN_ID}"
  --log-dir "${LOG_DIR}"
  --ckpt-path "${CKPT_PATH}"
  --nproc-per-node gpu
  --manual-rank-offsets
  --val-batch-size "${VAL_BATCH_SIZE}"
  --limit-val-batches "${LIMIT_VAL_BATCHES}"
  --master-port "${MASTER_PORT}"
  --checkpoint-sync-port "${CHECKPOINT_SYNC_PORT}"
  --pod-cache-root hsb-npc-training=/workspace/womd_v1_3/SMART_cache
  --pod-cache-root wo-pvc-2=/workspace/womd_v1_3/SMART_cache
  --extra-hydra-overrides "${extra_hydra_string}"
)

if [[ "${NO_PULL}" == "true" || "${NO_PULL}" == "1" ]]; then
  launcher_args+=(--no-pull)
else
  launcher_args+=(--git-ref "${GIT_REF}")
fi

python scripts/launch_h100x4_multinode_pretrain_tmux.py "${launcher_args[@]}" "$@"
