#!/usr/bin/env bash
set -Eeuo pipefail

# Epoch-62 preset for the H100 4+2 Flow control-space Waymo validation
# submission launcher. Epoch is zero-based: this targets epoch 62 from the
# run that trained through epoch 63.
#
# CKPT_PATH is intentionally required because the epoch_last.ckpt location can
# be either a pod-local training log path or a W&B-downloaded artifact path.

if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "ERROR: CKPT_PATH must point to the epoch 62 epoch_last.ckpt file." >&2
  echo "Example:" >&2
  echo "  CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/flow_control_epoch062/epoch_last.ckpt \\" >&2
  echo "  bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission_epoch062.sh" >&2
  exit 2
fi

export TASK_NAME="${TASK_NAME:-flow_control_waymo_val_epoch062_h100x4_h100x2}"

exec bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission.sh "$@"
