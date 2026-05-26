#!/usr/bin/env bash
set -Eeuo pipefail

# Epoch-58 preset for the generic testa/testaa Waymo validation submission
# launcher. This checkpoint is zero-based epoch 057, i.e. user-facing epoch 58.

export CKPT_PATH="${CKPT_PATH:-/mnt/nuplan/projects/catk/checkpoints/smart_ntp_rmm_sweep_rj5nc4v1/epoch_057.ckpt}"
export TASK_NAME="${TASK_NAME:-smart_ntp_waymo_val_epoch058_a100x4x2_main}"

exec bash scripts/start_smart_ntp_a100x4x2_testa_waymo_val_submission.sh "$@"
