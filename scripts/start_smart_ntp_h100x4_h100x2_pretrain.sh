#!/usr/bin/env bash
# Start SMART NTP pretrain on hsb-npc-training(H100x4) + wo-pvc-1(H100x2).
#
# This is a thin convenience wrapper around launch_smart_ntp_h100x4_h100x2.py.
# It never creates, deletes, or restarts pods.
set -Eeuo pipefail

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-13}"
TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_h100x4_h100x2_bs${TRAIN_BATCH_SIZE}_main}"

python scripts/launch_smart_ntp_h100x4_h100x2.py \
  --replace \
  --task-name "$TASK_NAME" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --val-batch-size "$TRAIN_BATCH_SIZE" \
  --test-batch-size "$TRAIN_BATCH_SIZE" \
  "$@"
