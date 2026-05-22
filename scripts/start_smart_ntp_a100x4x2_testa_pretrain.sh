#!/usr/bin/env bash
# Start the measured-safe SMART NTP A100x4x2 pretrain on testa/testaa.
#
# This is a thin convenience wrapper around launch_smart_ntp_a100x4x2_testa.py.
# It never creates, deletes, or restarts pods.
set -Eeuo pipefail

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs${TRAIN_BATCH_SIZE}_main}"

python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --task-name "$TASK_NAME" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --val-batch-size "$TRAIN_BATCH_SIZE" \
  --test-batch-size "$TRAIN_BATCH_SIZE" \
  "$@"
