#!/usr/bin/env bash
# Start the measured-safe MDG A100x4x2 pretrain on testa/testaa.
#
# This is a thin convenience wrapper around launch_smart_ntp_a100x4x2_testa.py.
# It never creates, deletes, or restarts pods.
set -Eeuo pipefail

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
TASK_NAME="${TASK_NAME:-mdg_wosac_pretrain_a100x4x2_bs${TRAIN_BATCH_SIZE}_main}"

python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --branch "${BRANCH:-MDG}" \
  --experiment "${EXPERIMENT:-mdg_pretrain}" \
  --cache-root "${CACHE_ROOT:-/workspace/womd_v1_3/MDG_cache}" \
  --task-name "$TASK_NAME" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  "$@"
