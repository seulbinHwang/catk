#!/usr/bin/env bash
# Convenience wrapper for goal-free RLFTSim on the testas A100x7 pod.
#
# Required:
#   CKPT_PATH=/path/to/SMART_BC_PRETRAINED.ckpt
#
# Smoke example:
#   CKPT_PATH=/path/to/SMART_BC_PRETRAINED.ckpt \
#   LIMIT_TRAIN_BATCHES=8 LIMIT_VAL_BATCHES=1 \
#   bash scripts/start_smart_rlftsim_testas_a100x7.sh
set -Eeuo pipefail

export POD="${POD:-testas}"
export BRANCH="${BRANCH:-main}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_rlftsim_testas_a100x7}"
export TASK_NAME="${TASK_NAME:-smart_rlftsim_testas_a100x7_batch8_topk32_closedval}"
export SESSION="${SESSION:-catk-smart-rlftsim-testas-a100x7}"
export MASTER_PORT="${MASTER_PORT:-29571}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export CATK_ACTION="${CATK_ACTION:-rlftsim_finetune}"
export CATK_EXPERIMENT="${CATK_EXPERIMENT:-rlftsim}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"

export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"

exec python scripts/launch_smart_rlftsim_testas_a100x7.py --branch "$BRANCH" "$@"
