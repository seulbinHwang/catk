#!/usr/bin/env bash
# Convenience wrapper for paper-aligned goal-free RLFTSim on 4 x V100x2 pods.
#
# Required:
#   CKPT_PATH=/path/to/SMART_BC_PRETRAINED.ckpt
#
# Smoke example:
#   CKPT_PATH=/path/to/SMART_BC_PRETRAINED.ckpt \
#   LIMIT_TRAIN_BATCHES=8 LIMIT_VAL_BATCHES=1 \
#   bash scripts/start_smart_rlftsim_v100x2x4.sh
set -Eeuo pipefail

export PODS="${PODS:-svvvv-2-1 svvvv-2-2 svvvv-2-3 svvvv-2-4}"
export BRANCH="${BRANCH:-traktok-rlftsim}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_rlftsim_v100x2x4}"
export TASK_NAME="${TASK_NAME:-smart_rlftsim_v100x2x4_paper_hparams}"
export SESSION="${SESSION:-catk-smart-rlftsim-v100x2x4}"
export MASTER_PORT="${MASTER_PORT:-29561}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export CATK_ACTION="${CATK_ACTION:-rlftsim_finetune}"
export CATK_EXPERIMENT="${CATK_EXPERIMENT:-rlftsim}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# Paper RLFTSim values. Direct in-memory RLFTSim batch 8 OOMs on 32GB V100s
# because each sample carries four closed-loop rollouts. The preset therefore
# keeps the optimizer-effective per-process batch at 8 with microbatch 1 and
# RLFTSim-internal gradient accumulation 8. Override TRAIN_BATCH_SIZE=8
# ACCUMULATE_GRAD_BATCHES=1 only if the model/config fits in memory.
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-0}"

exec python scripts/launch_smart_rlftsim_v100x2x4.py --branch "$BRANCH" "$@"
