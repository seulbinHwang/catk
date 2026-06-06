#!/usr/bin/env bash
# A100x4 wrapper around the generic 4-GPU self-forced OOM retry runner.
# The underlying script is hardware-agnostic; this file only pins A100/testa
# defaults so callers do not need to remember H100-named script details.

set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-self_forced_npfm_a100x4_testa}"
export TASK_NAME="${TASK_NAME:-flow_self_forced_dmd_a100x4_testa}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export INITIAL_BS="${INITIAL_BS:-160}"
export OOM_STEP="${OOM_STEP:-8}"
export MIN_BS="${MIN_BS:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
export TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.25}"
export TRAIN_MEMORY_BALANCED_BATCHES="${TRAIN_MEMORY_BALANCED_BATCHES:-true}"

exec bash scripts/self_forced_h100_4_with_oom_retry.sh "$@"
