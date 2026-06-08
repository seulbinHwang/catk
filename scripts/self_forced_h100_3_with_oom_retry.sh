#!/usr/bin/env bash
# H100x3 single-pod DMD self-forcing wrapper.
#
# The implementation intentionally delegates to the H100x4 OOM-retry runner,
# which already supports arbitrary `NPROC_PER_NODE` and CUDA device lists via
# environment variables. Defaults here encode the hsb-npc-training-3-1 recipe.

set -uo pipefail

export EXPERIMENT="${EXPERIMENT:-self_forced_npfm_h100_3_hsb31}"
export TASK_NAME="${TASK_NAME:-flow_self_forced_dmd_h100x3_hsb31}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export INITIAL_BS="${INITIAL_BS:-128}"
export OOM_STEP="${OOM_STEP:-16}"
export MIN_BS="${MIN_BS:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-16}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-2}"
export TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.25}"
export TRAIN_MEMORY_BALANCED_BATCHES="${TRAIN_MEMORY_BALANCED_BATCHES:-true}"
export CATK_LR="${CATK_LR:-1.0e-6}"
export ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-1}"
export SELF_FORCED_USE_STOP_MOTION="${SELF_FORCED_USE_STOP_MOTION:-false}"
export DECODER_USE_STOP_MOTION="${DECODER_USE_STOP_MOTION:-false}"
export UNFROZEN_RANGE="${UNFROZEN_RANGE:-middle}"
export RANDOM_TERMINAL_SCOPE="${RANDOM_TERMINAL_SCOPE:-global_batch}"
export RANDOM_TERMINAL_POLICY="${RANDOM_TERMINAL_POLICY:-all}"
export BACKPROP_LAST_K="${BACKPROP_LAST_K:-8}"
export ESTIMATOR_WARMUP_BANK_ENABLED="${ESTIMATOR_WARMUP_BANK_ENABLED:-true}"
export ESTIMATOR_WARMUP_BANK_ARTIFACT="${ESTIMATOR_WARMUP_BANK_ARTIFACT:-generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6:latest}"
export ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME="${ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME:-generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6}"

exec bash scripts/self_forced_h100_4_with_oom_retry.sh "$@"
