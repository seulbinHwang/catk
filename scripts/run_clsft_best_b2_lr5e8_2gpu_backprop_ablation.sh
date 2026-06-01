#!/usr/bin/env bash
# 2-GPU wrapper for the current best CLSFT/DMD b2 setting.
#
# It delegates to run_clsft_best_b2_lr5e8_valb_fallback.sh while pinning the
# run to two GPUs and exposing the training ODE gradient horizon through
# BACKPROP_LAST_K or SAMPLING_RTS_BACKPROP_LAST_K.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BACKPROP_LAST_K="${BACKPROP_LAST_K:-${SAMPLING_RTS_BACKPROP_LAST_K:-8}}"
case "${BACKPROP_LAST_K}" in
  ''|*[!0-9]*)
    echo "ERROR: BACKPROP_LAST_K must be a positive integer, got '${BACKPROP_LAST_K}'" >&2
    exit 1
    ;;
esac
if [[ "${BACKPROP_LAST_K}" -lt 1 ]]; then
  echo "ERROR: BACKPROP_LAST_K must be >= 1, got '${BACKPROP_LAST_K}'" >&2
  exit 1
fi

stamp="$(TZ=Asia/Seoul date +%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NUM_NODES="${NUM_NODES:-1}"

export CONDA_SH="${CONDA_SH:-/mnt/nuplan/miniforge/etc/profile.d/conda.sh}"
export CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"

export WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
export WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CLEAR_WANDB_API_KEY="${CLEAR_WANDB_API_KEY:-true}"
export WANDB_TAGS="${WANDB_TAGS:-[clsft,best_b2,2gpu,backprop_last_k_${BACKPROP_LAST_K}]}"

export TRAIN_B="${TRAIN_B:-2}"
export VAL_B_CANDIDATES="${VAL_B_CANDIDATES:-16 8 4}"
export LR="${LR:-5.0e-8}"
export ESTIMATOR_LR="${ESTIMATOR_LR:-5.0e-8}"
export DMD_BETA="${DMD_BETA:-1.0}"
export SF_UNFROZEN_RANGE="${SF_UNFROZEN_RANGE:-except_map_encoder}"
export SF_N_ROLLOUTS="${SF_N_ROLLOUTS:-1}"
export SF_N_ANCHORS="${SF_N_ANCHORS:-1}"
export ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-3}"
export ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"
export ESTIMATOR_WARMUP_STEPS="${ESTIMATOR_WARMUP_STEPS:-0}"

export SAMPLING_SAMPLE_STEPS="${SAMPLING_SAMPLE_STEPS:-16}"
export SAMPLING_SAMPLE_METHOD="${SAMPLING_SAMPLE_METHOD:-euler}"
export SAMPLING_RTS_POLICY="${SAMPLING_RTS_POLICY:-all}"
export SAMPLING_RTS_MIN_EXECUTED_STEPS="${SAMPLING_RTS_MIN_EXECUTED_STEPS:-${SAMPLING_SAMPLE_STEPS}}"
export SAMPLING_RTS_BACKPROP_LAST_K="${BACKPROP_LAST_K}"

export TASK_PREFIX="${TASK_PREFIX:-auto_clsft_best_b2_lr5e8_2gpu_bp${BACKPROP_LAST_K}_valbfallback_${stamp}}"

echo "[clsft-2gpu-backprop-ablation] TASK_PREFIX=${TASK_PREFIX}"
echo "[clsft-2gpu-backprop-ablation] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[clsft-2gpu-backprop-ablation] SAMPLING_RTS_BACKPROP_LAST_K=${SAMPLING_RTS_BACKPROP_LAST_K}"

exec bash scripts/run_clsft_best_b2_lr5e8_valb_fallback.sh
