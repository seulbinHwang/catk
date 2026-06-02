#!/usr/bin/env bash
# Run one node of the SMART branch A100x4x2 pretrain.
#
# This script is launched inside each existing pod by
# start_smart_a100x4x2_testa_pretrain.sh. It does not create or delete pods.
set -Eeuo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
: "${CACHE_ROOT:?CACHE_ROOT must be set}"
: "${TASK_NAME:?TASK_NAME must be set}"
: "${NNODES:?NNODES must be set}"
: "${NPROC_PER_NODE:?NPROC_PER_NODE must be set}"
: "${NODE_RANK:?NODE_RANK must be set}"
: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:?MASTER_PORT must be set}"

cd "$PROJECT_ROOT"

for split_dir in training validation testing validation_tfrecords_splitted; do
  if [[ ! -d "$CACHE_ROOT/$split_dir" ]]; then
    echo "[SMART_A100X4X2] missing cache directory: $CACHE_ROOT/$split_dir" >&2
    exit 2
  fi
done

if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV:-catk}" || true
fi

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"

HYDRA_OVERRIDES=(
  "experiment=${EXPERIMENT:-pre_bc_a100x4x2}"
  "trainer=ddp"
  "action=${ACTION:-fit}"
  "task_name=${TASK_NAME}"
  "hydra.run.dir=${PROJECT_ROOT}/logs/${TASK_NAME}/runs/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
  "paths.cache_root=${CACHE_ROOT}"
  "trainer.devices=${NPROC_PER_NODE}"
  "trainer.num_nodes=${NNODES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE:-10}"
  "data.val_batch_size=${VAL_BATCH_SIZE:-12}"
  "data.test_batch_size=${TEST_BATCH_SIZE:-12}"
)

if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  HYDRA_OVERRIDES+=("trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  HYDRA_OVERRIDES+=("trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
fi
if [[ -n "${LIMIT_TEST_BATCHES:-}" ]]; then
  HYDRA_OVERRIDES+=("trainer.limit_test_batches=${LIMIT_TEST_BATCHES}")
fi
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  HYDRA_OVERRIDES+=("trainer.max_epochs=${MAX_EPOCHS}")
fi
if [[ -n "${CKPT_PATH:-}" ]]; then
  HYDRA_OVERRIDES+=("ckpt_path=${CKPT_PATH}")
fi
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_OVERRIDES_ARRAY=(${EXTRA_HYDRA_OVERRIDES})
  HYDRA_OVERRIDES+=("${EXTRA_OVERRIDES_ARRAY[@]}")
fi

echo "[SMART_A100X4X2] task=${TASK_NAME}"
echo "[SMART_A100X4X2] project_root=${PROJECT_ROOT}"
echo "[SMART_A100X4X2] cache_root=${CACHE_ROOT}"
echo "[SMART_A100X4X2] node_rank=${NODE_RANK}/${NNODES}, nproc_per_node=${NPROC_PER_NODE}"
echo "[SMART_A100X4X2] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[SMART_A100X4X2] overrides=${HYDRA_OVERRIDES[*]}"

exec torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  -m src.run \
  "${HYDRA_OVERRIDES[@]}"
