#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-catk}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$HOME/womd_v1_3/scenario}"
SMART_CACHE_ROOT="${SMART_CACHE_ROOT:-$HOME/womd_v1_3/cache/SMART}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
MY_EXPERIMENT="${MY_EXPERIMENT:-pre_bc}"
MY_TASK_NAME="${MY_TASK_NAME:-smart_flow_7m_pre_bc_1gpu}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"
PRECISION="${PRECISION:-16-mixed}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-false}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-false}"

resolve_conda_profile() {
  local candidate=""

  if [ -n "${CONDA_EXE:-}" ]; then
    candidate="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh"
  do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  printf '%s\n' "Could not find conda.sh. Set CONDA_EXE or install Miniforge/Miniconda." >&2
  return 1
}

cache_ready() {
  local path="$1"
  local marker="$2"
  [ -f "$marker" ] && [ -d "$path" ] && [ -n "$(find "$path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]
}

source "$(resolve_conda_profile)"
conda activate "$CONDA_ENV_NAME"

export SMART_CACHE_ROOT

if ! cache_ready "$SMART_CACHE_ROOT/training" "$SMART_CACHE_ROOT/.training_cache_complete"; then
  INPUT_DIR="$RAW_DATA_ROOT" OUTPUT_DIR="$SMART_CACHE_ROOT" NUM_WORKERS="$CACHE_NUM_WORKERS" bash scripts/cache_womd.sh training
fi

if ! cache_ready "$SMART_CACHE_ROOT/validation" "$SMART_CACHE_ROOT/.validation_cache_complete" || ! cache_ready "$SMART_CACHE_ROOT/validation_tfrecords_splitted" "$SMART_CACHE_ROOT/.validation_cache_complete"; then
  INPUT_DIR="$RAW_DATA_ROOT" OUTPUT_DIR="$SMART_CACHE_ROOT" NUM_WORKERS="$CACHE_NUM_WORKERS" bash scripts/cache_womd.sh validation
fi

torchrun \
  --standalone \
  --nproc_per_node=1 \
  -m src.run \
  experiment="$MY_EXPERIMENT" \
  task_name="$MY_TASK_NAME" \
  logger="$LOGGER_NAME" \
  paths.cache_root="$SMART_CACHE_ROOT" \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.precision="$PRECISION" \
  trainer.limit_val_batches="$LIMIT_VAL_BATCHES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.test_batch_size="$TEST_BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  data.pin_memory=false \
  data.persistent_workers=false \
  model.model_config.val_open_loop="$VAL_OPEN_LOOP" \
  model.model_config.val_closed_loop="$VAL_CLOSED_LOOP" \
  "$@"

# 멀티 노드가 필요하면 아래 예시를 사용하면 된다.
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME
