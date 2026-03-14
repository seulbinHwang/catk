#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-catk}"
MY_EXPERIMENT="${MY_EXPERIMENT:-pre_bc}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"

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

first_existing_dir() {
  local candidate=""
  for candidate in "$@"; do
    if [ -n "$candidate" ] && [ -d "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

first_existing_parent() {
  local candidate=""
  for candidate in "$@"; do
    if [ -n "$candidate" ] && { [ -d "$candidate" ] || [ -d "$(dirname "$candidate")" ]; }; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

detect_default_raw_data_root() {
  first_existing_dir \
    "${RAW_DATA_ROOT:-}" \
    "/workspace/womd_v1_3/scenario" \
    "/scratch/data/womd/uncompressed/scenario" \
    "$HOME/womd_v1_3/scenario" \
    || true
}

detect_default_cache_root() {
  first_existing_parent \
    "${SMART_CACHE_ROOT:-}" \
    "/workspace/womd_v1_3/SMART_cache" \
    "/scratch/cache/SMART" \
    "$HOME/womd_v1_3/cache/SMART" \
    "$(pwd)/data/cache/SMART" \
    || printf '%s\n' "$HOME/womd_v1_3/cache/SMART"
}

count_visible_gpus() {
  local gpu_ids=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    printf '%s\n' "${#gpu_ids[@]}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | awk 'END {print NR}'
    return 0
  fi

  printf '%s\n' 0
}

cache_ready() {
  local path="$1"
  local complete_marker="$2"
  local partial_marker="$3"

  [ -f "$partial_marker" ] && return 1
  [ -d "$path" ] && [ -n "$(find "$path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ] || return 1
  [ -f "$complete_marker" ] && return 0

  # Legacy cache created before marker support.
  return 0
}

source "$(resolve_conda_profile)"
conda activate "$CONDA_ENV_NAME"

RAW_DATA_ROOT="${RAW_DATA_ROOT:-$(detect_default_raw_data_root)}"
SMART_CACHE_ROOT="${SMART_CACHE_ROOT:-$(detect_default_cache_root)}"
export SMART_CACHE_ROOT

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [ "$VISIBLE_GPU_COUNT" -lt 1 ]; then
  printf '%s\n' "No visible GPUs found. Set CUDA_VISIBLE_DEVICES or check nvidia-smi." >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$VISIBLE_GPU_COUNT}"

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  MY_TASK_NAME="${MY_TASK_NAME:-smart_flow_7m_pre_bc}"
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-12}"
  PRECISION="${PRECISION:-bf16-mixed}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
  TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4}"
  NUM_WORKERS="${NUM_WORKERS:-10}"
  LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
  VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-}"
  VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-}"
else
  MY_TASK_NAME="${MY_TASK_NAME:-smart_flow_7m_pre_bc_1gpu}"
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
  PRECISION="${PRECISION:-16-mixed}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
  TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
  VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-false}"
  VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-false}"
  if [ -z "${WANDB_MODE:-}" ]; then
    export WANDB_MODE=offline
  fi
fi

if ! cache_ready \
  "$SMART_CACHE_ROOT/training" \
  "$SMART_CACHE_ROOT/.training_cache_complete" \
  "$SMART_CACHE_ROOT/.training_cache_partial"
then
  if [ -z "$RAW_DATA_ROOT" ] || { [ ! -d "$RAW_DATA_ROOT" ] && [ ! -d "$RAW_DATA_ROOT/training" ]; }; then
    printf '%s\n' "Training cache is missing and RAW_DATA_ROOT could not be found. Set RAW_DATA_ROOT or SMART_CACHE_ROOT." >&2
    exit 1
  fi
  INPUT_DIR="$RAW_DATA_ROOT" OUTPUT_DIR="$SMART_CACHE_ROOT" NUM_WORKERS="$CACHE_NUM_WORKERS" bash scripts/cache_womd.sh training
fi

if ! cache_ready \
  "$SMART_CACHE_ROOT/validation" \
  "$SMART_CACHE_ROOT/.validation_cache_complete" \
  "$SMART_CACHE_ROOT/.validation_cache_partial" \
  || ! cache_ready \
    "$SMART_CACHE_ROOT/validation_tfrecords_splitted" \
    "$SMART_CACHE_ROOT/.validation_cache_complete" \
    "$SMART_CACHE_ROOT/.validation_cache_partial"
then
  if [ -z "$RAW_DATA_ROOT" ] || { [ ! -d "$RAW_DATA_ROOT" ] && [ ! -d "$RAW_DATA_ROOT/validation" ]; }; then
    printf '%s\n' "Validation cache is missing and RAW_DATA_ROOT could not be found. Set RAW_DATA_ROOT or SMART_CACHE_ROOT." >&2
    exit 1
  fi
  INPUT_DIR="$RAW_DATA_ROOT" OUTPUT_DIR="$SMART_CACHE_ROOT" NUM_WORKERS="$CACHE_NUM_WORKERS" bash scripts/cache_womd.sh validation
fi

torch_args=(
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  -m
  src.run
  experiment="$MY_EXPERIMENT"
  task_name="$MY_TASK_NAME"
  logger="$LOGGER_NAME"
  paths.cache_root="$SMART_CACHE_ROOT"
  trainer.precision="$PRECISION"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.test_batch_size="$TEST_BATCH_SIZE"
  data.num_workers="$NUM_WORKERS"
)

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  torch_args+=(
    trainer=ddp
  )
else
  torch_args+=(
    trainer.devices=1
    trainer.strategy=auto
    data.pin_memory=false
    data.persistent_workers=false
  )
fi

if [ -n "$LIMIT_VAL_BATCHES" ]; then
  torch_args+=(
    trainer.limit_val_batches="$LIMIT_VAL_BATCHES"
  )
fi

if [ -n "$VAL_OPEN_LOOP" ]; then
  torch_args+=(
    model.model_config.val_open_loop="$VAL_OPEN_LOOP"
  )
fi

if [ -n "$VAL_CLOSED_LOOP" ]; then
  torch_args+=(
    model.model_config.val_closed_loop="$VAL_CLOSED_LOOP"
  )
fi

torch_args+=("$@")

torchrun "${torch_args[@]}"
