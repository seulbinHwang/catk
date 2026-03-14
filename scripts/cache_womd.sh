#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-catk}"
DATA_SPLIT="${1:-${DATA_SPLIT:-training}}" # training, validation, testing
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_FILES="${MAX_FILES:-}"

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

detect_default_input_dir() {
  first_existing_dir \
    "${RAW_DATA_ROOT:-}" \
    "/workspace/womd_v1_3/scenario" \
    "/scratch/data/womd/uncompressed/scenario" \
    "$HOME/womd_v1_3/scenario" \
    || printf '%s\n' "$HOME/womd_v1_3/scenario"
}

detect_default_output_dir() {
  first_existing_parent \
    "${OUTPUT_DIR:-}" \
    "${SMART_CACHE_ROOT:-}" \
    "/workspace/womd_v1_3/SMART_cache" \
    "/scratch/cache/SMART" \
    "$HOME/womd_v1_3/cache/SMART" \
    "$(pwd)/data/cache/SMART" \
    || printf '%s\n' "$HOME/womd_v1_3/cache/SMART"
}

INPUT_DIR="${INPUT_DIR:-$(detect_default_input_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(detect_default_output_dir)}"
INPUT_SPLIT_DIR=""
TOTAL_INPUT_FILES=0

source "$(resolve_conda_profile)"
conda activate "$CONDA_ENV_NAME"

if [ "$(basename "$INPUT_DIR")" = "$DATA_SPLIT" ]; then
  INPUT_SPLIT_DIR="$INPUT_DIR"
  INPUT_DIR="$(dirname "$INPUT_DIR")"
else
  INPUT_SPLIT_DIR="$INPUT_DIR/$DATA_SPLIT"
fi

if [ ! -d "$INPUT_DIR" ] && [ ! -d "$INPUT_DIR/$DATA_SPLIT" ]; then
  printf '%s\n' "Input split directory not found. Set INPUT_DIR or RAW_DATA_ROOT. Tried: $INPUT_DIR/$DATA_SPLIT" >&2
  exit 1
fi

TOTAL_INPUT_FILES="$(find "$INPUT_SPLIT_DIR" -maxdepth 1 -type f | wc -l | tr -d '[:space:]')"

mkdir -p "$OUTPUT_DIR"
export SMART_CACHE_ROOT="$OUTPUT_DIR"

cache_done_marker="$OUTPUT_DIR/.${DATA_SPLIT}_cache_complete"
cache_partial_marker="$OUTPUT_DIR/.${DATA_SPLIT}_cache_partial"
rm -f "$cache_done_marker" "$cache_partial_marker"

args=(
  -m
  src.data_preprocess
  --split
  "$DATA_SPLIT"
  --num_workers
  "$NUM_WORKERS"
  --input_dir
  "$INPUT_DIR"
  --output_dir
  "$OUTPUT_DIR"
)

if [ -n "$MAX_FILES" ]; then
  args+=(
    --max_files
    "$MAX_FILES"
  )
fi

python "${args[@]}"

if [ -n "$MAX_FILES" ] && [ "$MAX_FILES" -lt "$TOTAL_INPUT_FILES" ]; then
  date --iso-8601=seconds > "$cache_partial_marker"
else
  date --iso-8601=seconds > "$cache_done_marker"
fi
