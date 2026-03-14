#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-catk}"
DATA_SPLIT="${1:-${DATA_SPLIT:-training}}" # training, validation, testing
INPUT_DIR="${INPUT_DIR:-$HOME/womd_v1_3/scenario}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/womd_v1_3/cache/SMART}"
NUM_WORKERS="${NUM_WORKERS:-4}"
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

source "$(resolve_conda_profile)"
conda activate "$CONDA_ENV_NAME"

if [ "$(basename "$INPUT_DIR")" = "$DATA_SPLIT" ]; then
  INPUT_DIR="$(dirname "$INPUT_DIR")"
fi

mkdir -p "$OUTPUT_DIR"
export SMART_CACHE_ROOT="$OUTPUT_DIR"

cache_done_marker="$OUTPUT_DIR/.${DATA_SPLIT}_cache_complete"
rm -f "$cache_done_marker"

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

if [ -z "$MAX_FILES" ]; then
  date --iso-8601=seconds > "$cache_done_marker"
fi
