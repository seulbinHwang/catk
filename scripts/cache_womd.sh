#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

SPLIT="${1:-${DATA_SPLIT:-validation}}"
RAW_ROOT="${2:-${RAW_ROOT:-}}"
CACHE_ROOT="${3:-${CACHE_ROOT:-}}"
NUM_WORKERS="${NUM_WORKERS:-12}"

if [[ -z "$RAW_ROOT" || -z "$CACHE_ROOT" ]]; then
  echo "Usage: bash scripts/cache_womd.sh [training|validation|testing] <raw_root> <cache_root>"
  echo "or set RAW_ROOT and CACHE_ROOT env vars."
  exit 1
fi

case "$SPLIT" in
  training|validation|testing) ;;
  *)
    echo "Invalid split: $SPLIT (expected one of: training, validation, testing)"
    exit 1
    ;;
esac

python \
  -m src.data_preprocess \
  --split "$SPLIT" \
  --num_workers "$NUM_WORKERS" \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"
