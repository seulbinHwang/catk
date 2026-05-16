#!/usr/bin/env bash
set -euo pipefail
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing

source "$(dirname "$0")/setup_runtime_env.sh"
WOMD_INPUT_DIR="${WOMD_INPUT_DIR:-/scratch/data/womd/uncompressed/scenario}"
python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir "$WOMD_INPUT_DIR" \
  --output_dir "$CACHE_ROOT"
