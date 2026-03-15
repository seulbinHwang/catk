#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
. "$(dirname "$0")/_activate_conda.sh"
python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir /scratch/data/womd/uncompressed/scenario \
  --output_dir /scratch/cache/SMART
