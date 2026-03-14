#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing
INPUT_DIR=/scratch/data/womd/uncompressed/scenario
OUTPUT_DIR=/scratch/cache/SMART
NUM_WORKERS=12

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers $NUM_WORKERS \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR

