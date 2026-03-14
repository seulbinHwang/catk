#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="pre_bc"
MY_TASK_NAME=$MY_EXPERIMENT"-flow-h100x6"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=ddp \
  task_name=$MY_TASK_NAME

echo "bash train.sh done!"
