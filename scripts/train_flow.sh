#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="pre_bc_flow"
MY_TASK_NAME=$MY_EXPERIMENT"-sim_agents_2025-debug"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

torchrun \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME

echo "bash train_flow.sh done!"
