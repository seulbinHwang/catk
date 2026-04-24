#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3

MY_EXPERIMENT="pre_bc_flow"
MY_TASK_NAME=$MY_EXPERIMENT"-sim_agents_2025-debug"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CATK_CONDA_ENV"

torchrun \
  --nproc_per_node=2 \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer.devices=2 \
  task_name=$MY_TASK_NAME

echo "bash train_flow.sh done!"
