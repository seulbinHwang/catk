#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val"
N_ROLLOUT=32
MY_TASK_NAME="smart_flow_7m_local_val"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  model.model_config.n_rollout_closed_val=$N_ROLLOUT \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME

