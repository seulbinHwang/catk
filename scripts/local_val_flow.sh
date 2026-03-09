#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
MY_EXPERIMENT="flow_local_val"
MY_TASK_NAME=$MY_EXPERIMENT
FLOW_CKPT=$1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  ckpt_path=$FLOW_CKPT \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME
