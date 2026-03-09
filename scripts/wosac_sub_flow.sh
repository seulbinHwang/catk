#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
ACTION=${2:-validate}
MY_EXPERIMENT="flow_wosac_sub"
MY_TASK_NAME=$MY_EXPERIMENT-$ACTION
FLOW_CKPT=$1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=$ACTION \
  ckpt_path=$FLOW_CKPT \
  task_name=$MY_TASK_NAME
