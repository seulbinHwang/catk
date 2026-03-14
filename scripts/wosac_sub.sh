#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION=validate # validate, test
MY_EXPERIMENT="wosac_sub"
MY_TASK_NAME="smart_flow_7m_wosac_${ACTION}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=$ACTION \
  task_name=$MY_TASK_NAME

