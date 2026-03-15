#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION=validate # validate or test
MY_EXPERIMENT="sim_agents_sub_flow"
MY_TASK_NAME="sim_agents_2025-$ACTION-debug"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=$ACTION \
  task_name=$MY_TASK_NAME

echo "bash $ACTION done!"
