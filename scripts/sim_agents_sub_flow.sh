#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION=validate # validate or test
MY_EXPERIMENT="sim_agents_sub_flow"
MY_TASK_NAME="sim_agents_2025-$ACTION-debug"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
. "$(dirname "$0")/_activate_conda.sh"

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=$ACTION \
  task_name=$MY_TASK_NAME

echo "bash $ACTION done!"
