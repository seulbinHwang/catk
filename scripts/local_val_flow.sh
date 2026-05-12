#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val_flow"
MY_TASK_NAME="sim_agents_2025_local_val-debug"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
. "$(dirname "$0")/_activate_conda.sh"
. "$(dirname "$0")/_wandb_env.sh"
require_wandb_env

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME

echo "bash local_val_flow.sh done!"
