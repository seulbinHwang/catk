#!/usr/bin/env bash
set -euo pipefail
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val"
VAL_K=12
MY_TASK_NAME=$MY_EXPERIMENT-K$VAL_K"-debug"

source "$(dirname "$0")/setup_runtime_env.sh"
if [ -z "${CKPT_PATH:-}" ]; then
  echo "CKPT_PATH=/path/to/model.ckpt 를 지정하세요." >&2
  exit 1
fi

# local_val runs on single GPU
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  ckpt_path="$CKPT_PATH" \
  paths.cache_root="$CACHE_ROOT" \
  trainer=default \
  model.model_config.validation_rollout_sampling.num_k=$VAL_K \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME

echo "bash local_val.sh done!"
