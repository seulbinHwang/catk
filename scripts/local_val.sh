#!/usr/bin/env bash
set -euo pipefail
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="mdg_pretrain"
MY_TASK_NAME="${TASK_NAME:-mdg-local-val-debug}"

source "$(dirname "$0")/setup_runtime_env.sh"
if [ -z "${CKPT_PATH:-}" ]; then
  echo "CKPT_PATH=/path/to/model.ckpt 를 지정하세요." >&2
  exit 1
fi

# local_val runs on single GPU
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=validate \
  ckpt_path="$CKPT_PATH" \
  paths.cache_root="$CACHE_ROOT" \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME

echo "bash local_val.sh done!"
