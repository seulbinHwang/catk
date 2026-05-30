#!/usr/bin/env bash
set -euo pipefail
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION=validate # validate, test
MY_EXPERIMENT="mdg_wosac_sub"
MY_TASK_NAME="${TASK_NAME:-$MY_EXPERIMENT-$ACTION-debug}"

source "$(dirname "$0")/setup_runtime_env.sh"
if [ -z "${CKPT_PATH:-}" ]; then
  echo "CKPT_PATH=/path/to/model.ckpt 를 지정하세요." >&2
  exit 1
fi

python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  ckpt_path="$CKPT_PATH" \
  paths.cache_root="$CACHE_ROOT" \
  action=$ACTION \
  task_name=$MY_TASK_NAME

# below is for training with ddp
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   action=$ACTION \
#   task_name=$MY_TASK_NAME

echo bash $ACTION done!
