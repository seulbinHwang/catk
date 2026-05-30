#!/usr/bin/env bash
set -euo pipefail
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="mdg_pretrain"
MY_TASK_NAME="${TASK_NAME:-$MY_EXPERIMENT-debug}"

source "$(dirname "$0")/setup_runtime_env.sh"
torchrun \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  paths.cache_root="$CACHE_ROOT" \
  task_name=$MY_TASK_NAME

# below is for multi-node/multi-GPU training with torchrun + Lightning DDP
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME

echo "bash train.sh done!"
