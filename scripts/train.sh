#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT="${EXPERIMENT:-pre_bc}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT-debug}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAINER_DEVICES="${TRAINER_DEVICES:-$NPROC_PER_NODE}"
CACHE_ROOT="${CACHE_ROOT:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"

cmd=(
  torchrun
  --nproc_per_node="$NPROC_PER_NODE"
  -m src.run
  experiment="$EXPERIMENT"
  trainer.devices="$TRAINER_DEVICES"
  logger.wandb.offline="$WANDB_OFFLINE"
  logger.wandb.entity="$WANDB_ENTITY"
  task_name="$TASK_NAME"
)

if [[ -n "$CACHE_ROOT" ]]; then
  cmd+=(paths.cache_root="$CACHE_ROOT")
fi

"${cmd[@]}"

# ! below is for training with ddp
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

echo "bash scripts/train.sh done!"
