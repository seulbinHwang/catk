#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT="${EXPERIMENT:-flow_pretrain_h1006}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
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
