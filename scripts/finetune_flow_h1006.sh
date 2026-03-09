#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FLOW_CKPT="${1:-${FLOW_CKPT:-}}"
if [[ -z "$FLOW_CKPT" ]]; then
  echo "Usage: bash scripts/finetune_flow_h1006.sh <flow_pretrain_ckpt>"
  echo "or set FLOW_CKPT env var."
  exit 1
fi

EXPERIMENT="${EXPERIMENT:-flow_clsft_h1006}"
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
  ckpt_path="$FLOW_CKPT"
  trainer.devices="$TRAINER_DEVICES"
  logger.wandb.offline="$WANDB_OFFLINE"
  logger.wandb.entity="$WANDB_ENTITY"
  task_name="$TASK_NAME"
)

if [[ -n "$CACHE_ROOT" ]]; then
  cmd+=(paths.cache_root="$CACHE_ROOT")
fi

"${cmd[@]}"
