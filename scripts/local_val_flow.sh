#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

FLOW_CKPT="${1:-${FLOW_CKPT:-}}"
if [[ -z "$FLOW_CKPT" ]]; then
  echo "Usage: bash scripts/local_val_flow.sh <model_ckpt>"
  echo "or set FLOW_CKPT env var."
  exit 1
fi

EXPERIMENT="${EXPERIMENT:-flow_local_val}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
CACHE_ROOT="${CACHE_ROOT:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"

cmd=(
  python
  -m src.run
  experiment="$EXPERIMENT"
  action=validate
  ckpt_path="$FLOW_CKPT"
  trainer=default
  trainer.accelerator=gpu
  trainer.devices="$TRAINER_DEVICES"
  trainer.strategy=auto
  logger.wandb.offline="$WANDB_OFFLINE"
  logger.wandb.entity="$WANDB_ENTITY"
  task_name="$TASK_NAME"
)

if [[ -n "$CACHE_ROOT" ]]; then
  cmd+=(paths.cache_root="$CACHE_ROOT")
fi

"${cmd[@]}"
