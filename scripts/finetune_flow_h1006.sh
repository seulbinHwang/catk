#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FLOW_CKPT="${1:-${FLOW_CKPT:-}}"
FLOW_CKPT_ARTIFACT="${FLOW_CKPT_ARTIFACT:-}"
if [[ -z "$FLOW_CKPT" && -z "$FLOW_CKPT_ARTIFACT" ]]; then
  echo "Usage: bash scripts/finetune_flow_h1006.sh <flow_pretrain_ckpt>"
  echo "or set FLOW_CKPT env var."
  echo "or set FLOW_CKPT_ARTIFACT='entity/project/artifact:alias'."
  exit 1
fi

EXPERIMENT="${EXPERIMENT:-flow_clsft_h1006}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
TRAINER_DEVICES="${TRAINER_DEVICES:-$NPROC_PER_NODE}"
CACHE_ROOT="${CACHE_ROOT:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-SMART-FLOW}"
WANDB_ENTITY="${WANDB_ENTITY:-jksg01019-naver-labs}"

if [[ -n "${WANDB_OFFLINE:-}" ]]; then
  _wandb_offline="$WANDB_OFFLINE"
elif [[ "$WANDB_MODE" == "offline" || "$WANDB_MODE" == "disabled" ]]; then
  _wandb_offline="True"
else
  _wandb_offline="False"
fi

cmd=(
  torchrun
  --nproc_per_node="$NPROC_PER_NODE"
  -m src.run
  experiment="$EXPERIMENT"
  trainer.devices="$TRAINER_DEVICES"
  logger.wandb.offline="$_wandb_offline"
  logger.wandb.project="$WANDB_PROJECT"
  logger.wandb.entity="$WANDB_ENTITY"
  task_name="$TASK_NAME"
)

if [[ -n "$FLOW_CKPT_ARTIFACT" ]]; then
  cmd+=(ckpt_artifact="$FLOW_CKPT_ARTIFACT")
else
  cmd+=(ckpt_path="$FLOW_CKPT")
fi

if [[ -n "$CACHE_ROOT" ]]; then
  cmd+=(paths.cache_root="$CACHE_ROOT")
fi

"${cmd[@]}"
