#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

FLOW_CKPT="${1:-${FLOW_CKPT:-}}"
FLOW_CKPT_ARTIFACT="${FLOW_CKPT_ARTIFACT:-}"
if [[ -z "$FLOW_CKPT" && -z "$FLOW_CKPT_ARTIFACT" ]]; then
  echo "Usage: bash scripts/wosac_sub_flow.sh <model_ckpt> [validate|test]"
  echo "or set FLOW_CKPT env var."
  echo "or set FLOW_CKPT_ARTIFACT='entity/project/artifact:alias'."
  exit 1
fi

ACTION="${2:-${ACTION:-validate}}"
case "$ACTION" in
  validate|test) ;;
  *)
    echo "Invalid action: $ACTION (expected: validate or test)"
    exit 1
    ;;
esac

EXPERIMENT="${EXPERIMENT:-flow_wosac_sub}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT-$ACTION}"
CACHE_ROOT="${CACHE_ROOT:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"

cmd=(
  python
  -m src.run
  experiment="$EXPERIMENT"
  action="$ACTION"
  logger.wandb.offline="$WANDB_OFFLINE"
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
