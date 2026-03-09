#!/usr/bin/env bash
set -euo pipefail

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION="${1:-${ACTION:-validate}}"
case "$ACTION" in
  validate|test) ;;
  *)
    echo "Invalid action: $ACTION (expected: validate or test)"
    exit 1
    ;;
esac

EXPERIMENT="${EXPERIMENT:-wosac_sub}"
TASK_NAME="${TASK_NAME:-$EXPERIMENT-$ACTION-debug}"
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

if [[ -n "$CACHE_ROOT" ]]; then
  cmd+=(paths.cache_root="$CACHE_ROOT")
fi

"${cmd[@]}"

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

echo "bash scripts/wosac_sub.sh $ACTION done!"
