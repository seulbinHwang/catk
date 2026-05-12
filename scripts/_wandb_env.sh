#!/usr/bin/env sh

require_wandb_env() {
  WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
  WANDB_ENTITY_VALUE="${WANDB_ENTITY:-jksg01019-naver-labs}"
  WANDB_PROJECT_VALUE="${WANDB_PROJECT:-SMART-FLOW}"
  WANDB_BASE_URL_VALUE="${WANDB_BASE_URL:-https://api.wandb.ai}"

  if [ -z "${WANDB_API_KEY_VALUE}" ]; then
    echo "WANDB_API_KEY is required." >&2
    echo "Example:" >&2
    echo "  export WANDB_API_KEY=your_api_key" >&2
    echo "  export WANDB_ENTITY=${WANDB_ENTITY_VALUE}" >&2
    echo "  export WANDB_PROJECT=${WANDB_PROJECT_VALUE}" >&2
    exit 1
  fi

  export WANDB_API_KEY="${WANDB_API_KEY_VALUE}"
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
  export WANDB_PROJECT="${WANDB_PROJECT_VALUE}"
  export WANDB_BASE_URL="${WANDB_BASE_URL_VALUE}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  unset WANDB_DISABLED
}
