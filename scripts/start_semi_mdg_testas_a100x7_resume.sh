#!/usr/bin/env bash
# Resume semi_mdg pretraining on the existing testas A100x7 pod.
#
# The checkpoint path must exist inside the pod. The first attempt resumes from
# RESUME_CKPT_PATH; after that, OOM retries prefer this task's latest
# epoch_last.ckpt so optimizer/scheduler state keeps advancing normally.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-/workspace/checkpoints/semi_mdg_resume/qplbq444_epoch_last_v45.ckpt}"
INITIAL_BS="${INITIAL_BS:-20}"
MIN_BS="${MIN_BS:-16}"
OOM_STEP="${OOM_STEP:-2}"
BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE:-108}"
BASE_LR="${BASE_LR:-0.0006}"
WANDB_MODE="${WANDB_MODE:-online}"
SESSION="${SESSION:-catk-semi-mdg-testas-a100x7}"
BRANCH="${BRANCH:-semi_mdg}"
CHECKOUT_REF="${CHECKOUT_REF:-770e3fb}"
TRAIN_SIDECAR_DIR="${TRAIN_SIDECAR_DIR:-/workspace/womd_v1_3/SMART_cache/semi_mdg_sidecar/training}"

STAMP="$(date +%Y%m%d_%H%M%S)"
CHECKOUT_TAG="$(printf "%s" "$CHECKOUT_REF" | tr -c 'A-Za-z0-9._-' '_')"
LR_TAG="$(awk -v base="$BASE_LR" -v bs="$INITIAL_BS" -v ref="$BASE_GLOBAL_BATCH_SIZE" 'BEGIN { lr = base * sqrt((bs * 7) / ref); printf "lr%.0fe-8", lr * 1e8 }')"
TASK_NAME="${TASK_NAME:-semi_mdg_resume_testas_a100x7_epoch44_${STAMP}_${CHECKOUT_TAG}_bs${INITIAL_BS}_${LR_TAG}}"

EXTRA_ARGS=()
if [[ -n "$TRAIN_SIDECAR_DIR" ]]; then
  EXTRA_ARGS+=(--train-sidecar-dir "$TRAIN_SIDECAR_DIR")
fi

python scripts/launch_mdg_testas_a100x7.py \
  --replace \
  --branch "$BRANCH" \
  --checkout-ref "$CHECKOUT_REF" \
  --session "$SESSION" \
  --task-name "$TASK_NAME" \
  --resume-ckpt-path "$RESUME_CKPT_PATH" \
  --initial-bs "$INITIAL_BS" \
  --min-bs "$MIN_BS" \
  --oom-step "$OOM_STEP" \
  --wandb-mode "$WANDB_MODE" \
  --auto-sqrt-lr \
  --base-lr "$BASE_LR" \
  --base-global-batch-size "$BASE_GLOBAL_BATCH_SIZE" \
  "${EXTRA_ARGS[@]}" \
  "$@"
