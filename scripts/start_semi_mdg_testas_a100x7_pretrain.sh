#!/usr/bin/env bash
# Start semi_mdg pretraining on the existing testas A100x7 pod.
#
# Defaults are the latest conservative testas probe result:
#   per-GPU train batch 20, global batch 140, sqrt-scaled LR 0.00068313.
# OOM retry lowers train batch by 2 and recomputes LR from the same sqrt rule.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INITIAL_BS="${INITIAL_BS:-20}"
MIN_BS="${MIN_BS:-16}"
OOM_STEP="${OOM_STEP:-2}"
BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE:-108}"
BASE_LR="${BASE_LR:-0.0006}"
WANDB_MODE="${WANDB_MODE:-online}"
SESSION="${SESSION:-catk-semi-mdg-testas-a100x7}"
BRANCH="${BRANCH:-semi_mdg}"

SHORT_SHA="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LR_TAG="$(awk -v base="$BASE_LR" -v bs="$INITIAL_BS" -v ref="$BASE_GLOBAL_BATCH_SIZE" 'BEGIN { lr = base * sqrt((bs * 7) / ref); printf "lr%.0fe-8", lr * 1e8 }')"
TASK_NAME="${TASK_NAME:-semi_mdg_pretrain_testas_a100x7_from_scratch_${STAMP}_${SHORT_SHA}_bs${INITIAL_BS}_${LR_TAG}}"

python scripts/launch_mdg_testas_a100x7.py \
  --replace \
  --branch "$BRANCH" \
  --session "$SESSION" \
  --task-name "$TASK_NAME" \
  --initial-bs "$INITIAL_BS" \
  --min-bs "$MIN_BS" \
  --oom-step "$OOM_STEP" \
  --wandb-mode "$WANDB_MODE" \
  --auto-sqrt-lr \
  --base-lr "$BASE_LR" \
  --base-global-batch-size "$BASE_GLOBAL_BATCH_SIZE" \
  "$@"
