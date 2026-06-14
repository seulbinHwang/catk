#!/usr/bin/env bash
# Launch SMART CAT-K/CLSFT fine-tuning on the static testas A100x7 pod
# from the zero-based epoch 21 checkpoint of
# smart_ntp_pretrain_a100x4x2_bs13_oom_retry_main_original_legacy_inputs_trainselectfalse_fresh_20260601.
#
# This wrapper refuses to start while testas has active GPU compute processes,
# unless ALLOW_BUSY_TESTAS=1 is set explicitly.
set -Eeuo pipefail

export POD="${POD:-testas}"
export BRANCH="${BRANCH:-main}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_clsft_testas_a100x7_epoch21}"
export TASK_NAME="${TASK_NAME:-smart_clsft_testas_a100x7_main_1iapr5ed_epoch21_bs10}"
export SESSION="${SESSION:-catk-smart-clsft-testas-a100x7-epoch21}"
export MASTER_PORT="${MASTER_PORT:-29573}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export CATK_ACTION="${CATK_ACTION:-finetune}"
export CATK_EXPERIMENT="${CATK_EXPERIMENT:-clsft}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"

export CKPT_ARTIFACT="${CKPT_ARTIFACT:-jksg01019-naver-labs/SMART-FLOW/epoch-last-1iapr5ed:v21}"
export CKPT_DOWNLOAD_DIR="${CKPT_DOWNLOAD_DIR:-/workspace/checkpoints/smart_clsft_epoch21_1iapr5ed_v21}"

export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-10}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-10}"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-2}"
export TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-false}"
export NUM_FREQ_BANDS="${NUM_FREQ_BANDS:-88}"
export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"
export WANDB_GROUP="${WANDB_GROUP:-smart_clsft_testas_a100x7_epoch21}"

launcher_args=(
  --branch "$BRANCH"
  --ckpt-artifact "$CKPT_ARTIFACT"
  --ckpt-download-dir "$CKPT_DOWNLOAD_DIR"
  --task-name "$TASK_NAME"
  --session "$SESSION"
  --master-port "$MASTER_PORT"
)

if [[ "${ALLOW_BUSY_TESTAS:-0}" != "1" ]]; then
  launcher_args+=(--require-idle-gpu)
fi

exec python scripts/launch_smart_clsft_testas_a100x7.py "${launcher_args[@]}" "$@"
