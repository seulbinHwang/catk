#!/bin/sh
# OCSC (Open-Closed Self-Consistency) 파인튜닝 — single-GPU smoke / debug 프리셋.
#
# 사용법:
#   sh scripts/train_flow_consistency_bptt_single.sh
#   OCSC_N_ROLLOUTS=4 MAX_EPOCHS=5 sh scripts/train_flow_consistency_bptt_single.sh
#
# 핵심 토글:
#   OCSC_GT_TARGET=true   → GT 궤적을 consistency target 으로 사용
#   OCSC_GT_TARGET=false  → open-loop sample 을 target 으로 사용 (default)

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_consistency_bptt}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-single}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

# ── Single-GPU 스모크 기본값 ───────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_B="${TRAIN_B:-4}"
VAL_B="${VAL_B:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-25}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-817}"
LR="${LR:-1e-6}"

# ── OCSC 파라미터 ───────────────────────────────────────────────────────────
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-2}"
OCSC_ANCHOR_STRIDE="${OCSC_ANCHOR_STRIDE:-4}"
OCSC_PRED_MAX_STEPS="${OCSC_PRED_MAX_STEPS:-4}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_USE_MMD="${OCSC_USE_MMD:-true}"
OCSC_GT_TARGET="${OCSC_GT_TARGET:-false}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-1.0}"
OCSC_REL_DISP_WEIGHT="${OCSC_REL_DISP_WEIGHT:-0.0}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.0}"
OCSC_EVAL_HARD_RMM="${OCSC_EVAL_HARD_RMM:-false}"
OCSC_EVAL_HARD_RMM_INTERVAL="${OCSC_EVAL_HARD_RMM_INTERVAL:-10}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[WARN] CKPT_PATH not found: ${CKPT_PATH}"
  echo "       OCSC 는 사전학습 체크포인트로부터의 fine-tuning 이 표준입니다."
  echo "       체크포인트 없이 진행하려면 위 경고를 무시하세요."
fi

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

PORT="$(get_free_port)"
echo "[ocsc-single] experiment=${MY_EXPERIMENT} task=${MY_TASK_NAME}"
echo "             OCSC_N_ROLLOUTS=${OCSC_N_ROLLOUTS} OCSC_GT_TARGET=${OCSC_GT_TARGET}"
echo "             OCSC_LOSS_TYPE=${OCSC_LOSS_TYPE} use_mmd=${OCSC_USE_MMD}"
echo "             LR=${LR} TRAIN_B=${TRAIN_B} MAX_EPOCHS=${MAX_EPOCHS}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  model.model_config.lr="${LR}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_anchor_stride="${OCSC_ANCHOR_STRIDE}" \
  model.model_config.finetune.ocsc_pred_max_steps="${OCSC_PRED_MAX_STEPS}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_use_mmd="${OCSC_USE_MMD}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_rel_disp_weight="${OCSC_REL_DISP_WEIGHT}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.finetune.ocsc_eval_hard_rmm="${OCSC_EVAL_HARD_RMM}" \
  model.model_config.finetune.ocsc_eval_hard_rmm_interval="${OCSC_EVAL_HARD_RMM_INTERVAL}" \
  ${EXTRA_ARGS}

echo "[ocsc-single] done"
