#!/bin/sh
# =============================================================================
# Flow-RWR Fine-tuning 실행 스크립트
# =============================================================================
# 목적:
#   - 동일 초기 상태에서 G개 closed-loop rollout → GPU RMM 점수 (TFRecord 불필요)
#   - Reward-Weighted Regression: w^g = softmax(R^g/β) 로 weighted FM loss
#   - 후반 anchor일수록 γ^k 로 temporal discount (closed-loop error accumulation)
# 엔트리:
#   torchrun … -m src.run → Hydra configs/experiment/flow_rwr_ft.yaml + 아래 CLI override
# 오버라이드:
#   거의 모든 값은 쉘 환경변수로 바꿀 수 있음 (예: TRAIN_B=16 sh scripts/train_flow_rwr_ft.sh)
# =============================================================================

# -----------------------------------------------------------------------------
# 로그 / 경고 / 스레드
# -----------------------------------------------------------------------------
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# -----------------------------------------------------------------------------
# 실험 식별 / Conda
# -----------------------------------------------------------------------------
MY_EXPERIMENT="${MY_EXPERIMENT:-flow_rwr_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-a100-rwrft}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# -----------------------------------------------------------------------------
# 데이터 / 체크포인트 경로
# -----------------------------------------------------------------------------
# Training split: 487k scenarios (training PKL dir), no tfrecords needed
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

# -----------------------------------------------------------------------------
# Trainer / DataLoader
# -----------------------------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"

TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-8}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

# -----------------------------------------------------------------------------
# 옵티마이저 / 스케줄
# -----------------------------------------------------------------------------
LR="${LR:-5e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"
N_VIS_BATCH="${N_VIS_BATCH:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

# -----------------------------------------------------------------------------
# Flow-RWR 전용 (model.model_config.finetune.*)
# -----------------------------------------------------------------------------
# G: 시나리오당 closed-loop rollout 수 (4 이상 권장)
RWR_N_ROLLOUTS="${RWR_N_ROLLOUTS:-4}"
# β: softmax temperature. 낮을수록 최고 reward rollout에 집중 (0.05~0.2 권장)
RWR_BETA="${RWR_BETA:-0.1}"
# MC samples for FM ELBO log-prob estimation
RWR_N_SAMPLES="${RWR_N_SAMPLES:-8}"
# γ: temporal discount per anchor step (1.0=uniform, 0.9=10% discount per step)
RWR_ANCHOR_DISCOUNT="${RWR_ANCHOR_DISCOUNT:-0.9}"
# true → residual_velocity_head만 학습 (trunk 동결, zero-initialized)
RWR_HEAD_ONLY="${RWR_HEAD_ONLY:-false}"

# -----------------------------------------------------------------------------
# W&B
# -----------------------------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

# -----------------------------------------------------------------------------
# Hydra 추가 인자
# -----------------------------------------------------------------------------
EXTRA_ARGS="${EXTRA_ARGS:-}"

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

echo "Experiment=${MY_EXPERIMENT}"
echo "Task=${MY_TASK_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CACHE_ROOT=${CACHE_ROOT}  (training split, no tfrecords)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TRAIN_B=${TRAIN_B} VAL_B=${VAL_B} NUM_WORKERS=${NUM_WORKERS}"
echo "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} MAX_EPOCHS=${MAX_EPOCHS}"
echo "RWR_N_ROLLOUTS=${RWR_N_ROLLOUTS} RWR_BETA=${RWR_BETA} RWR_N_SAMPLES=${RWR_N_SAMPLES}"
echo "RWR_ANCHOR_DISCOUNT=${RWR_ANCHOR_DISCOUNT} RWR_HEAD_ONLY=${RWR_HEAD_ONLY}"
echo "WANDB_MODE=${WANDB_MODE} WANDB_ENTITY=${WANDB_ENTITY}"

PORT="$(get_free_port)"
echo "==== Start training (flow_rwr_ft): train_batch_size=${TRAIN_B}, val_batch_size=${VAL_B} ===="

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRAD_CLIP_VAL}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.finetune.rollout_steps="${ROLLOUT_STEPS}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.rwr_n_rollouts="${RWR_N_ROLLOUTS}" \
  model.model_config.finetune.rwr_beta="${RWR_BETA}" \
  model.model_config.finetune.rwr_n_samples="${RWR_N_SAMPLES}" \
  model.model_config.finetune.rwr_anchor_discount="${RWR_ANCHOR_DISCOUNT}" \
  model.model_config.finetune.rwr_head_only="${RWR_HEAD_ONLY}" \
  ${EXTRA_ARGS}
