#!/bin/sh

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

# A100 2장 전용
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-am_finetune_flow_reward_final_step}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-a100-rewardfinalstep}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

# miniforge/miniconda 셸 초기화 스크립트가 없을 수 있으므로 존재할 때만 활성화합니다.
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi

# torchrun 등 학습 의존 라이브러리를 포함한 환경 활성화 시도
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

# 전체 데이터셋 사용 (limit_train_batches=1.0)
# batch_size=24: 유저 확인된 안정 베이스라인 (num_workers=24와 함께 ~3h 경험치)
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
TRAIN_B="${TRAIN_B:-24}"
VAL_B="${VAL_B:-$((TRAIN_B / 2))}"
if [ "$VAL_B" -lt 1 ]; then VAL_B=1; fi
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

get_free_port() {
  # 사용되지 않은 로컬 포트를 하나 뽑습니다.
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

echo "Experiment=${MY_EXPERIMENT}"
echo "Task=${MY_TASK_NAME}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TRAIN_B=${TRAIN_B} VAL_B=${VAL_B}"
echo "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} MAX_EPOCHS=${MAX_EPOCHS} LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}"
echo "WANDB_MODE=${WANDB_MODE} WANDB_ENTITY=${WANDB_ENTITY}"
echo "NUM_WORKERS=${NUM_WORKERS} PREFETCH_FACTOR=${PREFETCH_FACTOR} PERSISTENT_WORKERS=${PERSISTENT_WORKERS} PIN_MEMORY=${PIN_MEMORY}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"

PORT="$(get_free_port)"
echo "==== Start training: train_batch_size=${TRAIN_B}, val_batch_size=${VAL_B} ===="

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
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
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  logger.wandb.entity="${WANDB_ENTITY}"
