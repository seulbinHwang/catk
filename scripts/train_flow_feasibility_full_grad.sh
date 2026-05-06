#!/bin/sh

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-am_finetune_flow_feasibility_full_grad}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-a100}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

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
echo "CKPT_PATH=${CKPT_PATH}"
echo "TRAIN_B=${TRAIN_B} VAL_B=${VAL_B}"

PORT="$(get_free_port)"

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
