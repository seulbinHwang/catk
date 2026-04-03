#!/bin/sh
# Grid search: kinematic_reward_ft with varying flow_reg_lambda (BC regularization weight).
#
# 각 lambda 값에 대해 독립 실험을 순차 실행합니다.
# val_check_interval=500 (Video + RMM 계산 포함).
# GPU 2, 3 전용.
#
# 사용법:
#   ./scripts/gridsearch_kinematic_reward_ft.sh
#   ROLLOUT_STEPS=8 ./scripts/gridsearch_kinematic_reward_ft.sh   # rollout step 변경

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

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
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

# 공통 하이퍼파라미터
MAX_EPOCHS="${MAX_EPOCHS:-10}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"

# Lambda grid: BC 정규화 가중치 10가지
# 0.0 = reward only, 크면 GT BC 정규화 비중 커짐
LAMBDA_LIST="20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0"

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

echo "===== kinematic_reward_ft lambda grid search ====="
echo "CKPT_PATH=${CKPT_PATH}"
echo "ROLLOUT_STEPS=${ROLLOUT_STEPS}"
echo "TRAIN_B=${TRAIN_B}  VAL_B=${VAL_B}"
echo "max_steps=500  LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}"
echo "LAMBDA_LIST=${LAMBDA_LIST}"
echo "=================================================="

RUN_IDX=0
for LAMBDA in ${LAMBDA_LIST}; do
  RUN_IDX=$((RUN_IDX + 1))
  # lambda 값을 파일명에 안전하게 쓰기 (소수점 → p)
  LAMBDA_TAG=$(echo "${LAMBDA}" | sed 's/\./p/g')
  TASK_NAME="kinrewardft-lam${LAMBDA_TAG}-r${ROLLOUT_STEPS}-run${RUN_IDX}"
  PORT="$(get_free_port)"

  echo ""
  echo ">>> [${RUN_IDX}/${#LAMBDA_LIST}] lambda=${LAMBDA}  task=${TASK_NAME}  port=${PORT}"

  torchrun \
    --nproc_per_node=2 \
    --master_port="${PORT}" \
    --rdzv_endpoint="127.0.0.1:${PORT}" \
    -m src.run \
    experiment=kinematic_reward_ft \
    action=finetune \
    task_name="${TASK_NAME}" \
    ckpt_path="${CKPT_PATH}" \
    paths.cache_root="${CACHE_ROOT}" \
    data.train_batch_size="${TRAIN_B}" \
    data.val_batch_size="${VAL_B}" \
    data.train_max_num="${TRAIN_MAX_NUM}" \
    data.num_workers="${NUM_WORKERS}" \
    data.prefetch_factor="${PREFETCH_FACTOR}" \
    data.persistent_workers=true \
    data.pin_memory=true \
    +trainer.max_steps=500 \
    trainer.max_epochs=-1 \
    trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
    trainer.val_check_interval=500 \
    trainer.check_val_every_n_epoch=null \
    model.model_config.val_open_loop=true \
    model.model_config.val_closed_loop=true \
    model.model_config.finetune.rollout_steps="${ROLLOUT_STEPS}" \
    model.model_config.finetune.flow_reg_lambda="${LAMBDA}" \
    logger.wandb.entity="${WANDB_ENTITY}"

  EXIT_CODE=$?
  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "!!! lambda=${LAMBDA} FAILED (exit=${EXIT_CODE}). Continuing to next..."
  else
    echo "<<< lambda=${LAMBDA} done."
  fi
done

echo ""
echo "===== All grid search runs complete ====="
