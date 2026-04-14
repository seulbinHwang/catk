#!/bin/sh
# =============================================================================
# flow_bptt_ft: validation 전용 (학습 없음, action=validate)
# =============================================================================
# pretrained vs finetuned 비교 시 동일 LIMIT_VAL_BATCHES / VAL_B / CKPT_PATH 만 바꿔 재실행.
#
# 예:
#   sh scripts/val_flow_bptt_ft.sh
#   CKPT_PATH=/path/to/finetuned.ckpt MY_TASK_NAME=bptt-ft-epoch5-val sh scripts/val_flow_bptt_ft.sh
#   WANDB_MODE=offline LIMIT_VAL_BATCHES=50 sh scripts/val_flow_bptt_ft.sh
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_bptt_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-val-only}"
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

TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-10}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"

TRAIN_B="${TRAIN_B:-6}"
VAL_B="${VAL_B:-6}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-3}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-1}"

BPTT_N_ROLLOUTS="${BPTT_N_ROLLOUTS:-8}"
RMM_BPTT_USE_REF_MODEL="${RMM_BPTT_USE_REF_MODEL:-false}"
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_MAX_COARSE_STEPS="${BPTT_MAX_COARSE_STEPS:-16}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-true}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

# validate 경로에서도 옵티마 하이퍼가 cfg에 있으면 맞춰 두면 비교 시 동일 실험 블록과 합치기 쉬움
LR="${LR:-5e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
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

echo "Experiment=${MY_EXPERIMENT} action=validate"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "TRAIN_RAW_DIR=${TRAIN_RAW_DIR}"
echo "TRAIN_TFRECORDS_SPLITTED=${TRAIN_TFRECORDS_SPLITTED}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} WANDB_MODE=${WANDB_MODE}"
echo "VALIDATION_METRIC=${VALIDATION_METRIC} N_BATCH_SIM_AGENTS_METRIC=${N_BATCH_SIM_AGENTS_METRIC}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} NUM_WORKERS=${NUM_WORKERS}"

PORT="$(get_free_port)"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=validate \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.bptt_n_rollouts="${BPTT_N_ROLLOUTS}" \
  model.model_config.finetune.rmm_bptt_use_ref_model="${RMM_BPTT_USE_REF_MODEL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  ${BPTT_MAX_COARSE_STEPS:+model.model_config.finetune.bptt_max_coarse_steps="${BPTT_MAX_COARSE_STEPS}"} \
  ${EXTRA_ARGS}
