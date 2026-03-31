#!/bin/sh
# Validation: Predict-Project-Renoise (PPR) generation.
# 매 ODE step에서 x_1 예측 → kinematic projection → re-noise.

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-local_val_flow_ppr}"
MY_TASK_NAME="${MY_TASK_NAME:-val_ppr}"

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

VAL_B="${VAL_B:-4}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VEHICLE_DEADZONE="${VEHICLE_DEADZONE:-0.05}"
PED_DEADZONE="${PED_DEADZONE:-0.03}"
PED_MAX_SPEED="${PED_MAX_SPEED:-0.5}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

echo "Experiment=${MY_EXPERIMENT} (Predict-Project-Renoise)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "vehicle_deadzone=${VEHICLE_DEADZONE}  ped_deadzone=${PED_DEADZONE}"

python -m src.run \
  experiment="${MY_EXPERIMENT}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  "model.model_config.kinematic_projection.vehicle_deadzone=${VEHICLE_DEADZONE}" \
  "model.model_config.kinematic_projection.ped_deadzone=${PED_DEADZONE}" \
  "model.model_config.kinematic_projection.ped_max_speed=${PED_MAX_SPEED}" \
  logger.wandb.entity="${WANDB_ENTITY}"
