#!/bin/sh
set -eu

# Single-scenario overfit/debug run for Flow-BPTT.
# - Train on exactly one batch per epoch (batch_size=1, limit_train_batches=1)
# - Disable validation entirely
# - Deterministic sampling order (shuffle=false, fixed seed)

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_bptt_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-single-scenario-no-val}"
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

SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

# BPTT options (keep same knobs as train_flow_bptt_ft.sh)
BPTT_N_ROLLOUTS="${BPTT_N_ROLLOUTS:-3}"
RMM_BPTT_USE_REF_MODEL="${RMM_BPTT_USE_REF_MODEL:-false}"
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_MAX_COARSE_STEPS="${BPTT_MAX_COARSE_STEPS:-10}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-1.0}"
BPTT_DEBUG="${BPTT_DEBUG:-false}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

echo "Experiment=${MY_EXPERIMENT}"
echo "Task=${MY_TASK_NAME}"
echo "Train single scenario debug mode enabled"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TRAIN_RAW_DIR=${TRAIN_RAW_DIR}"
echo "TRAIN_TFRECORDS_SPLITTED=${TRAIN_TFRECORDS_SPLITTED}"
echo "SEED=${SEED} MAX_EPOCHS=${MAX_EPOCHS} LR=${LR}"
echo "BPTT_N_ROLLOUTS=${BPTT_N_ROLLOUTS} BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT} BPTT_MAX_COARSE_STEPS=${BPTT_MAX_COARSE_STEPS}"
echo "BPTT_SEQUENTIAL_ROLLOUTS=${BPTT_SEQUENTIAL_ROLLOUTS} BPTT_WARM_COARSE_STEPS=${BPTT_WARM_COARSE_STEPS} FLOW_VELOCITY_HEAD_ONLY=${FLOW_VELOCITY_HEAD_ONLY}"

python -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  seed="${SEED}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
  data.train_batch_size=1 \
  data.num_workers=8 \
  data.shuffle=false \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=0.0 \
  trainer.val_check_interval=1.0 \
  trainer.num_sanity_val_steps=0 \
  trainer.check_val_every_n_epoch=1 \
  trainer.log_every_n_steps=1 \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRAD_CLIP_VAL}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.finetune.bptt_n_rollouts="${BPTT_N_ROLLOUTS}" \
  model.model_config.finetune.rmm_bptt_use_ref_model="${RMM_BPTT_USE_REF_MODEL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_max_coarse_steps="${BPTT_MAX_COARSE_STEPS}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_debug="${BPTT_DEBUG}" \
  model.model_config.n_vis_batch=0 \
  model.model_config.n_batch_sim_agents_metric=0 \
  model.model_config.validation_metric=hard
