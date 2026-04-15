#!/bin/sh
set -eu

# =============================================================================
# Single-scenario overfit/debug run for Flow-BPTT (rmm_bptt_ft).
# - Train on exactly one batch per epoch (batch_size=1, limit_train_batches=1)
# - Disable validation entirely
# - Deterministic sampling order (shuffle=false, fixed seed)
#
# Knobs below mirror scripts/train_flow_bptt_ft.sh (env vars + Hydra overrides).
# Typical single-GPU: run with `python` (no torchrun). Use TRAIN_B=1 only.
# =============================================================================

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
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
LR="${LR:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-0}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

# GT FM 정규화 (rmm_bptt_ft): flow_train_clean_norm velocity FM MSE 가중치. 0 이면 비활성.
FLOW_REG_LAMBDA="${FLOW_REG_LAMBDA:-100.0}"

TRAIN_B="${TRAIN_B:-1}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

# BPTT / RMM (same keys as train_flow_bptt_ft.sh)
BPTT_N_ROLLOUTS="${BPTT_N_ROLLOUTS:-1}"
RMM_BPTT_USE_REF_MODEL="${RMM_BPTT_USE_REF_MODEL:-true}"
RMM_BPTT_REF_TRAIN="${RMM_BPTT_REF_TRAIN:-true}"
RMM_BPTT_REF_VAL="${RMM_BPTT_REF_VAL:-true}"
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
# 비우면 yaml 기본(bptt_max_coarse_steps=null = coarse 전부) 유지 (train_flow_bptt_ft.sh 와 동일)
BPTT_MAX_COARSE_STEPS="${BPTT_MAX_COARSE_STEPS:-4}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-0}"
BPTT_DEBUG="${BPTT_DEBUG:-true}"

# Gradient / debug echo (train_flow_bptt_ft.sh 와 동일 의미)
# GRAD_CLIP_VAL: Lightning 전역 grad norm clip (0 이면 대체로 클립 비활성)
# BPTT_GRAD_CLIP_TRAJ: pred_traj backward hook L2 상한 (0 이하면 비활성)

# val 비활성화 시에도 Hydra 키 호환용 (영향 거의 없음)
N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-1}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-4}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-0}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-1}"

WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
echo "BPTT_N_ROLLOUTS=${BPTT_N_ROLLOUTS} BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT} BPTT_MAX_COARSE_STEPS=${BPTT_MAX_COARSE_STEPS:-"(yaml)"}"
echo "BPTT_SEQUENTIAL_ROLLOUTS=${BPTT_SEQUENTIAL_ROLLOUTS} BPTT_WARM_COARSE_STEPS=${BPTT_WARM_COARSE_STEPS} FLOW_VELOCITY_HEAD_ONLY=${FLOW_VELOCITY_HEAD_ONLY}"
echo "[grad] GRAD_CLIP_VAL=${GRAD_CLIP_VAL} BPTT_GRAD_CLIP_TRAJ=${BPTT_GRAD_CLIP_TRAJ} FLOW_REG_LAMBDA=${FLOW_REG_LAMBDA} BPTT_DEBUG=${BPTT_DEBUG}"
echo "RMM_BPTT_REF_TRAIN=${RMM_BPTT_REF_TRAIN} RMM_BPTT_REF_VAL=${RMM_BPTT_REF_VAL}"

python -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  seed="${SEED}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
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
  logger.wandb.entity="${WANDB_ENTITY}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.finetune.flow_reg_lambda="${FLOW_REG_LAMBDA}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.bptt_n_rollouts="${BPTT_N_ROLLOUTS}" \
  model.model_config.finetune.rmm_bptt_use_ref_model="${RMM_BPTT_USE_REF_MODEL}" \
  model.model_config.finetune.rmm_bptt_ref_train="${RMM_BPTT_REF_TRAIN}" \
  model.model_config.finetune.rmm_bptt_ref_val="${RMM_BPTT_REF_VAL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  ${BPTT_MAX_COARSE_STEPS:+model.model_config.finetune.bptt_max_coarse_steps="${BPTT_MAX_COARSE_STEPS}"} \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_debug="${BPTT_DEBUG}" \
  ${EXTRA_ARGS}
