#!/usr/bin/env bash
# Standalone DMD step profiler launcher.
#
# Self-Forcing DMD `_run_flow_dmd_ft_step` 의 phase 별 GPU 시간 측정용 wrapper.
# train_flow_dmd_single.sh 와 동일한 env var / hydra override 를 사용하므로,
# 학습과 1:1 동일 설정에서의 단계별 시간을 잡을 수 있음.
#
# 사용:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/profile_dmd_step.sh
#
# 비교 실험 예 (Self-Forcing G=1, adjoint off):
#   CUDA_VISIBLE_DEVICES=3 DMD_N_ROLLOUTS=1 BPTT_USE_ADJOINT=false \
#     bash scripts/profile_dmd_step.sh

set -e

export LOGLEVEL=WARNING
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
# disable wosac/wandb to keep profiler standalone (no fork pool, no network).
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_dmd}"
MY_TASK_NAME="${MY_TASK_NAME:-profile-${MY_EXPERIMENT}}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then . "${CONDA_SH}"; fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"
TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"

# launcher 와 동일 default — 측정 결과가 학습 step 과 동일하도록.
TRAIN_B="${TRAIN_B:-16}"
VAL_B="${VAL_B:-16}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"   # profiler 는 single-shot 이라 worker 안 띄움
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PIN_MEMORY="${PIN_MEMORY:-true}"
SEED="${SEED:-817}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
PRECISION="${PRECISION:-32-true}"

LR="${LR:-1e-7}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-10000}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FLOW_SOLVER_METHOD="${FLOW_SOLVER_METHOD:-euler}"
FLOW_SOLVER_STEPS="${FLOW_SOLVER_STEPS:-16}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-0}"

DMD_BETA="${DMD_BETA:-1.0}"
DMD_BETA_WARMUP_STEPS="${DMD_BETA_WARMUP_STEPS:-0}"
DMD_BETA_ANNEAL_STEPS="${DMD_BETA_ANNEAL_STEPS:-0}"
DMD_N_ROLLOUTS="${DMD_N_ROLLOUTS:-4}"
DMD_PRED_MAX_STEPS="${DMD_PRED_MAX_STEPS:-4}"
DMD_USE_REAL_SCORE="${DMD_USE_REAL_SCORE:-true}"
DMD_FAKE_LR_SCALE="${DMD_FAKE_LR_SCALE:-1.0}"
DMD_NORMALIZE="${DMD_NORMALIZE:-true}"
DMD_ANCHOR_STRIDE="${DMD_ANCHOR_STRIDE:-4}"
DMD_STRICT_ACTIVE_MASK="${DMD_STRICT_ACTIVE_MASK:-true}"
DMD_WARMUP_FAKE_ONLY_STEPS="${DMD_WARMUP_FAKE_ONLY_STEPS:-0}"
DMD_GEN_GRAD_CLIP="${DMD_GEN_GRAD_CLIP:-10.0}"
DMD_GEN_UPDATE_RATIO="${DMD_GEN_UPDATE_RATIO:-3}"
DMD_ADAM_BETA1="${DMD_ADAM_BETA1:-0.0}"
DMD_ADAM_BETA2="${DMD_ADAM_BETA2:-0.999}"
DMD_EMA_WEIGHT="${DMD_EMA_WEIGHT:-0.0}"
DMD_EMA_START_STEP="${DMD_EMA_START_STEP:-0}"
DMD_FAKE_FT_SCOPE="${DMD_FAKE_FT_SCOPE:-full}"

BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
BPTT_LAST_N_COARSE_STEPS="${BPTT_LAST_N_COARSE_STEPS:-0}"
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-10.0}"
BPTT_LAST_COARSE_ONLY="${BPTT_LAST_COARSE_ONLY:-false}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"
FLOW_FT_TARGET="${FLOW_FT_TARGET:-full}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[profile-dmd] G=${DMD_N_ROLLOUTS} stride=${DMD_ANCHOR_STRIDE} pred=${DMD_PRED_MAX_STEPS}cs adjoint=${BPTT_USE_ADJOINT} B=${TRAIN_B}"
echo "[profile-dmd] PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-1} PROFILE_MEASURE_STEPS=${PROFILE_MEASURE_STEPS:-3}"
echo "[profile-dmd] CKPT=${CKPT_PATH}"

cd "$(dirname "$0")/.."

python scripts/profile_dmd_step.py \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  seed="${SEED}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  trainer.precision="${PRECISION}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.decoder.flow_solver_method="${FLOW_SOLVER_METHOD}" \
  model.model_config.decoder.flow_solver_steps="${FLOW_SOLVER_STEPS}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.flow_ft_target="${FLOW_FT_TARGET}" \
  model.model_config.finetune.dmd_beta="${DMD_BETA}" \
  model.model_config.finetune.dmd_beta_warmup_steps="${DMD_BETA_WARMUP_STEPS}" \
  model.model_config.finetune.dmd_beta_anneal_steps="${DMD_BETA_ANNEAL_STEPS}" \
  model.model_config.finetune.dmd_n_rollouts="${DMD_N_ROLLOUTS}" \
  model.model_config.finetune.dmd_pred_max_steps="${DMD_PRED_MAX_STEPS}" \
  model.model_config.finetune.dmd_use_real_score="${DMD_USE_REAL_SCORE}" \
  model.model_config.finetune.dmd_fake_lr_scale="${DMD_FAKE_LR_SCALE}" \
  model.model_config.finetune.dmd_normalize="${DMD_NORMALIZE}" \
  model.model_config.finetune.dmd_anchor_stride="${DMD_ANCHOR_STRIDE}" \
  model.model_config.finetune.dmd_strict_active_mask="${DMD_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.dmd_warmup_fake_only_steps="${DMD_WARMUP_FAKE_ONLY_STEPS}" \
  model.model_config.finetune.dmd_gen_grad_clip="${DMD_GEN_GRAD_CLIP}" \
  model.model_config.finetune.dmd_gen_update_ratio="${DMD_GEN_UPDATE_RATIO}" \
  model.model_config.finetune.dmd_adam_beta1="${DMD_ADAM_BETA1}" \
  model.model_config.finetune.dmd_adam_beta2="${DMD_ADAM_BETA2}" \
  model.model_config.finetune.dmd_ema_weight="${DMD_EMA_WEIGHT}" \
  model.model_config.finetune.dmd_ema_start_step="${DMD_EMA_START_STEP}" \
  model.model_config.finetune.dmd_fake_ft_scope="${DMD_FAKE_FT_SCOPE}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_coarse_steps="${BPTT_LAST_N_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_last_coarse_only="${BPTT_LAST_COARSE_ONLY}" \
  ${EXTRA_ARGS}
