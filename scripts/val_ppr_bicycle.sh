#!/bin/sh
# PPR 16-step + Kinematic Bicycle Model projection.
# Chunk 간 v_state 유지로 물리적 가속도/조향 제약 보장.

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then . "${CONDA_SH}"; fi
if command -v conda >/dev/null 2>&1; then conda activate "${CATK_CONDA_ENV}" || true; fi

GPU="${GPU:-1}"
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

VAL_B="${VAL_B:-4}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-4}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Bicycle model physical params (override 가능)
WHEELBASE="${WHEELBASE:-2.7}"
DELTA_MAX="${DELTA_MAX:-0.52}"
A_MAX="${A_MAX:-4.0}"
D_MAX="${D_MAX:-8.0}"
PED_A_MAX="${PED_A_MAX:-2.0}"

echo "CKPT_PATH=${CKPT_PATH}"
echo "GPU=${GPU}  wheelbase=${WHEELBASE}  delta_max=${DELTA_MAX}  a_max=${A_MAX}  d_max=${D_MAX}"
echo "Launching PPR bicycle model on GPU ${GPU} ..."

CUDA_VISIBLE_DEVICES=${GPU} python -m src.run \
  experiment=local_val_flow_ppr_bicycle \
  task_name=val_ppr_bicycle_vid \
  ckpt_path=${CKPT_PATH} \
  paths.cache_root=${CACHE_ROOT} \
  data.val_batch_size=${VAL_B} \
  data.num_workers=${NUM_WORKERS} \
  data.prefetch_factor=2 \
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO} \
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT} \
  model.model_config.kinematic_projection.wheelbase=${WHEELBASE} \
  model.model_config.kinematic_projection.delta_max=${DELTA_MAX} \
  model.model_config.kinematic_projection.a_max=${A_MAX} \
  model.model_config.kinematic_projection.d_max=${D_MAX} \
  model.model_config.kinematic_projection.ped_a_max=${PED_A_MAX} \
  logger.wandb.entity=${WANDB_ENTITY} \
  > /tmp/val_ppr_bicycle.log 2>&1 &

PID=$!
echo "PID=${PID} → /tmp/val_ppr_bicycle.log"
wait ${PID}
echo "done (exit $?)"
