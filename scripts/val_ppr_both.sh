#!/bin/sh
# PPR 4-step (GPU 2) + PPR 16-step (GPU 3) 동시 실행.
# 점수 없이 video만 생성 (n_vis_scenario=4, limit_val_batches=1).
#
# Deadzone 설정:
#   VEHICLE_DEADZONE  — Vehicle / Cyclist 종방향 속도 threshold (기본 0.05, 0이면 비활성)
#   PED_DEADZONE      — Pedestrian 이동 크기 threshold        (기본 0.03, 0이면 비활성)

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

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

GPU_4STEP="${GPU_4STEP:-2}"
GPU_16STEP="${GPU_16STEP:-3}"

VAL_B="${VAL_B:-4}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-4}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

VEHICLE_DEADZONE="${VEHICLE_DEADZONE:-0.1}"
PED_DEADZONE="${PED_DEADZONE:-0.025}"
PED_MAX_SPEED="${PED_MAX_SPEED:-0.5}"

COMMON_ARGS="
  ckpt_path=${CKPT_PATH}
  paths.cache_root=${CACHE_ROOT}
  data.val_batch_size=${VAL_B}
  data.num_workers=${NUM_WORKERS}
  data.prefetch_factor=2
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES}
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_batch_sim_agents_metric=1
  model.model_config.n_vis_batch=1
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO}
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT}
  model.model_config.kinematic_projection.vehicle_deadzone=${VEHICLE_DEADZONE}
  model.model_config.kinematic_projection.ped_deadzone=${PED_DEADZONE}
  model.model_config.kinematic_projection.ped_max_speed=${PED_MAX_SPEED}
  logger.wandb.entity=${WANDB_ENTITY}
"

echo "CKPT_PATH=${CKPT_PATH}"
echo "vehicle_deadzone=${VEHICLE_DEADZONE}  ped_deadzone=${PED_DEADZONE}"
echo "Launching 4-step on GPU ${GPU_4STEP}, 16-step on GPU ${GPU_16STEP} ..."

CUDA_VISIBLE_DEVICES=${GPU_4STEP} python -m src.run \
  experiment=local_val_flow_ppr \
  task_name=val_ppr_4step_vid \
  ${COMMON_ARGS} \
  > /tmp/val_ppr_4step_vid.log 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES=${GPU_16STEP} python -m src.run \
  experiment=local_val_flow_ppr_16step \
  task_name=val_ppr_16step_vid \
  ${COMMON_ARGS} \
  > /tmp/val_ppr_16step_vid.log 2>&1 &
PID16=$!

echo "4-step  PID=${PID4}  → /tmp/val_ppr_4step_vid.log"
echo "16-step PID=${PID16} → /tmp/val_ppr_16step_vid.log"

wait ${PID4}
echo "4-step  done (exit $?)"
wait ${PID16}
echo "16-step done (exit $?)"
