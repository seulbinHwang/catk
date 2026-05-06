#!/bin/sh
# Final-step kinematic projection 두 실험 동시 실행:
#   - proj ON  (GPU_PROJ):   표준 ODE 완료 후 마지막에 KinematicProjection 한 번 적용
#   - proj OFF (GPU_NOPROJ): 표준 ODE만, 아무 projection 없음
#
# PPR(val_ppr_both.sh)과의 차이: PPR은 매 ODE step마다 projection,
# 이 스크립트는 ODE 완료 이후 final step에만 한 번 projection.
#
# 비교 지표: val/ADE2s, val/FDE2s, val/yaw_ADE2s, val/yaw_FDE2s

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

GPU_PROJ="${GPU_PROJ:-2}"
GPU_NOPROJ="${GPU_NOPROJ:-3}"

VAL_B="${VAL_B:-4}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-4}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

COMMON_ARGS="
  ckpt_path=${CKPT_PATH}
  paths.cache_root=${CACHE_ROOT}
  data.val_batch_size=${VAL_B}
  data.num_workers=${NUM_WORKERS}
  data.prefetch_factor=2
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES}
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val=32
  model.model_config.n_batch_sim_agents_metric=100
  model.model_config.n_vis_batch=1
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO}
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT}
  logger.wandb.entity=${WANDB_ENTITY}
"

echo "CKPT_PATH=${CKPT_PATH}"
echo "Launching final-step kin proj ON on GPU ${GPU_PROJ}, proj OFF on GPU ${GPU_NOPROJ} ..."

# ON: ODE → KinematicProjection (final step only, predict_project_renoise=false 기본값)
CUDA_VISIBLE_DEVICES=${GPU_PROJ} python -m src.run \
  experiment=local_val_flow_kinematic_proj \
  task_name=val_final_kin_proj_on \
  ${COMMON_ARGS} \
  model.model_config.kinematic_projection.enabled=true \
  > /tmp/val_final_kin_proj_on.log 2>&1 &
PID_PROJ=$!

# OFF: ODE only, no projection
CUDA_VISIBLE_DEVICES=${GPU_NOPROJ} python -m src.run \
  experiment=local_val_flow_kinematic_proj \
  task_name=val_final_kin_proj_off \
  ${COMMON_ARGS} \
  model.model_config.kinematic_projection.enabled=false \
  > /tmp/val_final_kin_proj_off.log 2>&1 &
PID_NOPROJ=$!

echo "final-step kin proj ON  PID=${PID_PROJ}   → /tmp/val_final_kin_proj_on.log"
echo "final-step kin proj OFF PID=${PID_NOPROJ} → /tmp/val_final_kin_proj_off.log"

wait ${PID_PROJ}
echo "final-step kin proj ON  done (exit $?)"
wait ${PID_NOPROJ}
echo "final-step kin proj OFF done (exit $?)"
