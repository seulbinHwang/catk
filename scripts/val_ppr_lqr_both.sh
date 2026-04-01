#!/bin/sh
# PPR 16-step 두 실험 동시 실행 (kinematic projection ON 고정):
#   - TV-LQR ON  (GPU_LQR_ON)
#   - TV-LQR OFF (GPU_LQR_OFF)
# projection·그 외 세팅은 동일하게 유지하고 LQR 피드백만 비교.
#
# Deadzone 등은 val_ppr_both.sh 와 동일하게 환경변수로 줄 수 있음.

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

GPU_LQR_ON="${GPU_LQR_ON:-2}"
GPU_LQR_OFF="${GPU_LQR_OFF:-3}"

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
  model.model_config.n_batch_sim_agents_metric=100
  model.model_config.n_vis_batch=1
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO}
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT}
  logger.wandb.entity=${WANDB_ENTITY}
  model.model_config.kinematic_projection.enabled=true
"

echo "CKPT_PATH=${CKPT_PATH}"
echo "Launching 16-step LQR ON on GPU ${GPU_LQR_ON}, LQR OFF on GPU ${GPU_LQR_OFF} ..."

CUDA_VISIBLE_DEVICES=${GPU_LQR_ON} python -m src.run \
  experiment=local_val_flow_ppr_16step \
  task_name=val_ppr_16step_lqr_on \
  ${COMMON_ARGS} \
  model.model_config.kinematic_projection.use_lqr=true \
  > /tmp/val_ppr_16step_lqr_on.log 2>&1 &
PID_LQR_ON=$!

CUDA_VISIBLE_DEVICES=${GPU_LQR_OFF} python -m src.run \
  experiment=local_val_flow_ppr_16step \
  task_name=val_ppr_16step_lqr_off \
  ${COMMON_ARGS} \
  model.model_config.kinematic_projection.use_lqr=false \
  > /tmp/val_ppr_16step_lqr_off.log 2>&1 &
PID_LQR_OFF=$!

echo "16-step LQR ON  PID=${PID_LQR_ON}   → /tmp/val_ppr_16step_lqr_on.log"
echo "16-step LQR OFF PID=${PID_LQR_OFF} → /tmp/val_ppr_16step_lqr_off.log"

wait ${PID_LQR_ON}
echo "16-step LQR ON  done (exit $?)"
wait ${PID_LQR_OFF}
echo "16-step LQR OFF done (exit $?)"
