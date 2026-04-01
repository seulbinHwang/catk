#!/bin/sh
# Final-step projection 두 실험 동시 실행:
#   - final_proj ON  (GPU_PROJ)
#   - final_proj OFF (GPU_NOPROJ)
# projection 외 세팅은 완전히 동일하게 유지.

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
  model.model_config.n_batch_sim_agents_metric=100
  model.model_config.n_vis_batch=1
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO}
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT}
  logger.wandb.entity=${WANDB_ENTITY}
"

echo "CKPT_PATH=${CKPT_PATH}"
echo "Launching final_proj ON on GPU ${GPU_PROJ}, final_proj OFF on GPU ${GPU_NOPROJ} ..."

CUDA_VISIBLE_DEVICES=${GPU_PROJ} python -m src.run \
  experiment=local_val_flow_final_proj \
  task_name=val_final_proj_on \
  ${COMMON_ARGS} \
  model.model_config.final_projection.enabled=true \
  > /tmp/val_final_proj_on.log 2>&1 &
PID_PROJ=$!

CUDA_VISIBLE_DEVICES=${GPU_NOPROJ} python -m src.run \
  experiment=local_val_flow_final_proj \
  task_name=val_final_proj_off \
  ${COMMON_ARGS} \
  model.model_config.final_projection.enabled=false \
  > /tmp/val_final_proj_off.log 2>&1 &
PID_NOPROJ=$!

echo "final_proj ON  PID=${PID_PROJ}   → /tmp/val_final_proj_on.log"
echo "final_proj OFF PID=${PID_NOPROJ} → /tmp/val_final_proj_off.log"

wait ${PID_PROJ}
echo "final_proj ON  done (exit $?)"
wait ${PID_NOPROJ}
echo "final_proj OFF done (exit $?)"
