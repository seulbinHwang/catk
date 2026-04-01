#!/bin/sh
# 세 가지 bicycle model 조합 동시 실행:
#   A: PPR only         (GPU_A, experiment=local_val_flow_bicycle_ppr_only)
#   B: Postproc only    (GPU_B, experiment=local_val_flow_bicycle_postproc_only)
#   C: PPR + Postproc   (GPU_C, experiment=local_val_flow_bicycle_both)

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

GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"

VAL_B="${VAL_B:-4}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-4}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
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
  model.model_config.n_batch_sim_agents_metric=50
  model.model_config.n_vis_batch=1
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO}
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT}
  logger.wandb.entity=${WANDB_ENTITY}
"

echo "CKPT_PATH=${CKPT_PATH}"
echo "GPU_A=${GPU_A} (PPR only), GPU_B=${GPU_B} (Postproc only), GPU_C=${GPU_C} (Both)"
echo "Launching 3-way bicycle experiment ..."

CUDA_VISIBLE_DEVICES=${GPU_A} python -m src.run \
  experiment=local_val_flow_bicycle_ppr_only \
  task_name=val_bicycle_ppr_only \
  ${COMMON_ARGS} \
  > /tmp/val_bicycle_ppr_only.log 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=${GPU_B} python -m src.run \
  experiment=local_val_flow_bicycle_postproc_only \
  task_name=val_bicycle_postproc_only \
  ${COMMON_ARGS} \
  > /tmp/val_bicycle_postproc_only.log 2>&1 &
PID_B=$!

CUDA_VISIBLE_DEVICES=${GPU_C} python -m src.run \
  experiment=local_val_flow_bicycle_both \
  task_name=val_bicycle_both \
  ${COMMON_ARGS} \
  > /tmp/val_bicycle_both.log 2>&1 &
PID_C=$!

echo "PID_A=${PID_A} → /tmp/val_bicycle_ppr_only.log"
echo "PID_B=${PID_B} → /tmp/val_bicycle_postproc_only.log"
echo "PID_C=${PID_C} → /tmp/val_bicycle_both.log"

wait ${PID_A}
echo "A (PPR only)      done (exit $?)"
wait ${PID_B}
echo "B (Postproc only) done (exit $?)"
wait ${PID_C}
echo "C (Both)          done (exit $?)"
