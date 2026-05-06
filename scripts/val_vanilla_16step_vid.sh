#!/bin/sh
# Vanilla 16-step midpoint inference — projection 없이 원래 구현 그대로.
# 영상 생성 전용 (limit_val_batches=1, n_vis_scenario=4).

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
NUM_WORKERS="${NUM_WORKERS:-4}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-4}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"

echo "CKPT_PATH=${CKPT_PATH}"
echo "Launching vanilla 16-step midpoint on GPU ${GPU} ..."

CUDA_VISIBLE_DEVICES=${GPU} python -m src.run \
  experiment=local_val_flow_ppr_16step \
  task_name=val_vanilla_16step_vid \
  ckpt_path=${CKPT_PATH} \
  paths.cache_root=${CACHE_ROOT} \
  data.val_batch_size=${VAL_B} \
  data.num_workers=${NUM_WORKERS} \
  data.prefetch_factor=2 \
  trainer.limit_val_batches=1 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=${N_VIS_SCENARIO} \
  model.model_config.n_vis_rollout=${N_VIS_ROLLOUT} \
  model.model_config.kinematic_projection.enabled=false \
  model.model_config.kinematic_projection.predict_project_renoise=false \
  logger.wandb.entity=${WANDB_ENTITY} \
  > /tmp/val_vanilla_16step_vid.log 2>&1 &

PID=$!
echo "PID=${PID} → /tmp/val_vanilla_16step_vid.log"
wait ${PID}
echo "done (exit $?)"
