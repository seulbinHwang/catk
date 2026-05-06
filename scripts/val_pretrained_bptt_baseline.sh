#!/bin/sh
# =============================================================================
# Pretrained model baseline validation — flow_bptt_ft 설정과 1:1 대응
# =============================================================================
# bptt_ft 학습 중 validation과 동일한 조건:
#   - flow_solver_steps=4, kinematic_projection=false
#   - limit_val_batches=10, n_batch_sim_agents_metric=10
#   - val_open_loop=true, val_closed_loop=true, n_rollout_closed_val=16
#   - finetune.enabled=false (pretrained weight 그대로)
#   - precision=32-true (bptt_ft 기본값과 동일)
#   - 데이터: validation split (bptt_ft val loader와 동일)
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

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
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
MY_TASK_NAME="${MY_TASK_NAME:-val_pretrained_bptt_baseline}"

# ── bptt_ft 와 동일한 val 파라미터 ─────────────────────────────────────────
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-10}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10}"
N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"

VAL_B="${VAL_B:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-32-true}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

echo "=== Pretrained Baseline Validation (bptt_ft 설정 1:1 일치) ==="
echo "CKPT_PATH=${CKPT_PATH}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}  N_BATCH_SIM_AGENTS_METRIC=${N_BATCH_SIM_AGENTS_METRIC}"
echo "PRECISION=${PRECISION}  N_VIS_BATCH=${N_VIS_BATCH}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m src.run \
  experiment=flow_bptt_ft \
  action=validate \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor=2 \
  data.persistent_workers=true \
  data.pin_memory=true \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.precision="${PRECISION}" \
  model.model_config.finetune.enabled=false \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  logger.wandb.entity="${WANDB_ENTITY}"
