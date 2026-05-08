#!/bin/sh
# =============================================================================
# Pretrained ckpt validation only.
# OCSC single (`scripts/train_flow_consistency_bptt_single.sh`) 의 validation
# 셋팅과 동일하게 — 16 rollout / val=0.01 / hard RMM / 모든 val batch 에서
# RMM 계산 — 단 action=validate 라서 학습 없이 baseline 만 측정.
#
# 결과는 wandb online 으로 올라가 사용자가 OCSC 학습 시작 전 baseline 으로 참고.
#
# Run:
#   sh temp/ol_vs_cl_rmm/launch_pretrained_val.sh
#   CUDA_VISIBLE_DEVICES=2 sh temp/ol_vs_cl_rmm/launch_pretrained_val.sh
# =============================================================================

set -e

# GPU 정책
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
case "${CUDA_VISIBLE_DEVICES}" in
  *0*|*1*) echo "[ERROR] GPU 0/1 금지"; exit 1 ;;
esac

# 환경 (OCSC single launcher 와 동일)
export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export WOSAC_HARD_POOL_WORKERS="${WOSAC_HARD_POOL_WORKERS:-8}"
export WOSAC_REAL_POOL_WORKERS="${WOSAC_REAL_POOL_WORKERS:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

KST_NOW="$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)"

# ── conda env ──
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV}" || true

# ── 경로 ──
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"
[ ! -f "${CKPT_PATH}" ] && { echo "[ERROR] CKPT not found: ${CKPT_PATH}"; exit 1; }

# ── OCSC single 셋팅과 동일 ──
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10000}"
VAL_B="${VAL_B:-16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PRECISION="${PRECISION:-32-true}"
SEED="${SEED:-817}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"
FLOW_SOLVER_METHOD="${FLOW_SOLVER_METHOD:-euler}"
FLOW_SOLVER_STEPS="${FLOW_SOLVER_STEPS:-16}"

MY_TASK_NAME="${MY_TASK_NAME:-pretrained-val-rmm-${KST_NOW}-gpu${CUDA_VISIBLE_DEVICES}}"

cd "$(dirname "$0")/../.."

PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  # local_val_flow data config 에 prefetch_factor 가 없어 + 키추가 형식 사용.
  PREFETCH_ARG="+data.prefetch_factor=${PREFETCH_FACTOR}"
fi

echo "============================================================"
echo "[pretrained val] KST=${KST_NOW}  GPU=${CUDA_VISIBLE_DEVICES}"
echo "  CKPT=${CKPT_PATH}"
echo "  N_ROLLOUT_CLOSED_VAL=${N_ROLLOUT_CLOSED_VAL}  VAL_B=${VAL_B}"
echo "  LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}  VALIDATION_METRIC=${VALIDATION_METRIC}"
echo "  N_BATCH_SIM_AGENTS_METRIC=${N_BATCH_SIM_AGENTS_METRIC}"
echo "  task_name=${MY_TASK_NAME}  WANDB=${WANDB_MODE}"
echo "  WOSAC_HARD_POOL_WORKERS=${WOSAC_HARD_POOL_WORKERS}"
echo "============================================================"

exec python -m src.run \
  experiment=local_val_flow \
  action=validate \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  seed="${SEED}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  ${PREFETCH_ARG} \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.precision="${PRECISION}" \
  model.model_config.n_vis_batch=0 \
  model.model_config.n_vis_scenario=0 \
  model.model_config.n_vis_rollout=0 \
  model.model_config.val_open_loop=true \
  model.model_config.val_closed_loop=true \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.decoder.flow_solver_method="${FLOW_SOLVER_METHOD}" \
  model.model_config.decoder.flow_solver_steps="${FLOW_SOLVER_STEPS}" \
  model.model_config.eval_sampling_noise.noise_scale="${ROLLOUT_NOISE_SCALE}" \
  logger.wandb.entity="${WANDB_ENTITY}"
