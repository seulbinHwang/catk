#!/bin/sh
# =============================================================================
# OL vs CL Hard-RMM 시나리오별 비교 launcher.
#
#   - GPU 정책: 2 또는 3 만 사용 (CUDA 0/1 절대 금지).  default GPU=2.
#   - KST 타임스탬프로 wandb run name / artifacts dir 생성.
#   - experiment=local_val_flow (validate 셋팅) 위에 우리 비교 스크립트가
#     trainer.validate 를 사용하지 않고 직접 val_loader 를 돌린다.
#
# 사용:
#   sh temp/ol_vs_cl_rmm/launch_compare.sh
#   OLCL_G_ROLLOUTS=8 OLCL_LIMIT_VAL_BATCHES=0.005 sh temp/ol_vs_cl_rmm/launch_compare.sh
#   CUDA_VISIBLE_DEVICES=3 sh temp/ol_vs_cl_rmm/launch_compare.sh
#   OLCL_PAD_MODE=last sh temp/ol_vs_cl_rmm/launch_compare.sh   # GT 대신 hold-last
#
# =============================================================================

# ── shell strict ─────────────────────────────────────────────────────────────
set -e

# ── GPU 정책 ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
case "${CUDA_VISIBLE_DEVICES}" in
  *0*|*1*)
    echo "[ERROR] GPU 0/1 은 사용 금지 (CLAUDE.md 규칙).  CUDA_VISIBLE_DEVICES=2 또는 3 만 가능."
    exit 1
    ;;
esac

# ── 환경 ────────────────────────────────────────────────────────────────────
export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CPU thread (BLAS / OpenMP 가 worker 폭발하지 않도록 제한)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"

# WOSAC hard-RMM forkserver pool (사이즈 자동조정 = min(16, ncpu/2))
export WOSAC_HARD_POOL_WORKERS="${WOSAC_HARD_POOL_WORKERS:-8}"
export WOSAC_REAL_POOL_WORKERS="${WOSAC_REAL_POOL_WORKERS:-0}"   # real metric 안 씀
export WOSAC_HARD_LOG_CACHE_DIR="${WOSAC_HARD_LOG_CACHE_DIR:-/tmp/wosac_hard_log_feat_cache}"

# ── KST timestamp ───────────────────────────────────────────────────────────
KST_NOW="$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)"
export OLCL_WANDB_RUN_NAME="${OLCL_WANDB_RUN_NAME:-ol-vs-cl-2s-${KST_NOW}-gpu${CUDA_VISIBLE_DEVICES}}"
export OLCL_WANDB_PROJECT="${OLCL_WANDB_PROJECT:-project_3-ol-vs-cl-rmm}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

# ── 비교 스크립트 토글 ──────────────────────────────────────────────────────
export OLCL_G_ROLLOUTS="${OLCL_G_ROLLOUTS:-16}"             # WOSAC G
export OLCL_PRED_2S_COARSE="${OLCL_PRED_2S_COARSE:-4}"      # 4 × 0.5초 = 2초
export OLCL_LIMIT_VAL_BATCHES="${OLCL_LIMIT_VAL_BATCHES:-0.01}"
# RMM path 는 non-differentiable 2s native 고정 (hard_rmm_2s/ 본체 사용).

# ── ckpt (fix-hard-rmm 호환) ────────────────────────────────────────────────
CKPT_PATH="${OLCL_CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"
if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi
export OLCL_CKPT_PATH="${CKPT_PATH}"

# ── conda 환경 ──────────────────────────────────────────────────────────────
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# ── 데이터 경로 ─────────────────────────────────────────────────────────────
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"

# ── batch size 조정 가능 ────────────────────────────────────────────────────
VAL_B="${VAL_B:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

cd "$(dirname "$0")/../.."   # repo root

# pred horizon (초) — bc 없는 환경에서도 동작하도록 python 으로 계산
PRED_SEC="$(python3 -c "print(${OLCL_PRED_2S_COARSE} * 0.5)" 2>/dev/null || echo '?')"

echo "============================================================"
echo "[ol-vs-cl-rmm launch] KST=${KST_NOW}  GPU=${CUDA_VISIBLE_DEVICES}"
echo "  G=${OLCL_G_ROLLOUTS}  pred_max_steps=${OLCL_PRED_2S_COARSE} (= ${PRED_SEC} 초)"
echo "  limit_val_batches=${OLCL_LIMIT_VAL_BATCHES}  rmm_path=non_diff_2s_native"
echo "  CKPT=${CKPT_PATH}"
echo "  wandb: project=${OLCL_WANDB_PROJECT}  run=${OLCL_WANDB_RUN_NAME}  mode=${WANDB_MODE}"
echo "  WOSAC_HARD_POOL_WORKERS=${WOSAC_HARD_POOL_WORKERS}"
echo "  VAL_B=${VAL_B}  NUM_WORKERS=${NUM_WORKERS}  PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "============================================================"

# num_workers=0 일 땐 prefetch_factor 옵션을 주면 PyTorch DataLoader 가 에러를 낸다.
PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  # local_val_flow data config 에 prefetch_factor 가 없어서 +데이터필드 추가 형식 사용.
  PREFETCH_ARG="+data.prefetch_factor=${PREFETCH_FACTOR}"
fi

# Hydra config: experiment=local_val_flow (validate 모드용 base)
exec python temp/ol_vs_cl_rmm/compare_ol_vs_cl_rmm.py \
  experiment=local_val_flow \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  ${PREFETCH_ARG} \
  trainer.limit_val_batches="${OLCL_LIMIT_VAL_BATCHES}" \
  task_name="ol-vs-cl-${KST_NOW}" \
  "$@"
