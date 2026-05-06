#!/bin/sh
# =============================================================================
# OCSC single-scenario 버전 — 빠른 디버깅 / 알고리즘 검증용
# train_flow_consistency_bptt.sh 의 single-GPU, TRAIN_B=8 특화 프리셋
# =============================================================================
# 사용법:
#   sh scripts/train_flow_consistency_bptt_single.sh
#   OCSC_N_ROLLOUTS=4 MAX_EPOCHS=5 sh scripts/train_flow_consistency_bptt_single.sh
#   # validation worker / thread 조정 예:
#   WOSAC_HARD_POOL_WORKERS=2 OMP_NUM_THREADS=4 NUM_WORKERS=2 \
#     sh scripts/train_flow_consistency_bptt_single.sh
#
# 기본값 = fix-hard-rmm production 프리셋:
#   - GT-target consistency (rel_disp weight=1.0, position weight=0.0)
#   - Frozen pretrained reference decoder
#   - velocity_head 만 학습
#   - BPTT use_adjoint + last_coarse_only
#   - GT FM regularization λ=0.1
#   - validation: hard RMM
#
# 핵심 토글:
#   OCSC_GT_TARGET=false            → Open-loop sample target (default: true)
#   OCSC_USE_MMD=true               → MMD² (default: paired L2)
#   OCSC_USE_PRETRAINED_REF=false   → 현재 정책으로 OL 생성 (drift O)
#   OCSC_SHARE_NOISE_TAPE=false     → OL/CL noise 분리
#   BPTT_SEQUENTIAL_ROLLOUTS=true   → 메모리 ↓, manual optimization 자동 활성
#   BPTT_USE_ADJOINT=false          → ckpt 끔
#   BPTT_LAST_COARSE_ONLY=false     → 모든 coarse step grad
#   FLOW_VELOCITY_HEAD_ONLY=false   → step_refiner 도 같이 학습
#   VALIDATION_METRIC=real          → 공식 TF metric (default: hard)

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CPU thread 노브 (학습 main + forkserver 워커 모두 상속)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
# Validation 시 sim-agents metric forkserver pool 크기 (코드 default 16).
# - hard: HardSimAgentsMetrics, real: official TF SimAgentsMetrics.
export WOSAC_HARD_POOL_WORKERS="${WOSAC_HARD_POOL_WORKERS:-16}"
export WOSAC_REAL_POOL_WORKERS="${WOSAC_REAL_POOL_WORKERS:-16}"
# 1로 두면 hard vs real 결과 cross-check (느려짐, 디버깅용).
export WOSAC_VERIFY="${WOSAC_VERIFY:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_consistency_bptt}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-single}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# ── 데이터 / 체크포인트 ─────────────────────────────────────────────────────
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
# logs/pretrained/epoch_last.ckpt 는 이 repo 의 pretrained checkpoint 입니다.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${PROJECT_ROOT}/logs/pretrained/epoch_last.ckpt}"

TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"
FIXED_SCENARIO_MODE="${FIXED_SCENARIO_MODE:-false}"
FIXED_SCENARIO_INDEX="${FIXED_SCENARIO_INDEX:-0}"
FIXED_SCENARIO_COUNT="${FIXED_SCENARIO_COUNT:-4}"
FIXED_SCENARIO_PKL="${FIXED_SCENARIO_PKL:-}"

# ── Single-scenario 특화 기본값 (fix-hard-rmm 동일) ─────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-16}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-32}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
SEED="${SEED:-817}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
TRAINER_DETERMINISTIC="${TRAINER_DETERMINISTIC:-true}"

# single scenario: LR 조금 높여도 안정적
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"

# ── Flow ODE solver ─────────────────────────────────────────────────────────
FLOW_SOLVER_METHOD="${FLOW_SOLVER_METHOD:-euler}"
FLOW_SOLVER_STEPS="${FLOW_SOLVER_STEPS:-16}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

# ── Visualization ───────────────────────────────────────────────────────────
N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

# ── Validation backend ──────────────────────────────────────────────────────
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10000}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"

# ── OCSC 핵심 (fix-hard-rmm production defaults) ────────────────────────────
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-4}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_USE_MMD="${OCSC_USE_MMD:-false}"
OCSC_ANCHOR_STRIDE="${OCSC_ANCHOR_STRIDE:-1}"
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-true}"
OCSC_PRED_MAX_STEPS="${OCSC_PRED_MAX_STEPS:-2}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.0}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-0.0}"
OCSC_REL_DISP_WEIGHT="${OCSC_REL_DISP_WEIGHT:-1.0}"
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA:-0.1}"
# 사용자가 실수로 "=0.1" 형태로 넘긴 경우 보정.
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA#=}"
OCSC_GT_TARGET="${OCSC_GT_TARGET:-true}"
OCSC_SHARE_NOISE_TAPE="${OCSC_SHARE_NOISE_TAPE:-true}"
OCSC_SHARE_NOISE_ACROSS_TIME="${OCSC_SHARE_NOISE_ACROSS_TIME:-false}"

# ── HardRMM 모니터링 ────────────────────────────────────────────────────────
OCSC_EVAL_HARD_RMM="${OCSC_EVAL_HARD_RMM:-false}"
OCSC_EVAL_HARD_RMM_INTERVAL="${OCSC_EVAL_HARD_RMM_INTERVAL:-10}"

# ── Freeze 정책 ─────────────────────────────────────────────────────────────
TRAIN_FULL_FLOW_DECODER_ONLY="${TRAIN_FULL_FLOW_DECODER_ONLY:-false}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"

# ── BPTT 메모리 / 속도 토글 ────────────────────────────────────────────────
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
BPTT_LAST_N_COARSE_STEPS="${BPTT_LAST_N_COARSE_STEPS:-0}"
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-0.0}"
BPTT_LAST_COARSE_ONLY="${BPTT_LAST_COARSE_ONLY:-true}"

WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-SMART-FLOW}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  echo "        logs/pretrained/epoch_last.ckpt 를 이 repo 안에 두세요."
  exit 1
fi

# 고정 시나리오 모드:
TRAIN_RAW_DIR_EFFECTIVE="${TRAIN_RAW_DIR}"
FIXED_SCENARIO_TMP_DIR=""
if [ "${FIXED_SCENARIO_MODE}" = "true" ]; then
  FIXED_SCENARIO_TMP_DIR="$(mktemp -d "/tmp/ocsc_single_scenario.XXXXXX")"
  trap 'if [ -n "${FIXED_SCENARIO_TMP_DIR}" ] && [ -d "${FIXED_SCENARIO_TMP_DIR}" ]; then rm -rf "${FIXED_SCENARIO_TMP_DIR}"; fi' EXIT

  if [ -n "${FIXED_SCENARIO_PKL}" ]; then
    if [ ! -f "${FIXED_SCENARIO_PKL}" ]; then
      echo "[ERROR] FIXED_SCENARIO_PKL not found: ${FIXED_SCENARIO_PKL}"
      exit 1
    fi
    ln -s "${FIXED_SCENARIO_PKL}" "${FIXED_SCENARIO_TMP_DIR}/scenario.pkl"
  else
    SELECTED_PKL_PATHS="$(python3 - <<PY
import pathlib, sys
raw_dir = pathlib.Path("${TRAIN_RAW_DIR}")
paths = sorted([p for p in raw_dir.glob("*.pkl") if p.is_file()])
if not paths:
    print("")
    sys.exit(0)
start = int("${FIXED_SCENARIO_INDEX}")
count = int("${FIXED_SCENARIO_COUNT}")
if start < 0 or start >= len(paths) or count <= 0:
    print("")
    sys.exit(0)
end = min(len(paths), start + count)
selected = paths[start:end]
if not selected:
    print("")
    sys.exit(0)
print("\\n".join(p.as_posix() for p in selected))
PY
)"
    if [ -z "${SELECTED_PKL_PATHS}" ]; then
      echo "[ERROR] Could not select FIXED_SCENARIO_INDEX=${FIXED_SCENARIO_INDEX}, FIXED_SCENARIO_COUNT=${FIXED_SCENARIO_COUNT} under ${TRAIN_RAW_DIR}"
      exit 1
    fi
    i=0
    printf "%s\n" "${SELECTED_PKL_PATHS}" | while IFS= read -r p; do
      [ -n "${p}" ] || continue
      ln -s "${p}" "${FIXED_SCENARIO_TMP_DIR}/scenario_${i}.pkl"
      i=$((i + 1))
    done
  fi

  TRAIN_RAW_DIR_EFFECTIVE="${FIXED_SCENARIO_TMP_DIR}"
fi

PORT="$(get_free_port)"
echo "[ocsc-single] Experiment=${MY_EXPERIMENT} task=${MY_TASK_NAME}"
echo "  CKPT=${CKPT_PATH}"
echo "  CACHE=${CACHE_ROOT}"
echo "  TRAIN_RAW_DIR=${TRAIN_RAW_DIR_EFFECTIVE}"
echo "  FIXED_SCENARIO=${FIXED_SCENARIO_MODE} idx=${FIXED_SCENARIO_INDEX} count=${FIXED_SCENARIO_COUNT}"
echo "  NPROC=${NPROC_PER_NODE} TRAIN_B=${TRAIN_B} VAL_B=${VAL_B} MAX_EPOCHS=${MAX_EPOCHS}"
echo "  SEED=${SEED} DATA_SHUFFLE=${DATA_SHUFFLE} DETERMINISTIC=${TRAINER_DETERMINISTIC}"
echo "  OCSC: G=${OCSC_N_ROLLOUTS} stride=${OCSC_ANCHOR_STRIDE} pred=${OCSC_PRED_MAX_STEPS}"
echo "        gt_target=${OCSC_GT_TARGET} use_mmd=${OCSC_USE_MMD} loss=${OCSC_LOSS_TYPE}"
echo "        share_tape=${OCSC_SHARE_NOISE_TAPE} share_time=${OCSC_SHARE_NOISE_ACROSS_TIME}"
echo "        ref=${OCSC_USE_PRETRAINED_REF} fm_reg=${OCSC_FM_REG_LAMBDA}"
echo "        weights pos=${OCSC_POSITION_WEIGHT} disp=${OCSC_REL_DISP_WEIGHT} head=${OCSC_HEADING_WEIGHT}"
echo "  BPTT: adjoint=${BPTT_USE_ADJOINT} last_solver=${BPTT_LAST_N_SOLVER_STEPS}"
echo "        warm=${BPTT_WARM_COARSE_STEPS} last_only=${BPTT_LAST_COARSE_ONLY}"
echo "        last_n=${BPTT_LAST_N_COARSE_STEPS} grad_clip=${BPTT_GRAD_CLIP_TRAJ}"
echo "        sequential=${BPTT_SEQUENTIAL_ROLLOUTS}"
echo "  freeze: full_decoder=${TRAIN_FULL_FLOW_DECODER_ONLY} velocity_only=${FLOW_VELOCITY_HEAD_ONLY}"
echo "  validation: ${VALIDATION_METRIC} (n_batch=${N_BATCH_SIM_AGENTS_METRIC} n_rollout=${N_ROLLOUT_CLOSED_VAL})"
echo "  hard_rmm monitor: ${OCSC_EVAL_HARD_RMM} interval=${OCSC_EVAL_HARD_RMM_INTERVAL}"
echo "  LR=${LR} solver=${FLOW_SOLVER_METHOD}/${FLOW_SOLVER_STEPS}"
echo "  wandb: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} mode=${WANDB_MODE}"

PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  PREFETCH_ARG="data.prefetch_factor=${PREFETCH_FACTOR}"
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  seed="${SEED}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_raw_dir="${TRAIN_RAW_DIR_EFFECTIVE}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRAD_CLIP_VAL}" \
  trainer.deterministic="${TRAINER_DETERMINISTIC}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.decoder.flow_solver_method="${FLOW_SOLVER_METHOD}" \
  model.model_config.decoder.flow_solver_steps="${FLOW_SOLVER_STEPS}" \
  model.model_config.validation_rollout_sampling.noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.finetune.train_full_flow_decoder_only="${TRAIN_FULL_FLOW_DECODER_ONLY}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_use_mmd="${OCSC_USE_MMD}" \
  model.model_config.finetune.ocsc_anchor_stride="${OCSC_ANCHOR_STRIDE}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_pred_max_steps="${OCSC_PRED_MAX_STEPS}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_rel_disp_weight="${OCSC_REL_DISP_WEIGHT}" \
  model.model_config.finetune.ocsc_fm_reg_lambda="${OCSC_FM_REG_LAMBDA}" \
  model.model_config.finetune.ocsc_eval_hard_rmm="${OCSC_EVAL_HARD_RMM}" \
  model.model_config.finetune.ocsc_eval_hard_rmm_interval="${OCSC_EVAL_HARD_RMM_INTERVAL}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_share_noise_tape="${OCSC_SHARE_NOISE_TAPE}" \
  model.model_config.finetune.ocsc_share_noise_across_time="${OCSC_SHARE_NOISE_ACROSS_TIME}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_coarse_steps="${BPTT_LAST_N_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_last_coarse_only="${BPTT_LAST_COARSE_ONLY}" \
  ${PREFETCH_ARG} \
  ${EXTRA_ARGS}

echo "[ocsc-single] done"
