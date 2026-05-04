#!/bin/sh
# =============================================================================
# OCSC single-scenario 버전 — 빠른 디버깅 / 알고리즘 검증용
# train_flow_consistency_bptt.sh 의 single-GPU, TRAIN_B=1 특화 프리셋
# =============================================================================
# 사용법:
#   sh scripts/train_flow_consistency_bptt_single.sh
#   OCSC_N_ROLLOUTS=4 MAX_EPOCHS=5 sh scripts/train_flow_consistency_bptt_single.sh
#   # validation worker / thread 조정 예:
#   WOSAC_HARD_POOL_WORKERS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUM_WORKERS=2 \
#     sh scripts/train_flow_consistency_bptt_single.sh
#
# single scenario 특징:
#   - TRAIN_B=1: 시나리오 1개씩 학습 (gradient 방향 충돌 없음 → LR 좀 더 올릴 수 있음)
#   - NPROC_PER_NODE=1: 단일 GPU
#   - LIMIT_TRAIN_BATCHES=50: 빠른 스모크 (50 step 만)
#   - LIMIT_VAL_BATCHES=5: 빠른 val
#   - VAL_CHECK_INTERVAL=25: 25 step 마다 val
#   - LR=5e-6 (default): single scenario 에서는 multi 보다 높여도 됨 (1e-5 까지)
#   - OCSC_EVAL_HARD_RMM_INTERVAL=5: 5 step 마다 HardRMM (single B 는 계산이 빠름)
# =============================================================================

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
# - 학습 자체 self-val 은 보통 N_BATCH_SIM_AGENTS_METRIC=1 batch 만 본다.
# - 16 워커는 GPU 메모리 attach 만 ~6.7GB 차지하므로 학습 중에는 4 권장:
#     WOSAC_HARD_POOL_WORKERS=4 sh scripts/train_flow_consistency_bptt_single.sh
# - hard: HardSimAgentsMetrics, real: official TF SimAgentsMetrics.
export WOSAC_HARD_POOL_WORKERS="${WOSAC_HARD_POOL_WORKERS:-16}"
export WOSAC_REAL_POOL_WORKERS="${WOSAC_REAL_POOL_WORKERS:-16}"
# 1로 두면 hard vs real 결과 cross-check (느려짐, 디버깅용).
export WOSAC_VERIFY="${WOSAC_VERIFY:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

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

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"
FIXED_SCENARIO_MODE="${FIXED_SCENARIO_MODE:-false}"
FIXED_SCENARIO_INDEX="${FIXED_SCENARIO_INDEX:-0}"
FIXED_SCENARIO_COUNT="${FIXED_SCENARIO_COUNT:-4}"
FIXED_SCENARIO_PKL="${FIXED_SCENARIO_PKL:-}"

# ── Single-scenario 특화 기본값 ─────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# TRAIN_B / VAL_B: 각 train/val dataloader의 batch size (배치 "크기")
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-16}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-32}"
# LIMIT_*_BATCHES: "배치 개수 제한" (batch size와 다른 개념)
# - float(0~1]: 전체 dataloader 중 비율
# - int(>=1): 실제 배치 개수
# - val을 끄려면 LIMIT_VAL_BATCHES=0
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
# VAL_CHECK_INTERVAL: step(배치) 단위 validation 주기. 0 금지.
# CHECK_VAL_EVERY_N_EPOCH: epoch 단위 validation 주기. 0 금지.
# 실제 validation 실행 조건은 둘 다 만족해야 하며, val을 끄려면 LIMIT_VAL_BATCHES=0 사용.
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
if [ -z "${LR_TOTAL_STEPS}" ] || [ "${LR_TOTAL_STEPS}" = "-1" ]; then
  LR_TOTAL_STEPS=$(python3 - <<PY
import pathlib, math
p = pathlib.Path("${TRAIN_RAW_DIR}")
n = len(list(p.glob("*.pkl")))
if n > 0:
    steps_per_epoch = math.ceil(n / (${TRAIN_B} * ${NPROC_PER_NODE}))
    print(steps_per_epoch * ${MAX_EPOCHS})
else:
    print(1000)
PY
  )
  echo "[LR schedule] auto LR_TOTAL_STEPS=${LR_TOTAL_STEPS}"
fi

FLOW_SOLVER_METHOD="${FLOW_SOLVER_METHOD:-euler}"
FLOW_SOLVER_STEPS="${FLOW_SOLVER_STEPS:-16}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"
# N_ROLLOUT_CLOSED_VAL: validation에서 시나리오당 closed-loop rollout 샘플 개수.
#   - 0이면 closed-loop 결과 텐서가 비어 validation 단계에서 오류가 날 수 있음.
#   - validation을 켠다면 보통 1 이상 권장.
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
# N_BATCH_SIM_AGENTS_METRIC: val에서 sim-agents metric(WOSAC 계열)을 계산할 배치 수 제한.
#   - 0: metric 계산 스킵
#   - 1 이상: 앞에서 N개 val batch만 metric 계산 (속도/안정성 트레이드오프)
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10000}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-0}"

# ── OCSC 파라미터 ──────────────────────────────────────────────────────────
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-4}"
# OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-smooth_l1}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_USE_MMD="${OCSC_USE_MMD:-false}"
OCSC_ANCHOR_STRIDE="${OCSC_ANCHOR_STRIDE:-1}"
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-true}"
OCSC_TARGET_MAX_STEPS="${OCSC_TARGET_MAX_STEPS:-2}"
OCSC_PRED_MAX_STEPS="${OCSC_PRED_MAX_STEPS:-2}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.0}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-0.0}"
OCSC_REL_DISP_WEIGHT="${OCSC_REL_DISP_WEIGHT:-1.0}"
# GT FM regularization: velocity_head가 GT에서 drift하지 않도록 per-anchor FM loss를 MMD와 함께 backward.
# 0.0이면 기존 동작(MMD only). 권장 시작값: 0.1~1.0
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA:-0.1}"
# 사용자가 실수로 "=0.1" 형태로 넘긴 경우(Hydra override 파싱 오류) 보정.
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA#=}"
# single B: HardRMM 계산이 빠르므로 5 step 마다 (multi B 에서는 1 또는 더 높게)
OCSC_EVAL_HARD_RMM="${OCSC_EVAL_HARD_RMM:-false}"
OCSC_EVAL_HARD_RMM_INTERVAL="${OCSC_EVAL_HARD_RMM_INTERVAL:-10}"
# GT target: true → open-loop sample 대신 GT 궤적을 consistency target으로 사용.
# CL 예측을 2Hz로 다운샘플 후 GT와 비교 (ocsc_use_mmd에 따라 MMD 또는 masked L2).
OCSC_GT_TARGET="${OCSC_GT_TARGET:-true}"

# ── BPTT tricks ────────────────────────────────────────────────────────────
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
BPTT_LAST_N_COARSE_STEPS="${BPTT_LAST_N_COARSE_STEPS:-0}"
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-0.0}"
BPTT_LAST_COARSE_ONLY="${BPTT_LAST_COARSE_ONLY:-true}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"
FLOW_REG_LAMBDA="${FLOW_REG_LAMBDA:-0.0}"

WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
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
  exit 1
fi

# 고정 시나리오 모드:
# - FIXED_SCENARIO_PKL 이 주어지면 해당 파일 1개만 사용
# - 아니면 TRAIN_RAW_DIR 의 정렬된 pkl 목록에서
#   [FIXED_SCENARIO_INDEX, FIXED_SCENARIO_INDEX + FIXED_SCENARIO_COUNT) 구간 사용
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

echo "[single-scenario] Experiment=${MY_EXPERIMENT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "TRAIN_RAW_DIR=${TRAIN_RAW_DIR}"
echo "TRAIN_RAW_DIR_EFFECTIVE=${TRAIN_RAW_DIR_EFFECTIVE}"
echo "FIXED_SCENARIO_MODE=${FIXED_SCENARIO_MODE} FIXED_SCENARIO_INDEX=${FIXED_SCENARIO_INDEX} FIXED_SCENARIO_COUNT=${FIXED_SCENARIO_COUNT}"
if [ -n "${FIXED_SCENARIO_PKL}" ]; then
  echo "FIXED_SCENARIO_PKL=${FIXED_SCENARIO_PKL}"
fi
echo "CKPT_PATH=${CKPT_PATH}"
echo "NPROC=${NPROC_PER_NODE} TRAIN_B=${TRAIN_B} MAX_EPOCHS=${MAX_EPOCHS} LIMIT_TRAIN=${LIMIT_TRAIN_BATCHES}"
echo "SEED=${SEED} DATA_SHUFFLE=${DATA_SHUFFLE} DETERMINISTIC=${TRAINER_DETERMINISTIC}"
echo "OCSC_N_ROLLOUTS=${OCSC_N_ROLLOUTS} OCSC_LOSS_TYPE=${OCSC_LOSS_TYPE} OCSC_TARGET=${OCSC_TARGET_MAX_STEPS}cs OCSC_PRED=${OCSC_PRED_MAX_STEPS}cs"
echo "OCSC_HEADING_WEIGHT=${OCSC_HEADING_WEIGHT} OCSC_POSITION_WEIGHT=${OCSC_POSITION_WEIGHT} OCSC_REL_DISP_WEIGHT=${OCSC_REL_DISP_WEIGHT}"
echo "OCSC_FM_REG_LAMBDA=${OCSC_FM_REG_LAMBDA} OCSC_GT_TARGET=${OCSC_GT_TARGET}"
echo "OCSC_EVAL_HARD_RMM=${OCSC_EVAL_HARD_RMM} interval=${OCSC_EVAL_HARD_RMM_INTERVAL}"
echo "BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT} BPTT_GRAD_CLIP=${BPTT_GRAD_CLIP_TRAJ} LR=${LR}"
echo "OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUM_WORKERS=${NUM_WORKERS} WOSAC_HARD_POOL=${WOSAC_HARD_POOL_WORKERS} WOSAC_REAL_POOL=${WOSAC_REAL_POOL_WORKERS} WOSAC_VERIFY=${WOSAC_VERIFY}"

# num_workers=0 인 경우 torch DataLoader 제약상 prefetch_factor를 넘기면 안 됩니다.
PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  PREFETCH_ARG="data.prefetch_factor=${PREFETCH_FACTOR}"
fi

PORT="$(get_free_port)"
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
  trainer.deterministic="${TRAINER_DETERMINISTIC}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
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
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.decoder.flow_solver_method="${FLOW_SOLVER_METHOD}" \
  model.model_config.decoder.flow_solver_steps="${FLOW_SOLVER_STEPS}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.flow_reg_lambda="${FLOW_REG_LAMBDA}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_use_mmd="${OCSC_USE_MMD}" \
  model.model_config.finetune.ocsc_anchor_stride="${OCSC_ANCHOR_STRIDE}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_target_max_steps="${OCSC_TARGET_MAX_STEPS}" \
  model.model_config.finetune.ocsc_pred_max_steps="${OCSC_PRED_MAX_STEPS}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_rel_disp_weight="${OCSC_REL_DISP_WEIGHT}" \
  model.model_config.finetune.ocsc_fm_reg_lambda="${OCSC_FM_REG_LAMBDA}" \
  model.model_config.finetune.ocsc_eval_hard_rmm="${OCSC_EVAL_HARD_RMM}" \
  model.model_config.finetune.ocsc_eval_hard_rmm_interval="${OCSC_EVAL_HARD_RMM_INTERVAL}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_coarse_steps="${BPTT_LAST_N_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_last_coarse_only="${BPTT_LAST_COARSE_ONLY}" \
  ${PREFETCH_ARG} \
  ${EXTRA_ARGS} 
