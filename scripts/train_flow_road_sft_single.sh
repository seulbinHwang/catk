#!/bin/sh
# =============================================================================
# RoaD (Rollouts as Demonstrations) closed-loop SFT — single-GPU 런처
# OCSC 비교용 baseline.  train_flow_consistency_bptt_single.sh 의 RoaD 버전.
# =============================================================================
# 사용법:
#   sh scripts/train_flow_road_sft_single.sh
#   ROAD_SAMPLE_K=32 LR=1e-6 sh scripts/train_flow_road_sft_single.sh
#   WOSAC_HARD_POOL_WORKERS=4 OMP_NUM_THREADS=4 sh scripts/train_flow_road_sft_single.sh
#
# RoaD 알고리즘:
#   1. Expert-guided closed-loop rollout: 매 coarse step 마다 K 개 후보 샘플 →
#      GT continuation 에 weighted step-wise L2 최소인 후보 선택 → commit.
#   2. BC loss: 선택된 후보를 target 으로 flow-matching loss (BPTT 없음).
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export WOSAC_HARD_POOL_WORKERS="${WOSAC_HARD_POOL_WORKERS:-16}"
export WOSAC_REAL_POOL_WORKERS="${WOSAC_REAL_POOL_WORKERS:-16}"
export WOSAC_VERIFY="${WOSAC_VERIFY:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
# GPU 2/3 우선 (CLAUDE.md 규칙).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_road_sft}"
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
# OCSC_clean 은 반드시 85MB ckpt (CLAUDE.md).
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"

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
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
SEED="${SEED:-817}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
TRAINER_DETERMINISTIC="${TRAINER_DETERMINISTIC:-true}"

# LR 셋팅은 flow-matching pretraining (pre_bc_flow.yaml) 참조: lr=4e-4 / min_ratio=1e-2.
# RoaD BC loss = pretraining 과 동일 objective. fine-tuning 이라 warmup 만 약간 키움.
LR="${LR:-4e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# pretraining 과 동일하게 gradient clip 1.0 (road_ft 는 manual opt → finetune.gradient_clip_val).
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"
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

N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10000}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-0}"

# ── RoaD 파라미터 ──────────────────────────────────────────────────────────
# K: Sample-K 후보 개수 (논문 기본값 64).
ROAD_SAMPLE_K="${ROAD_SAMPLE_K:-64}"
# 시나리오당 expert-guided rollout 수 (N_roll). 논문 Table 4: 많을수록 단조 개선.
ROAD_N_ROLLOUTS="${ROAD_N_ROLLOUTS:-3}"
# expert-guided rollout coarse step 수 (16 = 8초 full episode).
ROAD_PRED_MAX_STEPS="${ROAD_PRED_MAX_STEPS:-16}"
# 후보 샘플링 noise scale (논문 temperature 0.8).
ROAD_TEMPERATURE="${ROAD_TEMPERATURE:-0.8}"
# d^g (Eq.6) position / heading 채널 가중치.
ROAD_POSITION_WEIGHT="${ROAD_POSITION_WEIGHT:-1.0}"
ROAD_HEADING_WEIGHT="${ROAD_HEADING_WEIGHT:-0.1}"
# d^g 비교 horizon H_t (fine step 수, 논문 first 20 = 2초).
ROAD_COMPARISON_HORIZON="${ROAD_COMPARISON_HORIZON:-20}"
# True → BC term 에 horizon 전체 GT valid 인 agent 만 포함.
ROAD_STRICT_ACTIVE_MASK="${ROAD_STRICT_ACTIVE_MASK:-true}"
# 학습 중 free-running closed-loop hard RMM 모니터링 (기본 off — 속도 저하 큼).
# RMM/CPD/CES 는 validation 에서 측정됨.
ROAD_EVAL_HARD_RMM="${ROAD_EVAL_HARD_RMM:-false}"
ROAD_EVAL_HARD_RMM_INTERVAL="${ROAD_EVAL_HARD_RMM_INTERVAL:-10}"

# 학습 대상 module 선택. RoaD 는 정책 전체 SFT 라 "full" 이 가장 충실.
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"
FLOW_FT_TARGET="${FLOW_FT_TARGET:-full}"

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

echo "[road-sft] Experiment=${MY_EXPERIMENT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "NPROC=${NPROC_PER_NODE} TRAIN_B=${TRAIN_B} MAX_EPOCHS=${MAX_EPOCHS} LR=${LR}"
echo "ROAD_SAMPLE_K=${ROAD_SAMPLE_K} ROAD_N_ROLLOUTS=${ROAD_N_ROLLOUTS} ROAD_PRED_MAX_STEPS=${ROAD_PRED_MAX_STEPS}"
echo "ROAD_TEMPERATURE=${ROAD_TEMPERATURE} ROAD_POS_W=${ROAD_POSITION_WEIGHT} ROAD_HEAD_W=${ROAD_HEADING_WEIGHT}"
echo "ROAD_COMPARISON_HORIZON=${ROAD_COMPARISON_HORIZON} ROAD_STRICT_ACTIVE_MASK=${ROAD_STRICT_ACTIVE_MASK}"
echo "ROAD_EVAL_HARD_RMM=${ROAD_EVAL_HARD_RMM} interval=${ROAD_EVAL_HARD_RMM_INTERVAL}"
echo "OMP=${OMP_NUM_THREADS} NUM_WORKERS=${NUM_WORKERS} WOSAC_HARD_POOL=${WOSAC_HARD_POOL_WORKERS}"

PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  PREFETCH_ARG="data.prefetch_factor=${PREFETCH_FACTOR}"
fi

PORT="$(get_free_port)"
ACTION="${ACTION:-finetune}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  seed="${SEED}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
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
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.flow_ft_target="${FLOW_FT_TARGET}" \
  model.model_config.finetune.gradient_clip_val="${GRAD_CLIP_VAL}" \
  model.model_config.finetune.road_sample_k="${ROAD_SAMPLE_K}" \
  model.model_config.finetune.road_n_rollouts="${ROAD_N_ROLLOUTS}" \
  model.model_config.finetune.road_pred_max_steps="${ROAD_PRED_MAX_STEPS}" \
  model.model_config.finetune.road_temperature="${ROAD_TEMPERATURE}" \
  model.model_config.finetune.road_position_weight="${ROAD_POSITION_WEIGHT}" \
  model.model_config.finetune.road_heading_weight="${ROAD_HEADING_WEIGHT}" \
  model.model_config.finetune.road_comparison_horizon="${ROAD_COMPARISON_HORIZON}" \
  model.model_config.finetune.road_strict_active_mask="${ROAD_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.road_eval_hard_rmm="${ROAD_EVAL_HARD_RMM}" \
  model.model_config.finetune.road_eval_hard_rmm_interval="${ROAD_EVAL_HARD_RMM_INTERVAL}" \
  ${PREFETCH_ARG} \
  ${EXTRA_ARGS}
