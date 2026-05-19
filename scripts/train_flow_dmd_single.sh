#!/bin/sh
# =============================================================================
# Self-Forcing DMD single-GPU launcher — quick smoke / β ablation
# train_flow_consistency_bptt_single.sh 의 DMD 버전.
# =============================================================================
# 사용법:
#   sh scripts/train_flow_dmd_single.sh
#   DMD_BETA=0.5 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=10 sh scripts/train_flow_dmd_single.sh
#
# DMD 특징 (vs OCSC):
#   - 2 optimizer alternating: opt_gen (main flow_decoder) + opt_fake (critic 사본).
#   - Memory: ~3× flow_decoder (main + ref + fake_score).  BPTT_USE_ADJOINT=true 권장.
#   - HardRMM 모니터링 주기 5 step (DMD step 자체가 OCSC 보다 비싸므로).
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
# GPU 2/3 default per repo policy; override with CUDA_VISIBLE_DEVICES=2 for example.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_dmd}"
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

# ── Single-scenario / smoke defaults ─────────────────────────────────────────
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

# DMD: pretrained 근처 stable 학습을 위해 보수적 LR.
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

N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10000}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-0}"

# ── DMD 핵심 파라미터 ───────────────────────────────────────────────────────
# entropy knob (spec β).  1.0 = vanilla DMD, <1 = diversity↑, >1 = sharpening.
DMD_BETA="${DMD_BETA:-1.0}"
# 시나리오당 closed-loop rollout 수 (G).  보통 1 충분.
DMD_N_ROLLOUTS="${DMD_N_ROLLOUTS:-1}"
# closed-loop rollout coarse(2Hz) step 수.  T_10hz=N×shift; flow_decoder T=20 hardcode → N=4 필수.
DMD_PRED_MAX_STEPS="${DMD_PRED_MAX_STEPS:-4}"
# frozen ref_flow_decoder 를 real_score teacher 로 사용 (true 권장).
DMD_USE_REAL_SCORE="${DMD_USE_REAL_SCORE:-true}"
# fake_score lr scale (lr_fake = lr_gen × scale).
DMD_FAKE_LR_SCALE="${DMD_FAKE_LR_SCALE:-1.0}"
# Self-Forcing abs-mean normalizer (synthetic grad scale 안정화).
DMD_NORMALIZE="${DMD_NORMALIZE:-true}"
# anchor stride.
DMD_ANCHOR_STRIDE="${DMD_ANCHOR_STRIDE:-1}"
# future fine step valid 한 agent 만 anchor 로.
DMD_STRICT_ACTIVE_MASK="${DMD_STRICT_ACTIVE_MASK:-true}"
# 초기 N step fake_score-only warmup.
DMD_WARMUP_FAKE_ONLY_STEPS="${DMD_WARMUP_FAKE_ONLY_STEPS:-0}"
# generator 별도 grad clip (0 = bptt_grad_clip_traj 따름).
DMD_GEN_GRAD_CLIP="${DMD_GEN_GRAD_CLIP:-0.0}"
# HardRMM 모니터링 (DMD step 비싸니 interval 5 권장).
DMD_EVAL_HARD_RMM="${DMD_EVAL_HARD_RMM:-true}"
DMD_EVAL_HARD_RMM_INTERVAL="${DMD_EVAL_HARD_RMM_INTERVAL:-5}"

# ── BPTT (OCSC 와 공유) ────────────────────────────────────────────────────
# Memory 3× decoder 이므로 adjoint 강력 권장.
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
BPTT_LAST_N_COARSE_STEPS="${BPTT_LAST_N_COARSE_STEPS:-0}"
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-1.0}"
BPTT_LAST_COARSE_ONLY="${BPTT_LAST_COARSE_ONLY:-false}"
# DMD generator 측 학습 대상: "full" 권장 (decoder 전체 update).
# "velocity_head" 등 부분 학습은 critic 도 같은 분기 따라가도록 자동 처리.
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

echo "[dmd-single] Experiment=${MY_EXPERIMENT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "TRAIN_RAW_DIR=${TRAIN_RAW_DIR}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "NPROC=${NPROC_PER_NODE} TRAIN_B=${TRAIN_B} MAX_EPOCHS=${MAX_EPOCHS} LIMIT_TRAIN=${LIMIT_TRAIN_BATCHES}"
echo "SEED=${SEED} DATA_SHUFFLE=${DATA_SHUFFLE} DETERMINISTIC=${TRAINER_DETERMINISTIC}"
echo "DMD_BETA=${DMD_BETA} DMD_N_ROLLOUTS=${DMD_N_ROLLOUTS} DMD_PRED=${DMD_PRED_MAX_STEPS}cs DMD_USE_REAL_SCORE=${DMD_USE_REAL_SCORE}"
echo "DMD_FAKE_LR_SCALE=${DMD_FAKE_LR_SCALE} DMD_NORMALIZE=${DMD_NORMALIZE} DMD_ANCHOR_STRIDE=${DMD_ANCHOR_STRIDE}"
echo "DMD_STRICT=${DMD_STRICT_ACTIVE_MASK} DMD_WARMUP=${DMD_WARMUP_FAKE_ONLY_STEPS} DMD_GEN_CLIP=${DMD_GEN_GRAD_CLIP}"
echo "DMD_EVAL_HARD_RMM=${DMD_EVAL_HARD_RMM} interval=${DMD_EVAL_HARD_RMM_INTERVAL}"
echo "BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT} BPTT_GRAD_CLIP=${BPTT_GRAD_CLIP_TRAJ} LR=${LR}"
echo "FLOW_VELOCITY_HEAD_ONLY=${FLOW_VELOCITY_HEAD_ONLY} FLOW_FT_TARGET=${FLOW_FT_TARGET}"
echo "OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUM_WORKERS=${NUM_WORKERS} WOSAC_HARD_POOL=${WOSAC_HARD_POOL_WORKERS}"

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
  model.model_config.finetune.dmd_beta="${DMD_BETA}" \
  model.model_config.finetune.dmd_n_rollouts="${DMD_N_ROLLOUTS}" \
  model.model_config.finetune.dmd_pred_max_steps="${DMD_PRED_MAX_STEPS}" \
  model.model_config.finetune.dmd_use_real_score="${DMD_USE_REAL_SCORE}" \
  model.model_config.finetune.dmd_fake_lr_scale="${DMD_FAKE_LR_SCALE}" \
  model.model_config.finetune.dmd_normalize="${DMD_NORMALIZE}" \
  model.model_config.finetune.dmd_anchor_stride="${DMD_ANCHOR_STRIDE}" \
  model.model_config.finetune.dmd_strict_active_mask="${DMD_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.dmd_warmup_fake_only_steps="${DMD_WARMUP_FAKE_ONLY_STEPS}" \
  model.model_config.finetune.dmd_gen_grad_clip="${DMD_GEN_GRAD_CLIP}" \
  model.model_config.finetune.dmd_eval_hard_rmm="${DMD_EVAL_HARD_RMM}" \
  model.model_config.finetune.dmd_eval_hard_rmm_interval="${DMD_EVAL_HARD_RMM_INTERVAL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_coarse_steps="${BPTT_LAST_N_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_last_coarse_only="${BPTT_LAST_COARSE_ONLY}" \
  ${PREFETCH_ARG} \
  ${EXTRA_ARGS}
