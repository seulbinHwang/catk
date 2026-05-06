#!/bin/sh
# OCSC (Open-Closed Self-Consistency) 파인튜닝 — single-GPU smoke / debug 프리셋.
#
# 사용법:
#   sh scripts/train_flow_consistency_bptt_single.sh
#   OCSC_N_ROLLOUTS=4 MAX_EPOCHS=5 sh scripts/train_flow_consistency_bptt_single.sh
#   # validation worker / thread 조정 예:
#   WOSAC_HARD_POOL_WORKERS=2 OMP_NUM_THREADS=4 NUM_WORKERS=2 \
#     sh scripts/train_flow_consistency_bptt_single.sh
#
# 핵심 토글 (자세히는 configs/experiment/flow_consistency_bptt.yaml 참조):
#   OCSC_GT_TARGET=true             → GT 궤적을 consistency target 으로 사용.
#   OCSC_USE_MMD=false              → paired L2 ablation.
#   OCSC_USE_PRETRAINED_REF=true    → frozen reference 로 OL 생성.
#   OCSC_SHARE_NOISE_TAPE=false     → OL/CL noise 분리.
#   BPTT_SEQUENTIAL_ROLLOUTS=true   → 메모리 ↓, manual optimization 자동 활성.
#   BPTT_USE_ADJOINT=true           → flow_ode model_fn ckpt.
#   BPTT_LAST_COARSE_ONLY=true      → 마지막 coarse step 만 grad.
#   VALIDATION_METRIC=real          → 공식 TF metric (default: hard).

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
# Validation 시 sim-agents metric forkserver pool 크기.
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

CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

# ── Single-GPU 스모크 기본값 ───────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_B="${TRAIN_B:-4}"
VAL_B="${VAL_B:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-25}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-817}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-0}"

# ── Validation backend ──────────────────────────────────────────────────────
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-2}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-4}"

# ── OCSC 핵심 ───────────────────────────────────────────────────────────────
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-2}"
OCSC_ANCHOR_STRIDE="${OCSC_ANCHOR_STRIDE:-4}"
OCSC_PRED_MAX_STEPS="${OCSC_PRED_MAX_STEPS:-4}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_USE_MMD="${OCSC_USE_MMD:-true}"
OCSC_GT_TARGET="${OCSC_GT_TARGET:-false}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-1.0}"
OCSC_REL_DISP_WEIGHT="${OCSC_REL_DISP_WEIGHT:-0.0}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.0}"
OCSC_SHARE_NOISE_TAPE="${OCSC_SHARE_NOISE_TAPE:-true}"
OCSC_SHARE_NOISE_ACROSS_TIME="${OCSC_SHARE_NOISE_ACROSS_TIME:-false}"
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-false}"
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA:-0.0}"

# ── Freeze 정책 ─────────────────────────────────────────────────────────────
TRAIN_FULL_FLOW_DECODER_ONLY="${TRAIN_FULL_FLOW_DECODER_ONLY:-false}"
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"

# ── BPTT 메모리 / 속도 토글 ────────────────────────────────────────────────
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-false}"
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
BPTT_LAST_COARSE_ONLY="${BPTT_LAST_COARSE_ONLY:-false}"
BPTT_LAST_N_COARSE_STEPS="${BPTT_LAST_N_COARSE_STEPS:-0}"
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-1.0}"
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"

# ── HardRMM 모니터링 (학습 step) ────────────────────────────────────────────
OCSC_EVAL_HARD_RMM="${OCSC_EVAL_HARD_RMM:-false}"
OCSC_EVAL_HARD_RMM_INTERVAL="${OCSC_EVAL_HARD_RMM_INTERVAL:-10}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[WARN] CKPT_PATH not found: ${CKPT_PATH}"
  echo "       OCSC 는 사전학습 체크포인트로부터의 fine-tuning 이 표준입니다."
fi

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

PORT="$(get_free_port)"
echo "[ocsc-single] experiment=${MY_EXPERIMENT} task=${MY_TASK_NAME}"
echo "  OCSC: G=${OCSC_N_ROLLOUTS} stride=${OCSC_ANCHOR_STRIDE} pred=${OCSC_PRED_MAX_STEPS}"
echo "        gt_target=${OCSC_GT_TARGET} use_mmd=${OCSC_USE_MMD} loss=${OCSC_LOSS_TYPE}"
echo "        share_tape=${OCSC_SHARE_NOISE_TAPE} share_time=${OCSC_SHARE_NOISE_ACROSS_TIME}"
echo "        ref=${OCSC_USE_PRETRAINED_REF} fm_reg=${OCSC_FM_REG_LAMBDA}"
echo "  BPTT: adjoint=${BPTT_USE_ADJOINT} last_solver=${BPTT_LAST_N_SOLVER_STEPS}"
echo "        warm=${BPTT_WARM_COARSE_STEPS} last_only=${BPTT_LAST_COARSE_ONLY}"
echo "        last_n=${BPTT_LAST_N_COARSE_STEPS} grad_clip=${BPTT_GRAD_CLIP_TRAJ}"
echo "        sequential=${BPTT_SEQUENTIAL_ROLLOUTS}"
echo "  freeze: full_decoder=${TRAIN_FULL_FLOW_DECODER_ONLY} velocity_only=${FLOW_VELOCITY_HEAD_ONLY}"
echo "  validation: ${VALIDATION_METRIC} (n_batch=${N_BATCH_SIM_AGENTS_METRIC} n_rollout=${N_ROLLOUT_CLOSED_VAL})"
echo "  hard_rmm monitor: ${OCSC_EVAL_HARD_RMM} interval=${OCSC_EVAL_HARD_RMM_INTERVAL}"
echo "  LR=${LR} TRAIN_B=${TRAIN_B} MAX_EPOCHS=${MAX_EPOCHS}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRAD_CLIP_VAL}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.finetune.train_full_flow_decoder_only="${TRAIN_FULL_FLOW_DECODER_ONLY}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_anchor_stride="${OCSC_ANCHOR_STRIDE}" \
  model.model_config.finetune.ocsc_pred_max_steps="${OCSC_PRED_MAX_STEPS}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_use_mmd="${OCSC_USE_MMD}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_rel_disp_weight="${OCSC_REL_DISP_WEIGHT}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.finetune.ocsc_share_noise_tape="${OCSC_SHARE_NOISE_TAPE}" \
  model.model_config.finetune.ocsc_share_noise_across_time="${OCSC_SHARE_NOISE_ACROSS_TIME}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_fm_reg_lambda="${OCSC_FM_REG_LAMBDA}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_coarse_only="${BPTT_LAST_COARSE_ONLY}" \
  model.model_config.finetune.bptt_last_n_coarse_steps="${BPTT_LAST_N_COARSE_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.ocsc_eval_hard_rmm="${OCSC_EVAL_HARD_RMM}" \
  model.model_config.finetune.ocsc_eval_hard_rmm_interval="${OCSC_EVAL_HARD_RMM_INTERVAL}" \
  ${EXTRA_ARGS}

echo "[ocsc-single] done"
