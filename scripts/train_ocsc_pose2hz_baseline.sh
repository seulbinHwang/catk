#!/bin/sh
# ==========================================================================
# OCSC (Open-Closed Self-Consistency) — pose 4-dim 2Hz coarse baseline.
#
# 현재 (2026-05-29) 가장 안정적으로 도는 세팅을 default 로 둔 wrapper.
# self_forced framework 와 격리된 별도 mode (finetune.mode=ocsc_ft).
# training_step 이 _run_flow_ocsc_ft_step 으로 분기.
#
# ── 알고리즘 한 줄 ────────────────────────────────────────────────────────
# 매 batch 마다 anchor 0 (history end) 기준으로
#   M=8 open-loop samples (frozen ref_flow_decoder, no_grad)
#   + G=4 closed-loop rollouts (student, with grad)
#   + per-CL nearest OL match (pose-norm flat L2 거리)
#   + paired pose-norm L2 (pos_w=1.0, head_w=0.01)
# 를 학습한다.  10 Hz fine 20 step 중 2 Hz coarse 4 points
# (0.5 / 1.0 / 1.5 / 2.0 s 끝점) 만 매칭한다.  pose 4-dim
# ``[x/20, y/20, cos, sin]`` in anchor 0 local frame.
#
# ── DMD (self_forced_npfm_pareto) 와의 비교 ───────────────────────────────
# DMD :  closed-loop generator vs critic (fake estimator) — 분포 매칭, R-F.
# OCSC:  closed-loop generator vs frozen ref decoder OL distribution —
#        critic 없음, OL sample 이 target.  mode 다양성 보존.
#
# ── 사용 예 ───────────────────────────────────────────────────────────────
#
# 1. 기본 (현재 도는 세팅 그대로, GPU 2, 3):
#    bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 2. smoke (1 GPU, 1 batch):
#    CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 \
#      MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=2 LIMIT_VAL_BATCHES=0 \
#      TRAIN_B=2 VAL_B=2 OCSC_N_ROLLOUTS=2 OCSC_N_OL_ROLLOUTS=4 \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 3. GT target mode (OL 대신 GT 1개 target, nearest 비활성):
#    OCSC_GT_TARGET=true OCSC_OL_NEAREST_MATCH=false \
#      MY_TASK_NAME=ocsc_gt_baseline \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 4. velocity_head 외 step_refiner 도 같이 학습:
#    OCSC_VELOCITY_HEAD_ONLY=false \
#      MY_TASK_NAME=ocsc_velhead_plus_refiner \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 5. full flow_decoder unfreeze (capacity max):
#    OCSC_VELOCITY_HEAD_ONLY=false OCSC_FULL_FLOW_DECODER=true \
#      MY_TASK_NAME=ocsc_full_decoder \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 6. M / G 변경 (sweep):
#    OCSC_N_ROLLOUTS=2 OCSC_N_OL_ROLLOUTS=16 \
#      MY_TASK_NAME=ocsc_g2_m16 \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# 7. heading weight sweep:
#    OCSC_HEADING_WEIGHT=0.1 MY_TASK_NAME=ocsc_head_w0.1 \
#      bash scripts/train_ocsc_pose2hz_baseline.sh
#
# ── 룰 (CLAUDE.md §6) ─────────────────────────────────────────────────────
# GPU 는 2, 3 만 사용.  default CUDA_VISIBLE_DEVICES=2,3 / NPROC_PER_NODE=2.
# 다른 GPU 는 동료가 점유 중일 수 있음.
# ==========================================================================

# ────────────────────────────────────────────────────────────────────────
# 0. Shell / 환경
# ────────────────────────────────────────────────────────────────────────
export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"

# CLAUDE.md §6: 이 repo 의 모든 학습/평가는 GPU 2, 3 만 사용.
# default 는 2 GPU DDP (gpu 2,3) — 가장 빠른 sweep / 본 학습 세팅.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# WandB 모드.  offline / disabled 도 가능.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"

# ────────────────────────────────────────────────────────────────────────
# 1. Conda env activate
# ────────────────────────────────────────────────────────────────────────
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# ────────────────────────────────────────────────────────────────────────
# 2. Top-level Hydra entry & 경로 / 이름
# ────────────────────────────────────────────────────────────────────────
MY_EXPERIMENT="${MY_EXPERIMENT:-ocsc_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-ocsc_pose2hz_baseline}"

# Hydra action:
#   - finetune : weight-only 로딩(strict=False) + 새 optimizer.  ★ default
#   - fit      : Lightning 전체 resume (optimizer/scheduler 포함, ckpt 에서 재개)
#   - validate : trainer.validate (RMM + CPD 로컬 계산만)
#   - test     : trainer.test
ACTION="${ACTION:-finetune}"

# Random seed.
SEED="${SEED:-817}"

# WOMD cache root.  cache 안에는 training/ validation/ testing/
# validation_tfrecords_splitted/ 폴더가 있어야 함.
# Prefer the Ubuntu server cache and fall back to mounted pod caches.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -f "${SCRIPT_DIR}/resolve_womd_cache_root.sh" ]; then
  . "${SCRIPT_DIR}/resolve_womd_cache_root.sh"
fi
if command -v resolve_womd_cache_root >/dev/null 2>&1; then
  RESOLVED_CACHE_ROOT="$(resolve_womd_cache_root)" || exit 1
  CACHE_ROOT="${RESOLVED_CACHE_ROOT}"
else
  CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
fi
export CACHE_ROOT

# Pretrained backbone checkpoint.  fine-tune 의 출발점.
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

TMUX_LOG_TAIL="${TMUX_LOG_TAIL:-false}"
TMUX_LOG_PATH="${TMUX_LOG_PATH:-/tmp/${MY_TASK_NAME}.log}"
TMUX_LOG_WINDOW="${TMUX_LOG_WINDOW:-${MY_TASK_NAME}:log}"
TMUX_LOG_SESSION="${TMUX_LOG_SESSION:-}"

is_true() {
  case "$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

quote_sh() {
  printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

setup_tmux_log_tail() {
  if ! is_true "${TMUX_LOG_TAIL}"; then
    return 0
  fi
  if [ -z "${TMUX:-}" ] || [ -z "${TMUX_PANE:-}" ]; then
    return 0
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p "$(dirname "${TMUX_LOG_PATH}")"
  : > "${TMUX_LOG_PATH}"

  quoted_log="$(quote_sh "${TMUX_LOG_PATH}")"
  tmux pipe-pane -o -t "${TMUX_PANE}" "cat >> ${quoted_log}" >/dev/null 2>&1 || true
  trap 'tmux pipe-pane -t "${TMUX_PANE}" >/dev/null 2>&1 || true' EXIT HUP INT TERM

  session="${TMUX_LOG_SESSION}"
  if [ -z "${session}" ]; then
    session="$(tmux display-message -p '#S' 2>/dev/null || true)"
  fi
  if [ -n "${session}" ]; then
    TMUX_TAIL_WORKDIR="$(pwd)" "${SCRIPT_DIR}/tmux_tail_log.sh" "${TMUX_LOG_PATH}" "${TMUX_LOG_WINDOW}" "${session}" >/dev/null 2>&1 || true
  fi
}

setup_tmux_log_tail

# ────────────────────────────────────────────────────────────────────────
# 3. Trainer (Lightning)
# ────────────────────────────────────────────────────────────────────────
# 분산 학습 — default 2 GPU DDP.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_NODES="${NUM_NODES:-1}"

# Epoch / step
MAX_EPOCHS="${MAX_EPOCHS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"   # 1.0 = 전체, smoke 시 정수도 가능
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"       # default 10 %, val 시간 ~23 분/cycle

# Validation 빈도:
#   step 단위 평가 (sweep 빠른 탐색용; default 200 step):
#     VAL_CHECK_INTERVAL=<int>  +  CHECK_VAL_EVERY_N_EPOCH=null
#   epoch 단위 평가 (긴 학습용):
#     VAL_CHECK_INTERVAL=null   +  CHECK_VAL_EVERY_N_EPOCH=<int>
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"

# precision: bf16-mixed (default, H100 권장), fp16-mixed, 32-true (재현 디버그).
PRECISION="${PRECISION:-bf16-mixed}"

# DDP strategy.  optimizer/DDP는 requires_grad=True 파라미터만 보므로 OCSC
# velocity_head_only에서도 find_unused scan 없이 기본 DDP를 쓴다.
if [ "${NPROC_PER_NODE}" -gt 1 ] || [ "${NUM_NODES}" -gt 1 ]; then
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
else
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-auto}"
fi

# OCSC mode 는 automatic_optimization (manual 아님) → Lightning gradient clip 적용.
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-false}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"

# ────────────────────────────────────────────────────────────────────────
# 4. Data (configs/data/waymo.yaml)
# ────────────────────────────────────────────────────────────────────────
# 현재 도는 baseline: train_B=16 / val_B=32 (2 GPU 합산 effective 32 / 64).
# val_B 늘려도 fast WOSAC scoring 이 per-scenario serial 이라
# total validation 시간은 별로 안 줄어듦 (~23 분/cycle 고정).
TRAIN_B="${TRAIN_B:-16}"
VAL_B="${VAL_B:-32}"
TEST_B="${TEST_B:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-${NUM_WORKERS}}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
EVAL_PERSISTENT_WORKERS="${EVAL_PERSISTENT_WORKERS:-true}"
EVAL_PIN_MEMORY="${EVAL_PIN_MEMORY:-${PIN_MEMORY}}"
EVAL_MULTIPROCESSING_CONTEXT="${EVAL_MULTIPROCESSING_CONTEXT:-spawn}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"

# OCSC 학습은 매 batch 의 closed-loop rollout 비용 큼 (G=4 CL).
# 0.5 = 한 epoch 에 train scenario 의 50 % 만 샘플 (1.0 = 전체).
TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.5}"
TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-true}"

# ────────────────────────────────────────────────────────────────────────
# 5. Generator 학습 (model.model_config)
# ────────────────────────────────────────────────────────────────────────
# OCSC mode 는 LambdaLR (cosine warmup/decay) 없이 self.lr 고정 사용.
# (AdamW 의 효과적 step size).
LR="${LR:-2.0e-6}"                # 현재 baseline.  velocity_head 만 학습이라 작게.
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-${MAX_EPOCHS}}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1.0}"   # 1.0 = decay 없음

# Validation rollout 비용 / closed-loop 평가
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-100000}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1728}"
SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-0}"

VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-true}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"

# ────────────────────────────────────────────────────────────────────────
# 6. OCSC core knob — 알고리즘 본체 ★
# ────────────────────────────────────────────────────────────────────────
#
# G (closed-loop rollouts):  student 가 매 step 만드는 closed-loop rollout 수.
#   매 g 마다 새 noise seed 로 다른 rollout (mode 다양성 확보).  메모리/시간
#   비용은 G 에 거의 linear.  default 4.
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-4}"

# M (open-loop samples):  frozen ref_flow_decoder 가 만드는 OL sample 수.
#   -1 이면 M = G (paired).  M > G 면 candidate pool 확장 → 더 다양한 mode 매칭.
#   nearest_match=True 면 M >= G 필수.  default 8 (M=2G).
OCSC_N_OL_ROLLOUTS="${OCSC_N_OL_ROLLOUTS:-8}"

# Nearest match:  각 CL g 에 대해 M OL 중 pose-norm flat L2 가 가장 작은 OL 을
# target 으로 매칭 → paired L2.  False 면 단순 mean (M == G 필수).  default true.
OCSC_OL_NEAREST_MATCH="${OCSC_OL_NEAREST_MATCH:-true}"

# Loss type — 현재 l2 만 지원 (smooth_l1 / l1 / pwil 미지원, 최소 port).
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"

# GT target mode:  True 면 OL sample 무시하고 GT 궤적 1 개를 target 으로 사용.
#   M=1, nearest_match 비활성.  diversity 보존 X, 단순 closed-loop GT BC.
#   default false (OCSC 본래 동작).
OCSC_GT_TARGET="${OCSC_GT_TARGET:-false}"

# Pretrained ref decoder:  True 면 OL 생성에 frozen pretrained flow_decoder deepcopy 사용.
#   False 면 student 자신의 flow_decoder 로 OL → 학습 진행할수록 OL distribution 도 같이 drift.
#   default true (안정성).
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-true}"

# Anchor index:  사용할 anchor (0 = history end).
#   현재 single anchor 만 지원 (multi-anchor 는 후속 port).  default 0.
OCSC_ANCHOR_IDX="${OCSC_ANCHOR_IDX:-0}"

# Strict active mask: future 2s가 모두 valid인 agent만 OCSC loss에 포함.
OCSC_STRICT_ACTIVE_MASK="${OCSC_STRICT_ACTIVE_MASK:-true}"

# Pose / heading weight:  pose 4-dim L2 의 두 component 가중치.
#   pos_w * mean((Δx/20)²+(Δy/20)²) + head_w * mean((Δcos)²+(Δsin)²)
#   OCSC clean 정합: pos_w=1.0, head_w=0.01 (heading 적게 — stopped agent noisy GT 영향 ↓).
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-1.0}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.01}"

# ────────────────────────────────────────────────────────────────────────
# 7. 학습 가능 범위 (Flow decoder 의 어떤 모듈만 unfreeze 할지)
# ────────────────────────────────────────────────────────────────────────
# velocity_head_only=True (default):
#   flow_decoder.velocity_head 만 trainable (9.6 K params).
#   capacity 가장 작음.  baseline RMM 안 망가뜨리고 미세 조정.
#   set_model_for_finetuning 이 step_refiner 도 unfreeze 하지만 __init__ 에서 다시 freeze.
#
# velocity_head_only=False + full_flow_decoder=False (default):
#   flow_decoder.step_refiner + flow_decoder.velocity_head (set_model_for_finetuning 의 default).
#
# velocity_head_only=False + full_flow_decoder=True:
#   flow_decoder 전체 trainable.  capacity 최대.  RMM ↑ 가능성 / 위험성도.
OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-true}"
OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-false}"

# ────────────────────────────────────────────────────────────────────────
# 8. Checkpoint / Callbacks
# ────────────────────────────────────────────────────────────────────────
# best ckpt 선정 monitor key.  RMM 기준 default (mode=max).
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"

# ────────────────────────────────────────────────────────────────────────
# 9. Logger (WandB)
# ────────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
# entity 빈 값으로 두면 wandb 의 user default entity 사용.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-[]}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-all}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

# ────────────────────────────────────────────────────────────────────────
# 10. Hydra CLI extras
# ────────────────────────────────────────────────────────────────────────
# 추가 override 가 필요하면 EXTRA_ARGS 에 공백 분리.
#   예:  EXTRA_ARGS="model.model_config.lr_warmup_steps=200"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ────────────────────────────────────────────────────────────────────────
# 11. Sanity check
# ────────────────────────────────────────────────────────────────────────
if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  echo "        Set CKPT_PATH=<path-to-pretrained.ckpt> or use ACTION=fit (resume)."
  exit 1
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

# ────────────────────────────────────────────────────────────────────────
# 12. Banner
# ────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[ocsc_pose2hz_baseline] launching ..."
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NPROC=${NPROC_PER_NODE}  NUM_NODES=${NUM_NODES}  ACTION=${ACTION}"
echo "  EXPERIMENT=${MY_EXPERIMENT}  TASK=${MY_TASK_NAME}"
echo "  CKPT_PATH=${CKPT_PATH}"
echo "------------------------------------------------------------"
echo "  Trainer: max_epochs=${MAX_EPOCHS} precision=${PRECISION} strategy=${TRAINER_STRATEGY}"
echo "    limit_train=${LIMIT_TRAIN_BATCHES} limit_val=${LIMIT_VAL_BATCHES}"
echo "    val_check_interval=${VAL_CHECK_INTERVAL} grad_clip=${GRADIENT_CLIP_VAL}"
echo "  Data: train_B=${TRAIN_B} val_B=${VAL_B} workers=${NUM_WORKERS}"
echo "    eval_workers=${EVAL_NUM_WORKERS} eval_mp=${EVAL_MULTIPROCESSING_CONTEXT}"
echo "    epoch_sample_frac=${TRAIN_EPOCH_SAMPLE_FRACTION}"
echo "  Generator: lr=${LR}"
echo "  OCSC core ★:"
echo "    G(n_rollouts)=${OCSC_N_ROLLOUTS}  M(n_ol_rollouts)=${OCSC_N_OL_ROLLOUTS}"
echo "    nearest_match=${OCSC_OL_NEAREST_MATCH}  loss_type=${OCSC_LOSS_TYPE}"
echo "    gt_target=${OCSC_GT_TARGET}  use_ref=${OCSC_USE_PRETRAINED_REF}"
echo "    anchor_idx=${OCSC_ANCHOR_IDX}  strict_active=${OCSC_STRICT_ACTIVE_MASK}"
echo "    pos_w=${OCSC_POSITION_WEIGHT}  head_w=${OCSC_HEADING_WEIGHT}"
echo "  Trainable range:"
echo "    velocity_head_only=${OCSC_VELOCITY_HEAD_ONLY}  full_flow_decoder=${OCSC_FULL_FLOW_DECODER}"
echo "  Checkpoint: monitor=${CHECKPOINT_MONITOR} mode=${CHECKPOINT_MODE}"
echo "  WandB: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} offline=${WANDB_OFFLINE}"
echo "============================================================"

# ────────────────────────────────────────────────────────────────────────
# 13. Launch
# ────────────────────────────────────────────────────────────────────────
PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  PREFETCH_ARG="data.prefetch_factor=${PREFETCH_FACTOR}"
fi
EVAL_PREFETCH_ARG=""
if [ "${EVAL_NUM_WORKERS}" -gt 0 ]; then
  EVAL_PREFETCH_ARG="data.eval_prefetch_factor=${EVAL_PREFETCH_FACTOR}"
fi
EVAL_MP_ARG=""
if [ -n "${EVAL_MULTIPROCESSING_CONTEXT}" ]; then
  EVAL_MP_ARG="data.eval_multiprocessing_context=${EVAL_MULTIPROCESSING_CONTEXT}"
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NUM_NODES}" \
  --master_port="${PORT}" \
  --rdzv_endpoint="127.0.0.1:${PORT}" \
  -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${NPROC_PER_NODE}" \
  trainer.num_nodes="${NUM_NODES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.precision="${PRECISION}" \
  trainer.strategy="${TRAINER_STRATEGY}" \
  trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}" \
  trainer.sync_batchnorm="${SYNC_BATCHNORM}" \
  trainer.num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.test_batch_size="${TEST_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  data.eval_num_workers="${EVAL_NUM_WORKERS}" \
  data.eval_persistent_workers="${EVAL_PERSISTENT_WORKERS}" \
  data.eval_pin_memory="${EVAL_PIN_MEMORY}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_use_eval_agent_selection="${TRAIN_USE_EVAL_AGENT_SELECTION}" \
  data.train_epoch_sample_fraction="${TRAIN_EPOCH_SAMPLE_FRACTION}" \
  callbacks.model_checkpoint.monitor="${CHECKPOINT_MONITOR}" \
  callbacks.model_checkpoint.mode="${CHECKPOINT_MODE}" \
  callbacks.model_checkpoint.save_top_k="${CHECKPOINT_SAVE_TOP_K}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.tags="${WANDB_TAGS}" \
  logger.wandb.log_model="${WANDB_LOG_MODEL}" \
  logger.wandb.offline="${WANDB_OFFLINE}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}" \
  model.model_config.sim_agents_metric_workers="${SIM_AGENTS_METRIC_WORKERS}" \
  model.model_config.val_open_loop="${VAL_OPEN_LOOP}" \
  model.model_config.val_closed_loop="${VAL_CLOSED_LOOP}" \
  model.model_config.finetune.enabled=true \
  model.model_config.finetune.mode=ocsc_ft \
  model.model_config.finetune.velocity_head_only="${OCSC_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.train_full_flow_decoder_only="${OCSC_FULL_FLOW_DECODER}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_n_ol_rollouts="${OCSC_N_OL_ROLLOUTS}" \
  model.model_config.finetune.ocsc_ol_nearest_match="${OCSC_OL_NEAREST_MATCH}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_anchor_idx="${OCSC_ANCHOR_IDX}" \
  model.model_config.finetune.ocsc_strict_active_mask="${OCSC_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.self_forced.enabled=false \
  ${PREFETCH_ARG} \
  ${EVAL_PREFETCH_ARG} \
  ${EVAL_MP_ARG} \
  ${EXTRA_ARGS}

echo "bash $(basename "$0") done!"
