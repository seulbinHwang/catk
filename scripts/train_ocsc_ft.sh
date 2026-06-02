#!/bin/sh
# ==========================================================================
# OCSC (Open-Closed Self-Consistency) fine-tuning launcher.
#
# self_forced framework 와 격리된 별도 mode (finetune.mode=ocsc_ft).
# training_step 이 _run_flow_ocsc_ft_step 으로 분기 — anchor 0 기준
# G CL rollouts + M OL samples + nearest match + paired L2.
#
# ── 사용 예 ───────────────────────────────────────────────────────────────
#  1. 1 GPU smoke (cuda:3):
#     CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 \
#       MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=2 LIMIT_VAL_BATCHES=2 \
#       bash scripts/train_ocsc_ft.sh
#
#  2. 2 GPU 본 학습 (GPU 2, 3):
#     CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
#       MY_TASK_NAME=ocsc_ft_v1 \
#       bash scripts/train_ocsc_ft.sh
#
#  3. GT target mode (single GT 1개와 paired L2, nearest 비활성):
#     OCSC_GT_TARGET=true OCSC_OL_NEAREST_MATCH=false \
#       bash scripts/train_ocsc_ft.sh
#
# ── 룰 (CLAUDE.md §6) ─────────────────────────────────────────────────────
# GPU 는 2, 3 만 사용.  default CUDA_VISIBLE_DEVICES=3.
# ==========================================================================

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# ── Hydra entry / paths ─────────────────────────────────────────────────
MY_EXPERIMENT="${MY_EXPERIMENT:-ocsc_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-debug}"
ACTION="${ACTION:-finetune}"
SEED="${SEED:-817}"
# WOMD cache root. Prefer the Ubuntu server cache and fall back to mounted pod caches.
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

# ── Trainer ─────────────────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_NODES="${NUM_NODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
PRECISION="${PRECISION:-bf16-mixed}"
if [ "${NPROC_PER_NODE}" -gt 1 ] || [ "${NUM_NODES}" -gt 1 ]; then
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
else
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-auto}"
fi
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-false}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"

# ── Data ────────────────────────────────────────────────────────────────
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-16}"
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
TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.5}"
TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-true}"

# ── Model / LR ──────────────────────────────────────────────────────────
LR="${LR:-1.0e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-${MAX_EPOCHS}}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1.0}"

N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-100000}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1728}"
SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-8}"
export CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS="${CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS:-50000}"
export CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS="${CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS:-50000}"

VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-true}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"

# ── OCSC core knob ──────────────────────────────────────────────────────
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-4}"               # G
OCSC_N_OL_ROLLOUTS="${OCSC_N_OL_ROLLOUTS:--1}"        # M; -1 = G
OCSC_OL_NEAREST_MATCH="${OCSC_OL_NEAREST_MATCH:-true}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_GT_TARGET="${OCSC_GT_TARGET:-false}"
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-true}"
OCSC_ANCHOR_IDX="${OCSC_ANCHOR_IDX:-0}"
OCSC_MATCH_SPACE="${OCSC_MATCH_SPACE:-pose}"
OCSC_LOSS_WINDOW_STEPS="${OCSC_LOSS_WINDOW_STEPS:--1}"
OCSC_LOSS_TEMPORAL_STRIDE="${OCSC_LOSS_TEMPORAL_STRIDE:--1}"
OCSC_STRICT_ACTIVE_MASK="${OCSC_STRICT_ACTIVE_MASK:-true}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-1.0}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-0.01}"
OCSC_EXCEPT_MAP_ENCODER="${OCSC_EXCEPT_MAP_ENCODER:-false}"
OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-true}"
OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-false}"

CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
CHECKPOINT_SAVE_LAST="${CHECKPOINT_SAVE_LAST:-true}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
WANDB_TAGS="${WANDB_TAGS:-[]}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-all}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
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

echo "============================================================"
echo "[ocsc_ft] launching ..."
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
echo "  Generator: lr=${LR} weight_decay=${WEIGHT_DECAY}"
echo "  OCSC ★:"
echo "    G(n_rollouts)=${OCSC_N_ROLLOUTS}  M(n_ol_rollouts)=${OCSC_N_OL_ROLLOUTS}"
echo "    nearest_match=${OCSC_OL_NEAREST_MATCH}  gt_target=${OCSC_GT_TARGET}"
echo "    use_ref=${OCSC_USE_PRETRAINED_REF}  anchor_idx=${OCSC_ANCHOR_IDX}  match_space=${OCSC_MATCH_SPACE}  loss_window_steps=${OCSC_LOSS_WINDOW_STEPS}  loss_stride=${OCSC_LOSS_TEMPORAL_STRIDE}  strict_active=${OCSC_STRICT_ACTIVE_MASK}"
echo "    pos_w=${OCSC_POSITION_WEIGHT}  head_w=${OCSC_HEADING_WEIGHT}"
echo "    except_map_encoder=${OCSC_EXCEPT_MAP_ENCODER}  velocity_head_only=${OCSC_VELOCITY_HEAD_ONLY}  full_flow_decoder=${OCSC_FULL_FLOW_DECODER}"
echo "  Checkpoint: monitor=${CHECKPOINT_MONITOR} mode=${CHECKPOINT_MODE} save_last=${CHECKPOINT_SAVE_LAST}"
echo "============================================================"

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
  callbacks.model_checkpoint.save_last="${CHECKPOINT_SAVE_LAST}" \
  callbacks.model_checkpoint.save_top_k="${CHECKPOINT_SAVE_TOP_K}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.tags="${WANDB_TAGS}" \
  logger.wandb.log_model="${WANDB_LOG_MODEL}" \
  logger.wandb.offline="${WANDB_OFFLINE}" \
  model.model_config.lr="${LR}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
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
  model.model_config.finetune.train_except_map_encoder="${OCSC_EXCEPT_MAP_ENCODER}" \
  model.model_config.finetune.velocity_head_only="${OCSC_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.train_full_flow_decoder_only="${OCSC_FULL_FLOW_DECODER}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_n_ol_rollouts="${OCSC_N_OL_ROLLOUTS}" \
  model.model_config.finetune.ocsc_ol_nearest_match="${OCSC_OL_NEAREST_MATCH}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_anchor_idx="${OCSC_ANCHOR_IDX}" \
  model.model_config.finetune.ocsc_match_space="${OCSC_MATCH_SPACE}" \
  model.model_config.finetune.ocsc_loss_window_steps="${OCSC_LOSS_WINDOW_STEPS}" \
  model.model_config.finetune.ocsc_loss_temporal_stride="${OCSC_LOSS_TEMPORAL_STRIDE}" \
  model.model_config.finetune.ocsc_strict_active_mask="${OCSC_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  model.model_config.self_forced.enabled=false \
  ${PREFETCH_ARG} \
  ${EVAL_PREFETCH_ARG} \
  ${EVAL_MP_ARG} \
  ${EXTRA_ARGS}

status=$?
echo "bash $(basename "$0") done! status=${status}"
exit "${status}"
