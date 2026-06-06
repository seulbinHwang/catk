#!/bin/sh
# Self-forcing(DMD) fine-tuning — cadence 기반 fake:gen 업데이트.
#
# 새 cadence 의미: fake(critic)는 매 batch 1회, generator 는 N batch 마다 1회(서로 다른
# batch 들에서). 같은 시나리오를 여러 번 돌리지 않는다.  (estimator_updates_per_step=1)
#
# 기본 런치: cadence 5:1, gen lr=fake lr=1e-7, EMA off, DMD, train B=8, val 1000 batch,
#            scorer 440 scene + val_b 16 (scene 수 자동 조정), GPU 2,3.
#   bash scripts/_self_forcing_update.sh
set -e

is_true() { case "$(printf %s "$1" | tr A-Z a-z)" in 1|true|yes|on) return 0;; *) return 1;; esac; }
get_free_port() { python - <<'PY'
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
}

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV:-catk}" || true
cd "${REPO_ROOT}"

# --- core knobs ---
MY_EXPERIMENT="${MY_EXPERIMENT:-self_forced_npfm}"
ACTION="${ACTION:-finetune}"
SEED="${SEED:-817}"
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/catk_cache}"
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

GPU="${GPU:-2,3}"
export CUDA_VISIBLE_DEVICES="${GPU}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(printf %s "${GPU}" | awk -F, '{print NF}')}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"

# self-forcing cadence / lr / ema / objective
CADENCE="${CADENCE:-5}"                         # fake:gen = N:1
ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-1}"   # fake updates per batch (1 = distinct batches)
GEN_LR="${GEN_LR:-1e-7}"
FAKE_LR="${FAKE_LR:-1e-7}"
USE_EMA="${USE_EMA:-false}"
DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"
ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-1}"

# data / trainer
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-16}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-440}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"      # scorer_scene_num 가 자동 상향
MAX_EPOCHS="${MAX_EPOCHS:-16}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-32}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"

# wandb
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"

TS="$(date +%m%d_%H%M%S)"
TASK_DEFAULT="sfupdate_cad${CADENCE}_gen${GEN_LR}_fake${FAKE_LR}_${DM_OBJECTIVE}_b${TRAIN_B}x${NPROC_PER_NODE}_${TS}"
TASK="${MY_TASK_NAME:-${TASK_DEFAULT}}"
LOG="${LOG:-artifacts/${TASK}.log}"
mkdir -p artifacts

if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"; exit 1
fi

set -- \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${TASK}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${NPROC_PER_NODE}" \
  trainer.precision="${PRECISION}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch=null \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.shuffle="${DATA_SHUFFLE}" \
  model.model_config.lr="${GEN_LR}" \
  model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.self_forced.distribution_matching_objective="${DM_OBJECTIVE}" \
  model.model_config.self_forced.cadence="${CADENCE}" \
  model.model_config.self_forced.estimator_updates_per_step="${ESTIMATOR_UPDATES_PER_STEP}" \
  model.model_config.self_forced.estimator_lr="${FAKE_LR}" \
  model.model_config.self_forced.use_ema="${USE_EMA}" \
  model.model_config.self_forced.estimator_warmup_epochs="${ESTIMATOR_WARMUP_EPOCHS}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  "~callbacks.model_checkpoint" \
  "~callbacks.epoch_last_checkpoint"

echo "============================================================"
echo "[sf-update] task=${TASK}"
echo "  GPU=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}  ckpt=${CKPT_PATH}"
echo "  cadence(fake:gen)=${CADENCE}:1  est_updates/batch=${ESTIMATOR_UPDATES_PER_STEP}  gen_lr=${GEN_LR} fake_lr=${FAKE_LR}"
echo "  objective=${DM_OBJECTIVE} use_ema=${USE_EMA} warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}"
echo "  train_B=${TRAIN_B} val_B=${VAL_B} scorer_scene=${SCORER_SCENE_NUM} val_check=${VAL_CHECK_INTERVAL} precision=${PRECISION}"
echo "  log=${LOG}"
echo "============================================================"

if is_true "${DRY_RUN:-false}"; then
  printf "torchrun --standalone --nproc_per_node=%s -m src.run" "${NPROC_PER_NODE}"
  for a in "$@"; do printf " %s" "$a"; done; printf "\n"; exit 0
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  -m src.run "$@" > "${LOG}" 2>&1
echo "[sf-update] done status=$? log=${LOG}"
