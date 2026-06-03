#!/bin/sh
# ============================================================================
# DMD 방향 검증: val scene 1개에만 DMD overfit → 그 scene 의 RMM 이 오르는지 확인.
#
#   - train = val = 동일한 val scene 1개 (RMM scorer 용 tfrecord 가 있는 val 에서 선택).
#   - 매 step 그 scene 에 DMD 적용, check_val_every_n_epoch 마다 그 scene RMM 측정.
#   - RMM 이 step 따라 상승하면 DMD gradient 방향 OK.
#
# 사용: CUDA_VISIBLE_DEVICES=3 bash scripts/overfit_single_scene_dmd.sh
#       SCENE_ID=<id> GEN_LR=1e-6 bash scripts/overfit_single_scene_dmd.sh
# ============================================================================
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV}" || true

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/catk_cache}"
VAL_PKL_DIR="${CACHE_ROOT}/validation"
VAL_TFR_DIR="${CACHE_ROOT}/validation_tfrecords_splitted"

# ── 1) 단일 scene 데이터셋 준비 ──────────────────────────────────────────────
SCENE_ID="${SCENE_ID:-$(ls "${VAL_PKL_DIR}" | head -1 | sed 's/\.pkl$//')}"
WORK="${WORK:-/home2/pnc2/repos_python/datasets/catk_cache/_single_scene_${SCENE_ID}}"
SCENE_DIR="${WORK}/scene"
TFR_DIR="${WORK}/tfrecords"
mkdir -p "${SCENE_DIR}" "${TFR_DIR}"
if [ ! -f "${VAL_PKL_DIR}/${SCENE_ID}.pkl" ]; then echo "[ERROR] no pkl for ${SCENE_ID}"; exit 1; fi
if [ ! -f "${VAL_TFR_DIR}/${SCENE_ID}.tfrecords" ]; then echo "[ERROR] no tfrecords for ${SCENE_ID}"; exit 1; fi
cp -f "${VAL_PKL_DIR}/${SCENE_ID}.pkl" "${SCENE_DIR}/"
cp -f "${VAL_TFR_DIR}/${SCENE_ID}.tfrecords" "${TFR_DIR}/"
echo "[overfit] scene=${SCENE_ID}  scene_dir=${SCENE_DIR}  tfr_dir=${TFR_DIR}"

# ── 2) overfit 하이퍼파라미터 ────────────────────────────────────────────────
GEN_LR="${GEN_LR:-1.0e-6}"            # generator lr (낮게)
FAKE_LR="${FAKE_LR:-1.0e-4}"          # critic lr
UNFROZEN_RANGE="${UNFROZEN_RANGE:-full_flow_decoder}"   # overfit 용량 위해 full
N_ANCHORS="${N_ANCHORS:-4}"
# critic cadence: anchor 당 critic FM update 수.  실효 critic:gen = n_anchors×updates : 1.
# (updates=1, n_anchors=4 → 4:1;  updates=3, n_anchors=4 → 12:1.)
ESTIMATOR_UPDATES="${ESTIMATOR_UPDATES:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"       # limit_train_batches=1 이라 = 학습 step 수
VAL_EVERY="${VAL_EVERY:-10}"          # 10 step 마다 그 scene RMM
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-32}"   # 1 scene RMM 노이즈↓

TS="$(date +%m%d_%H%M)"
TASK="${TASK:-overfit1scene_${SCENE_ID}_${UNFROZEN_RANGE}_lr${GEN_LR}_${TS}}"
mkdir -p artifacts
LOG="artifacts/${TASK}.log"
echo "[overfit] STDOUT_LOG=$(pwd)/${LOG}"
echo "[overfit] scope=${UNFROZEN_RANGE} gen_lr=${GEN_LR} fake_lr=${FAKE_LR} cadence=${N_ANCHORS}x${ESTIMATOR_UPDATES}:1 max_steps=${MAX_EPOCHS} val_every=${VAL_EVERY} n_rollout=${N_ROLLOUT_CLOSED_VAL}"

python -m src.run \
  experiment=flow_dmd action=finetune \
  ckpt_path=logs/pretrained/pretrained.ckpt \
  task_name="${TASK}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_raw_dir="${SCENE_DIR}" \
  data.val_raw_dir="${SCENE_DIR}" \
  data.val_tfrecords_splitted="${TFR_DIR}" \
  data.train_batch_size=1 data.val_batch_size=1 data.num_workers=1 \
  data.train_epoch_sample_fraction=1.0 \
  trainer.devices=1 trainer.strategy=auto \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1.0 \
  trainer.val_check_interval=null \
  trainer.check_val_every_n_epoch="${VAL_EVERY}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  model.model_config.lr="${GEN_LR}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.scorer_scene_num=null \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.self_forced.unfrozen_range="${UNFROZEN_RANGE}" \
  model.model_config.self_forced.estimator_lr="${FAKE_LR}" \
  model.model_config.self_forced.estimator_updates_per_step="${ESTIMATOR_UPDATES}" \
  model.model_config.self_forced.estimator_warmup_steps=0 \
  model.model_config.self_forced.n_anchors="${N_ANCHORS}" \
  model.model_config.self_forced.sampling.sample_steps="${SAMPLE_STEPS}" \
  logger.wandb.entity=se99an logger.wandb.project=clsft-catk \
  ${EXTRA_ARGS} \
  > "${LOG}" 2>&1
echo "[overfit] done. log: ${LOG}"
