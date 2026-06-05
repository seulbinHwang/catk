#!/bin/sh
# ============================================================================
# DMD 단일 run 실행기 (수동 sweep 용)
# ----------------------------------------------------------------------------
# 사용법:
#   1) 아래 ★ SWEEP 칸을 채운다 (비워두면 DEFAULT 값 사용).
#   2) bash scripts/dmd_run.sh
#   또는 env 로 덮어쓰기:  GEN_LR=3e-5 CRITIC_LR=1e-4 N_ANCHORS=4 bash scripts/dmd_run.sh
#
# MODE:
#   MODE=overfit  → 단일 scene overfit (방향 검증, 빠름)   [기본]
#   MODE=full     → 전체 multi-scene 학습 (DDP, GPU 2,3)
#
# repo 정책: GPU 2,3 만 사용.  overfit=single GPU(기본 3), full=2,3 DDP.
# ============================================================================
set -e

# ──────────────────────────────────────────────────────────────────────────
# ★★★ SWEEP 대상 — 여기 빈칸을 채워서 서치 (비우면 아래 DEFAULT 사용) ★★★
# ──────────────────────────────────────────────────────────────────────────
GEN_LR="1e-6"              # generator(main encoder) lr            예) 1e-5 / 3e-5 / 5e-5
CRITIC_LR="1e-5"           # critic(fake_score) lr                 예) 1e-4 / 3e-5
N_ANCHORS="4"           # GT-grounded time-anchor 수            예) 1 / 2 / 4
ANCHOR_STRIDE="4"       # anchor 간격(coarse step, 4=2초)        예) 2 / 4 / 8
ESTIMATOR_UPDATES="3"   # anchor 당 critic update (cadence=NA×U) 예) 1 / 2 / 4
UNFROZEN_RANGE="except_map_encoder"      # 학습 scope                            velocity_head_only / full_flow_decoder / except_map_encoder
DMD_BETA="1"            # entropy knob (1=vanilla, <1 div↑)     예) 0.8 / 1.0 / 1.2
N_ROLLOUTS="4"          # G (variance reduction, VRAM ~G배)     예) 1 / 2 / 4
SAMPLE_STEPS="16"        # 학습 closed-loop ODE step             예) 8 / 16
NOISE_SCALE="1.0"         # rollout noise scale (>1 div↑)         예) 1.0 / 1.2
GRAD_CLIP="10.0"           # generator manual grad clip            예) 1.0 / 10.0
FAKE_WARMUP="0"         # critic-only warmup step (full 모드)    예) 0 / 200

# ──────────────────────────────────────────────────────────────────────────
# DEFAULT (SWEEP 칸이 비어 있을 때 사용) — flow_dmd.yaml 검증 세팅
# ──────────────────────────────────────────────────────────────────────────
: "${MODE:=overfit}"                                    # overfit | full

GEN_LR="${GEN_LR:-5.0e-5}"
CRITIC_LR="${CRITIC_LR:-1.0e-4}"
N_ANCHORS="${N_ANCHORS:-4}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-4}"
ESTIMATOR_UPDATES="${ESTIMATOR_UPDATES:-4}"
UNFROZEN_RANGE="${UNFROZEN_RANGE:-velocity_head_only}"
DMD_BETA="${DMD_BETA:-1.0}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-16}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
FAKE_WARMUP="${FAKE_WARMUP:-200}"
# full backprop: random_terminal_step 비활성 → 16 ODE step 전체 grad.
FULL_BACKPROP="${FULL_BACKPROP:-true}"
if [ "${FULL_BACKPROP}" = "true" ]; then RT_ENABLED=false; else RT_ENABLED=true; fi

# ── 환경 / 경로 ──────────────────────────────────────────────────────────────
export LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV}" || true
cd "$(dirname "$0")/.." || exit 1

CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/catk_cache}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
ACTION="${ACTION:-finetune}"          # fresh=finetune, resume=fit
mkdir -p artifacts
[ -f "${CKPT_PATH}" ] || { echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"; exit 1; }

TS="$(date +%m%d_%H%M%S)"

# ── 공통 self_forced override (overfit/full 동일) ────────────────────────────
SF_ARGS="\
  model.model_config.lr=${GEN_LR} \
  model.model_config.self_forced.estimator_lr=${CRITIC_LR} \
  model.model_config.self_forced.unfrozen_range=${UNFROZEN_RANGE} \
  model.model_config.self_forced.estimator_updates_per_step=${ESTIMATOR_UPDATES} \
  model.model_config.self_forced.n_anchors=${N_ANCHORS} \
  model.model_config.self_forced.anchor_stride=${ANCHOR_STRIDE} \
  model.model_config.self_forced.dmd_beta=${DMD_BETA} \
  model.model_config.self_forced.n_rollouts=${N_ROLLOUTS} \
  model.model_config.self_forced.gradient_clip_val=${GRAD_CLIP} \
  model.model_config.self_forced.sampling.sample_steps=${SAMPLE_STEPS} \
  model.model_config.self_forced.sampling.noise_scale=${NOISE_SCALE} \
  model.model_config.self_forced.sampling.random_terminal_step.enabled=${RT_ENABLED}"

print_hp() {
  echo "[dmd_run] MODE=${MODE} ACTION=${ACTION} GPU=${CUDA_VISIBLE_DEVICES}"
  echo "  GEN_LR=${GEN_LR}  CRITIC_LR=${CRITIC_LR}  scope=${UNFROZEN_RANGE}"
  echo "  anchor: n=${N_ANCHORS} stride=${ANCHOR_STRIDE}  cadence(critic:gen)=$((N_ANCHORS*ESTIMATOR_UPDATES)):1 (updates=${ESTIMATOR_UPDATES})"
  echo "  dmd_beta=${DMD_BETA}  G(n_rollouts)=${N_ROLLOUTS}  ODE=${SAMPLE_STEPS}(full_bp=${FULL_BACKPROP})  noise=${NOISE_SCALE}  clip=${GRAD_CLIP}  warmup=${FAKE_WARMUP}"
  echo "  ckpt=${CKPT_PATH}  wandb=${WANDB_ENTITY}/${WANDB_PROJECT}"
  echo "  STDOUT_LOG=$(pwd)/${LOG}"
}

# ============================================================================
# MODE = overfit  : 단일 scene overfit (방향 검증)
# ============================================================================
if [ "${MODE}" = "overfit" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
  VAL_PKL_DIR="${CACHE_ROOT}/validation"
  VAL_TFR_DIR="${CACHE_ROOT}/validation_tfrecords_splitted"
  SCENE_ID="${SCENE_ID:-$(ls "${VAL_PKL_DIR}" | head -1 | sed 's/\.pkl$//')}"
  WORK="${CACHE_ROOT}/_single_scene_${SCENE_ID}"
  SCENE_DIR="${WORK}/scene"; TFR_DIR="${WORK}/tfrecords"
  mkdir -p "${SCENE_DIR}" "${TFR_DIR}"
  cp -f "${VAL_PKL_DIR}/${SCENE_ID}.pkl" "${SCENE_DIR}/"
  cp -f "${VAL_TFR_DIR}/${SCENE_ID}.tfrecords" "${TFR_DIR}/"

  MAX_EPOCHS="${MAX_EPOCHS:-180}"        # limit_train_batches=1 → = 학습 step 수
  VAL_EVERY="${VAL_EVERY:-10}"           # N step 마다 그 scene RMM
  N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
  TASK="${TASK:-dmd_overfit_${SCENE_ID}_g${GEN_LR}_c${CRITIC_LR}_a${N_ANCHORS}u${ESTIMATOR_UPDATES}_${TS}}"
  LOG="artifacts/${TASK}.log"
  echo "[dmd_run] scene=${SCENE_ID}"
  print_hp
  python -m src.run \
    experiment=flow_dmd action=finetune \
    ckpt_path="${CKPT_PATH}" task_name="${TASK}" \
    paths.cache_root="${CACHE_ROOT}" \
    data.train_raw_dir="${SCENE_DIR}" data.val_raw_dir="${SCENE_DIR}" \
    data.val_tfrecords_splitted="${TFR_DIR}" \
    data.train_batch_size=1 data.val_batch_size=1 data.num_workers=1 \
    data.train_epoch_sample_fraction=1.0 \
    trainer.devices=1 trainer.strategy=auto \
    trainer.limit_train_batches=1 trainer.limit_val_batches=1.0 \
    trainer.val_check_interval=null \
    trainer.check_val_every_n_epoch="${VAL_EVERY}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
    model.model_config.scorer_scene_num=null \
    model.model_config.n_batch_sim_agents_metric=1 \
    model.model_config.self_forced.estimator_warmup_steps=0 \
    ${SF_ARGS} \
    logger.wandb.entity="${WANDB_ENTITY}" logger.wandb.project="${WANDB_PROJECT}" \
    ${EXTRA_ARGS} \
    > "${LOG}" 2>&1
  echo "[dmd_run] done. log=${LOG}"
  echo "[dmd_run] 추세 평가:  conda run -n catk python tools/eval_rmm_trend.py \$(grep -oE 'clsft-catk/runs/[a-z0-9]+' ${LOG} | tail -1 | sed 's#.*/##')"
  exit 0
fi

# ============================================================================
# MODE = full  : 전체 multi-scene 학습 (DDP)
# ============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
N_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
if [ "${N_DEVICES}" -gt 1 ]; then STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"; else STRATEGY="${STRATEGY:-auto}"; fi

TRAIN_B="${TRAIN_B:-32}"; VAL_B="${VAL_B:-16}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-400}"
MAX_EPOCHS="${MAX_EPOCHS:-16}"; PRECISION="${PRECISION:-bf16-mixed}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"; NUM_WORKERS="${NUM_WORKERS:-4}"
TASK="${TASK:-dmd_full_g${GEN_LR}_c${CRITIC_LR}_a${N_ANCHORS}u${ESTIMATOR_UPDATES}_${UNFROZEN_RANGE}_${TS}}"
LOG="artifacts/${TASK}.log"
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
print_hp

torchrun --nproc_per_node="${N_DEVICES}" --master_port="${PORT}" -m src.run \
  experiment=flow_dmd action="${ACTION}" task_name="${TASK}" \
  ckpt_path="${CKPT_PATH}" paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${N_DEVICES}" trainer.strategy="${STRATEGY}" \
  trainer.precision="${PRECISION}" trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  data.train_batch_size="${TRAIN_B}" data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.self_forced.estimator_warmup_steps="${FAKE_WARMUP}" \
  ${SF_ARGS} \
  logger.wandb.entity="${WANDB_ENTITY}" logger.wandb.project="${WANDB_PROJECT}" \
  ${EXTRA_ARGS} \
  2>&1 | tee "${LOG}"
echo "[dmd_run] done. log=${LOG}"
