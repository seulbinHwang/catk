#!/bin/sh
# ============================================================================
# sf_cadence25_best.sh — BEST self-forcing DMD fine-tuning preset (cadence 25)
# ============================================================================
# 2026-06-08 cadence sweep {25,50,100,200} 결과 cadence=25 세팅이 RMM 최고로
# 단조 상승(0.77973 -> 0.78028 -> 0.78060 @ step1k/2k/3k, baseline 0.77776).
# 더 작은 cadence(=generator 를 더 자주 업데이트)일수록 RMM 이 높고 상승,
# cadence=200 은 단조 하락해 baseline 으로 수렴. 이 스크립트는 그 "이긴 세팅"을
# 한 줄로 정확히 재현한다.
#
# 핵심 세팅 (전부 고정):
#   experiment=self_forced_npfm  action=finetune  ckpt=logs/pretrained/pretrained.ckpt
#   precision=32-true (fp32, V100)  GPU=0,1 (nproc=2)  train_B=2  val_B=16
#   gen_lr=fake_lr=estimator_lr=1e-7   cadence=25:1 (fake critic 매 배치 / gen 25배치마다)
#   estimator_updates_per_step=1       NO warmup (estimator_warmup_epochs=0, init_ckpt 없음)
#   distribution_matching_objective=dmd  normalize_direction=false (raw teacher-fake)
#   path_step_size=2.0   use_ema=false
#   scorer_scene_num=440 (-> n_batch_sim_agents_metric 자동보정 -> 448 scene 채점)
#   sim_agents_metric_workers=8   n_rollout_closed_val=32   val_check_interval=1000
#   shuffle=false  num_workers=4  max_epochs=16  seed=817
#   DDP find_unused_parameters=true gradient_as_bucket_view=true timeout=14400s
#   wandb se99an/clsft-catk  (pod env 의 stale jksg01019 무시하도록 강제)
#   model_checkpoint / epoch_last_checkpoint 콜백 제거
#
# 위 세팅이 만들어내는 정확한 torchrun 커맨드(검증된 DRY_RUN 출력):
#   torchrun --standalone --nproc_per_node=2 -m src.run \
#     experiment=self_forced_npfm action=finetune \
#     ckpt_path=logs/pretrained/pretrained.ckpt seed=817 \
#     paths.cache_root=$CACHE_ROOT trainer.devices=2 \
#     ~trainer.strategy +trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy \
#     +trainer.strategy.find_unused_parameters=true \
#     +trainer.strategy.gradient_as_bucket_view=true \
#     +trainer.strategy.timeout._target_=datetime.timedelta \
#     +trainer.strategy.timeout.seconds=14400 \
#     trainer.precision=32-true trainer.max_epochs=16 ++trainer.val_check_interval=1000 \
#     trainer.check_val_every_n_epoch=1 trainer.limit_val_batches=1 \
#     data.train_batch_size=2 data.val_batch_size=16 data.num_workers=4 data.shuffle=false \
#     model.model_config.lr=1e-7 model.model_config.scorer_scene_num=440 \
#     model.model_config.n_rollout_closed_val=32 \
#     model.model_config.self_forced.distribution_matching_objective=dmd \
#     model.model_config.self_forced.path_step_size=2.0 \
#     model.model_config.self_forced.normalize_direction=false \
#     model.model_config.sim_agents_metric_workers=8 \
#     model.model_config.self_forced.cadence=25 \
#     model.model_config.self_forced.estimator_updates_per_step=1 \
#     model.model_config.self_forced.estimator_lr=1e-7 \
#     model.model_config.self_forced.use_ema=false \
#     model.model_config.self_forced.estimator_warmup_epochs=0 \
#     logger.wandb.entity=se99an logger.wandb.project=clsft-catk \
#     ~callbacks.model_checkpoint ~callbacks.epoch_last_checkpoint
#
# 사용:
#   bash scripts/sf_cadence25_best.sh                # 백그라운드 런치 + tmux 로그창
#   DRY_RUN=true bash scripts/sf_cadence25_best.sh   # torchrun 커맨드만 출력(런치 안함)
# 환경(클러스터)에 맞춰 CONDA_SH / CACHE_ROOT 만 override 하면 됨. 예:
#   CONDA_SH=/path/conda.sh CACHE_ROOT=/path/cache bash scripts/sf_cadence25_best.sh
# ============================================================================
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 이긴 세팅(cadence 25) 고정 프리셋 — 모두 _self_forcing_update.sh 노브로 매핑 ---
export GPU="${GPU:-0,1}"                       # nproc=2 (V100 2장)
export PRECISION="${PRECISION:-32-true}"       # fp32
export TRAIN_B="${TRAIN_B:-2}"
export VAL_B="${VAL_B:-16}"
export CADENCE="${CADENCE:-25}"                # ★ 핵심: 25:1
export ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-1}"
export GEN_LR="${GEN_LR:-1e-7}"
export FAKE_LR="${FAKE_LR:-1e-7}"
export DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"
export NORMALIZE_DIRECTION="${NORMALIZE_DIRECTION:-false}"   # raw teacher-fake
export PATH_STEP_SIZE="${PATH_STEP_SIZE:-2.0}"
export USE_EMA="${USE_EMA:-false}"
export ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"  # NO warmup
export ESTIMATOR_INIT_CKPT="${ESTIMATOR_INIT_CKPT:-}"          # warm critic 없음(빈값)
export SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-440}"
export N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-32}"
export SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-8}"
export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
export ACTION="${ACTION:-finetune}"
export CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

# --- 클러스터별 경로 (override 권장) ---
export CONDA_SH="${CONDA_SH:-/mnt/nuplan/miniforge/etc/profile.d/conda.sh}"
export CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"

# --- wandb 강제(se99an/clsft-catk) : pod env 의 stale jksg01019 덮어쓰기 ---
export WANDB_ENTITY="se99an"
export WANDB_PROJECT="clsft-catk"
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_API_KEY 2>/dev/null || true   # 키 노출/오계정 로그인 방지

# --- task / 로그 경로 ---
TS="$(date +%m%d_%H%M%S)"
TASK="${MY_TASK_NAME:-sfupd_cad${CADENCE}_g${GEN_LR}_f${FAKE_LR}_eu${ESTIMATOR_UPDATES_PER_STEP}_nowarm_RAWstep${PATH_STEP_SIZE}_fp32_b${TRAIN_B}_BEST_${TS}}"
export MY_TASK_NAME="${TASK}"
BOOT_LOG="artifacts/${TASK}.boot.log"
RUN_LOG="artifacts/${TASK}.log"
mkdir -p artifacts

# DRY_RUN 은 그대로 위임 (torchrun 커맨드만 출력)
if [ "${DRY_RUN:-false}" = "true" ]; then
  exec bash scripts/_self_forcing_update.sh
fi

# 백그라운드 런치
nohup bash scripts/_self_forcing_update.sh > "${BOOT_LOG}" 2>&1 &
LAUNCH_PID=$!
echo "$TASK" > /tmp/sfupd_cad25_task.txt

# tmux 로그창 (kinematic 세션) 재생성 — 있으면
tmux kill-window -t kinematic:sflog 2>/dev/null || true
sleep 1
tmux new-window -t kinematic -n sflog "tail -F ${RUN_LOG}" 2>/dev/null || true

echo "============================================================"
echo "[sf_cadence25_best] launched (pid=${LAUNCH_PID})"
echo "  task     = ${TASK}"
echo "  preset   = cadence25:1 eu1 gen=${GEN_LR} fake=${FAKE_LR} step=${PATH_STEP_SIZE} RAWdir fp32 b${TRAIN_B} NOwarm EMA=off"
echo "  ckpt     = ${CKPT_PATH}"
echo "  scorer   = ${SCORER_SCENE_NUM} (-> 448 scene)  val_check=${VAL_CHECK_INTERVAL}"
echo "  boot log = ${BOOT_LOG}"
echo "  run log  = ${RUN_LOG}"
echo "  wandb    = ${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "  tmux     = kinematic:sflog (tail)"
echo "============================================================"
