#!/bin/sh
# Self-forcing DMD fine-tuning — "A-v1 + warmup critic" 프리셋.
#
# 배경(2026-06-08): 발산 진단 결과 driver=fake(critic) drift. 대응으로
#   (1) critic 강화: estimator_updates_per_step 1→5 + fake_lr 1e-7 복원
#   (2) critic warmup: F_psi 초기값을 1 epoch warmup된 ckpt로 override
# generator/teacher 는 pretrained 에서 시작, fake critic 만 warmup 상태로 출발.
# 나머지는 raw direction(거리-나눗셈 off) + path_step_size 2.0 + cadence 5 + EMA off.
#
# 사용:
#   bash scripts/sf_warmcritic.sh                 # 백그라운드 런치 + tmux 로그창
#   DRY_RUN=true bash scripts/sf_warmcritic.sh    # torchrun 커맨드만 출력
# 개별 노브는 env 로 override 가능 (예: FAKE_LR=1e-6 bash scripts/sf_warmcritic.sh).
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- A-v1 + warmup critic 프리셋 (env 로 override 가능) ---
export GPU="${GPU:-2,3}"
export CADENCE="${CADENCE:-5}"
export ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-5}"   # critic 강화
export GEN_LR="${GEN_LR:-1e-7}"
export FAKE_LR="${FAKE_LR:-1e-7}"
export USE_EMA="${USE_EMA:-false}"
export DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"
export PATH_STEP_SIZE="${PATH_STEP_SIZE:-2.0}"
export NORMALIZE_DIRECTION="${NORMALIZE_DIRECTION:-false}"             # raw teacher-fake
export ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"
export TRAIN_B="${TRAIN_B:-8}"
# warmup된 fake critic(F_psi) override ckpt
export ESTIMATOR_INIT_CKPT="${ESTIMATOR_INIT_CKPT:-logs/fake_1e_7/fake_warmup_epoch0.ckpt}"

# --- task / 로그 경로 ---
TS="$(date +%m%d_%H%M%S)"
TASK="${MY_TASK_NAME:-sfupdate_cad${CADENCE}_gen${GEN_LR}_fake${FAKE_LR}_${DM_OBJECTIVE}_b${TRAIN_B}_w${ESTIMATOR_WARMUP_EPOCHS}_RAWdir_step${PATH_STEP_SIZE}_EU${ESTIMATOR_UPDATES_PER_STEP}_WARMcritic_${TS}}"
export MY_TASK_NAME="${TASK}"
BOOT_LOG="artifacts/${TASK}.boot.log"
RUN_LOG="artifacts/${TASK}.log"
mkdir -p artifacts

# ckpt 존재 확인 (override 지정 시)
if [ -n "${ESTIMATOR_INIT_CKPT}" ] && [ ! -f "${ESTIMATOR_INIT_CKPT}" ]; then
  echo "[ERROR] ESTIMATOR_INIT_CKPT not found: ${ESTIMATOR_INIT_CKPT}"; exit 1
fi

# DRY_RUN 은 그대로 위임
if [ "${DRY_RUN:-false}" = "true" ]; then
  exec bash scripts/_self_forcing_update.sh
fi

# 백그라운드 런치
nohup bash scripts/_self_forcing_update.sh > "${BOOT_LOG}" 2>&1 &
LAUNCH_PID=$!
echo "$TASK" > /tmp/sfupdate_task.txt

# tmux 로그창 (kinematic 세션) 재생성
tmux kill-window -t kinematic:sflog 2>/dev/null || true
sleep 1
tmux new-window -t kinematic -n sflog "tail -F ${RUN_LOG}" 2>/dev/null || true

echo "============================================================"
echo "[sf_warmcritic] launched (pid=${LAUNCH_PID})"
echo "  task         = ${TASK}"
echo "  preset       = cadence${CADENCE}:1 EU${ESTIMATOR_UPDATES_PER_STEP} gen=${GEN_LR} fake=${FAKE_LR} step=${PATH_STEP_SIZE} raw EMA=${USE_EMA} warmup=${ESTIMATOR_WARMUP_EPOCHS}"
echo "  warm critic  = ${ESTIMATOR_INIT_CKPT}"
echo "  stdout log   = ${BOOT_LOG}"
echo "  run log      = ${RUN_LOG}"
echo "  hydra dir    = logs/${TASK}/runs/<ts>"
echo "  wandb        = se99an/clsft-catk (run id 는 run log 의 'View run at' 참조)"
echo "  tmux         = kinematic:sflog (tail)"
echo "============================================================"
