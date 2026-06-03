#!/bin/sh
# ============================================================================
# DMD 방향 자동 sweep: 단일 val scene overfit 에서 RMM 이 뚜렷이 오르는 (cadence, lr, ODE)
# 조합을 순차 탐색.  RISING(=RMM 추세 상승) 조합을 찾으면 멈추고 결과 파일에 기록.
#
# 각 config: scripts/overfit_single_scene_dmd.sh 를 20 val 점(200 step)까지 돌린 뒤
#            tools/eval_rmm_trend.py 로 RMM 추세 판정.
#
# 사용: CUDA_VISIBLE_DEVICES=3 bash scripts/overfit_dmd_sweep.sh
# 결과: artifacts/sweep_results.txt  (찾으면 'FOUND:' 라인)
# ============================================================================
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV}" || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

cd "$(dirname "$0")/.." || exit 1
RESULTS="artifacts/sweep_results.txt"
mkdir -p artifacts
echo "===== DMD overfit sweep 시작 $(date) (GPU=${CUDA_VISIBLE_DEVICES}) =====" > "${RESULTS}"

# 공통 sweep 설정
MAX_EPOCHS="${MAX_EPOCHS:-200}"      # = 학습 step 수 (val 20점 @ val_every 10)
VAL_EVERY="${VAL_EVERY:-10}"
NRCV="${NRCV:-16}"                   # val n_rollout_closed_val (속도/노이즈 균형)
PER_CFG_TIMEOUT="${PER_CFG_TIMEOUT:-3000}"   # config 당 최대 50분

# config 목록: "label GEN_LR FAKE_LR EST_UPDATES N_ANCHORS SAMPLE_STEPS"
#   cadence(critic:gen) = N_ANCHORS × EST_UPDATES : 1
set -- \
  "c4x1_g1e4_f1e4_ode16 1.0e-4 1.0e-4 1 4 16" \
  "c1x3_g1e4_f1e4_ode16 1.0e-4 1.0e-4 3 1 16" \
  "c4x1_g5e4_f1e4_ode16 5.0e-4 1.0e-4 1 4 16" \
  "c1x3_g5e4_f1e3_ode16 5.0e-4 1.0e-3 3 1 16" \
  "c4x1_g1e4_f1e3_ode16 1.0e-4 1.0e-3 1 4 16" \
  "c4x1_g1e4_f1e4_ode4  1.0e-4 1.0e-4 1 4 4" \
  "c4x1_g1e3_f1e3_ode16 1.0e-3 1.0e-3 1 4 16" \
  "c1x3_g1e4_f1e4_ode8  1.0e-4 1.0e-4 3 1 8"

FOUND=""
for cfg in "$@"; do
  set -- $cfg
  LABEL="$1"; GLR="$2"; FLR="$3"; EU="$4"; NA="$5"; SS="$6"
  TS="$(date +%m%d_%H%M%S)"
  TASK="sweep_${LABEL}_${TS}"
  LOG="artifacts/${TASK}.log"
  echo ">>> [$LABEL] gen_lr=$GLR fake_lr=$FLR cadence=${NA}x${EU}:1 ode=$SS  ($(date +%H:%M))" | tee -a "${RESULTS}"
  GEN_LR="$GLR" FAKE_LR="$FLR" ESTIMATOR_UPDATES="$EU" N_ANCHORS="$NA" SAMPLE_STEPS="$SS" \
    MAX_EPOCHS="$MAX_EPOCHS" VAL_EVERY="$VAL_EVERY" N_ROLLOUT_CLOSED_VAL="$NRCV" TASK="$TASK" \
    timeout "${PER_CFG_TIMEOUT}" bash scripts/overfit_single_scene_dmd.sh >/dev/null 2>&1
  RID="$(grep -oE 'runs/[a-z0-9]+' "$LOG" 2>/dev/null | tail -1 | sed 's#runs/##')"
  if [ -z "$RID" ]; then echo "    [$LABEL] no wandb run id (launch fail?)" | tee -a "${RESULTS}"; continue; fi
  V="$(python tools/eval_rmm_trend.py "$RID" 2>/dev/null)"
  echo "    [$LABEL] run=$RID  -> $V" | tee -a "${RESULTS}"
  case "$V" in
    RISING*) FOUND="$LABEL :: $V :: wandb=se99an/clsft-catk/runs/$RID :: gen_lr=$GLR fake_lr=$FLR cadence=${NA}x${EU}:1 ode=$SS"
             echo "FOUND: $FOUND" | tee -a "${RESULTS}"; break ;;
  esac
done

echo "===== SWEEP DONE $(date).  FOUND=[$FOUND] =====" | tee -a "${RESULTS}"
