#!/bin/sh
# ============================================================================
# DMD lr×clip 정밀 sweep: 단일 scene overfit 에서 "안정적으로 상승"하는 (gen lr, grad clip)
# 최적점 탐색.  cadence 4:1(n_anchors=4, updates=1)·critic lr·ODE 는 고정(검증됨).
#
# 각 config 를 18 val 점까지 돌린 뒤 tools/eval_rmm_trend.py 로 (d, min, std, score) 평가.
# 전부 돌린 뒤 score(=상승폭 − 크래시 페널티) 내림차순으로 랭킹.
#
# 사용(단일):  CUDA_VISIBLE_DEVICES=3 bash scripts/overfit_dmd_sweep.sh
#     (인자로 config 직접 지정 → 2 GPU 분할 병렬 가능)
#   CUDA_VISIBLE_DEVICES=2 bash scripts/overfit_dmd_sweep.sh "g1e5_c10 1.0e-5 10.0" "g1e5_c1 1.0e-5 1.0"
# 결과: artifacts/sweep_results_gpu${CUDA_VISIBLE_DEVICES}.txt
# ============================================================================
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV}" || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
cd "$(dirname "$0")/.." || exit 1

# 고정(검증된) 세팅
FAKE_LR="${FAKE_LR:-1.0e-4}"; EU="${EU:-1}"; NA="${NA:-4}"; SS="${SS:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-180}"; VAL_EVERY="${VAL_EVERY:-10}"; NRCV="${NRCV:-16}"
PER_CFG_TIMEOUT="${PER_CFG_TIMEOUT:-3000}"

# config 미지정 시 기본 grid (gen lr × grad clip)
if [ "$#" -eq 0 ]; then
  set -- \
    "g1e5_c10 1.0e-5 10.0" "g1e5_c1 1.0e-5 1.0" \
    "g3e5_c10 3.0e-5 10.0" "g3e5_c1 3.0e-5 1.0" \
    "g1e4_c10 1.0e-4 10.0" "g1e4_c1 1.0e-4 1.0"
fi

RES="artifacts/sweep_results_gpu${CUDA_VISIBLE_DEVICES}.txt"
RANK="artifacts/sweep_rank_gpu${CUDA_VISIBLE_DEVICES}.txt"
: > "${RANK}"
echo "===== lr×clip sweep 시작 $(date) GPU=${CUDA_VISIBLE_DEVICES} (cadence ${NA}x${EU}:1, critic ${FAKE_LR}, ode ${SS}) =====" | tee "${RES}"

for cfg in "$@"; do
  set -- $cfg
  LABEL="$1"; GLR="$2"; GCLIP="$3"
  TASK="sweepL_${LABEL}_$(date +%m%d_%H%M%S)"
  LOG="artifacts/${TASK}.log"
  echo ">>> [$LABEL] gen_lr=$GLR grad_clip=$GCLIP  ($(date +%H:%M))" | tee -a "${RES}"
  GEN_LR="$GLR" FAKE_LR="$FAKE_LR" ESTIMATOR_UPDATES="$EU" N_ANCHORS="$NA" SAMPLE_STEPS="$SS" \
    GRAD_CLIP="$GCLIP" MAX_EPOCHS="$MAX_EPOCHS" VAL_EVERY="$VAL_EVERY" N_ROLLOUT_CLOSED_VAL="$NRCV" \
    TASK="$TASK" timeout "${PER_CFG_TIMEOUT}" bash scripts/overfit_single_scene_dmd.sh >/dev/null 2>&1
  RID="$(grep -oE 'clsft-catk/runs/[a-z0-9]+' "$LOG" 2>/dev/null | tail -1 | sed 's#.*/##')"
  if [ -z "$RID" ]; then echo "    [$LABEL] no wandb run id" | tee -a "${RES}"; continue; fi
  V="$(python tools/eval_rmm_trend.py "$RID" 2>/dev/null)"
  echo "    [$LABEL] run=$RID -> $V" | tee -a "${RES}"
  SCORE="$(printf '%s' "$V" | grep -oE 'score=[+-][0-9.]+' | sed 's/score=//')"
  [ -n "$SCORE" ] && printf '%s\t%s\tgen_lr=%s clip=%s run=%s -> %s\n' "$SCORE" "$LABEL" "$GLR" "$GCLIP" "$RID" "$V" >> "${RANK}"
done

echo "===== sweep DONE $(date) — score 내림차순 랭킹 =====" | tee -a "${RES}"
sort -k1 -rn "${RANK}" 2>/dev/null | tee -a "${RES}"
BEST="$(sort -k1 -rn "${RANK}" 2>/dev/null | head -1)"
echo "BEST: ${BEST}" | tee -a "${RES}"
