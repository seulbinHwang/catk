#!/bin/sh
# =============================================================================
# OCSC_clean (= origin/fix-hard-rmm verbatim) 현재 도는 세팅 pin 본
# - GT-target consistency, single GPU
# - launcher default 와 동일하지 않은 부분만 명시 export
# - 다른 default (LR=1e-6, OCSC_USE_MMD/OCSC_USE_PRETRAINED_REF/ocsc_n_rollouts=4
#   등) 는 scripts/train_flow_consistency_bptt_single.sh 의 default 그대로
#
# 사용:
#   sh scripts/launch_ocsc_clean_current.sh
#
# OL-target 으로 바꾸려면 OCSC_GT_TARGET=false 만 override
# =============================================================================

# GPU / experiment 메타 ─────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MY_TASK_NAME="${MY_TASK_NAME:-ocsc-gt-gpu3-clean}"

# OCSC 모드 ─────────────────────────────────────────────────────────────────
export OCSC_GT_TARGET="${OCSC_GT_TARGET:-true}"      # GT consistency target

# Trainer 노브 ──────────────────────────────────────────────────────────────
export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
export TRAIN_B="${TRAIN_B:-8}"
export VAL_B="${VAL_B:-8}"
export TRAINER_DETERMINISTIC="${TRAINER_DETERMINISTIC:-false}"

# 위 외 모든 default 는 launcher 가 결정.  핵심 default 요약 (참고용):
#   LR=1e-6, LR_TOTAL_STEPS=auto, LR_MIN_RATIO=0.1
#   FLOW_VELOCITY_HEAD_ONLY=true
#   OCSC_N_ROLLOUTS=4, OCSC_ANCHOR_STRIDE=1, OCSC_PRED_MAX_STEPS=2
#   OCSC_USE_MMD=true, OCSC_USE_PRETRAINED_REF=true, OCSC_FM_REG_LAMBDA=0.1
#   OCSC_REL_DISP_WEIGHT=1.0 (POSITION/HEADING=0.0)
#   BPTT_USE_ADJOINT=true, BPTT_LAST_COARSE_ONLY=true
#   FLOW_SOLVER_METHOD=euler, FLOW_SOLVER_STEPS=16
#   VALIDATION_METRIC=hard, N_ROLLOUT_CLOSED_VAL=16
#   limit_val_batches=0.01, max_epochs=20, precision=32-true
#   CKPT_PATH=/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt
#                                       ^^ fix-hard-rmm 호환 ckpt (85MB)

bash scripts/train_flow_consistency_bptt_single.sh
