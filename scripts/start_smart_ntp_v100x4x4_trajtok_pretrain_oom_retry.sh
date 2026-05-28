#!/usr/bin/env bash
# Launch the TrajTok SMART NTP pretrain fair-comparison run on
# sv/svv/testsv/testsvvv, each with 4 V100 GPUs.
#
# The comparison reference is
# smart_ntp_pretrain_a100x4x2_bs14_oom_retry_main_original_legacy_inputs_trainselectfalse_20260528.
# That run used 8 GPUs with per-GPU train batch size 14, so its effective
# global train batch was 112. This wrapper defaults to 16 GPUs with per-GPU
# train batch size 7 to keep the effective global train batch at 112 while
# running the latest trajtok branch and TrajTok methodology.
#
# Matched recipe intent:
#   - pre_bc_a100x4x2 experiment preset
#   - effective global train batch 112
#   - VAL/TEST effective global batch 96 via per-GPU batch 6
#   - trainer.accumulate_grad_batches=1 from the preset
#   - data.train_use_eval_agent_selection=false
#   - mixed precision intent, using fp16 because V100 does not natively support
#     bf16 training
#   - DDP unused-parameter detection, because TrajTok type-specific classifier
#     heads are not guaranteed to be used on every rank in every batch
#
# Intentional differences:
#   - latest trajtok branch instead of main
#   - TrajTok vocab, type-specific decoder heads, and spatial-aware smoothing
#   - four V100x4 pods instead of two A100x4 pods
set -Eeuo pipefail

export PODS="${PODS:-sv svv testsv testsvvv}"
export BRANCH="${BRANCH:-trajtok}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_v100x4x4_trajtok_fair}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_v100x4x4_globalbs112_oom_retry_trajtok_legacy_inputs_trainselectfalse_20260528}"
export SESSION="${SESSION:-catk-smart-ntp-v100x4x4-trajtok}"
export MASTER_PORT="${MASTER_PORT:-29541}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export INITIAL_BS="${INITIAL_BS:-7}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-6}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-6}"

FAIR_COMPARISON_OVERRIDES=(
  "trainer.precision=16-mixed"
  "trainer.strategy.find_unused_parameters=true"
  "data.train_use_eval_agent_selection=false"
  "logger.wandb.group=smart_ntp_pretrain_trajtok_fair_compare"
  "logger.wandb.job_type=pretrain_trajtok_fair_compare"
)

if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${FAIR_COMPARISON_OVERRIDES[*]} ${EXTRA_HYDRA_OVERRIDES}"
else
  export EXTRA_HYDRA_OVERRIDES="${FAIR_COMPARISON_OVERRIDES[*]}"
fi

exec bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh "$@"
