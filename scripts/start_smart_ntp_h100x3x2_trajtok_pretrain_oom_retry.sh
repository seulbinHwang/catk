#!/usr/bin/env bash
# Launch the TrajTok SMART NTP pretrain run on
# hsb-npc-training-3-1/hsb-npc-training-3-2, each with 3 H100 GPUs.
#
# This mirrors scripts/start_smart_ntp_v100x4x4_trajtok_pretrain_oom_retry.sh
# but targets a homogeneous H100x3x2 layout. Six GPUs cannot exactly reproduce
# the V100 wrapper's effective global train batch of 112 with
# accumulate_grad_batches=1, so this wrapper starts at per-GPU batch 18
# (effective global train batch 108) and uses the shared OOM retry loop to lower
# the batch if an out-of-memory marker appears.
#
# Matched recipe intent:
#   - pre_bc_a100x4x2 experiment preset
#   - latest trajtok branch
#   - TrajTok vocab, type-specific decoder heads, and spatial-aware smoothing
#   - data.train_use_eval_agent_selection=false
#   - trainer.accumulate_grad_batches=1 from the preset
#   - zero-gradient-touched type-specific classifier heads with DDP
#     unused-parameter detection disabled
#
# H100-specific runtime choice:
#   - bf16 mixed precision, because H100 supports bf16 natively.
set -Eeuo pipefail

export PODS="${PODS:-hsb-npc-training-3-1 hsb-npc-training-3-2}"
export BRANCH="${BRANCH:-trajtok}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_h100x3x2_trajtok_fair}"
export TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_h100x3x2_globalbs108_oom_retry_trajtok_legacy_inputs_trainselectfalse_20260529}"
export SESSION="${SESSION:-catk-smart-ntp-h100x3x2-trajtok}"
export MASTER_PORT="${MASTER_PORT:-29551}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export INITIAL_BS="${INITIAL_BS:-18}"
export OOM_STEP="${OOM_STEP:-1}"
export MIN_BS="${MIN_BS:-14}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"

FAIR_COMPARISON_OVERRIDES=(
  "trainer.precision=bf16-mixed"
  "trainer.strategy.find_unused_parameters=false"
  "data.train_use_eval_agent_selection=false"
  "logger.wandb.group=smart_ntp_pretrain_trajtok_fair_compare"
  "logger.wandb.job_type=pretrain_trajtok_h100x3x2_fair_compare"
)

if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${FAIR_COMPARISON_OVERRIDES[*]} ${EXTRA_HYDRA_OVERRIDES}"
else
  export EXTRA_HYDRA_OVERRIDES="${FAIR_COMPARISON_OVERRIDES[*]}"
fi

exec bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh "$@"
