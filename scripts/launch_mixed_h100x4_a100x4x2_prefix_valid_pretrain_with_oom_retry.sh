#!/usr/bin/env bash
# Launch mixed H100x4 + A100x4x2 FW30 prefix-valid pretrain with OOM fallback.
#
# This script runs on the local machine with kubectl access. It never creates,
# deletes, or restarts pods; it only replaces the configured tmux session inside
# the already-running pods.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PODS="${PODS:-wo-pvc-800 testa testaa}"
export BRANCH="${BRANCH:-self_forcing_anchor_new}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_flow_mixed_h100x4_a100x4x2_prefix_valid}"
export TASK_NAME="${TASK_NAME:-flow_pretrain_prefix_valid_fw30_maskaware_mixed_h100x4_a100x4x2_bs26}"
export SESSION="${SESSION:-catk-pretrain-mixed-h100-a100-prefix-fw30}"
export INITIAL_BS="${INITIAL_BS:-26}"
export OOM_STEP="${OOM_STEP:-2}"
export MIN_BS="${MIN_BS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-5.0e-4}"
export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-model.model_config.decoder.flow_window_steps=30 model.model_config.token_processor.use_prefix_valid_future_loss_mask=true}"

exec bash scripts/h100x4_multinode_pretrain_with_oom_retry.sh "$@"
