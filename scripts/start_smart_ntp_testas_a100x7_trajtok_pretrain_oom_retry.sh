#!/usr/bin/env bash
# Launch TrajTok SMART NTP pretrain on the single testas A100x7 pod.
#
# This is a testas-specific preset on top of the generic A100 OOM-retry
# wrapper. It keeps the TrajTok branch recipe and recalculates LR by sqrt
# scaling whenever OOM retry lowers the per-rank train batch.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INITIAL_BS="${INITIAL_BS:-18}"
EFFECTIVE_BS=$(( INITIAL_BS * 7 ))

export NAMESPACE="${NAMESPACE:-p-pnc}"
export CONTAINER="${CONTAINER:-main}"
export PODS="${PODS:-testas}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_testas_a100x7_trajtok_20260605}"
export BRANCH="${BRANCH:-trajtok}"
export TASK_NAME="${TASK_NAME:-trajtok_pretrain_testas_a100x7_gbs${EFFECTIVE_BS}_lr595e4_oom_retry_20260605}"
export SESSION="${SESSION:-catk-smart-ntp-testas-a100x7-trajtok}"
export EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
export REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export MASTER_PORT="${MASTER_PORT:-29617}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export INITIAL_BS
export OOM_STEP="${OOM_STEP:-2}"
export MIN_BS="${MIN_BS:-14}"
export TOTAL_GPU_COUNT="${TOTAL_GPU_COUNT:-7}"
export BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-128}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-6e-4}"
export LEARNING_RATE="${LEARNING_RATE:-auto}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export A100_MAX_TRAIN_BATCH_SIZE="${A100_MAX_TRAIN_BATCH_SIZE:-64}"

TRAJTOK_OVERRIDES=(
  "trainer.precision=bf16-mixed"
  "trainer.strategy.find_unused_parameters=false"
  "data.train_use_eval_agent_selection=false"
  "data.train_memory_balanced_batching=true"
  "logger.wandb.group=smart_ntp_pretrain_trajtok_testas_a100x7"
  "logger.wandb.job_type=pretrain_trajtok_testas_a100x7"
)

if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  export EXTRA_HYDRA_OVERRIDES="${TRAJTOK_OVERRIDES[*]} ${EXTRA_HYDRA_OVERRIDES}"
else
  export EXTRA_HYDRA_OVERRIDES="${TRAJTOK_OVERRIDES[*]}"
fi

exec bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh "$@"
