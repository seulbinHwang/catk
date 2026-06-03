#!/usr/bin/env bash
# Run MDG pretrain on one pod with 7 A100 GPUs.
#
# Intended default target:
#   pod: testas
#   cache: /workspace/womd_v1_3/MDG_cache
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

main() {
  export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/MDG_cache}"
  export NNODES="${NNODES:-1}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
  export TRAINER_DEVICES="${TRAINER_DEVICES:-7}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29671}"

  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-28}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
  export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
  export MAX_EPOCHS="${MAX_EPOCHS:-64}"
  export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
  export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"
  export TASK_NAME="${TASK_NAME:-mdg_wosac_pretrain_testas_a100x7_bs${TRAIN_BATCH_SIZE}}"
  export WANDB_MODE="${WANDB_MODE:-online}"

  local precision="${PRECISION:-bf16-mixed}"
  local data_num_workers="${DATA_NUM_WORKERS:-4}"
  local val_closed_loop="${VAL_CLOSED_LOOP:-true}"
  local n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC:-10}"
  local scorer_scene_num="${SCORER_SCENE_NUM:-1680}"
  local checkpoint_monitor="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
  local checkpoint_mode="${CHECKPOINT_MODE:-max}"
  local memory_balanced_batching="${TRAIN_MEMORY_BALANCED_BATCHING:-true}"
  local memory_metadata_num_workers="${TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS:-32}"
  local memory_build_on_missing="${TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING:-true}"

  local overrides=(
    ++trainer.use_distributed_sampler=false
    trainer.num_sanity_val_steps=0
    trainer.precision="$precision"
    data.num_workers="$data_num_workers"
    model.model_config.val_closed_loop="$val_closed_loop"
    model.model_config.n_batch_sim_agents_metric="$n_batch_sim_agents_metric"
    model.model_config.scorer_scene_num="$scorer_scene_num"
    callbacks.model_checkpoint.monitor="$checkpoint_monitor"
    callbacks.model_checkpoint.mode="$checkpoint_mode"
    data.train_memory_balanced_batching="$memory_balanced_batching"
    data.train_memory_balance_metadata_num_workers="$memory_metadata_num_workers"
    data.train_memory_balance_build_on_missing="$memory_build_on_missing"
  )
  if [[ -n "${TRAIN_MEMORY_BALANCE_METADATA_CACHE:-}" ]]; then
    overrides+=(data.train_memory_balance_metadata_cache="$TRAIN_MEMORY_BALANCE_METADATA_CACHE")
  fi

  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    local user_overrides=()
    read -r -a user_overrides <<< "$CATK_HYDRA_OVERRIDES"
    overrides+=("${user_overrides[@]}")
  fi
  export CATK_HYDRA_OVERRIDES="${overrides[*]}"

  log "MDG A100x7 pretrain"
  log "  cache_root:       $CACHE_ROOT"
  log "  train_batch_size: $TRAIN_BATCH_SIZE per GPU"
  log "  global_batch:     $((TRAIN_BATCH_SIZE * NPROC_PER_NODE * NNODES))"
  log "  val_batch_size:   $VAL_BATCH_SIZE"
  log "  max_epochs:       $MAX_EPOCHS"
  log "  precision:        $precision"
  log "  wandb_mode:       $WANDB_MODE"
  log "  val_closed_loop:  $val_closed_loop"
  log "  scorer_scene_num: $scorer_scene_num"
  log "  checkpoint:       $checkpoint_monitor ($checkpoint_mode)"
  log "  mem balanced:     $memory_balanced_batching"
  log "  metadata workers: $memory_metadata_num_workers"

  exec bash scripts/smart_ntp_a100x4x2_pretrain.sh "$@"
}

main "$@"
