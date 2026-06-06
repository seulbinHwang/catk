#!/usr/bin/env bash
set -euo pipefail

cd /home2/pnc2/repos_python/kinematic_flow

WATCHDOG_LOG="artifacts/road_oom_watchdog.log"
GPU_SET="2,3"
VAL_B=16
SCORER_SCENE_NUM=880
BASE_TASK="road_ft_main_prealign_lr1e5"
SUFFIX="v16_val200_lval002_scorer880_nofixednoise_2gpu_ddpunused_oomfallback"
INITIAL_TASK="road_ft_main_prealign_lr1e5_b32_v16_val200_lval002_nbatch27_nofixednoise_2gpu_ddpunused"
INITIAL_LOG="artifacts/road_ft_main_prealign_lr1e5_b32_v16_val200_lval002_nbatch27_nofixednoise_2gpu_ddpunused.log"
BATCHES=(32 16 8 4)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCHDOG_LOG"
}

task_for_batch() {
  local batch="$1"
  if [[ "$batch" == "32" ]]; then
    printf '%s\n' "$INITIAL_TASK"
  else
    printf '%s_b%s_%s\n' "$BASE_TASK" "$batch" "$SUFFIX"
  fi
}

log_for_batch() {
  local batch="$1"
  if [[ "$batch" == "32" ]]; then
    printf '%s\n' "$INITIAL_LOG"
  else
    printf 'artifacts/%s_b%s_%s.log\n' "$BASE_TASK" "$batch" "$SUFFIX"
  fi
}

is_running() {
  local task="$1"
  pgrep -f "task_name=${task}" >/dev/null 2>&1
}

kill_task() {
  local task="$1"
  pkill -TERM -f "task_name=${task}" 2>/dev/null || true
  sleep 8
  pkill -KILL -f "task_name=${task}" 2>/dev/null || true
}

has_oom() {
  local run_log="$1"
  [[ -f "$run_log" ]] && grep -Eqi 'CUDA out of memory|out of memory|OutOfMemoryError|CUDA error: out of memory|\bOOM\b' "$run_log"
}

has_non_oom_failure() {
  local run_log="$1"
  [[ -f "$run_log" ]] && grep -Eqi 'Traceback|Error executing job|RuntimeError|ValueError|TypeError|FileNotFoundError' "$run_log"
}

has_clean_finish() {
  local run_log="$1"
  [[ -f "$run_log" ]] && grep -Eqi 'Trainer.fit stopped|Closing wandb|`Trainer.fit` stopped' "$run_log"
}

launch_batch() {
  local batch="$1"
  local task
  local run_log
  task="$(task_for_batch "$batch")"
  run_log="$(log_for_batch "$batch")"
  : > "$run_log"
  log "Launching fallback task=${task} train_batch_size=${batch} val_batch_size=${VAL_B} scorer_scene_num=${SCORER_SCENE_NUM}"
  setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_SET" \
    WANDB_MODE=online \
    WANDB_SILENT=false \
    HYDRA_FULL_ERROR=1 \
    LOGLEVEL=INFO \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    /home2/pnc2/miniforge3/envs/catk/bin/python -u -m src.run \
      experiment=road_ft \
      action=finetune \
      task_name="$task" \
      ckpt_path=logs/pretrained/pretrained.ckpt \
      paths.cache_root=/home2/pnc2/repos_python/datasets/catk_cache \
      trainer.devices=2 \
      trainer.strategy=ddp_find_unused_parameters_true \
      trainer.limit_train_batches=1.0 \
      trainer.limit_val_batches=0.02 \
      trainer.val_check_interval=200 \
      trainer.check_val_every_n_epoch=null \
      trainer.log_every_n_steps=1 \
      trainer.max_epochs=16 \
      trainer.num_sanity_val_steps=0 \
      data.train_batch_size="$batch" \
      data.val_batch_size="$VAL_B" \
      data.test_batch_size="$VAL_B" \
      data.train_epoch_sample_fraction=0.5 \
      data.num_workers=4 \
      data.prefetch_factor=1 \
      model.model_config.lr=1e-5 \
      model.model_config.lr_warmup_steps=0 \
      model.model_config.lr_total_steps=16 \
      model.model_config.lr_min_ratio=1.0 \
      model.model_config.weight_decay=1e-4 \
      model.model_config.validation_fixed_flow_noise=false \
      model.model_config.scorer_scene_num="$SCORER_SCENE_NUM" \
      model.model_config.n_batch_sim_agents_metric=28 \
      model.model_config.n_rollout_closed_val=16 \
      model.model_config.finetune.train_except_map_encoder=true \
      model.model_config.finetune.flow_velocity_head_only=false \
      model.model_config.finetune.flow_ft_target=except_map_encoder \
      model.model_config.finetune.road_sample_k=64 \
      model.model_config.finetune.road_n_rollouts=1 \
      model.model_config.finetune.road_pred_max_steps=16 \
      model.model_config.finetune.road_temperature=0.8 \
      model.model_config.finetune.road_comparison_horizon=20 \
      model.model_config.finetune.road_strict_active_mask=true \
      callbacks.model_checkpoint.save_top_k=1 \
      callbacks.model_checkpoint.save_last=true \
      callbacks.epoch_last_checkpoint.upload_to_wandb=false \
      logger.wandb.log_model=false \
      >> "$run_log" 2>&1 < /dev/null &
  log "Fallback launched pid=$! log=${run_log}"
}

log "watchdog start: current task=${INITIAL_TASK}; fallback chain: ${BATCHES[*]}"

idx=0
while true; do
  batch="${BATCHES[$idx]}"
  task="$(task_for_batch "$batch")"
  run_log="$(log_for_batch "$batch")"

  if has_oom "$run_log"; then
    log "OOM detected for task=${task} batch=${batch}; killing task and preparing fallback"
    kill_task "$task"
    next_idx=$((idx + 1))
    if (( next_idx >= ${#BATCHES[@]} )); then
      log "No smaller fallback batch configured after batch=${batch}; watchdog stopping"
      exit 1
    fi
    idx="$next_idx"
    launch_batch "${BATCHES[$idx]}"
    sleep 60
    continue
  fi

  if is_running "$task"; then
    sleep 30
    continue
  fi

  if has_clean_finish "$run_log"; then
    log "Task appears to have finished cleanly: task=${task}; watchdog stopping"
    exit 0
  fi

  if has_non_oom_failure "$run_log"; then
    log "Non-OOM failure detected for task=${task}; watchdog will not relaunch"
    exit 2
  fi

  log "Task=${task} is not running and no OOM marker found yet; waiting for log flush"
  sleep 30
done
