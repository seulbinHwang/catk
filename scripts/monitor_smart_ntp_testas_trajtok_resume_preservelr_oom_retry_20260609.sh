#!/usr/bin/env bash
# Monitor the 20260609 TrajTok testas resume run and retry with a lower
# per-rank batch size if CUDA OOM appears. This is intentionally separate from
# the launcher so it can attach to an already-running tmux job without
# interrupting it.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
POD="${POD:-testas}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_testas_a100x7_trajtok_resume_preservelr_20260609}"
BRANCH="${BRANCH:-trajtok}"
TASK_NAME="${TASK_NAME:-smart_ntp_resume_testas_a100x7_from_h100x4x2_epoch44_20260609_6bad4615_bs16_lr581e-6_preservelr}"
SESSION="${SESSION:-catk-smart-ntp-testas-a100x7-trajtok-resume-preservelr}"
EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
MASTER_PORT="${MASTER_PORT:-29643}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
CURRENT_BS="${CURRENT_BS:-16}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
SOURCE_CKPT_PATH="${SOURCE_CKPT_PATH:-/mnt/nuplan/projects/catk/checkpoints/trajtok_resume_h100x4x2_20260605/epoch_last_from_wandb.ckpt}"
LEARNING_RATE="${LEARNING_RATE:-0.0005809475}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
remote_quote() { printf '%q' "$1"; }

remote_run_root() {
  printf '%s/tmux_smart_ntp_a100x4x2/%s' "${REMOTE_LOG_DIR%/}" "$TASK_NAME"
}

remote_tmux_log() {
  printf '%s/%s.tmux.log' "$(remote_run_root)" "$POD"
}

remote_runs_dir() {
  printf '%s/%s/runs' "${REMOTE_LOG_DIR%/}" "$TASK_NAME"
}

find_latest_epoch_last_ckpt() {
  local runs_dir_q
  runs_dir_q="$(remote_quote "$(remote_runs_dir)")"
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc \
    "{ ls -t ${runs_dir_q}/*/checkpoints/epoch_last.ckpt 2>/dev/null; ls -t ${runs_dir_q}/*/checkpoints/last.ckpt 2>/dev/null; } | head -1" \
    2>/dev/null | tr -d '\r'
}

remote_log_has_oom() {
  local log_q regex_q
  log_q="$(remote_quote "$(remote_tmux_log)")"
  regex_q="$(remote_quote "$OOM_REGEX")"
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc \
    "grep -Eq ${regex_q} ${log_q} 2>/dev/null" >/dev/null 2>&1
}

remote_exit_status() {
  local log_q
  log_q="$(remote_quote "$(remote_tmux_log)")"
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc \
    "grep -E '\\[tmux-run\\] exited with status [0-9]+' ${log_q} 2>/dev/null | tail -1" \
    2>/dev/null | tr -d '\r'
}

stop_session() {
  python3 scripts/launch_smart_ntp_a100x4x2_testa.py \
    --namespace "$NAMESPACE" \
    --pods "$POD" \
    --container "$CONTAINER" \
    --project-root "$PROJECT_ROOT" \
    --no-pull \
    --branch "$BRANCH" \
    --task-name "$TASK_NAME" \
    --session "$SESSION" \
    --stop >/dev/null 2>&1 || true
}

clear_remote_tmux_log() {
  local log_q
  log_q="$(remote_quote "$(remote_tmux_log)")"
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc "rm -f ${log_q}" || true
}

start_attempt() {
  local bs="$1"
  local ckpt_path="$2"
  local extra_overrides
  extra_overrides="trainer.precision=bf16-mixed trainer.strategy.find_unused_parameters=false data.train_use_eval_agent_selection=false data.train_memory_balanced_batching=true data.train_agent_token_sidecar_dir=${CACHE_ROOT}/trajtok_agent_token_sidecar/training data.train_agent_token_sidecar_required=true model.model_config.token_processor.agent_sidecar_dir=${CACHE_ROOT}/trajtok_agent_token_sidecar/training model.model_config.token_processor.agent_sidecar_required=true logger.wandb.group=smart_ntp_pretrain_trajtok_testas_a100x7 logger.wandb.job_type=pretrain_trajtok_testas_a100x7"

  python3 scripts/launch_smart_ntp_a100x4x2_testa.py \
    --namespace "$NAMESPACE" \
    --pods "$POD" \
    --container "$CONTAINER" \
    --project-root "$PROJECT_ROOT" \
    --no-pull \
    --branch "$BRANCH" \
    --experiment "$EXPERIMENT" \
    --task-name "$TASK_NAME" \
    --session "$SESSION" \
    --master-port "$MASTER_PORT" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --log-dir "$REMOTE_LOG_DIR" \
    --train-batch-size "$bs" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --test-batch-size "$TEST_BATCH_SIZE" \
    --replace \
    --cache-root "$CACHE_ROOT" \
    --ckpt-path "$ckpt_path" \
    --learning-rate "$LEARNING_RATE" \
    --extra-hydra-overrides "$extra_overrides"
}

bs="$CURRENT_BS"
attempt=1
log "attached monitor: task=${TASK_NAME}, current_bs=${bs}, min_bs=${MIN_BS}, lr=${LEARNING_RATE}"
log "remote log: $(remote_tmux_log)"

while true; do
  if remote_log_has_oom; then
    next_bs=$(( bs - OOM_STEP ))
    log "OOM detected at bs=${bs}; stopping current session."
    stop_session
    if (( next_bs < MIN_BS )); then
      log "next bs=${next_bs} is below MIN_BS=${MIN_BS}; leaving stopped and exiting."
      exit 1
    fi

    latest_ckpt="$(find_latest_epoch_last_ckpt)"
    if [[ -z "$latest_ckpt" ]]; then
      latest_ckpt="$SOURCE_CKPT_PATH"
    fi
    log "retrying with bs=${next_bs}, ckpt=${latest_ckpt}"
    clear_remote_tmux_log
    start_attempt "$next_bs" "$latest_ckpt"
    bs="$next_bs"
    attempt=$(( attempt + 1 ))
    sleep "$POLL_INTERVAL"
    continue
  fi

  status_line="$(remote_exit_status)"
  if [[ "$status_line" =~ exited\ with\ status\ ([0-9]+) ]]; then
    status="${BASH_REMATCH[1]}"
    if [[ "$status" == "0" ]]; then
      log "training completed successfully: ${status_line}"
      exit 0
    fi
    log "training exited without OOM marker: ${status_line}"
    exit "$status"
  fi

  log "monitoring bs=${bs}; attach: kubectl exec -it -n ${NAMESPACE} ${POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
  sleep "$POLL_INTERVAL"
done
