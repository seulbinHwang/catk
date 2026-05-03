#!/usr/bin/env bash
# Launch the existing hsb-npc-training / hsb-npc-training2 H100x4x2 pretrain
# pipeline with automatic batch-size fallback on CUDA OOM.
#
# This script runs on the local machine that has kubectl access. It never
# creates, deletes, or restarts pods. Each attempt starts/replaces only the
# configured tmux session inside the already-running pods.
#
# Default behavior:
#   * first attempt: data.train_batch_size=26
#   * on CUDA OOM: reduce batch by 2 and resume from the latest epoch_last.ckpt
#   * stop at MIN_BS=20 unless overridden

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-hsb-npc-training hsb-npc-training2}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
BRANCH="${BRANCH:-self_forcing_bugfix}"
TASK_NAME="${TASK_NAME:-flow_semi_continuous_pretrain_h100x4x2_bs26}"
SESSION="${SESSION:-catk-h100x4-pretrain}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
MASTER_PORT="${MASTER_PORT:-29511}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
INITIAL_BS="${INITIAL_BS:-26}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-20}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
LEARNING_RATE="${LEARNING_RATE:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"

read -r -a POD_ARRAY <<< "$PODS"
if (( ${#POD_ARRAY[@]} < 2 )); then
  echo "ERROR: PODS must contain at least two pod names." >&2
  exit 2
fi
MASTER_POD="${POD_ARRAY[0]}"

LOCAL_RETRY_LOG_DIR="${REPO_ROOT}/logs/_h100x4_multinode_pretrain_oom_retry/${TASK_NAME}"
mkdir -p "$LOCAL_RETRY_LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

remote_quote() {
  printf '%q' "$1"
}

remote_run_root() {
  local safe_task="${TASK_NAME//\//_}"
  printf '%s/tmux_h100x4_multinode_pretrain/%s' "${REMOTE_LOG_DIR%/}" "$safe_task"
}

remote_master_tmux_log() {
  printf '%s/%s.tmux.log' "$(remote_run_root)" "$MASTER_POD"
}

find_latest_epoch_last_ckpt() {
  local runs_dir runs_dir_q cmd
  runs_dir="${REMOTE_LOG_DIR%/}/${TASK_NAME}/runs"
  runs_dir_q="$(remote_quote "$runs_dir")"
  cmd="ls -t ${runs_dir_q}/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1"
  kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$cmd" 2>/dev/null | tr -d '\r'
}

start_attempt() {
  local bs="$1"
  local ckpt_path="$2"
  local -a cmd=(
    python scripts/launch_h100x4_multinode_pretrain_tmux.py
    --namespace "$NAMESPACE"
    --pods "${POD_ARRAY[@]}"
    --container "$CONTAINER"
    --project-root "$PROJECT_ROOT"
    --branch "$BRANCH"
    --task-name "$TASK_NAME"
    --session "$SESSION"
    --master-port "$MASTER_PORT"
    --nproc-per-node "$NPROC_PER_NODE"
    --log-dir "$REMOTE_LOG_DIR"
    --train-batch-size "$bs"
    --replace
  )

  if [[ -n "$ckpt_path" ]]; then
    cmd+=(--ckpt-path "$ckpt_path")
  fi
  if [[ -n "$LEARNING_RATE" ]]; then
    cmd+=(--learning-rate "$LEARNING_RATE")
  fi
  if [[ -n "$VAL_BATCH_SIZE" ]]; then
    cmd+=(--val-batch-size "$VAL_BATCH_SIZE")
  fi
  if [[ -n "$LIMIT_TRAIN_BATCHES" ]]; then
    cmd+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
  fi
  if [[ -n "$LIMIT_VAL_BATCHES" ]]; then
    cmd+=(--limit-val-batches "$LIMIT_VAL_BATCHES")
  fi
  if [[ -n "$MAX_EPOCHS" ]]; then
    cmd+=(--max-epochs "$MAX_EPOCHS")
  fi
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    cmd+=(--extra-hydra-overrides "$EXTRA_HYDRA_OVERRIDES")
  fi

  log "launcher command:"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

wait_for_attempt_exit() {
  local remote_log remote_log_q grep_cmd status_line
  remote_log="$(remote_master_tmux_log)"
  remote_log_q="$(remote_quote "$remote_log")"
  grep_cmd="grep -E '\\[tmux-run\\] exited with status [0-9]+' ${remote_log_q} 2>/dev/null | tail -1"

  while true; do
    status_line="$(
      kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$grep_cmd" 2>/dev/null || true
    )"
    status_line="${status_line//$'\r'/}"
    if [[ "$status_line" =~ exited\ with\ status\ ([0-9]+) ]]; then
      ATTEMPT_EXIT_CODE="${BASH_REMATCH[1]}"
      return 0
    fi
    log "waiting for attempt to finish; attach: kubectl exec -it -n ${NAMESPACE} ${MASTER_POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
    sleep "$POLL_INTERVAL"
  done
}

copy_attempt_log() {
  local destination="$1"
  local remote_log remote_log_q
  remote_log="$(remote_master_tmux_log)"
  remote_log_q="$(remote_quote "$remote_log")"
  if ! kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "cat ${remote_log_q}" > "$destination" 2>/dev/null; then
    log "warning: failed to copy tmux log from ${MASTER_POD}:${remote_log}"
    : > "$destination"
  fi
}

bs="$INITIAL_BS"
attempt=0
while (( bs >= MIN_BS )); do
  attempt=$(( attempt + 1 ))
  attempt_log="${LOCAL_RETRY_LOG_DIR}/attempt_$(printf '%03d' "$attempt")_bs${bs}.log"
  latest_ckpt="$(find_latest_epoch_last_ckpt)"

  if [[ -n "$latest_ckpt" ]]; then
    log "Attempt #${attempt}: bs=${bs}, resume ckpt=${latest_ckpt}"
  else
    log "Attempt #${attempt}: bs=${bs}, fresh fit (no epoch_last.ckpt found yet)"
  fi
  log "  local attempt log -> ${attempt_log}"

  if ! start_attempt "$bs" "$latest_ckpt"; then
    log "launcher failed before torchrun completed."
    exit 1
  fi

  ATTEMPT_EXIT_CODE=""
  wait_for_attempt_exit
  copy_attempt_log "$attempt_log"

  if [[ "$ATTEMPT_EXIT_CODE" == "0" ]]; then
    log "Training completed successfully (attempt #${attempt}, bs=${bs})."
    exit 0
  fi

  if grep -Eq "$OOM_REGEX" "$attempt_log"; then
    new_bs=$(( bs - OOM_STEP ))
    log "OOM detected at bs=${bs} (exit=${ATTEMPT_EXIT_CODE}). Lowering to bs=${new_bs}."
    if (( new_bs < MIN_BS )); then
      log "Next bs=${new_bs} is below MIN_BS=${MIN_BS}; aborting."
      exit 1
    fi
    bs="$new_bs"
    continue
  fi

  log "Non-OOM failure (exit=${ATTEMPT_EXIT_CODE}). See ${attempt_log}. Aborting retry loop."
  exit "$ATTEMPT_EXIT_CODE"
done

log "Reached MIN_BS=${MIN_BS} without a successful run. Aborting."
exit 1
