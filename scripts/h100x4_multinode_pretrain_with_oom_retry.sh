#!/usr/bin/env bash
# Launch the existing hsb-npc-training / hsb-npc-training-2 H100x4x2 pretrain
# pipeline with automatic batch-size fallback on CUDA OOM.
#
# This script runs on the local machine that has kubectl access. It never
# creates, deletes, or restarts pods. Each attempt starts/replaces only the
# configured tmux session inside the already-running pods.
#
# Default behavior:
#   * first attempt: data.train_batch_size=26
#   * on CUDA OOM: reduce batch by OOM_STEP and resume from the latest epoch_last.ckpt.
#     If OOM_STEP=0, keep the same batch size; set MAX_OOM_ATTEMPTS to cap retries.
#   * on retryable external exits such as SIGTERM/SIGABRT: keep the batch size
#     and resume from the latest epoch_last.ckpt
#   * stop at MIN_BS=20 unless overridden, or at MAX_OOM_ATTEMPTS when set

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-hsb-npc-training hsb-npc-training-2}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
BRANCH="${BRANCH:-self_forcing_bugfix}"
GIT_REF="${GIT_REF:-}"
TASK_NAME="${TASK_NAME:-flow_semi_continuous_pretrain_h100x4x2_bs26}"
SESSION="${SESSION:-catk-h100x4-pretrain}"
EXPERIMENT="${EXPERIMENT:-pre_bc_flow_2x4_h100}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
CACHE_ROOT="${CACHE_ROOT:-}"
POD_CACHE_ROOTS="${POD_CACHE_ROOTS:-}"
MASTER_PORT="${MASTER_PORT:-29511}"
CHECKPOINT_SYNC_PORT="${CHECKPOINT_SYNC_PORT:-29512}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MANUAL_RANK_OFFSETS="${MANUAL_RANK_OFFSETS:-0}"
INITIAL_BS="${INITIAL_BS:-26}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-20}"
MAX_OOM_ATTEMPTS="${MAX_OOM_ATTEMPTS:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
RETRY_NON_OOM_EXIT_CODES="${RETRY_NON_OOM_EXIT_CODES:-134,143}"
MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-3}"
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

oom_attempt_limit_label() {
  if (( MAX_OOM_ATTEMPTS > 0 )); then
    printf '%s' "$MAX_OOM_ATTEMPTS"
  else
    printf '%s' "unlimited"
  fi
}

remote_quote() {
  printf '%q' "$1"
}

remote_run_root() {
  local safe_task="${TASK_NAME//\//_}"
  printf '%s/tmux_h100x4_multinode_pretrain/%s' "${REMOTE_LOG_DIR%/}" "$safe_task"
}

remote_tmux_log_for_pod() {
  local pod="$1"
  printf '%s/%s.tmux.log' "$(remote_run_root)" "$pod"
}

remote_master_tmux_log() {
  remote_tmux_log_for_pod "$MASTER_POD"
}

remote_status_file_for_pod() {
  local pod="$1"
  printf '%s/%s.torchrun_status' "$(remote_run_root)" "$pod"
}

is_retryable_non_oom_exit() {
  local status="$1"
  local code
  local -a retry_codes
  IFS=',' read -r -a retry_codes <<< "$RETRY_NON_OOM_EXIT_CODES"
  for code in "${retry_codes[@]}"; do
    code="${code//[[:space:]]/}"
    if [[ -n "$code" && "$status" == "$code" ]]; then
      return 0
    fi
  done
  return 1
}

find_latest_epoch_last_ckpt() {
  local runs_dir runs_dir_q cmd
  runs_dir="${REMOTE_LOG_DIR%/}/${TASK_NAME}/runs"
  runs_dir_q="$(remote_quote "$runs_dir")"
  cmd="{ ls -t ${runs_dir_q}/*/checkpoints/epoch_last.ckpt 2>/dev/null; ls -t ${runs_dir_q}/*/checkpoints/last.ckpt 2>/dev/null; } | head -1"
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
    --experiment "$EXPERIMENT"
    --task-name "$TASK_NAME"
    --session "$SESSION"
    --master-port "$MASTER_PORT"
    --checkpoint-sync-port "$CHECKPOINT_SYNC_PORT"
    --nproc-per-node "$NPROC_PER_NODE"
    --log-dir "$REMOTE_LOG_DIR"
    --train-batch-size "$bs"
    --replace
  )

  if [[ -n "$GIT_REF" ]]; then
    cmd+=(--git-ref "$GIT_REF")
  fi

  if [[ -n "$CACHE_ROOT" ]]; then
    cmd+=(--cache-root "$CACHE_ROOT")
  fi
  if [[ "$MANUAL_RANK_OFFSETS" == "1" ]]; then
    cmd+=(--manual-rank-offsets)
  fi
  if [[ -n "$POD_CACHE_ROOTS" ]]; then
    read -r -a pod_cache_root_array <<< "$POD_CACHE_ROOTS"
    local mapping
    for mapping in "${pod_cache_root_array[@]}"; do
      cmd+=(--pod-cache-root "$mapping")
    done
  fi
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
  local remote_log remote_log_q grep_cmd status_line oom_pod status_file status_file_q status_cmd
  remote_log="$(remote_master_tmux_log)"
  remote_log_q="$(remote_quote "$remote_log")"
  grep_cmd="grep -E '\\[tmux-run\\] exited with status [0-9]+' ${remote_log_q} 2>/dev/null | tail -1"
  status_file="$(remote_status_file_for_pod "$MASTER_POD")"
  status_file_q="$(remote_quote "$status_file")"
  status_cmd="cat ${status_file_q} 2>/dev/null | tail -1"

  while true; do
    if oom_pod="$(find_remote_oom_pod)"; then
      log "OOM marker observed on ${oom_pod}; stopping all node sessions before retry."
      ATTEMPT_EXIT_CODE="1"
      ATTEMPT_EXIT_REASON="oom"
      stop_attempt_sessions
      return 0
    fi
    status_line="$(
      kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$grep_cmd" 2>/dev/null || true
    )"
    status_line="${status_line//$'\r'/}"
    if [[ "$status_line" =~ exited\ with\ status\ ([0-9]+) ]]; then
      ATTEMPT_EXIT_CODE="${BASH_REMATCH[1]}"
      ATTEMPT_EXIT_REASON="exit"
      return 0
    fi
    status_line="$(
      kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$status_cmd" 2>/dev/null || true
    )"
    status_line="${status_line//$'\r'/}"
    if [[ "$status_line" =~ ^[0-9]+$ ]]; then
      ATTEMPT_EXIT_CODE="$status_line"
      ATTEMPT_EXIT_REASON="status_file"
      return 0
    fi
    log "waiting for attempt to finish; attach: kubectl exec -it -n ${NAMESPACE} ${MASTER_POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
    sleep "$POLL_INTERVAL"
  done
}

find_remote_oom_pod() {
  local pod remote_log remote_log_q oom_regex_q cmd
  oom_regex_q="$(remote_quote "$OOM_REGEX")"
  for pod in "${POD_ARRAY[@]}"; do
    remote_log="$(remote_tmux_log_for_pod "$pod")"
    remote_log_q="$(remote_quote "$remote_log")"
    cmd="grep -Eq ${oom_regex_q} ${remote_log_q} 2>/dev/null"
    if kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "$cmd" >/dev/null 2>&1; then
      printf '%s\n' "$pod"
      return 0
    fi
  done
  return 1
}

stop_attempt_sessions() {
  local pod run_root run_root_q session_q pod_q task_q grace
  run_root="$(remote_run_root)"
  run_root_q="$(remote_quote "$run_root")"
  session_q="$(remote_quote "$SESSION")"
  task_q="$(remote_quote "$TASK_NAME")"
  grace="${REMOTE_KILL_GRACE_SEC:-20}"
  for pod in "${POD_ARRAY[@]}"; do
    pod_q="$(remote_quote "$pod")"
    kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set +e
TASK_NAME_TO_STOP=${task_q}
task_process_pids() {
  pgrep -f \"task_name=\${TASK_NAME_TO_STOP}\" 2>/dev/null | while read -r pid; do
    if [[ -n \"\$pid\" && \"\$pid\" != \"\$\$\" && \"\$pid\" != \"\${BASHPID:-}\" ]]; then
      echo \"\$pid\"
    fi
  done
}
terminate_task_processes() {
  local pids=()
  mapfile -t pids < <(task_process_pids || true)
  if (( \${#pids[@]} > 0 )); then
    echo \"terminating task processes for \${TASK_NAME_TO_STOP}: \${pids[*]}\"
    kill -TERM \"\${pids[@]}\" 2>/dev/null || true
    sleep ${grace}
    mapfile -t pids < <(task_process_pids || true)
    if (( \${#pids[@]} > 0 )); then
      echo \"force killing task processes for \${TASK_NAME_TO_STOP}: \${pids[*]}\"
      kill -KILL \"\${pids[@]}\" 2>/dev/null || true
    fi
  fi
}
pgid_file=${run_root_q}/${pod_q}.torchrun_pgid
if [[ -f \"\$pgid_file\" ]]; then
  pgid=\"\$(cat \"\$pgid_file\" 2>/dev/null || true)\"
  if [[ -n \"\$pgid\" && \"\$pgid\" != \"0\" ]]; then
    kill -TERM -- \"-\$pgid\" 2>/dev/null || true
    sleep ${grace}
    kill -KILL -- \"-\$pgid\" 2>/dev/null || true
  fi
fi
terminate_task_processes
tmux kill-session -t ${session_q} 2>/dev/null || true
" >/dev/null 2>&1 || true
  done
}

copy_attempt_log() {
  local destination="$1"
  local pod remote_log remote_log_q
  : > "$destination"
  for pod in "${POD_ARRAY[@]}"; do
    remote_log="$(remote_tmux_log_for_pod "$pod")"
    remote_log_q="$(remote_quote "$remote_log")"
    {
      printf '===== %s:%s =====\n' "$pod" "$remote_log"
      if ! kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "cat ${remote_log_q}" 2>/dev/null; then
        printf 'warning: failed to copy tmux log from %s:%s\n' "$pod" "$remote_log"
      fi
      printf '\n'
    } >> "$destination"
  done
}

bs="$INITIAL_BS"
attempt=0
oom_attempt_count=0
non_oom_retry_count=0
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

  stop_attempt_sessions
  if ! start_attempt "$bs" "$latest_ckpt"; then
    log "launcher failed before torchrun completed."
    exit 1
  fi

  ATTEMPT_EXIT_CODE=""
  ATTEMPT_EXIT_REASON=""
  wait_for_attempt_exit
  copy_attempt_log "$attempt_log"

  if [[ "$ATTEMPT_EXIT_CODE" == "0" ]]; then
    log "Training completed successfully (attempt #${attempt}, bs=${bs})."
    exit 0
  fi

  if [[ "$ATTEMPT_EXIT_REASON" == "oom" ]] || grep -Eq "$OOM_REGEX" "$attempt_log"; then
    non_oom_retry_count=0
    oom_attempt_count=$(( oom_attempt_count + 1 ))
    if (( MAX_OOM_ATTEMPTS > 0 && oom_attempt_count >= MAX_OOM_ATTEMPTS )); then
      log "OOM detected at bs=${bs} (exit=${ATTEMPT_EXIT_CODE}); reached MAX_OOM_ATTEMPTS=${MAX_OOM_ATTEMPTS}. Stopping without lowering batch size."
      stop_attempt_sessions
      exit 1
    fi
    if (( OOM_STEP == 0 )); then
      log "OOM detected at bs=${bs} (exit=${ATTEMPT_EXIT_CODE}). Keeping bs=${bs} and resuming from the latest checkpoint (${oom_attempt_count}/$(oom_attempt_limit_label) OOM attempts used)."
      continue
    fi
    new_bs=$(( bs - OOM_STEP ))
    log "OOM detected at bs=${bs} (exit=${ATTEMPT_EXIT_CODE}). Lowering to bs=${new_bs}."
    if (( new_bs < MIN_BS )); then
      log "Next bs=${new_bs} is below MIN_BS=${MIN_BS}; aborting."
      exit 1
    fi
    bs="$new_bs"
    continue
  fi

  if is_retryable_non_oom_exit "$ATTEMPT_EXIT_CODE" && (( non_oom_retry_count < MAX_NON_OOM_RETRIES )); then
    non_oom_retry_count=$(( non_oom_retry_count + 1 ))
    latest_ckpt="$(find_latest_epoch_last_ckpt)"
    if [[ -n "$latest_ckpt" ]]; then
      log "Retryable non-OOM exit=${ATTEMPT_EXIT_CODE}; retrying bs=${bs} from latest ckpt=${latest_ckpt} (${non_oom_retry_count}/${MAX_NON_OOM_RETRIES})."
    else
      log "Retryable non-OOM exit=${ATTEMPT_EXIT_CODE}; retrying bs=${bs} with no checkpoint found (${non_oom_retry_count}/${MAX_NON_OOM_RETRIES})."
    fi
    stop_attempt_sessions
    continue
  fi

  log "Non-OOM failure (exit=${ATTEMPT_EXIT_CODE}). See ${attempt_log}. Aborting retry loop."
  exit "$ATTEMPT_EXIT_CODE"
done

log "Reached MIN_BS=${MIN_BS} without a successful run. Aborting."
exit 1
