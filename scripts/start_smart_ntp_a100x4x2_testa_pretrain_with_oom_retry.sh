#!/usr/bin/env bash
# Launch SMART NTP pretrain on existing x4 GPU pods with
# automatic CUDA OOM fallback.
#
# The first attempt starts at INITIAL_BS=16 by default. If any pod log contains
# a CUDA OOM marker, the script stops the distributed job, finds the newest
# epoch_last.ckpt under the same task name, lowers data.train_batch_size by
# OOM_STEP=1, and starts the next attempt from that checkpoint.
#
# This script runs on the local machine with kubectl access. It never creates,
# deletes, or restarts pods. Defaults target the historical testa/testaa x4x2
# recipe, but PODS can contain any positive number of x4 GPU pods.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-testa testaa}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_a100x4x2_oom_retry_main}"
REPO_URL="${REPO_URL:-https://github.com/seulbinHwang/catk.git}"
BRANCH="${BRANCH:-main}"
GIT_REF="${GIT_REF:-}"
TASK_NAME="${TASK_NAME:-smart_ntp_pretrain_a100x4x2_bs16_oom_retry_main}"
SESSION="${SESSION:-catk-smart-ntp-a100x4x2}"
EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
CACHE_ROOT="${CACHE_ROOT:-}"
POD_CACHE_ROOTS="${POD_CACHE_ROOTS:-}"
MASTER_PORT="${MASTER_PORT:-29521}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
INITIAL_BS="${INITIAL_BS:-16}"
OOM_STEP="${OOM_STEP:-1}"
MIN_BS="${MIN_BS:-8}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-2}"
RETRY_NON_OOM_EXIT_CODES="${RETRY_NON_OOM_EXIT_CODES:-134,143}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a POD_ARRAY <<< "$PODS"
if (( ${#POD_ARRAY[@]} < 1 )); then
  echo "ERROR: PODS must contain at least one x4 GPU pod name." >&2
  exit 2
fi
MASTER_POD="${POD_ARRAY[0]}"

if ! [[ "$INITIAL_BS" =~ ^[0-9]+$ && "$OOM_STEP" =~ ^[0-9]+$ && "$MIN_BS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: INITIAL_BS, OOM_STEP, and MIN_BS must be non-negative integers." >&2
  exit 2
fi
if (( INITIAL_BS < 1 || OOM_STEP < 1 || MIN_BS < 1 )); then
  echo "ERROR: INITIAL_BS, OOM_STEP, and MIN_BS must be >= 1." >&2
  exit 2
fi
if (( INITIAL_BS < MIN_BS )); then
  echo "ERROR: INITIAL_BS=${INITIAL_BS} is below MIN_BS=${MIN_BS}." >&2
  exit 2
fi

LOCAL_RETRY_LOG_DIR="${REPO_ROOT}/logs/_smart_ntp_a100x4x2_oom_retry/${TASK_NAME}"
mkdir -p "$LOCAL_RETRY_LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
remote_quote() { printf '%q' "$1"; }

prepare_project_root() {
  local pod repo_q root_q branch_q git_ref_q script
  repo_q="$(remote_quote "$REPO_URL")"
  root_q="$(remote_quote "$PROJECT_ROOT")"
  branch_q="$(remote_quote "$BRANCH")"
  git_ref_q="$(remote_quote "$GIT_REF")"

  for pod in "${POD_ARRAY[@]}"; do
    script="
set -Eeuo pipefail
repo=${repo_q}
root=${root_q}
branch=${branch_q}
git_ref=${git_ref_q}
mkdir -p \"\$(dirname \"\$root\")\"
if [ ! -d \"\$root/.git\" ]; then
  git clone \"\$repo\" \"\$root\"
fi
cd \"\$root\"
git config --global --add safe.directory \"\$root\" || true
git fetch origin --prune
if [ -n \"\$git_ref\" ]; then
  git checkout --detach \"\$git_ref\"
else
  git checkout -B \"\$branch\" \"origin/\$branch\"
fi
git status --short --branch
git rev-parse --short HEAD
"
    log "preparing project root on ${pod}: ${PROJECT_ROOT}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'kubectl exec -n %q %q -c %q -- bash -lc %q\n' \
        "$NAMESPACE" "$pod" "$CONTAINER" "$script"
    else
      kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "$script"
    fi
  done
}

safe_task_name() {
  printf '%s\n' "${TASK_NAME//\//_}"
}

remote_run_root() {
  printf '%s/tmux_smart_ntp_a100x4x2/%s' "${REMOTE_LOG_DIR%/}" "$(safe_task_name)"
}

remote_tmux_log_for_pod() {
  local pod="$1"
  printf '%s/%s.tmux.log' "$(remote_run_root)" "$pod"
}

remote_master_tmux_log() {
  remote_tmux_log_for_pod "$MASTER_POD"
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

stop_attempt_sessions() {
  local -a cmd=(
    python scripts/launch_smart_ntp_a100x4x2_testa.py
    --namespace "$NAMESPACE"
    --pods "${POD_ARRAY[@]}"
    --container "$CONTAINER"
    --project-root "$PROJECT_ROOT"
    --no-pull
    --branch "$BRANCH"
    --task-name "$TASK_NAME"
    --session "$SESSION"
    --stop
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run stop command:"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}" >/dev/null 2>&1 || true
}

start_attempt() {
  local bs="$1"
  local ckpt_path="$2"
  local val_bs="$VAL_BATCH_SIZE"
  local test_bs="$TEST_BATCH_SIZE"
  local -a cmd=(
    python scripts/launch_smart_ntp_a100x4x2_testa.py
    --namespace "$NAMESPACE"
    --pods "${POD_ARRAY[@]}"
    --container "$CONTAINER"
    --project-root "$PROJECT_ROOT"
    --no-pull
    --branch "$BRANCH"
    --experiment "$EXPERIMENT"
    --task-name "$TASK_NAME"
    --session "$SESSION"
    --master-port "$MASTER_PORT"
    --nproc-per-node "$NPROC_PER_NODE"
    --log-dir "$REMOTE_LOG_DIR"
    --train-batch-size "$bs"
    --val-batch-size "$val_bs"
    --test-batch-size "$test_bs"
    --replace
  )

  if [[ -n "$GIT_REF" ]]; then
    cmd+=(--git-ref "$GIT_REF")
  fi
  if [[ -n "$CACHE_ROOT" ]]; then
    cmd+=(--cache-root "$CACHE_ROOT")
  fi
  if [[ -n "$POD_CACHE_ROOTS" ]]; then
    local mapping
    local -a pod_cache_root_array
    read -r -a pod_cache_root_array <<< "$POD_CACHE_ROOTS"
    for mapping in "${pod_cache_root_array[@]}"; do
      cmd+=(--pod-cache-root "$mapping")
    done
  fi
  if [[ -n "$ckpt_path" ]]; then
    cmd+=(--ckpt-path "$ckpt_path")
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
  if [[ -n "$LEARNING_RATE" ]]; then
    cmd+=(--learning-rate "$LEARNING_RATE")
  fi
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    cmd+=(--extra-hydra-overrides "$EXTRA_HYDRA_OVERRIDES")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  log "launcher command:"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
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

wait_for_attempt_exit() {
  local remote_log remote_log_q grep_cmd status_line oom_pod
  remote_log="$(remote_master_tmux_log)"
  remote_log_q="$(remote_quote "$remote_log")"
  grep_cmd="grep -E '\\[tmux-run\\] exited with status [0-9]+' ${remote_log_q} 2>/dev/null | tail -1"

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

    log "waiting for attempt to finish; attach: kubectl exec -it -n ${NAMESPACE} ${MASTER_POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
    sleep "$POLL_INTERVAL"
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
non_oom_retry_count=0

prepare_project_root

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
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run completed after rendering the first attempt."
    exit 0
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
    log "Retryable non-OOM exit=${ATTEMPT_EXIT_CODE}; retrying bs=${bs} from latest checkpoint if available (${non_oom_retry_count}/${MAX_NON_OOM_RETRIES})."
    stop_attempt_sessions
    continue
  fi

  log "Non-OOM failure (exit=${ATTEMPT_EXIT_CODE}). See ${attempt_log}. Aborting retry loop."
  exit "$ATTEMPT_EXIT_CODE"
done

log "Reached MIN_BS=${MIN_BS} without a successful run. Aborting."
exit 1
