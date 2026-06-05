#!/usr/bin/env bash
# Launch UniMM Anchor-Based-4s on hsb-npc-training-3-{1,2} with automatic
# CUDA OOM fallback.
#
# The first attempt starts at INITIAL_BS=28 by default. If any pod log contains
# a CUDA OOM marker, the script stops the distributed job, finds the newest
# epoch_last.ckpt under the same task name, lowers data.train_batch_size by
# OOM_STEP, and starts the next attempt from that checkpoint. Non-OOM exits such
# as SIGTERM/SIGABRT can be retried at the same batch size.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-hsb-npc-training-3-1 hsb-npc-training-3-2}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_unimm_h100x3x2}"
REPO_URL="${REPO_URL:-https://github.com/seulbinHwang/catk.git}"
BRANCH="${BRANCH:-UniMM}"
TASK_NAME="${TASK_NAME:-unimm_anchor_based_4s_h100x3x2_pretrain_globalbs168_oom_retry}"
SESSION="${SESSION:-unimm-h100x3x2}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
ANCHOR_FILE="${ANCHOR_FILE:-}"
MASTER_PORT="${MASTER_PORT:-29551}"
INITIAL_BS="${INITIAL_BS:-28}"
OOM_STEP="${OOM_STEP:-2}"
MAX_SAME_BS_OOM_RETRIES="${MAX_SAME_BS_OOM_RETRIES:-3}"
MIN_BS="${MIN_BS:-16}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-3}"
RETRY_NON_OOM_EXIT_CODES="${RETRY_NON_OOM_EXIT_CODES:-134,143}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-0.0005}"
BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE:-32}"
RESUME_LR_POLICY="${RESUME_LR_POLICY:-checkpoint}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a POD_ARRAY <<< "$PODS"
if (( ${#POD_ARRAY[@]} != 2 )); then
  echo "ERROR: PODS must contain exactly two H100 x3 pod names." >&2
  exit 2
fi
MASTER_POD="${POD_ARRAY[0]}"

if ! [[ "$INITIAL_BS" =~ ^[0-9]+$ && "$OOM_STEP" =~ ^[0-9]+$ && "$MIN_BS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: INITIAL_BS, OOM_STEP, and MIN_BS must be non-negative integers." >&2
  exit 2
fi
if (( INITIAL_BS < 1 || MIN_BS < 1 )); then
  echo "ERROR: INITIAL_BS and MIN_BS must be >= 1." >&2
  exit 2
fi
if (( INITIAL_BS < MIN_BS )); then
  echo "ERROR: INITIAL_BS=${INITIAL_BS} is below MIN_BS=${MIN_BS}." >&2
  exit 2
fi

LOCAL_RETRY_LOG_DIR="${REPO_ROOT}/logs/_unimm_h100x3x2_oom_retry/${TASK_NAME}"
mkdir -p "$LOCAL_RETRY_LOG_DIR"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
remote_quote() { printf '%q' "$1"; }
safe_task_name() { printf '%s\n' "${TASK_NAME//\//_}"; }

checkpoint_base_learning_rate() {
  local ckpt_path="$1"
  local ckpt_q py py_q cmd
  if [[ -z "$ckpt_path" || "$DRY_RUN" == "1" || "$RESUME_LR_POLICY" != "checkpoint" ]]; then
    return 1
  fi
  ckpt_q="$(remote_quote "$ckpt_path")"
  py="$(cat <<'PY'
import sys
import torch

ckpt = sys.argv[1]
data = torch.load(ckpt, map_location="cpu", weights_only=False)
values = []
for scheduler in data.get("lr_schedulers") or []:
    values.extend(scheduler.get("base_lrs") or [])
for optimizer in data.get("optimizer_states") or []:
    for group in optimizer.get("param_groups") or []:
        value = group.get("initial_lr", group.get("lr"))
        if value is not None:
            values.append(value)
if not values:
    raise SystemExit(1)
print(f"{float(values[0]):.12f}")
PY
)"
  py_q="$(remote_quote "$py")"
  cmd="python -c ${py_q} ${ckpt_q}"
  kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$cmd" 2>/dev/null | tr -d '\r'
}

learning_rate_for_batch() {
  local bs="$1"
  if [[ -n "$LEARNING_RATE" ]]; then
    printf '%s\n' "$LEARNING_RATE"
    return 0
  fi
  python3 - "$bs" "${#POD_ARRAY[@]}" "$GPUS_PER_NODE" "$BASE_LEARNING_RATE" "$BASE_GLOBAL_BATCH_SIZE" <<'PY'
import math
import sys

batch_size = int(sys.argv[1])
num_nodes = int(sys.argv[2])
gpus_per_node = int(sys.argv[3])
base_lr = float(sys.argv[4])
base_global_batch = float(sys.argv[5])
global_batch = batch_size * num_nodes * gpus_per_node
print(f"{base_lr * math.sqrt(global_batch / base_global_batch):.12f}")
PY
}

learning_rate_for_attempt() {
  local bs="$1"
  local ckpt_path="$2"
  local ckpt_lr
  if [[ -z "$LEARNING_RATE" ]]; then
    ckpt_lr="$(checkpoint_base_learning_rate "$ckpt_path" || true)"
    if [[ -n "$ckpt_lr" ]]; then
      printf '%s\n' "$ckpt_lr"
      return 0
    fi
  fi
  learning_rate_for_batch "$bs"
}

remote_run_root() {
  printf '%s/tmux_unimm_h100x3x2/%s' "${REMOTE_LOG_DIR%/}" "$(safe_task_name)"
}

remote_tmux_log_for_pod() {
  local pod="$1"
  printf '%s/%s.tmux.log' "$(remote_run_root)" "$pod"
}

remote_status_file_for_pod() {
  local pod="$1"
  printf '%s/%s.torchrun_status' "$(remote_run_root)" "$pod"
}

remote_pgid_file_for_pod() {
  local pod="$1"
  printf '%s/%s.torchrun_pgid' "$(remote_run_root)" "$pod"
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
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  runs_dir="${REMOTE_LOG_DIR%/}/${TASK_NAME}/runs"
  runs_dir_q="$(remote_quote "$runs_dir")"
  cmd="{ ls -t ${runs_dir_q}/*/checkpoints/epoch_last.ckpt 2>/dev/null; ls -t ${runs_dir_q}/*/checkpoints/last.ckpt 2>/dev/null; } | head -1"
  kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "$cmd" 2>/dev/null | tr -d '\r'
}

start_attempt() {
  local bs="$1"
  local ckpt_path="$2"
  local attempt_lr
  attempt_lr="$(learning_rate_for_attempt "$bs" "$ckpt_path")"
  local -a cmd=(
    python3 scripts/launch_unimm_h100x3x2.py
    --namespace "$NAMESPACE"
    --pods "${POD_ARRAY[@]}"
    --container "$CONTAINER"
    --project-root "$PROJECT_ROOT"
    --repo-url "$REPO_URL"
    --branch "$BRANCH"
    --cache-root "$CACHE_ROOT"
    --log-dir "$REMOTE_LOG_DIR"
    --task-name "$TASK_NAME"
    --session "$SESSION"
    --master-port "$MASTER_PORT"
    --train-batch-size "$bs"
    --val-batch-size "$VAL_BATCH_SIZE"
    --test-batch-size "$TEST_BATCH_SIZE"
    --learning-rate "$attempt_lr"
    --wandb-mode "$WANDB_MODE"
    --replace
  )

  if [[ -n "$ANCHOR_FILE" ]]; then
    cmd+=(--anchor-file "$ANCHOR_FILE")
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
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    cmd+=(--extra-hydra-overrides "$EXTRA_HYDRA_OVERRIDES")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  log "launcher command:"
  log "attempt learning_rate=${attempt_lr}"
  if [[ -n "$ckpt_path" && -z "$LEARNING_RATE" && "$RESUME_LR_POLICY" == "checkpoint" ]]; then
    log "resume LR policy=checkpoint; scheduler/optimizer progress is restored from ckpt_path."
  fi
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

stop_attempt_sessions() {
  local pod pod_q run_root run_root_q session_q task_q grace
  run_root="$(remote_run_root)"
  run_root_q="$(remote_quote "$run_root")"
  session_q="$(remote_quote "$SESSION")"
  task_q="$(remote_quote "$TASK_NAME")"
  grace="${REMOTE_KILL_GRACE_SEC:-20}"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: skip stopping existing remote tmux sessions."
    return 0
  fi

  for pod in "${POD_ARRAY[@]}"; do
    pod_q="$(remote_quote "$pod")"
    kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set +e
TASK_NAME_TO_STOP=${task_q}
task_process_pids() {
  ps -eo pid=,cmd= | awk -v task=\"task_name=\${TASK_NAME_TO_STOP}\" '
    \$0 ~ task && (\$0 ~ /(^|[ /])python([0-9.]*)?([[:space:]]|$)/ || \$0 ~ /(^|[ /])torchrun([[:space:]]|$)/) { print \$1 }
  ' |
    while read -r pid; do
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
  pgid=\"\${pgid//[[:space:]]/}\"
  current_pgid=\"\$(ps -o pgid= -p \"\$\$\" 2>/dev/null | tr -d '[:space:]')\"
  if [[ \"\$pgid\" =~ ^[0-9]+$ && \"\$pgid\" != \"0\" && \"\$pgid\" != \"\$current_pgid\" ]] &&
     ps -eo pgid=,cmd= | awk -v pgid=\"\$pgid\" -v task=\"\$TASK_NAME_TO_STOP\" \
       '\$1 == pgid && index(\$0, task) > 0 { found = 1 } END { exit found ? 0 : 1 }'; then
    kill -TERM -- \"-\$pgid\" 2>/dev/null || true
    sleep ${grace}
    kill -KILL -- \"-\$pgid\" 2>/dev/null || true
  fi
  rm -f \"\$pgid_file\" 2>/dev/null || true
fi
terminate_task_processes
tmux kill-session -t ${session_q} 2>/dev/null || true
" >/dev/null 2>&1 || true
  done
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
same_bs_oom_retry_count=0

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
    log "launcher failed before training completed."
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
    if (( OOM_STEP == 0 )); then
      same_bs_oom_retry_count=$(( same_bs_oom_retry_count + 1 ))
      if (( same_bs_oom_retry_count > MAX_SAME_BS_OOM_RETRIES )); then
        log "OOM detected at bs=${bs} for ${same_bs_oom_retry_count} same-batch attempts; MAX_SAME_BS_OOM_RETRIES=${MAX_SAME_BS_OOM_RETRIES}. Aborting."
        exit 1
      fi
      log "OOM detected at bs=${bs} (exit=${ATTEMPT_EXIT_CODE}). Keeping bs=${bs} and resuming from the latest checkpoint."
      continue
    fi
    same_bs_oom_retry_count=0
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
    continue
  fi

  log "Non-OOM failure (exit=${ATTEMPT_EXIT_CODE}). See ${attempt_log}. Aborting retry loop."
  exit "$ATTEMPT_EXIT_CODE"
done

log "Reached MIN_BS=${MIN_BS} without a successful run. Aborting."
exit 1
