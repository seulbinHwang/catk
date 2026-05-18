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
#   * on CUDA OOM: reduce batch by OOM_STEP and resume from the latest epoch_last.ckpt
#     If OOM_STEP=0, keep the same batch size and only resume.
#     In same-batch mode, abort after MAX_SAME_BS_OOM_RETRIES repeated OOM retries.
#   * on retryable external exits such as SIGTERM/SIGABRT: keep the batch size
#     and resume from the latest epoch_last.ckpt
#   * stop at MIN_BS=20 unless overridden

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-hsb-npc-training hsb-npc-training2}"
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
MAX_SAME_BS_OOM_RETRIES="${MAX_SAME_BS_OOM_RETRIES:-3}"
MIN_BS="${MIN_BS:-20}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
RETRY_NON_OOM_EXIT_CODES="${RETRY_NON_OOM_EXIT_CODES:-134,143}"
MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-3}"
LEARNING_RATE="${LEARNING_RATE:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
MEMORY_BALANCE_PREFLIGHT="${MEMORY_BALANCE_PREFLIGHT:-0}"
MEMORY_BALANCE_METADATA_CACHE="${MEMORY_BALANCE_METADATA_CACHE:-}"
MEMORY_BALANCE_METADATA_NUM_WORKERS="${MEMORY_BALANCE_METADATA_NUM_WORKERS:-8}"
MEMORY_BALANCE_METADATA_FORCE_REBUILD="${MEMORY_BALANCE_METADATA_FORCE_REBUILD:-0}"
DEFAULT_CACHE_ROOT="${DEFAULT_CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"

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

cache_root_for_pod() {
  local pod="$1"
  local mapping mapping_pod mapping_path
  local -a pod_cache_root_array
  if [[ -n "$POD_CACHE_ROOTS" ]]; then
    read -r -a pod_cache_root_array <<< "$POD_CACHE_ROOTS"
    for mapping in "${pod_cache_root_array[@]}"; do
      if [[ "$mapping" == *=* ]]; then
        mapping_pod="${mapping%%=*}"
        mapping_path="${mapping#*=}"
        if [[ "$mapping_pod" == "$pod" && -n "$mapping_path" ]]; then
          printf '%s\n' "$mapping_path"
          return 0
        fi
      fi
    done
  fi
  if [[ -n "$CACHE_ROOT" ]]; then
    printf '%s\n' "$CACHE_ROOT"
    return 0
  fi
  case "$pod" in
    hsb-npc-training)
      printf '%s\n' "/mnt/nuplan/womd_v1_3/SMART_cache"
      ;;
    *)
      printf '%s\n' "$DEFAULT_CACHE_ROOT"
      ;;
  esac
}

sync_project_for_pod() {
  local pod="$1"
  local git_cmd project_root_q branch_q git_ref_q
  project_root_q="$(remote_quote "$PROJECT_ROOT")"
  branch_q="$(remote_quote "$BRANCH")"
  git_ref_q="$(remote_quote "$GIT_REF")"

  if [[ -n "$GIT_REF" ]]; then
    git_cmd="git fetch origin ${branch_q} && git checkout ${git_ref_q}"
  else
    git_cmd="git fetch origin ${branch_q} && { git checkout ${branch_q} 2>/dev/null || git checkout -B ${branch_q} origin/${branch_q}; } && git pull --ff-only origin ${branch_q}"
  fi

  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set -euo pipefail
cd ${project_root_q}
${git_cmd}
"
}

prebuild_memory_balance_metadata_for_pod() {
  local pod="$1"
  local cache_root metadata_cache raw_dir force_arg metadata_cache_q raw_dir_q project_root_q workers_q
  cache_root="$(cache_root_for_pod "$pod")"
  metadata_cache="${MEMORY_BALANCE_METADATA_CACHE:-${REMOTE_LOG_DIR%/}/dataset_metadata/womd_training_memory_balance_v1.pt}"
  raw_dir="${cache_root%/}/training"
  force_arg=""
  if [[ "$MEMORY_BALANCE_METADATA_FORCE_REBUILD" == "1" ]]; then
    force_arg="--force"
  fi

  metadata_cache_q="$(remote_quote "$metadata_cache")"
  raw_dir_q="$(remote_quote "$raw_dir")"
  project_root_q="$(remote_quote "$PROJECT_ROOT")"
  workers_q="$(remote_quote "$MEMORY_BALANCE_METADATA_NUM_WORKERS")"

  log "prebuilding memory-balance metadata on ${pod}: raw_dir=${raw_dir} cache=${metadata_cache}"
  sync_project_for_pod "$pod"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set -euo pipefail
cd ${project_root_q}
mkdir -p \"\$(dirname ${metadata_cache_q})\"
test -d ${raw_dir_q}
python tools/build_memory_balance_metadata.py \
  --raw-dir ${raw_dir_q} \
  --cache-path ${metadata_cache_q} \
  --num-workers ${workers_q} \
  ${force_arg}
"
}

copy_memory_balance_metadata_from_master() {
  local pod="$1"
  local metadata_cache tmp_cache metadata_cache_q tmp_cache_q
  if [[ "$pod" == "$MASTER_POD" ]]; then
    return 0
  fi

  metadata_cache="${MEMORY_BALANCE_METADATA_CACHE:-${REMOTE_LOG_DIR%/}/dataset_metadata/womd_training_memory_balance_v1.pt}"
  tmp_cache="${metadata_cache}.tmp.${MASTER_POD}.$$"
  metadata_cache_q="$(remote_quote "$metadata_cache")"
  tmp_cache_q="$(remote_quote "$tmp_cache")"

  log "copying memory-balance metadata from ${MASTER_POD} to ${pod}: cache=${metadata_cache}"
  kubectl exec -n "$NAMESPACE" "$MASTER_POD" -c "$CONTAINER" -- bash -lc "
set -euo pipefail
test -s ${metadata_cache_q}
cat ${metadata_cache_q}
" | kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set -euo pipefail
mkdir -p \"\$(dirname ${metadata_cache_q})\"
cat > ${tmp_cache_q}
mv ${tmp_cache_q} ${metadata_cache_q}
"
}

validate_memory_balance_metadata_on_pod() {
  local pod="$1"
  local cache_root metadata_cache raw_dir metadata_cache_q raw_dir_q project_root_q workers_q
  cache_root="$(cache_root_for_pod "$pod")"
  metadata_cache="${MEMORY_BALANCE_METADATA_CACHE:-${REMOTE_LOG_DIR%/}/dataset_metadata/womd_training_memory_balance_v1.pt}"
  raw_dir="${cache_root%/}/training"
  metadata_cache_q="$(remote_quote "$metadata_cache")"
  raw_dir_q="$(remote_quote "$raw_dir")"
  project_root_q="$(remote_quote "$PROJECT_ROOT")"
  workers_q="$(remote_quote "$MEMORY_BALANCE_METADATA_NUM_WORKERS")"

  log "validating memory-balance metadata on ${pod}: raw_dir=${raw_dir} cache=${metadata_cache}"
  sync_project_for_pod "$pod"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
set -euo pipefail
cd ${project_root_q}
test -d ${raw_dir_q}
python tools/build_memory_balance_metadata.py \
  --raw-dir ${raw_dir_q} \
  --cache-path ${metadata_cache_q} \
  --num-workers ${workers_q}
"
}

prebuild_memory_balance_metadata() {
  if [[ "$MEMORY_BALANCE_PREFLIGHT" != "1" ]]; then
    return 0
  fi

  local pod status failed=0 master_cache_root pod_cache_root
  master_cache_root="$(cache_root_for_pod "$MASTER_POD")"

  prebuild_memory_balance_metadata_for_pod "$MASTER_POD"
  status=$?
  if (( status != 0 )); then
    log "memory-balance metadata preflight failed on ${MASTER_POD} (exit=${status})"
    return "$status"
  fi
  log "memory-balance metadata preflight ready on ${MASTER_POD}"

  for pod in "${POD_ARRAY[@]}"; do
    if [[ "$pod" == "$MASTER_POD" ]]; then
      continue
    fi
    pod_cache_root="$(cache_root_for_pod "$pod")"
    if [[ "$pod_cache_root" == "$master_cache_root" ]]; then
      if copy_memory_balance_metadata_from_master "$pod" && validate_memory_balance_metadata_on_pod "$pod"; then
        log "memory-balance metadata preflight ready on ${pod}"
      else
        status=$?
        log "memory-balance metadata preflight failed on ${pod} (exit=${status})"
        failed=1
      fi
    elif prebuild_memory_balance_metadata_for_pod "$pod"; then
      log "memory-balance metadata preflight ready on ${pod}"
    else
      status=$?
      log "memory-balance metadata preflight failed on ${pod} (exit=${status})"
      failed=1
    fi
  done

  if (( failed != 0 )); then
    return 1
  fi
  log "memory-balance metadata preflight ready on all ${#POD_ARRAY[@]} pods"
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
non_oom_retry_count=0
same_bs_oom_retry_count=0
if ! prebuild_memory_balance_metadata; then
  log "memory-balance metadata preflight failed. Aborting before distributed training."
  exit 1
fi
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
