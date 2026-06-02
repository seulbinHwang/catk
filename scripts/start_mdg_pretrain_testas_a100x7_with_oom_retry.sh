#!/usr/bin/env bash
# Launch MDG pretrain on the existing testas A100x7 pod with automatic
# train_batch_size fallback on CUDA OOM.
#
# Run this script on ssh user@10.60.188.83, where kubectl can access testas.
# It never creates, deletes, or restarts pods. It only replaces the configured
# tmux session inside the already-running testas pod.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NAMESPACE="${NAMESPACE:-p-pnc}"
POD="${POD:-testas}"
CONTAINER="${CONTAINER:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk_mdg_pretrain}"
BRANCH="${BRANCH:-MDG}"
TASK_NAME="${TASK_NAME:-mdg_wosac_pretrain_testas_a100x7_oom_retry_bs${INITIAL_BS:-32}}"
SESSION="${SESSION:-mdg-pretrain-a100x7-oom-retry}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/MDG_cache}"
REMOTE_ATTEMPT_LOG_DIR="${REMOTE_ATTEMPT_LOG_DIR:-${PROJECT_ROOT}/logs/testas_mdg_pretrain_a100x7_oom_retry}"
LOCAL_RETRY_LOG_DIR="${LOCAL_RETRY_LOG_DIR:-${REPO_ROOT}/logs/_mdg_testas_a100x7_oom_retry/${TASK_NAME}}"

INITIAL_BS="${INITIAL_BS:-34}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-24}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_NON_OOM_RETRIES="${MAX_NON_OOM_RETRIES:-2}"
RETRY_NON_OOM_EXIT_CODES="${RETRY_NON_OOM_EXIT_CODES:-134,143}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-64}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
WANDB_MODE="${WANDB_MODE:-online}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1680}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
TRAIN_MEMORY_BALANCED_BATCHING="${TRAIN_MEMORY_BALANCED_BATCHING:-true}"
TRAIN_MEMORY_BALANCE_METADATA_CACHE="${TRAIN_MEMORY_BALANCE_METADATA_CACHE:-}"
TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS="${TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS:-32}"
TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING="${TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING:-true}"
MASTER_PORT="${MASTER_PORT:-29671}"
CATK_RESUME_CHECKPOINT_NAME="${CATK_RESUME_CHECKPOINT_NAME:-epoch_last.ckpt}"
CATK_HYDRA_OVERRIDES="${CATK_HYDRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
remote_quote() { printf '%q' "$1"; }

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "ERROR: ${name} must be a positive integer; got ${value}." >&2
    exit 2
  fi
}

validate_positive_int INITIAL_BS "$INITIAL_BS"
validate_positive_int OOM_STEP "$OOM_STEP"
validate_positive_int MIN_BS "$MIN_BS"
if (( INITIAL_BS < MIN_BS )); then
  echo "ERROR: INITIAL_BS=${INITIAL_BS} is below MIN_BS=${MIN_BS}." >&2
  exit 2
fi

mkdir -p "$LOCAL_RETRY_LOG_DIR"

kubectl_exec() {
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc "$1"
}

remote_runs_dir() {
  printf '%s/logs/%s/runs' "${PROJECT_ROOT%/}" "$TASK_NAME"
}

find_latest_epoch_last_ckpt() {
  local runs_dir runs_dir_q checkpoint_name_q cmd
  runs_dir="$(remote_runs_dir)"
  runs_dir_q="$(remote_quote "$runs_dir")"
  checkpoint_name_q="$(remote_quote "$CATK_RESUME_CHECKPOINT_NAME")"
  cmd="{ ls -t ${runs_dir_q}/*/checkpoints/${checkpoint_name_q} 2>/dev/null; ls -t ${runs_dir_q}/*/checkpoints/last.ckpt 2>/dev/null; } | head -1"
  kubectl_exec "$cmd" 2>/dev/null | tr -d '\r'
}

stop_attempt_session() {
  local session_q
  session_q="$(remote_quote "$SESSION")"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run stop: tmux kill-session -t ${SESSION}"
    return 0
  fi
  kubectl_exec "tmux kill-session -t ${session_q} 2>/dev/null || true" >/dev/null 2>&1 || true
}

attempt_log_dir() {
  local attempt="$1"
  local bs="$2"
  printf '%s/attempt_%03d_bs%s' "${REMOTE_ATTEMPT_LOG_DIR%/}" "$attempt" "$bs"
}

find_remote_attempt_log() {
  local log_dir="$1"
  local log_dir_q task_name_q cmd
  log_dir_q="$(remote_quote "$log_dir")"
  task_name_q="$(remote_quote "$TASK_NAME")"
  cmd="ls -t ${log_dir_q}/${task_name_q}_*.log 2>/dev/null | head -1"
  kubectl_exec "$cmd" 2>/dev/null | tr -d '\r'
}

copy_remote_log() {
  local remote_log="$1"
  local local_log="$2"
  local remote_log_q
  remote_log_q="$(remote_quote "$remote_log")"
  if ! kubectl_exec "cat ${remote_log_q}" > "$local_log" 2>/dev/null; then
    printf 'warning: failed to copy remote log %s\n' "$remote_log" > "$local_log"
  fi
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

start_attempt() {
  local attempt="$1"
  local bs="$2"
  local ckpt_path="$3"
  local auto_resume="false"
  local resume_require_checkpoint="true"
  local log_dir
  log_dir="$(attempt_log_dir "$attempt" "$bs")"
  if [[ -n "$ckpt_path" ]]; then
    auto_resume="true"
  else
    resume_require_checkpoint="false"
  fi

  local -a env_args=(
    "NAMESPACE=$NAMESPACE"
    "POD=$POD"
    "CONTAINER=$CONTAINER"
    "PROJECT_ROOT=$PROJECT_ROOT"
    "BRANCH=$BRANCH"
    "SESSION=$SESSION"
    "LOG_DIR=$log_dir"
    "REPLACE_SESSION=1"
    "CACHE_ROOT=$CACHE_ROOT"
    "TRAIN_BATCH_SIZE=$bs"
    "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
    "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
    "MAX_EPOCHS=$MAX_EPOCHS"
    "LIMIT_TRAIN_BATCHES=$LIMIT_TRAIN_BATCHES"
    "LIMIT_VAL_BATCHES=$LIMIT_VAL_BATCHES"
    "DATA_NUM_WORKERS=$DATA_NUM_WORKERS"
    "PRECISION=$PRECISION"
    "WANDB_MODE=$WANDB_MODE"
    "VAL_CLOSED_LOOP=$VAL_CLOSED_LOOP"
    "N_BATCH_SIM_AGENTS_METRIC=$N_BATCH_SIM_AGENTS_METRIC"
    "SCORER_SCENE_NUM=$SCORER_SCENE_NUM"
    "CHECKPOINT_MONITOR=$CHECKPOINT_MONITOR"
    "CHECKPOINT_MODE=$CHECKPOINT_MODE"
    "TRAIN_MEMORY_BALANCED_BATCHING=$TRAIN_MEMORY_BALANCED_BATCHING"
    "TRAIN_MEMORY_BALANCE_METADATA_CACHE=$TRAIN_MEMORY_BALANCE_METADATA_CACHE"
    "TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS=$TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS"
    "TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING=$TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING"
    "MASTER_PORT=$MASTER_PORT"
    "TASK_NAME=$TASK_NAME"
    "CATK_AUTO_RESUME=$auto_resume"
    "CATK_RESUME_TASK_NAME=$TASK_NAME"
    "CATK_RESUME_CHECKPOINT_NAME=$CATK_RESUME_CHECKPOINT_NAME"
    "CATK_RESUME_REQUIRE_CHECKPOINT=$resume_require_checkpoint"
    "CATK_HYDRA_OVERRIDES=$CATK_HYDRA_OVERRIDES"
  )

  log "starting attempt #${attempt}: bs=${bs}, auto_resume=${auto_resume}, log_dir=${log_dir}"
  if [[ -n "$ckpt_path" ]]; then
    log "  latest checkpoint: ${ckpt_path}"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  env'
    printf ' %q' "${env_args[@]}"
    printf ' bash scripts/start_mdg_pretrain_testas_a100x7.sh\n'
    return 0
  fi
  env "${env_args[@]}" bash scripts/start_mdg_pretrain_testas_a100x7.sh
}

wait_for_attempt_exit() {
  local attempt="$1"
  local bs="$2"
  local log_dir remote_log status_line oom_regex_q remote_log_q cmd
  log_dir="$(attempt_log_dir "$attempt" "$bs")"
  oom_regex_q="$(remote_quote "$OOM_REGEX")"

  remote_log=""
  for _ in {1..60}; do
    remote_log="$(find_remote_attempt_log "$log_dir")"
    if [[ -n "$remote_log" ]]; then
      break
    fi
    sleep 2
  done
  if [[ -z "$remote_log" ]]; then
    log "failed to find remote attempt log under ${log_dir}"
    ATTEMPT_EXIT_CODE="1"
    ATTEMPT_EXIT_REASON="missing_log"
    return 0
  fi

  log "  remote log: ${remote_log}"
  remote_log_q="$(remote_quote "$remote_log")"
  while true; do
    cmd="grep -Eq ${oom_regex_q} ${remote_log_q} 2>/dev/null"
    if kubectl_exec "$cmd" >/dev/null 2>&1; then
      log "OOM marker observed in ${remote_log}; stopping ${SESSION} before retry."
      ATTEMPT_EXIT_CODE="1"
      ATTEMPT_EXIT_REASON="oom"
      stop_attempt_session
      return 0
    fi

    status_line="$(kubectl_exec "grep -E '\\[pretrain-exit\\].*status=[0-9]+' ${remote_log_q} 2>/dev/null | tail -1" 2>/dev/null || true)"
    status_line="${status_line//$'\r'/}"
    if [[ "$status_line" =~ status=([0-9]+) ]]; then
      ATTEMPT_EXIT_CODE="${BASH_REMATCH[1]}"
      ATTEMPT_EXIT_REASON="exit"
      return 0
    fi

    log "waiting; attach: kubectl exec -it -n ${NAMESPACE} ${POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
    sleep "$POLL_INTERVAL"
  done
}

bs="$INITIAL_BS"
attempt=0
non_oom_retry_count=0

while (( bs >= MIN_BS )); do
  attempt=$(( attempt + 1 ))
  latest_ckpt="$(find_latest_epoch_last_ckpt)"
  local_attempt_log="${LOCAL_RETRY_LOG_DIR}/attempt_$(printf '%03d' "$attempt")_bs${bs}.log"

  stop_attempt_session
  if ! start_attempt "$attempt" "$bs" "$latest_ckpt"; then
    log "launcher failed before training started."
    exit 1
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run completed after rendering first attempt."
    exit 0
  fi

  ATTEMPT_EXIT_CODE=""
  ATTEMPT_EXIT_REASON=""
  wait_for_attempt_exit "$attempt" "$bs"
  remote_log="$(find_remote_attempt_log "$(attempt_log_dir "$attempt" "$bs")")"
  if [[ -n "$remote_log" ]]; then
    copy_remote_log "$remote_log" "$local_attempt_log"
    log "  copied attempt log: ${local_attempt_log}"
  fi

  if [[ "$ATTEMPT_EXIT_CODE" == "0" ]]; then
    log "Training completed successfully at bs=${bs} on attempt #${attempt}."
    exit 0
  fi

  if [[ "$ATTEMPT_EXIT_REASON" == "oom" ]] || { [[ -f "$local_attempt_log" ]] && grep -Eq "$OOM_REGEX" "$local_attempt_log"; }; then
    non_oom_retry_count=0
    new_bs=$(( bs - OOM_STEP ))
    log "OOM detected at bs=${bs}; lowering train_batch_size to ${new_bs}."
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

  log "Non-OOM failure exit=${ATTEMPT_EXIT_CODE}. See ${local_attempt_log}. Aborting."
  exit "${ATTEMPT_EXIT_CODE:-1}"
done

log "Reached MIN_BS=${MIN_BS} without success."
exit 1
