#!/usr/bin/env bash
# Start UniMM H100 x3x2 pretrain only when both target pods look idle.
#
# This guard is intentionally conservative: if any target pod has the UniMM
# tmux session already open, visible GPU compute processes, high GPU memory, or
# high GPU utilization, it refuses to launch unless FORCE=1 or --force is used.

set -Eeuo pipefail

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-hsb-npc-training-3-1 hsb-npc-training-3-2}"
SESSION="${SESSION:-unimm-h100x3x2}"
TASK_NAME="${TASK_NAME:-unimm_anchor_based_4s_h100x3x2_pretrain_globalbs168_guarded_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29578}"
INITIAL_BS="${INITIAL_BS:-28}"
OOM_STEP="${OOM_STEP:-2}"
MIN_BS="${MIN_BS:-16}"
WANDB_MODE="${WANDB_MODE:-online}"
IDLE_MAX_MEMORY_MIB="${IDLE_MAX_MEMORY_MIB:-1024}"
IDLE_MAX_UTIL_PCT="${IDLE_MAX_UTIL_PCT:-10}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY=0

DEFAULT_EXTRA_OVERRIDES="model.model_config.inference_temperature=1.0 model.model_config.inference_top_k=0 model.model_config.inference_top_p=1.0 model.model_config.scorer_scene_num=1680"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-$DEFAULT_EXTRA_OVERRIDES}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/start_unimm_h100x3x2_pretrain_if_idle.sh [--status] [--dry-run] [--force]

Environment overrides:
  TASK_NAME, SESSION, MASTER_PORT, INITIAL_BS, OOM_STEP, MIN_BS, WANDB_MODE
  IDLE_MAX_MEMORY_MIB, IDLE_MAX_UTIL_PCT, EXTRA_HYDRA_OVERRIDES

Default experiment:
  UniMM Anchor-Based-4s, 2 nodes x 3 H100, per-GPU train batch 28,
  OOM retry step 2, inference_temperature=1.0, top_k=0, scorer_scene_num=1680.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)
      STATUS_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 2
  fi
}

remote_quote() { printf '%q' "$1"; }

tmux_session_exists() {
  local pod="$1"
  local session_q
  session_q="$(remote_quote "$SESSION")"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- \
    bash -lc "tmux has-session -t ${session_q} 2>/dev/null" >/dev/null 2>&1
}

gpu_status_for_pod() {
  local pod="$1"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc '
set -Eeuo pipefail
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo "--- compute-apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
'
}

pod_is_idle() {
  local pod="$1"
  local output gpu_lines apps_lines busy=0

  if tmux_session_exists "$pod"; then
    log "BUSY ${pod}: tmux session '${SESSION}' already exists."
    return 1
  fi

  output="$(gpu_status_for_pod "$pod")"
  printf '%s\n' "===== ${pod} GPU status ====="
  printf '%s\n' "$output"

  gpu_lines="$(printf '%s\n' "$output" | awk '/--- compute-apps ---/{exit} {print}')"
  apps_lines="$(printf '%s\n' "$output" | awk 'seen {print} /--- compute-apps ---/{seen=1}')"

  if printf '%s\n' "$apps_lines" | awk '
    NF &&
    $0 !~ /^[[:space:]]*$/ &&
    $0 !~ /No running processes found/ &&
    $0 !~ /^N\/A/ {found=1}
    END {exit found ? 0 : 1}
  '; then
    log "BUSY ${pod}: visible GPU compute process exists."
    busy=1
  fi

  while IFS=, read -r gpu_idx util_pct mem_used mem_total; do
    gpu_idx="${gpu_idx//[[:space:]]/}"
    util_pct="${util_pct//[[:space:]]/}"
    mem_used="${mem_used//[[:space:]]/}"
    mem_total="${mem_total//[[:space:]]/}"
    [[ -z "$gpu_idx" ]] && continue
    if [[ "$util_pct" =~ ^[0-9]+$ ]] && (( util_pct > IDLE_MAX_UTIL_PCT )); then
      log "BUSY ${pod}: GPU ${gpu_idx} util ${util_pct}% > ${IDLE_MAX_UTIL_PCT}%."
      busy=1
    fi
    if [[ "$mem_used" =~ ^[0-9]+$ ]] && (( mem_used > IDLE_MAX_MEMORY_MIB )); then
      log "BUSY ${pod}: GPU ${gpu_idx} memory ${mem_used}MiB > ${IDLE_MAX_MEMORY_MIB}MiB."
      busy=1
    fi
  done <<< "$gpu_lines"

  (( busy == 0 ))
}

require_command kubectl

read -r -a POD_ARRAY <<< "$PODS"
if (( ${#POD_ARRAY[@]} != 2 )); then
  echo "ERROR: PODS must contain exactly two pod names; got: ${PODS}" >&2
  exit 2
fi

all_idle=1
for pod in "${POD_ARRAY[@]}"; do
  if ! pod_is_idle "$pod"; then
    all_idle=0
  fi
done

if (( STATUS_ONLY == 1 )); then
  if (( all_idle == 1 )); then
    log "All target pods look idle."
    exit 0
  fi
  log "At least one target pod looks busy."
  exit 1
fi

if (( all_idle != 1 && FORCE != 1 )); then
  log "Refusing to launch UniMM because target pods are not idle."
  log "Use --status to inspect, or --force only after confirming with the current GPU user."
  exit 1
fi

export NAMESPACE CONTAINER PODS SESSION TASK_NAME MASTER_PORT
export INITIAL_BS OOM_STEP MIN_BS WANDB_MODE EXTRA_HYDRA_OVERRIDES DRY_RUN

log "Launching guarded UniMM H100 x3x2 pretrain."
log "  task_name=${TASK_NAME}"
log "  session=${SESSION}"
log "  initial_bs=${INITIAL_BS}, oom_step=${OOM_STEP}, min_bs=${MIN_BS}"
log "  extra_overrides=${EXTRA_HYDRA_OVERRIDES}"

exec bash scripts/launch_unimm_h100x3x2_with_oom_retry.sh
