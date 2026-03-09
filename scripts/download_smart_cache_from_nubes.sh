#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------- CPU / 동시성 설정 ----------------
CPUSET="${CPUSET:-}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ------------------------------------------------------------------------

REMOTE_DIR="${1:-${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache}}"
LOCAL_DIR="${2:-${LOCAL_DIR:-/workspace/womd_v1_3/SMART_cache}}"

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

if [[ -z "$REMOTE_DIR" || -z "$LOCAL_DIR" ]]; then
  echo "Usage: bash scripts/download_smart_cache_from_nubes.sh <remote_dir> <local_dir>"
  echo "or set REMOTE_DIR and LOCAL_DIR env vars."
  exit 1
fi

mkdir -p "$LOCAL_DIR"

_count_cpus_in_set() {
  local cpu_set="$1"
  local total=0
  local part
  local start
  local end

  if [[ -z "$cpu_set" ]]; then
    echo "0"
    return
  fi

  IFS=',' read -ra parts <<< "$cpu_set"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      start="${part%-*}"
      end="${part#*-}"
      total=$(( total + end - start + 1 ))
    else
      total=$(( total + 1 ))
    fi
  done

  echo "$total"
}

_detect_cpuset() {
  if [[ -n "$CPUSET" ]]; then
    echo "$CPUSET"
    return
  fi

  if command -v taskset >/dev/null 2>&1; then
    taskset -pc $$ 2>/dev/null | awk -F': ' 'NR==1 {print $2}'
    return
  fi

  echo ""
}

_run_pinned() {
  if [[ -n "$ACTIVE_CPUSET" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$ACTIVE_CPUSET" "$@"
  else
    "$@"
  fi
}

ACTIVE_CPUSET="$(_detect_cpuset)"
AVAILABLE_CPUS="$(_count_cpus_in_set "$ACTIVE_CPUSET")"
if [[ "$AVAILABLE_CPUS" -le 0 ]]; then
  AVAILABLE_CPUS="$(nproc)"
fi

export DP_MAX_CPUS="${AVAILABLE_CPUS}"

if [[ -z "${NUBES_JOBS:-}" ]]; then
  if [[ "$AVAILABLE_CPUS" -le 4 ]]; then
    NUBES_JOBS="$AVAILABLE_CPUS"
  elif [[ "$AVAILABLE_CPUS" -le 8 ]]; then
    NUBES_JOBS=$(( AVAILABLE_CPUS - 1 ))
  elif [[ "$AVAILABLE_CPUS" -le 16 ]]; then
    NUBES_JOBS=$(( AVAILABLE_CPUS - 2 ))
  else
    NUBES_JOBS=$(( (AVAILABLE_CPUS * 3) / 4 ))
  fi
fi

if [[ "$NUBES_JOBS" -lt 1 ]]; then
  NUBES_JOBS=1
fi

echo "[SMART_CACHE_DOWNLOAD] CPUSET=${ACTIVE_CPUSET:-auto}, DP_MAX_CPUS=${DP_MAX_CPUS}, NUBES_JOBS=${NUBES_JOBS}, PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC}"

_format_hours_minutes() {
  local seconds="$1"
  local total_minutes
  local hours
  local minutes

  if ! awk "BEGIN { exit !($seconds >= 0) }"; then
    echo "unknown"
    return
  fi

  total_minutes=$(( seconds / 60 ))
  hours=$(( total_minutes / 60 ))
  minutes=$(( total_minutes % 60 ))
  printf "%dh %dm" "$hours" "$minutes"
}

_count_existing_files() {
  local manifest_path="$1"
  local count=0
  local remote_path
  local rel_path

  while IFS= read -r remote_path; do
    [[ -z "$remote_path" ]] && continue
    rel_path="${remote_path#${REMOTE_DIR}/}"
    if [[ "$rel_path" == "$remote_path" ]]; then
      rel_path="${remote_path##*/}"
    fi
    if [[ -f "$LOCAL_DIR/$rel_path" ]]; then
      count=$(( count + 1 ))
    fi
  done < "$manifest_path"

  echo "$count"
}

_print_progress() {
  local total_expected="$1"
  local current_count="$2"
  local start_count="$3"
  local start_epoch="$4"
  local percent=0
  local elapsed_seconds
  local processed_this_run
  local rate
  local remaining
  local eta_seconds

  if [[ "$total_expected" -gt 0 ]]; then
    percent=$(awk "BEGIN { printf \"%.2f\", 100 * $current_count / $total_expected }")
  fi

  elapsed_seconds=$(( $(date +%s) - start_epoch ))
  processed_this_run=$(( current_count - start_count ))
  if [[ "$processed_this_run" -lt 0 ]]; then
    processed_this_run=0
  fi

  if [[ "$elapsed_seconds" -gt 0 && "$processed_this_run" -gt 0 ]]; then
    rate=$(awk "BEGIN { printf \"%.6f\", $processed_this_run / $elapsed_seconds }")
    remaining=$(( total_expected - current_count ))
    if [[ "$remaining" -lt 0 ]]; then
      remaining=0
    fi
    eta_seconds=$(awk "BEGIN { printf \"%d\", $remaining / $rate }")
  else
    eta_seconds=-1
  fi

  echo "[download-progress] total_expected=${total_expected} existing=${current_count} percent=${percent}% elapsed=$(_format_hours_minutes "$elapsed_seconds") eta=$(_format_hours_minutes "$eta_seconds")"
}

_monitor_progress() {
  local manifest_path="$1"
  local total_expected="$2"
  local start_count="$3"
  local start_epoch="$4"
  local current_count

  while true; do
    sleep "$PROGRESS_INTERVAL_SEC"
    current_count=$(_count_existing_files "$manifest_path")
    _print_progress "$total_expected" "$current_count" "$start_count" "$start_epoch"
  done
}

manifest_path="$(mktemp)"
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  rm -f "$manifest_path"
}
trap cleanup EXIT

echo "[LIST] reading remote object list from $REMOTE_DIR [start]"
_run_pinned nubescli list "$REMOTE_DIR" -R -o -f > "$manifest_path"
echo "[LIST] reading remote object list from $REMOTE_DIR [end]"

total_expected=$(grep -c . "$manifest_path" || true)
start_count=$(_count_existing_files "$manifest_path")
start_epoch=$(date +%s)

echo "[PRECHECK] total_expected=${total_expected}, already_existing=${start_count}, missing=$(( total_expected - start_count ))"
_print_progress "$total_expected" "$start_count" "$start_count" "$start_epoch"

if [[ "$start_count" -ge "$total_expected" ]]; then
  echo "[DOWNLOAD] skipped because all files already exist under $LOCAL_DIR"
  exit 0
fi

_monitor_progress "$manifest_path" "$total_expected" "$start_count" "$start_epoch" &
monitor_pid=$!

echo "[DOWNLOAD] NUBES SMART_cache missing files -> $LOCAL_DIR [start]"
_run_pinned nubescli dir-download \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -j "$NUBES_JOBS" \
  -s \
  --no-progress
echo "[DOWNLOAD] NUBES SMART_cache missing files -> $LOCAL_DIR [end]"

final_count=$(_count_existing_files "$manifest_path")
_print_progress "$total_expected" "$final_count" "$start_count" "$start_epoch"
echo "Download complete."
