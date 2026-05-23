#!/usr/bin/env bash
set -Eeuo pipefail

# Download a SMART cache directory from Nubes.
#
# Default usage:
#   bash scripts/download_smart_cache_from_nubes.sh \
#     labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
#     "$CACHE_ROOT"
#
# The default Nubes transfer parallelism is intentionally fixed at 96 jobs.
# Override with NUBES_JOBS only when the runtime environment needs a different
# value.

CPUSET="${CPUSET:-}"
NUBES_JOBS="${NUBES_JOBS:-96}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

REMOTE_DIR="${1:-${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache}}"
LOCAL_DIR="${2:-${LOCAL_DIR:-/workspace/womd_v1_3/SMART_cache}}"

if [[ -z "$REMOTE_DIR" || -z "$LOCAL_DIR" ]]; then
  echo "Usage: bash scripts/download_smart_cache_from_nubes.sh <remote_dir> <local_dir>"
  echo "or set REMOTE_DIR and LOCAL_DIR env vars."
  exit 1
fi

if ! [[ "$NUBES_JOBS" =~ ^[0-9]+$ ]] || (( NUBES_JOBS < 1 )); then
  echo "ERROR: NUBES_JOBS must be a positive integer; got: $NUBES_JOBS"
  exit 2
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

mkdir -p "$LOCAL_DIR"

detect_cpuset() {
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

run_pinned() {
  if [[ -n "$ACTIVE_CPUSET" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$ACTIVE_CPUSET" "$@"
  else
    "$@"
  fi
}

format_hours_minutes() {
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

count_local_files() {
  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "0"
    return
  fi

  find "$LOCAL_DIR" -type f 2>/dev/null | wc -l | awk '{print $1}'
}

print_progress() {
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

  if [[ "$current_count" -gt "$total_expected" ]]; then
    current_count="$total_expected"
  fi

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

  echo "[download-progress] total_expected=${total_expected} existing=${current_count} percent=${percent}% elapsed=$(format_hours_minutes "$elapsed_seconds") eta=$(format_hours_minutes "$eta_seconds")"
}

monitor_progress() {
  local total_expected="$1"
  local start_count="$2"
  local start_epoch="$3"
  local current_count

  while true; do
    sleep "$PROGRESS_INTERVAL_SEC"
    current_count=$(count_local_files)
    print_progress "$total_expected" "$current_count" "$start_count" "$start_epoch"
  done
}

ACTIVE_CPUSET="$(detect_cpuset)"
export DP_MAX_CPUS="${DP_MAX_CPUS:-$NUBES_JOBS}"

echo "[SMART_CACHE_DOWNLOAD] CPUSET=${ACTIVE_CPUSET:-auto}, PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC}"
echo "[SMART_CACHE_DOWNLOAD] chosen_download_jobs=${NUBES_JOBS}"

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
run_pinned nubescli list "$REMOTE_DIR" -R -o -f > "$manifest_path"
echo "[LIST] reading remote object list from $REMOTE_DIR [end]"

if [[ -s "$manifest_path" ]]; then
  total_expected=$(awk 'NR == 1 && $0 == "Path" {next} NF {count++} END {print count + 0}' "$manifest_path")
else
  total_expected=0
fi

start_count=$(count_local_files)
start_epoch=$(date +%s)
missing=$(( total_expected - start_count ))
if [[ "$missing" -lt 0 ]]; then
  missing=0
fi

echo "[PRECHECK] total_expected=${total_expected}, already_existing=${start_count}, missing=${missing}"
print_progress "$total_expected" "$start_count" "$start_count" "$start_epoch"

if [[ "$total_expected" -gt 0 && "$start_count" -ge "$total_expected" ]]; then
  echo "[DOWNLOAD] skipped because all files already exist under $LOCAL_DIR"
  exit 0
fi

monitor_progress "$total_expected" "$start_count" "$start_epoch" &
monitor_pid=$!

echo "[DOWNLOAD] NUBES SMART_cache missing files -> $LOCAL_DIR [start]"
run_pinned nubescli dir-download \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -j "$NUBES_JOBS" \
  -s \
  --no-progress
echo "[DOWNLOAD] NUBES SMART_cache missing files -> $LOCAL_DIR [end]"

final_count=$(count_local_files)
print_progress "$total_expected" "$final_count" "$start_count" "$start_epoch"
echo "Download complete."
