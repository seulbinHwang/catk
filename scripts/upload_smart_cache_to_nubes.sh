#!/bin/bash
set -euo pipefail

# ---------------- CPU 분리 설정 ----------------
CPUSET="0-31,64-95"
NUM_CPUS=64
NUM_CPUS_FOR_USE=56
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"

export DP_MAX_CPUS=${NUM_CPUS}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ----------------------------------------------

# 스크립트(현재 쉘) 자체를 CPUSET에 고정
# 이후 실행되는 하위 작업들도 동일 CPUSET을 사용
if command -v taskset >/dev/null 2>&1; then
  taskset -cp "${CPUSET}" $$ >/dev/null
fi

CPU_USAGE_PERCENT=$(awk "BEGIN { printf \"%.1f\", 100 * $NUM_CPUS_FOR_USE / $NUM_CPUS }")

echo "[SMART_CACHE_UPLOAD] CPUSET=${CPUSET}, PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC}"
echo "[SMART_CACHE_UPLOAD] available_cpus=${NUM_CPUS}, chosen_upload_jobs=${NUM_CPUS_FOR_USE}, cpu_usage_percent=${CPU_USAGE_PERCENT}%"

LOCAL_DIR="${LOCAL_DIR:-/media/user/E/dataset/womd_v1_3/SMART_cache}"
REMOTE_DIR="${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache}"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "ERROR: Local directory not found: $LOCAL_DIR"
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

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

_write_local_manifest() {
  find "$LOCAL_DIR" -type f | sed "s#^$LOCAL_DIR/##" | LC_ALL=C sort > "$LOCAL_MANIFEST"
}

_write_remote_manifest() {
  taskset -c "${CPUSET}" \
  nubescli list "$REMOTE_DIR" -R -o -f | \
    awk -v remote_dir="$REMOTE_DIR" '
      NR == 1 && $0 == "Path" {next}
      !NF {next}
      index($0, remote_dir "/") == 1 {print substr($0, length(remote_dir) + 2); next}
      {print $0}
    ' | LC_ALL=C sort > "$REMOTE_MANIFEST"
}

_count_uploaded_files() {
  comm -12 "$LOCAL_MANIFEST" "$REMOTE_MANIFEST" | wc -l | awk '{print $1}'
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

  echo "[upload-progress] total_expected=${total_expected} existing=${current_count} percent=${percent}% elapsed=$(_format_hours_minutes "$elapsed_seconds") eta=$(_format_hours_minutes "$eta_seconds")"
}

_monitor_progress() {
  local total_expected="$1"
  local start_count="$2"
  local start_epoch="$3"
  local current_count

  while true; do
    sleep "$PROGRESS_INTERVAL_SEC"
    _write_remote_manifest
    current_count=$(_count_uploaded_files)
    _print_progress "$total_expected" "$current_count" "$start_count" "$start_epoch"
  done
}

LOCAL_MANIFEST="$(mktemp)"
REMOTE_MANIFEST="$(mktemp)"
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  rm -f "$LOCAL_MANIFEST" "$REMOTE_MANIFEST"
}
trap cleanup EXIT

echo "[LIST] reading local object list from $LOCAL_DIR [start]"
_write_local_manifest
echo "[LIST] reading local object list from $LOCAL_DIR [end]"

echo "[LIST] reading remote object list from $REMOTE_DIR [start]"
_write_remote_manifest
echo "[LIST] reading remote object list from $REMOTE_DIR [end]"

total_expected=$(grep -c . "$LOCAL_MANIFEST" || true)
start_count=$(_count_uploaded_files)
start_epoch=$(date +%s)

echo "[PRECHECK] total_expected=${total_expected}, already_existing=${start_count}, missing=$(( total_expected - start_count ))"
_print_progress "$total_expected" "$start_count" "$start_count" "$start_epoch"

if [[ "$start_count" -ge "$total_expected" ]]; then
  echo "[UPLOAD] skipped because all local files already exist under $REMOTE_DIR"
  exit 0
fi

_monitor_progress "$total_expected" "$start_count" "$start_epoch" &
monitor_pid=$!

echo "[UPLOAD] SMART_cache missing files -> $REMOTE_DIR [start]"
taskset -c "${CPUSET}" \
nubescli dir-upload "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -e \
  -s \
  -j ${NUM_CPUS_FOR_USE} \
  --no-progress
echo "[UPLOAD] SMART_cache missing files -> $REMOTE_DIR [end]"

kill "$monitor_pid" >/dev/null 2>&1 || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""

_write_remote_manifest
final_count=$(_count_uploaded_files)
_print_progress "$total_expected" "$final_count" "$start_count" "$start_epoch"

echo "Upload complete."
