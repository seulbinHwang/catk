#!/usr/bin/env bash
set -Eeuo pipefail

CPUSET="${CPUSET:-}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"
DEFAULT_NUBES_JOBS=96
SKIP_REMOTE_LIST="${SKIP_REMOTE_LIST:-0}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

REMOTE_DIR="${1:-${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_RAW_cache}}"
LOCAL_DIR="${2:-${LOCAL_DIR:-/workspace/womd_v1_3/SMART_RAW_cache}}"
NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
export NUBES_GATEWAY_ADDRESS

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

if [[ -z "$REMOTE_DIR" || -z "$LOCAL_DIR" ]]; then
  echo "Usage: bash scripts/download_smart_raw_cache_from_nubes.sh <remote_dir> <local_dir>"
  echo "or set REMOTE_DIR and LOCAL_DIR env vars."
  exit 1
fi

mkdir -p "$LOCAL_DIR"

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

_count_cpus_in_list() {
  local cpu_list="$1"
  local total=0
  local part
  local start
  local end

  if [[ -z "$cpu_list" ]]; then
    echo "0"
    return
  fi

  IFS=',' read -ra parts <<< "$cpu_list"
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

_detect_available_cpus() {
  local quota
  local period
  local cpus

  if [[ -n "${NUM_CPUS:-}" ]]; then
    CPU_DETECTION_SOURCE="NUM_CPUS"
    AVAILABLE_CPUS="$NUM_CPUS"
    return
  fi

  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max || true
    if [[ -n "${quota:-}" && -n "${period:-}" && "$quota" != "max" && "$period" -gt 0 ]]; then
      cpus=$(( quota / period ))
      if [[ "$cpus" -gt 0 ]]; then
        CPU_DETECTION_SOURCE="cgroup_cpu.max"
        AVAILABLE_CPUS="$cpus"
        return
      fi
    fi
  fi

  if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
    quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
    period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    if [[ -n "${quota:-}" && -n "${period:-}" && "$quota" -gt 0 && "$period" -gt 0 ]]; then
      cpus=$(( quota / period ))
      if [[ "$cpus" -gt 0 ]]; then
        CPU_DETECTION_SOURCE="cgroup_cfs"
        AVAILABLE_CPUS="$cpus"
        return
      fi
    fi
  fi

  cpus="$(nproc 2>/dev/null || true)"
  if [[ -n "${cpus:-}" && "$cpus" -gt 0 ]]; then
    CPU_DETECTION_SOURCE="nproc"
    AVAILABLE_CPUS="$cpus"
    return
  fi

  cpus="$(_count_cpus_in_list "$ACTIVE_CPUSET")"
  if [[ "$cpus" -gt 0 ]]; then
    CPU_DETECTION_SOURCE="taskset"
    AVAILABLE_CPUS="$cpus"
    return
  fi

  CPU_DETECTION_SOURCE="fallback"
  AVAILABLE_CPUS="1"
}

_run_pinned() {
  if [[ -n "$ACTIVE_CPUSET" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$ACTIVE_CPUSET" "$@"
  else
    "$@"
  fi
}

ACTIVE_CPUSET="$(_detect_cpuset)"
CPU_DETECTION_SOURCE=""
AVAILABLE_CPUS=""
_detect_available_cpus

export DP_MAX_CPUS="${AVAILABLE_CPUS}"

NUBES_JOBS="${NUBES_JOBS:-$DEFAULT_NUBES_JOBS}"
if [[ "$NUBES_JOBS" -lt 1 ]]; then
  NUBES_JOBS=1
fi

CPU_USAGE_PERCENT=$(awk "BEGIN { printf \"%.1f\", 100 * $NUBES_JOBS / $AVAILABLE_CPUS }")

echo "[SMART_RAW_CACHE_DOWNLOAD] remote=${REMOTE_DIR}"
echo "[SMART_RAW_CACHE_DOWNLOAD] local=${LOCAL_DIR}"
echo "[SMART_RAW_CACHE_DOWNLOAD] nubes=${NUBES_GATEWAY_ADDRESS}"
echo "[SMART_RAW_CACHE_DOWNLOAD] skip_remote_list=${SKIP_REMOTE_LIST}"
echo "[SMART_RAW_CACHE_DOWNLOAD] CPUSET=${ACTIVE_CPUSET:-auto}, PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC}"
echo "[SMART_RAW_CACHE_DOWNLOAD] available_cpus=${AVAILABLE_CPUS}, chosen_download_jobs=${NUBES_JOBS}, cpu_usage_percent=${CPU_USAGE_PERCENT}%, cpu_detection_source=${CPU_DETECTION_SOURCE}"

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

_count_local_files() {
  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "0"
    return
  fi

  find "$LOCAL_DIR" -type f 2>/dev/null | wc -l | awk '{print $1}'
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

  echo "[download-progress] total_expected=${total_expected} existing=${current_count} percent=${percent}% elapsed=$(_format_hours_minutes "$elapsed_seconds") eta=$(_format_hours_minutes "$eta_seconds")"
}

_monitor_progress() {
  local total_expected="$1"
  local start_count="$2"
  local start_epoch="$3"
  local current_count

  while true; do
    sleep "$PROGRESS_INTERVAL_SEC"
    current_count=$(_count_local_files)
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

if [[ "$SKIP_REMOTE_LIST" == "1" ]]; then
  start_count=$(_count_local_files)
  start_epoch=$(date +%s)
  echo "[PRECHECK] remote listing skipped for fastest bulk download; already_existing=${start_count}"
  echo "[DOWNLOAD] NUBES SMART_RAW_cache missing files -> $LOCAL_DIR [start]"
  _run_pinned nubescli dir-download \
    "$REMOTE_DIR" \
    "$LOCAL_DIR" \
    -j "$NUBES_JOBS" \
    -s \
    --no-progress
  echo "[DOWNLOAD] NUBES SMART_RAW_cache missing files -> $LOCAL_DIR [end]"
  final_count=$(_count_local_files)
  elapsed_seconds=$(( $(date +%s) - start_epoch ))
  echo "[DOWNLOAD] final_local_files=${final_count}, elapsed=$(_format_hours_minutes "$elapsed_seconds")"
  echo "Download complete."
  exit 0
fi

echo "[LIST] reading remote object list from $REMOTE_DIR [start]"
_run_pinned nubescli list "$REMOTE_DIR" -R -o -f > "$manifest_path"
echo "[LIST] reading remote object list from $REMOTE_DIR [end]"

if [[ -s "$manifest_path" ]]; then
  total_expected=$(awk 'NR == 1 && $0 == "Path" {next} NF {count++} END {print count + 0}' "$manifest_path")
else
  total_expected=0
fi

if [[ "$total_expected" -le 0 ]]; then
  echo "ERROR: no remote files found under $REMOTE_DIR"
  exit 1
fi

start_count=$(_count_local_files)
start_epoch=$(date +%s)

echo "[PRECHECK] total_expected=${total_expected}, already_existing=${start_count}, missing=$(( total_expected - start_count ))"
_print_progress "$total_expected" "$start_count" "$start_count" "$start_epoch"

if [[ "$start_count" -ge "$total_expected" ]]; then
  echo "[DOWNLOAD] skipped because all files already exist under $LOCAL_DIR"
  exit 0
fi

_monitor_progress "$total_expected" "$start_count" "$start_epoch" &
monitor_pid=$!

echo "[DOWNLOAD] NUBES SMART_RAW_cache missing files -> $LOCAL_DIR [start]"
_run_pinned nubescli dir-download \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -j "$NUBES_JOBS" \
  -s \
  --no-progress
echo "[DOWNLOAD] NUBES SMART_RAW_cache missing files -> $LOCAL_DIR [end]"

final_count=$(_count_local_files)
_print_progress "$total_expected" "$final_count" "$start_count" "$start_epoch"

if [[ "$final_count" -lt "$total_expected" ]]; then
  echo "ERROR: download finished but local file count is incomplete: ${final_count}/${total_expected}"
  exit 1
fi

echo "Download complete."
