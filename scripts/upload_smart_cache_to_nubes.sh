#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------- CPU / concurrency settings ----------------
CPUSET="${CPUSET:-}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-300}"
MAX_UPLOAD_JOBS="${MAX_UPLOAD_JOBS:-16}"
LARGE_TREE_THRESHOLD="${LARGE_TREE_THRESHOLD:-200000}"
LARGE_TREE_PROGRESS_INTERVAL_SEC="${LARGE_TREE_PROGRESS_INTERVAL_SEC:-600}"
REMOTE_PROGRESS="${REMOTE_PROGRESS:-1}"
NUBES_RETRY="${NUBES_RETRY:-5}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ------------------------------------------------------------

LOCAL_DIR="${LOCAL_DIR:-/media/user/E/dataset/womd_v1_3/SMART_cache}"
REMOTE_DIR="${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache}"

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "ERROR: Local directory not found: $LOCAL_DIR"
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

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

  cpus="$(_count_cpus_in_list "$ACTIVE_CPUSET")"
  if [[ "$cpus" -gt 0 ]]; then
    CPU_DETECTION_SOURCE="cpuset"
    AVAILABLE_CPUS="$cpus"
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
  _run_pinned nubescli -r "$NUBES_RETRY" list "$REMOTE_DIR" -R -o -f | \
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
  local progress_interval="$4"
  local current_count

  while true; do
    sleep "$progress_interval"
    if _write_remote_manifest; then
      current_count=$(_count_uploaded_files)
      _print_progress "$total_expected" "$current_count" "$start_count" "$start_epoch"
    else
      echo "[upload-progress] remote manifest refresh failed; keeping previous progress snapshot"
    fi
  done
}

ACTIVE_CPUSET="$(_detect_cpuset)"
CPU_DETECTION_SOURCE=""
AVAILABLE_CPUS=""
_detect_available_cpus

export DP_MAX_CPUS="${AVAILABLE_CPUS}"

if [[ -n "$ACTIVE_CPUSET" ]] && command -v taskset >/dev/null 2>&1; then
  taskset -cp "$ACTIVE_CPUSET" $$ >/dev/null 2>&1 || true
fi

if [[ -z "${NUBES_JOBS:-}" ]]; then
  if [[ "$AVAILABLE_CPUS" -le 4 ]]; then
    NUBES_JOBS="$AVAILABLE_CPUS"
  elif [[ "$AVAILABLE_CPUS" -le 8 ]]; then
    NUBES_JOBS=$(( AVAILABLE_CPUS - 1 ))
  elif [[ "$AVAILABLE_CPUS" -le 16 ]]; then
    NUBES_JOBS=$(( AVAILABLE_CPUS - 2 ))
  else
    NUBES_JOBS=$(( (AVAILABLE_CPUS * 4) / 5 ))
  fi
fi

if [[ "$NUBES_JOBS" -lt 1 ]]; then
  NUBES_JOBS=1
fi

if [[ "$MAX_UPLOAD_JOBS" -gt 0 && "$NUBES_JOBS" -gt "$MAX_UPLOAD_JOBS" ]]; then
  NUBES_JOBS="$MAX_UPLOAD_JOBS"
fi

CPU_USAGE_PERCENT=$(awk "BEGIN { printf \"%.1f\", 100 * $NUBES_JOBS / $AVAILABLE_CPUS }")

echo "[SMART_CACHE_UPLOAD] CPUSET=${ACTIVE_CPUSET:-auto}, PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC}"
echo "[SMART_CACHE_UPLOAD] available_cpus=${AVAILABLE_CPUS}, chosen_upload_jobs=${NUBES_JOBS}, cpu_usage_percent=${CPU_USAGE_PERCENT}%, cpu_detection_source=${CPU_DETECTION_SOURCE}, max_upload_jobs=${MAX_UPLOAD_JOBS}, nubes_retry=${NUBES_RETRY}"

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

effective_progress_interval="$PROGRESS_INTERVAL_SEC"
if [[ "$total_expected" -ge "$LARGE_TREE_THRESHOLD" && "$effective_progress_interval" -lt "$LARGE_TREE_PROGRESS_INTERVAL_SEC" ]]; then
  effective_progress_interval="$LARGE_TREE_PROGRESS_INTERVAL_SEC"
  echo "[UPLOAD] large tree detected; progress interval raised to ${effective_progress_interval}s to reduce recursive list pressure"
fi

if [[ "$REMOTE_PROGRESS" == "1" ]]; then
  _monitor_progress "$total_expected" "$start_count" "$start_epoch" "$effective_progress_interval" &
  monitor_pid=$!
else
  echo "[UPLOAD] remote progress monitor disabled"
fi

upload_flags=(-e -j "$NUBES_JOBS" --no-progress)
if [[ "$start_count" -gt 0 ]]; then
  upload_flags+=(-s)
  echo "[UPLOAD] enabling --skip because remote manifest already contains ${start_count} objects"
else
  echo "[UPLOAD] remote manifest is empty; omitting --skip to avoid per-file HEAD checks"
fi

echo "[UPLOAD] SMART_cache missing files -> $REMOTE_DIR [start]"
_run_pinned nubescli -r "$NUBES_RETRY" dir-upload \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  "${upload_flags[@]}"
echo "[UPLOAD] SMART_cache missing files -> $REMOTE_DIR [end]"

if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
fi

_write_remote_manifest
final_count=$(_count_uploaded_files)
_print_progress "$total_expected" "$final_count" "$start_count" "$start_epoch"

echo "Upload complete."
