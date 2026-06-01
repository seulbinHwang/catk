#!/usr/bin/env bash
set -Eeuo pipefail

# Download the MDG WOMD cache from Nubes into a training pod.

REMOTE_DIR="${1:-${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache_0601}}"
LOCAL_DIR="${2:-${LOCAL_DIR:-/workspace/womd_v1_3/MDG_cache}}"
NUBES_JOBS="${NUBES_JOBS:-96}"
NUBES_RETRY="${NUBES_RETRY:-3}"
NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"
EXPECTED_TRAINING="${EXPECTED_TRAINING:-486995}"
EXPECTED_VALIDATION="${EXPECTED_VALIDATION:-44097}"
EXPECTED_TESTING="${EXPECTED_TESTING:-44920}"
EXPECTED_VALIDATION_TFRECORDS="${EXPECTED_VALIDATION_TFRECORDS:-44097}"
EXPECTED_TOTAL_FILES="${EXPECTED_TOTAL_FILES:-$((EXPECTED_TRAINING + EXPECTED_VALIDATION + EXPECTED_TESTING + EXPECTED_VALIDATION_TFRECORDS))}"
VERIFY_REMOTE="${VERIFY_REMOTE:-1}"
SPLITS=(training validation testing validation_tfrecords_splitted)

export NUBES_GATEWAY_ADDRESS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_mdg_cache_from_nubes.sh [remote_dir] [local_dir]

Defaults:
  remote_dir = labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache_0601
  local_dir  = /workspace/womd_v1_3/MDG_cache

Environment:
  NUBES_JOBS=96
  NUBES_GATEWAY_ADDRESS=c.nubes.sto.navercorp.com:8000
  PROGRESS_INTERVAL_SEC=60
  VERIFY_REMOTE=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! [[ "$NUBES_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUBES_JOBS must be a positive integer: $NUBES_JOBS" >&2
  exit 2
fi

if ! [[ "$NUBES_RETRY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUBES_RETRY must be a positive integer: $NUBES_RETRY" >&2
  exit 2
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH" >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"

count_files() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -type f 2>/dev/null | wc -l | awk '{print $1}'
  else
    echo "0"
  fi
}

split_total_count() {
  local total=0
  local split
  local count
  for split in "${SPLITS[@]}"; do
    count="$(count_files "$LOCAL_DIR/$split")"
    total=$((total + count))
  done
  echo "$total"
}

check_count() {
  local name="$1"
  local observed="$2"
  local expected="$3"
  if [[ "$observed" != "$expected" ]]; then
    echo "ERROR: $name count mismatch: observed=$observed expected=$expected" >&2
    exit 1
  fi
}

remote_count() {
  local manifest
  manifest="$(mktemp)"
  nubescli --retry "$NUBES_RETRY" list "$REMOTE_DIR" -R -o -f >"$manifest"
  awk 'NR == 1 && $0 == "Path" {next} NF {count++} END {print count + 0}' "$manifest"
  rm -f "$manifest"
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
  total_minutes=$((seconds / 60))
  hours=$((total_minutes / 60))
  minutes=$((total_minutes % 60))
  printf "%dh %dm" "$hours" "$minutes"
}

print_progress() {
  local total_expected="$1"
  local current_count="$2"
  local start_count="$3"
  local start_epoch="$4"
  local percent="0.00"
  local elapsed_seconds
  local processed_this_run
  local rate
  local remaining
  local eta_seconds=-1

  if (( current_count > total_expected )); then
    current_count="$total_expected"
  fi
  if (( total_expected > 0 )); then
    percent="$(awk "BEGIN { printf \"%.2f\", 100 * $current_count / $total_expected }")"
  fi
  elapsed_seconds=$(($(date +%s) - start_epoch))
  processed_this_run=$((current_count - start_count))
  if (( processed_this_run < 0 )); then
    processed_this_run=0
  fi
  if (( elapsed_seconds > 0 && processed_this_run > 0 )); then
    rate="$(awk "BEGIN { printf \"%.6f\", $processed_this_run / $elapsed_seconds }")"
    remaining=$((total_expected - current_count))
    if (( remaining < 0 )); then
      remaining=0
    fi
    eta_seconds="$(awk "BEGIN { printf \"%d\", $remaining / $rate }")"
  fi
  echo "[download-progress] total_expected=${total_expected} existing=${current_count} percent=${percent}% elapsed=$(format_hours_minutes "$elapsed_seconds") eta=$(format_hours_minutes "$eta_seconds")"
}

monitor_progress() {
  local total_expected="$1"
  local start_count="$2"
  local start_epoch="$3"
  while true; do
    sleep "$PROGRESS_INTERVAL_SEC"
    print_progress "$total_expected" "$(split_total_count)" "$start_count" "$start_epoch"
  done
}

monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MDG_CACHE_DOWNLOAD] start=$(date '+%F %T')"
echo "[MDG_CACHE_DOWNLOAD] remote=$REMOTE_DIR"
echo "[MDG_CACHE_DOWNLOAD] local=$LOCAL_DIR"
echo "[MDG_CACHE_DOWNLOAD] nubes_jobs=$NUBES_JOBS retry=$NUBES_RETRY gateway=$NUBES_GATEWAY_ADDRESS"
df -h "$LOCAL_DIR" || true
df -ih "$LOCAL_DIR" || true

if [[ "$VERIFY_REMOTE" == "1" ]]; then
  echo "[MDG_CACHE_DOWNLOAD] remote_count_check start=$(date '+%F %T')"
  remote_files="$(remote_count)"
  echo "[MDG_CACHE_DOWNLOAD] remote_files=$remote_files expected=$EXPECTED_TOTAL_FILES"
  check_count remote_total "$remote_files" "$EXPECTED_TOTAL_FILES"
fi

start_count="$(split_total_count)"
start_epoch="$(date +%s)"
missing=$((EXPECTED_TOTAL_FILES - start_count))
if (( missing < 0 )); then
  missing=0
fi
echo "[MDG_CACHE_DOWNLOAD] local_existing=$start_count expected=$EXPECTED_TOTAL_FILES missing=$missing"
print_progress "$EXPECTED_TOTAL_FILES" "$start_count" "$start_count" "$start_epoch"

if (( start_count >= EXPECTED_TOTAL_FILES )); then
  echo "[MDG_CACHE_DOWNLOAD] skipped because all files already exist under $LOCAL_DIR"
  exit 0
fi

monitor_progress "$EXPECTED_TOTAL_FILES" "$start_count" "$start_epoch" &
monitor_pid=$!

echo "[MDG_CACHE_DOWNLOAD] dir-download start=$(date '+%F %T')"
nubescli --retry "$NUBES_RETRY" dir-download \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -j "$NUBES_JOBS" \
  -s \
  --no-progress
echo "[MDG_CACHE_DOWNLOAD] dir-download end=$(date '+%F %T')"

training_count="$(count_files "$LOCAL_DIR/training")"
validation_count="$(count_files "$LOCAL_DIR/validation")"
testing_count="$(count_files "$LOCAL_DIR/testing")"
validation_tf_count="$(count_files "$LOCAL_DIR/validation_tfrecords_splitted")"
final_total=$((training_count + validation_count + testing_count + validation_tf_count))

echo "[MDG_CACHE_DOWNLOAD] local_counts training=$training_count validation=$validation_count testing=$testing_count validation_tfrecords_splitted=$validation_tf_count"
check_count training "$training_count" "$EXPECTED_TRAINING"
check_count validation "$validation_count" "$EXPECTED_VALIDATION"
check_count testing "$testing_count" "$EXPECTED_TESTING"
check_count validation_tfrecords_splitted "$validation_tf_count" "$EXPECTED_VALIDATION_TFRECORDS"
check_count total "$final_total" "$EXPECTED_TOTAL_FILES"

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
if (( elapsed < 1 )); then
  elapsed=1
fi
local_bytes="$(du -sb "$LOCAL_DIR/training" "$LOCAL_DIR/validation" "$LOCAL_DIR/testing" "$LOCAL_DIR/validation_tfrecords_splitted" | awk '{sum += $1} END {print sum + 0}')"
rate_mib_s="$(awk "BEGIN { printf \"%.2f\", ($local_bytes / 1048576) / $elapsed }")"
print_progress "$EXPECTED_TOTAL_FILES" "$final_total" "$start_count" "$start_epoch"
echo "[MDG_CACHE_DOWNLOAD] COMPLETE $(date '+%F %T') elapsed_seconds=$elapsed avg_rate_mib_s=$rate_mib_s"
