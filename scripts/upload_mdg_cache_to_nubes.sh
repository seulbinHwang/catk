#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_DIR="${1:-${LOCAL_DIR:-/media/user/F/dataset/womd_v1_3/MDG_cache}}"
REMOTE_DIR="${2:-${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache}}"
NUBES_JOBS="${NUBES_JOBS:-96}"
NUBES_RETRY="${NUBES_RETRY:-3}"
NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
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
  bash scripts/upload_mdg_cache_to_nubes.sh [local_dir] [remote_dir]

Defaults:
  local_dir  = /media/user/F/dataset/womd_v1_3/MDG_cache
  remote_dir = labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache

Environment:
  NUBES_JOBS=96
  NUBES_GATEWAY_ADDRESS=c.nubes.sto.navercorp.com:8000
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

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "ERROR: Local MDG cache directory not found: $LOCAL_DIR" >&2
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH" >&2
  exit 1
fi

count_files() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -type f | wc -l | awk '{print $1}'
  else
    echo "0"
  fi
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

split_bytes_sum() {
  du -sb \
    "$LOCAL_DIR/training" \
    "$LOCAL_DIR/validation" \
    "$LOCAL_DIR/testing" \
    "$LOCAL_DIR/validation_tfrecords_splitted" | awk '{sum += $1} END {print sum + 0}'
}

split_size_human() {
  du -sch \
    "$LOCAL_DIR/training" \
    "$LOCAL_DIR/validation" \
    "$LOCAL_DIR/testing" \
    "$LOCAL_DIR/validation_tfrecords_splitted" | awk '$2 == "total" {print $1}'
}

upload_split() {
  local split="$1"
  echo "[MDG_CACHE_UPLOAD] upload_split=$split start=$(date '+%F %T')"
  nubescli --retry "$NUBES_RETRY" dir-upload "$REMOTE_DIR/$split" \
    "$LOCAL_DIR/$split" \
    -s \
    -j "$NUBES_JOBS"
  echo "[MDG_CACHE_UPLOAD] upload_split=$split end=$(date '+%F %T')"
}

local_training="$(count_files "$LOCAL_DIR/training")"
local_validation="$(count_files "$LOCAL_DIR/validation")"
local_testing="$(count_files "$LOCAL_DIR/testing")"
local_validation_tfrecords="$(count_files "$LOCAL_DIR/validation_tfrecords_splitted")"
local_total=$((local_training + local_validation + local_testing + local_validation_tfrecords))
local_bytes="$(split_bytes_sum)"
local_size="$(split_size_human)"

echo "[MDG_CACHE_UPLOAD] start=$(date '+%F %T')"
echo "[MDG_CACHE_UPLOAD] local=$LOCAL_DIR"
echo "[MDG_CACHE_UPLOAD] remote=$REMOTE_DIR"
echo "[MDG_CACHE_UPLOAD] nubes_jobs=$NUBES_JOBS retry=$NUBES_RETRY gateway=$NUBES_GATEWAY_ADDRESS"
echo "[MDG_CACHE_UPLOAD] local_size=$local_size bytes=$local_bytes files=$local_total"
echo "[MDG_CACHE_UPLOAD] local_counts training=$local_training validation=$local_validation testing=$local_testing validation_tfrecords_splitted=$local_validation_tfrecords"

check_count training "$local_training" "$EXPECTED_TRAINING"
check_count validation "$local_validation" "$EXPECTED_VALIDATION"
check_count testing "$local_testing" "$EXPECTED_TESTING"
check_count validation_tfrecords_splitted "$local_validation_tfrecords" "$EXPECTED_VALIDATION_TFRECORDS"
check_count total "$local_total" "$EXPECTED_TOTAL_FILES"

start_epoch="$(date +%s)"

for split in "${SPLITS[@]}"; do
  upload_split "$split"
done

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
if (( elapsed < 1 )); then
  elapsed=1
fi
rate_mib_s="$(awk "BEGIN { printf \"%.2f\", ($local_bytes / 1048576) / $elapsed }")"

echo "[MDG_CACHE_UPLOAD] upload_end=$(date '+%F %T') elapsed_seconds=$elapsed avg_rate_mib_s=$rate_mib_s"

if [[ "$VERIFY_REMOTE" == "1" ]]; then
  echo "[MDG_CACHE_UPLOAD] remote_count_check start=$(date '+%F %T')"
  uploaded_count="$(remote_count)"
  echo "[MDG_CACHE_UPLOAD] remote_files=$uploaded_count expected=$EXPECTED_TOTAL_FILES"
  check_count remote_total "$uploaded_count" "$EXPECTED_TOTAL_FILES"
fi

echo "[MDG_CACHE_UPLOAD] COMPLETE $(date '+%F %T')"
