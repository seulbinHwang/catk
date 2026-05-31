#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW_ROOT="${RAW_ROOT:-/media/user/E/dataset/womd_v1_3/scenario}"
CACHE_ROOT="${CACHE_ROOT:-/media/user/F/dataset/womd_v1_3/MDG_cache}"
LOG_DIR="${LOG_DIR:-$CACHE_ROOT/logs}"
CONDA_SH="${CONDA_SH:-/media/user/E/miniforge/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-catk}"
MIN_FREE_GB="${MIN_FREE_GB:-300}"
MIN_FREE_INODES="${MIN_FREE_INODES:-1000000}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-60}"
EXPECTED_TRAINING="${EXPECTED_TRAINING:-486995}"
EXPECTED_VALIDATION="${EXPECTED_VALIDATION:-44097}"
EXPECTED_TESTING="${EXPECTED_TESTING:-44920}"
CHECK_EXPECTED_COUNTS="${CHECK_EXPECTED_COUNTS:-1}"

TOTAL_WORKERS="${TOTAL_WORKERS:-$(nproc)}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$((TOTAL_WORKERS * 83 / 100))}"
VAL_WORKERS="${VAL_WORKERS:-$((TOTAL_WORKERS * 10 / 100))}"
if (( TRAIN_WORKERS < 1 )); then TRAIN_WORKERS=1; fi
if (( VAL_WORKERS < 1 )); then VAL_WORKERS=1; fi
TEST_WORKERS="${TEST_WORKERS:-$((TOTAL_WORKERS - TRAIN_WORKERS - VAL_WORKERS))}"
if (( TEST_WORKERS < 1 )); then TEST_WORKERS=1; fi

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

disk_available_gb() {
  df -Pk "$1" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'
}

inode_available() {
  df -Pi "$1" | awk 'NR == 2 {print $4}'
}

count_files() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -type f | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

check_count() {
  local name="$1"
  local observed="$2"
  local expected="$3"
  if [[ "$CHECK_EXPECTED_COUNTS" == "1" && "$observed" != "$expected" ]]; then
    echo "[supervisor] ERROR: $name count mismatch: observed=$observed expected=$expected"
    return 1
  fi
}

run_split() {
  local split="$1"
  local workers="$2"
  local log_file="$LOG_DIR/$split.log"
  echo "[supervisor] launch split=$split workers=$workers log=$log_file" >&2
  (
    set -euo pipefail
    cd "$REPO_ROOT"
    if [[ -f "$CONDA_SH" ]]; then
      # shellcheck source=/dev/null
      source "$CONDA_SH"
      conda activate "$CONDA_ENV"
    fi
    echo "[$split] start $(date '+%F %T') workers=$workers"
    if command -v ionice >/dev/null 2>&1 && [[ -x /usr/bin/time ]]; then
      /usr/bin/time -v ionice -c2 -n0 python -u -m src.data_preprocess \
        --input_dir "$RAW_ROOT" \
        --output_dir "$CACHE_ROOT" \
        --split "$split" \
        --num_workers "$workers"
    elif [[ -x /usr/bin/time ]]; then
      /usr/bin/time -v python -u -m src.data_preprocess \
        --input_dir "$RAW_ROOT" \
        --output_dir "$CACHE_ROOT" \
        --split "$split" \
        --num_workers "$workers"
    else
      python -u -m src.data_preprocess \
        --input_dir "$RAW_ROOT" \
        --output_dir "$CACHE_ROOT" \
        --split "$split" \
        --num_workers "$workers"
    fi
    echo "[$split] done $(date '+%F %T')"
  ) >"$log_file" 2>&1 &
  RUN_SPLIT_PID="$!"
}

echo "[supervisor] start $(date '+%F %T')"
echo "[supervisor] repo=$REPO_ROOT"
echo "[supervisor] raw=$RAW_ROOT"
echo "[supervisor] cache=$CACHE_ROOT"
echo "[supervisor] workers training=$TRAIN_WORKERS validation=$VAL_WORKERS testing=$TEST_WORKERS total=$((TRAIN_WORKERS + VAL_WORKERS + TEST_WORKERS))"
echo "[supervisor] git branch=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true) head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
git -C "$REPO_ROOT" status --short --branch || true

for split in training validation testing; do
  if [[ ! -d "$RAW_ROOT/$split" ]]; then
    echo "[supervisor] ERROR: missing raw split directory: $RAW_ROOT/$split"
    exit 1
  fi
done

echo "[supervisor] raw file counts"
for split in training validation testing; do
  printf '  %s ' "$split"
  find "$RAW_ROOT/$split" -type f | wc -l
done

mkdir -p "$CACHE_ROOT"
available_gb="$(disk_available_gb "$CACHE_ROOT")"
available_inodes="$(inode_available "$CACHE_ROOT")"
if (( available_gb < MIN_FREE_GB )); then
  echo "[supervisor] ERROR: not enough free space under $CACHE_ROOT: ${available_gb}GB < ${MIN_FREE_GB}GB"
  exit 1
fi
if (( available_inodes < MIN_FREE_INODES )); then
  echo "[supervisor] ERROR: not enough free inodes under $CACHE_ROOT: ${available_inodes} < ${MIN_FREE_INODES}"
  exit 1
fi

echo "[supervisor] disk before"
df -h "$RAW_ROOT" "$CACHE_ROOT"
df -ih "$CACHE_ROOT"

RUN_SPLIT_PID=""
run_split training "$TRAIN_WORKERS"
training_pid="$RUN_SPLIT_PID"
run_split validation "$VAL_WORKERS"
validation_pid="$RUN_SPLIT_PID"
run_split testing "$TEST_WORKERS"
testing_pid="$RUN_SPLIT_PID"
echo "[supervisor] pids training=$training_pid validation=$validation_pid testing=$testing_pid"

while kill -0 "$training_pid" 2>/dev/null || kill -0 "$validation_pid" 2>/dev/null || kill -0 "$testing_pid" 2>/dev/null; do
  training_count="$(count_files "$CACHE_ROOT/training")"
  validation_count="$(count_files "$CACHE_ROOT/validation")"
  testing_count="$(count_files "$CACHE_ROOT/testing")"
  validation_tf_count="$(count_files "$CACHE_ROOT/validation_tfrecords_splitted")"
  echo "[monitor] $(date '+%F %T') counts training=$training_count validation=$validation_count testing=$testing_count split_tfrecords=$validation_tf_count"
  df -h "$CACHE_ROOT" | tail -n 1
  sleep "$MONITOR_INTERVAL_SEC"
done

status=0
wait "$training_pid" || status=$?
wait "$validation_pid" || status=$?
wait "$testing_pid" || status=$?
if (( status != 0 )); then
  echo "[supervisor] FAILED status=$status"
  exit "$status"
fi

training_count="$(count_files "$CACHE_ROOT/training")"
validation_count="$(count_files "$CACHE_ROOT/validation")"
testing_count="$(count_files "$CACHE_ROOT/testing")"
validation_tf_count="$(count_files "$CACHE_ROOT/validation_tfrecords_splitted")"
echo "[supervisor] final counts training=$training_count validation=$validation_count testing=$testing_count split_tfrecords=$validation_tf_count"
check_count training "$training_count" "$EXPECTED_TRAINING"
check_count validation "$validation_count" "$EXPECTED_VALIDATION"
check_count testing "$testing_count" "$EXPECTED_TESTING"
check_count validation_tfrecords_splitted "$validation_tf_count" "$EXPECTED_VALIDATION"

echo "[supervisor] disk after"
df -h "$CACHE_ROOT"
df -ih "$CACHE_ROOT"
du -sh "$CACHE_ROOT" || true
echo "[supervisor] ALL_SPLITS_DONE $(date '+%F %T')"
