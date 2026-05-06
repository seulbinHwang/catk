#!/usr/bin/env bash
# train_with_tfrecords: cache_womd_split_parallel.sh 와 동일하게
#   - job당 --num_workers (샤드 병렬) + --num_jobs (프로세스 병렬)
#   - TF/OMP 스레드 과점유 방지
# 환경변수: NJ (기본 8), NW (기본 4), CACHE, IN, PY
set -euo pipefail

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PY="${PY:-/home2/pnc2/miniforge3/envs/catk/bin/python}"
CACHE="${CACHE:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
IN="${IN:-/home2/pnc2/repos_python/datasets/smart_data/waymo/scenario}"
NJ="${NJ:-8}"
NW="${NW:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

pkill -f "python -m src.data_preprocess" 2>/dev/null || true
sleep 2
rm -rf "${CACHE}/train_with_tfrecords" "${CACHE}/train_with_tfrecords_tfrecords_splitted"

for ((R=0; R<NJ; R++)); do
  nohup "$PY" -u -m src.data_preprocess \
    --input_dir "$IN" --output_dir "$CACHE" \
    --split training --output_split train_with_tfrecords \
    --write_tfrecords always \
    --num_workers "$NW" \
    --num_jobs "$NJ" --job_rank "$R" \
    > logs/preprocess_train_with_tfrecords_j${R}.log 2>&1 &
done
echo "Started num_jobs=${NJ} num_workers=${NW} (logs: logs/preprocess_train_with_tfrecords_j*.log)"
