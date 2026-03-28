#!/bin/sh
set -eu

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# one process 당 thread 과점유 방지
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA_SPLIT="${DATA_SPLIT:-training}" # training, validation, testing
NUM_JOBS="${NUM_JOBS:-8}"            # 전체 shard를 나눌 병렬 job 수
NUM_WORKERS_PER_JOB="${NUM_WORKERS_PER_JOB:-4}"
INPUT_DIR="${INPUT_DIR:-/home2/pnc2/repos_python/datasets/smart_data/waymo/scenario}"
OUTPUT_DIR="${OUTPUT_DIR:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_parallel}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"

if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
conda activate "${CATK_CONDA_ENV}"

echo "DATA_SPLIT=${DATA_SPLIT}"
echo "NUM_JOBS=${NUM_JOBS}"
echo "NUM_WORKERS_PER_JOB=${NUM_WORKERS_PER_JOB}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

pids=""
job=0
while [ "${job}" -lt "${NUM_JOBS}" ]; do
  echo "[launch] job_rank=${job}"
  python -u -m src.data_preprocess \
    --split "${DATA_SPLIT}" \
    --num_workers "${NUM_WORKERS_PER_JOB}" \
    --num_jobs "${NUM_JOBS}" \
    --job_rank "${job}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" &
  pids="${pids} $!"
  job=$((job + 1))
done

status=0
for pid in ${pids}; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"

