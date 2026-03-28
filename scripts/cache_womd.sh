#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# TensorFlow intra/inter op thread 과다 점유를 막아 멀티프로세스 효율을 올립니다.
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA_SPLIT="${DATA_SPLIT:-validation}" # training, validation, testing
NUM_WORKERS="${NUM_WORKERS:-24}"
INPUT_DIR="${INPUT_DIR:-/scratch/data/womd/uncompressed/scenario}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/cache/SMART}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
conda activate "${CATK_CONDA_ENV}"

echo "DATA_SPLIT=${DATA_SPLIT}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS}"
echo "TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS}"

python \
  -m src.data_preprocess \
  --split "${DATA_SPLIT}" \
  --num_workers "${NUM_WORKERS}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"