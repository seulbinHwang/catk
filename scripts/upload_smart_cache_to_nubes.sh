#!/bin/bash
set -euo pipefail

# ---------------- CPU 분리 설정 ----------------
CPUSET="0-31,64-95"
NUM_CPUS=64
NUM_CPUS_FOR_USE=56

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

echo "[SMART_CACHE_UPLOAD] Using CPUSET=${CPUSET}, NUM_CPUS=${NUM_CPUS}"

LOCAL_DIR="/media/user/E/dataset/womd_v1_3/SMART_cache"
REMOTE_DIR="labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "ERROR: Local directory not found: $LOCAL_DIR"
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

echo "Step: Uploading SMART_cache directory..."

taskset -c "${CPUSET}" \
nubescli dir-upload "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -s -j ${NUM_CPUS_FOR_USE}

echo "Upload complete."
