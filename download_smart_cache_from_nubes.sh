#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------- CPU 분리 설정 (업로드 스크립트와 동일) ----------------
CPUSET="0-31,64-95"
NUM_CPUS=64

export DP_MAX_CPUS=${NUM_CPUS}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ------------------------------------------------------------------------

# 스크립트 자체를 CPUSET에 고정
if command -v taskset >/dev/null 2>&1; then
  taskset -cp "${CPUSET}" $$ >/dev/null
fi

REMOTE_DIR="labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache"
LOCAL_DIR="/workspace/womd_v1_3/SMART_cache"

echo "[SMART_CACHE_DOWNLOAD] CPUSET=${CPUSET}, DP_MAX_CPUS=${DP_MAX_CPUS}"

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

# nubes_to_mlx_only_womd.sh 방식: 전체 코어의 4/5 사용
NUBES_JOBS=$(( ( $(nproc) * 4 ) / 5 ))
if [ "$NUBES_JOBS" -lt 1 ]; then
  NUBES_JOBS=1
fi

echo "[NUBES_JOBS] total_cores=$(nproc), use_cores=${NUBES_JOBS} (4/5)"

# 대상 디렉터리 준비(내용만 비우고 디렉터리는 유지)
if [ -d "$LOCAL_DIR" ]; then
  echo "[PREPARE] clean contents under $LOCAL_DIR [start]"
  if [[ "$LOCAL_DIR" == "/" || "$LOCAL_DIR" == "" ]]; then
    echo "Refusing to clean unsafe directory: '$LOCAL_DIR'"
    exit 1
  fi
  find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "[PREPARE] clean contents under $LOCAL_DIR [end]"
else
  echo "[PREPARE] create $LOCAL_DIR"
  mkdir -p "$LOCAL_DIR"
fi

echo "[DOWNLOAD] NUBES SMART_cache -> $LOCAL_DIR [start]"
taskset -c "${CPUSET}" \
nubescli dir-download \
  "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -j "$NUBES_JOBS" \
  -s \
  --no-progress
echo "[DOWNLOAD] NUBES SMART_cache -> $LOCAL_DIR [end]"

echo "Download complete."
