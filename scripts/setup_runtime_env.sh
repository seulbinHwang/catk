#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_ENV:=catk}"

if [ -z "${CONDA_SH:-}" ]; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniforge/etc/profile.d/conda.sh" \
    "/media/user/E/miniforge/etc/profile.d/conda.sh"
  do
    if [ -f "$candidate" ]; then
      CONDA_SH="$candidate"
      break
    fi
  done
fi

if [ -z "${CONDA_SH:-}" ] || [ ! -f "$CONDA_SH" ]; then
  echo "conda.sh를 찾을 수 없습니다. CONDA_SH=/path/to/conda.sh 를 지정하세요." >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [ -z "${CACHE_ROOT:-}" ]; then
  if [ -d "/workspace/womd_v1_3/MDG_cache" ]; then
    CACHE_ROOT="/workspace/womd_v1_3/MDG_cache"
  elif [ -d "/media/user/F/dataset/womd_v1_3/MDG_cache" ]; then
    CACHE_ROOT="/media/user/F/dataset/womd_v1_3/MDG_cache"
  else
    CACHE_ROOT="/workspace/womd_v1_3/MDG_cache"
  fi
fi
export CACHE_ROOT
