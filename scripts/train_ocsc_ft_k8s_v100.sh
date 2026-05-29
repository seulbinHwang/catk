#!/bin/sh
# Kubernetes V100 pod launcher for OCSC fine-tuning.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export CONDA_SH="${CONDA_SH:-/mnt/nuplan/miniforge/etc/profile.d/conda.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CACHE_ROOT_CANDIDATES="${CACHE_ROOT_CANDIDATES:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1 /workspace/womd_v1_3/SMART_cache /mnt/nuplan/womd_v1_3/SMART_cache /scratch/cache/SMART}"

# Defaults in train_ocsc_ft.sh run Open-loop nearest L2 matching.
# OCSC_OL_NEAREST_MATCH=true, OCSC_LOSS_TYPE=l2.
exec "${SCRIPT_DIR}/train_ocsc_ft.sh" "$@"
