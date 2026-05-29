#!/bin/sh
# Kubernetes V100 pod launcher for pareto DMD / anchor-FM fine-tuning.
#
# This wrapper keeps the Ubuntu server cache path first, then falls back to the
# SMART cache mounted in the current pods such as testsv/testsvv/testsvvvv.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export CONDA_SH="${CONDA_SH:-/mnt/nuplan/miniforge/etc/profile.d/conda.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CACHE_ROOT_CANDIDATES="${CACHE_ROOT_CANDIDATES:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1 /workspace/womd_v1_3/SMART_cache /mnt/nuplan/womd_v1_3/SMART_cache /scratch/cache/SMART}"

# DMD is the base default in train_self_forced_npfm_pareto.sh. To run the
# open-loop anchor FM regularizer, pass USE_ANCHOR_FM=true ANCHOR_WEIGHT=...
# before this wrapper.
exec "${SCRIPT_DIR}/train_self_forced_npfm_pareto.sh" "$@"
