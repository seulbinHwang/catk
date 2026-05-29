#!/bin/sh
# Resolve the WOMD SMART cache root across the Ubuntu server and Kubernetes pods.
#
# Expected cache layout:
#   <root>/training
#   <root>/validation
#   <root>/testing
#   <root>/validation_tfrecords_splitted

cache_root_has_required_splits() {
  _catk_cache_root="${1:-}"

  [ -n "${_catk_cache_root}" ] || return 1
  [ -d "${_catk_cache_root}/training" ] || return 1
  [ -d "${_catk_cache_root}/validation" ] || return 1
  [ -d "${_catk_cache_root}/testing" ] || return 1
  [ -d "${_catk_cache_root}/validation_tfrecords_splitted" ] || return 1
}

resolve_womd_cache_root() {
  _catk_explicit_cache_root="${CACHE_ROOT:-}"

  if [ -n "${_catk_explicit_cache_root}" ]; then
    if cache_root_has_required_splits "${_catk_explicit_cache_root}"; then
      printf '%s\n' "${_catk_explicit_cache_root}"
      return 0
    fi

    echo "[WARN] CACHE_ROOT does not contain the required WOMD cache splits: ${_catk_explicit_cache_root}" >&2
    echo "       Falling back to CACHE_ROOT_CANDIDATES." >&2
  fi

  _catk_cache_candidates="${CACHE_ROOT_CANDIDATES:-}"
  if [ -z "${_catk_cache_candidates}" ]; then
    _catk_cache_candidates="${OCSC_UBUNTU_CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1} ${K8S_WOMD_CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache} ${NUPLAN_WOMD_CACHE_ROOT:-/mnt/nuplan/womd_v1_3/SMART_cache} ${LEGACY_WOMD_CACHE_ROOT:-/scratch/cache/SMART}"
  fi

  for _catk_candidate_cache_root in ${_catk_cache_candidates}; do
    [ -n "${_catk_candidate_cache_root}" ] || continue
    if cache_root_has_required_splits "${_catk_candidate_cache_root}"; then
      printf '%s\n' "${_catk_candidate_cache_root}"
      return 0
    fi
  done

  echo "[ERROR] No usable WOMD SMART cache root found." >&2
  echo "        Set CACHE_ROOT to a directory containing training/, validation/, testing/, and validation_tfrecords_splitted/." >&2
  return 1
}
