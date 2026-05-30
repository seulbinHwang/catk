#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/setup_runtime_env.sh"

export PROJECT_ROOT="${ROOT_DIR}"
export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"

python "${ROOT_DIR}/src/run.py" \
  experiment=unimm_anchor_based_4s \
  paths.cache_root="${CACHE_ROOT}" \
  task_name="${TASK_NAME:-unimm_anchor_based_4s}" \
  "$@"
