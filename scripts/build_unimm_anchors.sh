#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/setup_runtime_env.sh"

OUTPUT="${OUTPUT:-${ROOT_DIR}/src/unimm/anchors/unimm_anchors_8s_k2048.pkl}"

python "${ROOT_DIR}/scripts/build_unimm_anchors.py" \
  --train-cache-dir "${CACHE_ROOT}/training" \
  --output "${OUTPUT}" \
  "$@"
