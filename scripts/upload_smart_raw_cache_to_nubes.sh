#!/bin/bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-/media/user/F/dataset/womd_v1_3/SMART_RAW_cache}"
REMOTE_DIR="${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_RAW_cache}"
JOBS="${JOBS:-96}"
NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
export NUBES_GATEWAY_ADDRESS

if [ ! -d "$LOCAL_DIR" ]; then
  echo "ERROR: Local directory not found: $LOCAL_DIR"
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

echo "Uploading SMART_RAW_cache directory..."
echo "  local:  $LOCAL_DIR"
echo "  remote: $REMOTE_DIR"
echo "  jobs:   $JOBS"
echo "  nubes:  $NUBES_GATEWAY_ADDRESS"

nubescli dir-upload "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -s -j "$JOBS"

echo "Upload complete."
