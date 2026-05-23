#!/bin/bash
set -euo pipefail

LOCAL_DIR="/media/user/D/dataset/womd_v1_3/SMART_cache_0523"
REMOTE_DIR="labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache_0523"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "ERROR: Local directory not found: $LOCAL_DIR"
  exit 1
fi

if ! command -v nubescli >/dev/null 2>&1; then
  echo "ERROR: nubescli not found in PATH"
  exit 1
fi

echo "Step: Uploading SMART_cache directory..."

nubescli dir-upload "$REMOTE_DIR" \
  "$LOCAL_DIR" \
  -s -j 96

echo "Upload complete."
