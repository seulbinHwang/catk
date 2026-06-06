#!/usr/bin/env bash
# Precompute deterministic semi_mdg token/flow training sidecars on one multi-GPU pod.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
SPLIT="${SPLIT:-training}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${CACHE_ROOT}/semi_mdg_sidecar}"
OUTPUT_DIR="${OUTPUT_DIR:-${SIDECAR_ROOT}/${SPLIT}}"
EXPERIMENT="${EXPERIMENT:-mdg_pretrain_h100x3x2}"
NUM_SHARDS="${NUM_SHARDS:-}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -z "$NUM_SHARDS" ]]; then
  NUM_SHARDS="$(python - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)"
fi

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be a positive integer, got: $NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
echo "[sidecar] cache_dir=${CACHE_ROOT}/${SPLIT}"
echo "[sidecar] output_dir=${OUTPUT_DIR}"
echo "[sidecar] experiment=${EXPERIMENT}"
echo "[sidecar] num_shards=${NUM_SHARDS}"

PIDS=()
for (( shard = 0; shard < NUM_SHARDS; shard++ )); do
  log_file="${OUTPUT_DIR}/shard_${shard}_of_${NUM_SHARDS}.log"
  args=(
    tools/precompute_semi_mdg_sidecar.py
    --cache-dir "${CACHE_ROOT}/${SPLIT}"
    --output-dir "$OUTPUT_DIR"
    --experiment "$EXPERIMENT"
    --device cuda
    --num-shards "$NUM_SHARDS"
    --shard-index "$shard"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    python "${args[@]}" 2>&1 | tee "$log_file"
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[sidecar] one or more shards failed" >&2
  exit "$status"
fi

count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.pkl' | wc -l)"
echo "[sidecar] complete: files=${count}, output_dir=${OUTPUT_DIR}"
