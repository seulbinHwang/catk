#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH=${1:?"Usage: scripts/road_flow_finetune.sh /path/to/flow_pretrained.ckpt [extra hydra args...]"}
shift || true

python src/run.py \
  experiment=road_flow \
  ckpt_path="${CKPT_PATH}" \
  "$@"
