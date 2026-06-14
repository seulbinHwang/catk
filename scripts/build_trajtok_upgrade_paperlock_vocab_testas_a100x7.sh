#!/usr/bin/env bash
# Build the trajtok_upgrade paper-lock motion vocabulary on the testas A100x7 pod.
#
# This script intentionally writes a new token file, trajtok_paperlock_vocab.pkl,
# and a new grid-stat cache. It does not overwrite the legacy trajtok_vocab.pkl
# used by the original trajtok branch.
set -Eeuo pipefail

NAMESPACE="${NAMESPACE:-p-pnc}"
POD="${POD:-testas}"
CONTAINER="${CONTAINER:-main}"
REPO_URL="${REPO_URL:-https://github.com/seulbinHwang/catk.git}"
BRANCH="${BRANCH:-trajtok_upgrade}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_trajtok_upgrade_paperlock_vocab}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
RAW_DATA_PATH="${RAW_DATA_PATH:-${CACHE_ROOT}/training}"
GRID_STATS_CACHE="${GRID_STATS_CACHE:-${CACHE_ROOT}/trajtok_upgrade_paperlock_grid_stats.pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/src/smart/tokens/trajtok_paperlock_vocab.pkl}"
SESSION="${SESSION:-catk-trajtok-upgrade-paperlock-vocab-testas}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6}"
MAX_WORKERS="${MAX_WORKERS:-28}"
USE_CACHE="${USE_CACHE:-0}"

remote_quote() { printf '%q' "$1"; }

if [[ "${STOP:-0}" == "1" ]]; then
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- \
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  exit 0
fi

repo_q="$(remote_quote "$REPO_URL")"
branch_q="$(remote_quote "$BRANCH")"
root_q="$(remote_quote "$PROJECT_ROOT")"
cache_root_q="$(remote_quote "$CACHE_ROOT")"
raw_data_q="$(remote_quote "$RAW_DATA_PATH")"
grid_cache_q="$(remote_quote "$GRID_STATS_CACHE")"
output_q="$(remote_quote "$OUTPUT_PATH")"
session_q="$(remote_quote "$SESSION")"
gpu_devices_q="$(remote_quote "$GPU_DEVICES")"
max_workers_q="$(remote_quote "$MAX_WORKERS")"
use_cache_q="$(remote_quote "$USE_CACHE")"

remote_script="
set -Eeuo pipefail
repo=${repo_q}
branch=${branch_q}
root=${root_q}
cache_root=${cache_root_q}
raw_data=${raw_data_q}
grid_cache=${grid_cache_q}
output_path=${output_q}
session=${session_q}
gpu_devices=${gpu_devices_q}
max_workers=${max_workers_q}
use_cache=${use_cache_q}

mkdir -p \"\$(dirname \"\$root\")\"
if [ ! -d \"\$root/.git\" ]; then
  git clone \"\$repo\" \"\$root\"
fi
cd \"\$root\"
git config --global --add safe.directory \"\$root\" || true
git fetch origin --prune
git checkout -B \"\$branch\" \"origin/\$branch\"
git status --short --branch
git rev-parse --short HEAD

mkdir -p \"\$cache_root\" \"\$(dirname \"\$grid_cache\")\" \"\$(dirname \"\$output_path\")\"
if [ ! -d \"\$raw_data\" ]; then
  echo \"ERROR: raw data path not found: \$raw_data\" >&2
  exit 2
fi

tmux kill-session -t \"\$session\" 2>/dev/null || true
cache_flag=--no-cache
if [ \"\$use_cache\" = \"1\" ]; then
  cache_flag=
fi
tmux new-session -d -s \"\$session\" \"
set -Eeuo pipefail
cd \\\"\$root\\\"
source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
conda activate catk
export SMART_CACHE_ROOT=\\\"\$cache_root\\\"
export CUDA_VISIBLE_DEVICES=\\\"\$gpu_devices\\\"
python -m src.smart.tokens.trajtok \\
  --raw-data-path \\\"\$raw_data\\\" \\
  --traj-data-path \\\"\$grid_cache\\\" \\
  --output-path \\\"\$output_path\\\" \\
  --use-grid-stats \\
  --gpu-devices \\\"\$gpu_devices\\\" \\
  --grid-stats-worker-backend process \\
  --max-workers \\\"\$max_workers\\\" \\
  \$cache_flag
\"
echo \"Started tmux session: \$session\"
echo \"Attach: kubectl exec -it -n ${NAMESPACE} ${POD} -c ${CONTAINER} -- tmux attach -t \$session\"
"

kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc "$remote_script"
