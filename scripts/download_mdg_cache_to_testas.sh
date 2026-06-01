#!/usr/bin/env bash
set -Eeuo pipefail

# Start an MDG cache download inside the existing testas pod.
# This script is intended to be run on ssh user@10.60.188.78.

NAMESPACE="${NAMESPACE:-p-pnc}"
POD="${POD:-testas}"
CONTAINER="${CONTAINER:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
BRANCH="${BRANCH:-MDG}"
REMOTE_DIR="${REMOTE_DIR:-labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache_0601}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/MDG_cache}"
NUBES_JOBS="${NUBES_JOBS:-96}"
NUBES_RETRY="${NUBES_RETRY:-3}"
NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
SESSION="${SESSION:-mdg-cache-download}"
LOG_DIR="${LOG_DIR:-/workspace/womd_v1_3/logs}"
REPLACE_SESSION="${REPLACE_SESSION:-1}"

remote_quote() {
  printf '%q' "$1"
}

kubectl_exec() {
  kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc "$1"
}

prepare_script="
set -Eeuo pipefail
mkdir -p $(remote_quote "$(dirname "$PROJECT_ROOT")")
if [ ! -d $(remote_quote "$PROJECT_ROOT")/.git ]; then
  git clone https://github.com/seulbinHwang/catk.git $(remote_quote "$PROJECT_ROOT")
fi
cd $(remote_quote "$PROJECT_ROOT")
git config --global --add safe.directory $(remote_quote "$PROJECT_ROOT") || true
git fetch origin +$(remote_quote "$BRANCH"):refs/remotes/origin/$(remote_quote "$BRANCH")
git checkout -B $(remote_quote "$BRANCH") refs/remotes/origin/$(remote_quote "$BRANCH")
bash -n scripts/download_mdg_cache_from_nubes.sh
git status --short --branch --untracked-files=no
git rev-parse --short HEAD
"

echo "[launcher] preparing $POD:$PROJECT_ROOT on branch $BRANCH"
kubectl_exec "$prepare_script"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR%/}/download_mdg_cache_from_nubes_${timestamp}.log"

inner="
set -Eeuo pipefail
cd $(remote_quote "$PROJECT_ROOT")
mkdir -p $(remote_quote "$LOG_DIR")
export NUBES_GATEWAY_ADDRESS=$(remote_quote "$NUBES_GATEWAY_ADDRESS")
export NUBES_JOBS=$(remote_quote "$NUBES_JOBS")
export NUBES_RETRY=$(remote_quote "$NUBES_RETRY")
echo '[download-start]' \$(date '+%F %T') pod=$(remote_quote "$POD") remote=$(remote_quote "$REMOTE_DIR") local=$(remote_quote "$CACHE_ROOT") jobs=$(remote_quote "$NUBES_JOBS") | tee -a $(remote_quote "$log_file")
bash scripts/download_mdg_cache_from_nubes.sh $(remote_quote "$REMOTE_DIR") $(remote_quote "$CACHE_ROOT") 2>&1 | tee -a $(remote_quote "$log_file")
status=\${PIPESTATUS[0]}
echo '[download-exit]' \$(date '+%F %T') status=\$status | tee -a $(remote_quote "$log_file")
exec bash
"
tmux_command="bash -lc $(remote_quote "$inner")"

launcher="
set -Eeuo pipefail
mkdir -p $(remote_quote "$LOG_DIR")
if [ $(remote_quote "$REPLACE_SESSION") = '1' ] && tmux has-session -t $(remote_quote "$SESSION") 2>/dev/null; then
  tmux kill-session -t $(remote_quote "$SESSION")
fi
tmux new-session -d -s $(remote_quote "$SESSION") -c $(remote_quote "$PROJECT_ROOT") $(remote_quote "$tmux_command")
echo '[launcher] started tmux session: $(remote_quote "$SESSION")'
echo '[launcher] log: $(remote_quote "$log_file")'
"

echo "[launcher] starting download in $POD tmux session $SESSION"
kubectl_exec "$launcher"

cat <<EOF
[launcher] attach:
  kubectl exec -it -n $NAMESPACE $POD -c $CONTAINER -- tmux attach -t $SESSION
[launcher] tail log:
  kubectl exec -n $NAMESPACE $POD -c $CONTAINER -- bash -lc 'tail -f $(remote_quote "$log_file")'
EOF
