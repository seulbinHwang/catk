#!/usr/bin/env bash
# Start MDG A100x7 pretrain inside the existing testas pod.
#
# This script is intended to be run on ssh user@10.60.188.78.
set -Eeuo pipefail

NAMESPACE="${NAMESPACE:-p-pnc}"
POD="${POD:-testas}"
CONTAINER="${CONTAINER:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
BRANCH="${BRANCH:-MDG}"
SESSION="${SESSION:-mdg-pretrain-a100x7}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/testas_mdg_pretrain_a100x7}"
REPLACE_SESSION="${REPLACE_SESSION:-0}"

CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/MDG_cache}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-64}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
WANDB_MODE="${WANDB_MODE:-online}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1680}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
TRAIN_MEMORY_BALANCED_BATCHING="${TRAIN_MEMORY_BALANCED_BATCHING:-true}"
TRAIN_MEMORY_BALANCE_METADATA_CACHE="${TRAIN_MEMORY_BALANCE_METADATA_CACHE:-}"
TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS="${TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS:-32}"
TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING="${TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING:-true}"
MASTER_PORT="${MASTER_PORT:-29671}"
TASK_NAME="${TASK_NAME:-mdg_wosac_pretrain_testas_a100x7_bs${TRAIN_BATCH_SIZE}}"
CATK_AUTO_RESUME="${CATK_AUTO_RESUME:-false}"
CATK_RESUME_TASK_NAME="${CATK_RESUME_TASK_NAME:-}"
CATK_RESUME_CHECKPOINT_NAME="${CATK_RESUME_CHECKPOINT_NAME:-epoch_last.ckpt}"
CATK_RESUME_REQUIRE_CHECKPOINT="${CATK_RESUME_REQUIRE_CHECKPOINT:-true}"
CATK_HYDRA_OVERRIDES="${CATK_HYDRA_OVERRIDES:-}"

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
bash -n scripts/mdg_pretrain_a100x7.sh
git status --short --branch --untracked-files=no
git rev-parse --short HEAD
"

echo "[launcher] preparing $POD:$PROJECT_ROOT on branch $BRANCH"
kubectl_exec "$prepare_script"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR%/}/${TASK_NAME}_${timestamp}.log"

inner="
set -Eeuo pipefail
cd $(remote_quote "$PROJECT_ROOT")
mkdir -p $(remote_quote "$LOG_DIR")
export CACHE_ROOT=$(remote_quote "$CACHE_ROOT")
export TRAIN_BATCH_SIZE=$(remote_quote "$TRAIN_BATCH_SIZE")
export VAL_BATCH_SIZE=$(remote_quote "$VAL_BATCH_SIZE")
export TEST_BATCH_SIZE=$(remote_quote "$TEST_BATCH_SIZE")
export MAX_EPOCHS=$(remote_quote "$MAX_EPOCHS")
export LIMIT_TRAIN_BATCHES=$(remote_quote "$LIMIT_TRAIN_BATCHES")
export LIMIT_VAL_BATCHES=$(remote_quote "$LIMIT_VAL_BATCHES")
export DATA_NUM_WORKERS=$(remote_quote "$DATA_NUM_WORKERS")
export PRECISION=$(remote_quote "$PRECISION")
export WANDB_MODE=$(remote_quote "$WANDB_MODE")
export VAL_CLOSED_LOOP=$(remote_quote "$VAL_CLOSED_LOOP")
export N_BATCH_SIM_AGENTS_METRIC=$(remote_quote "$N_BATCH_SIM_AGENTS_METRIC")
export SCORER_SCENE_NUM=$(remote_quote "$SCORER_SCENE_NUM")
export CHECKPOINT_MONITOR=$(remote_quote "$CHECKPOINT_MONITOR")
export CHECKPOINT_MODE=$(remote_quote "$CHECKPOINT_MODE")
export TRAIN_MEMORY_BALANCED_BATCHING=$(remote_quote "$TRAIN_MEMORY_BALANCED_BATCHING")
export TRAIN_MEMORY_BALANCE_METADATA_CACHE=$(remote_quote "$TRAIN_MEMORY_BALANCE_METADATA_CACHE")
export TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS=$(remote_quote "$TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS")
export TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING=$(remote_quote "$TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING")
export MASTER_PORT=$(remote_quote "$MASTER_PORT")
export TASK_NAME=$(remote_quote "$TASK_NAME")
export CATK_AUTO_RESUME=$(remote_quote "$CATK_AUTO_RESUME")
export CATK_RESUME_TASK_NAME=$(remote_quote "$CATK_RESUME_TASK_NAME")
export CATK_RESUME_CHECKPOINT_NAME=$(remote_quote "$CATK_RESUME_CHECKPOINT_NAME")
export CATK_RESUME_REQUIRE_CHECKPOINT=$(remote_quote "$CATK_RESUME_REQUIRE_CHECKPOINT")
export CATK_HYDRA_OVERRIDES=$(remote_quote "$CATK_HYDRA_OVERRIDES")
echo '[pretrain-start]' \$(date '+%F %T') task=$(remote_quote "$TASK_NAME") train_bs=$(remote_quote "$TRAIN_BATCH_SIZE") cache=$(remote_quote "$CACHE_ROOT") | tee -a $(remote_quote "$log_file")
bash scripts/mdg_pretrain_a100x7.sh 2>&1 | tee -a $(remote_quote "$log_file")
status=\${PIPESTATUS[0]}
echo '[pretrain-exit]' \$(date '+%F %T') status=\$status | tee -a $(remote_quote "$log_file")
exec bash
"
tmux_command="bash -lc $(remote_quote "$inner")"

launcher="
set -Eeuo pipefail
mkdir -p $(remote_quote "$LOG_DIR")
if tmux has-session -t $(remote_quote "$SESSION") 2>/dev/null; then
  if [ $(remote_quote "$REPLACE_SESSION") = '1' ]; then
    tmux kill-session -t $(remote_quote "$SESSION")
  else
    echo '[launcher] tmux session already exists: $(remote_quote "$SESSION")'
    echo '[launcher] set REPLACE_SESSION=1 to replace it.'
    exit 3
  fi
fi
tmux new-session -d -s $(remote_quote "$SESSION") -c $(remote_quote "$PROJECT_ROOT") $(remote_quote "$tmux_command")
echo '[launcher] started tmux session: $(remote_quote "$SESSION")'
echo '[launcher] log: $(remote_quote "$log_file")'
"

echo "[launcher] starting pretrain in $POD tmux session $SESSION"
kubectl_exec "$launcher"

cat <<EOF
[launcher] attach:
  kubectl exec -it -n $NAMESPACE $POD -c $CONTAINER -- tmux attach -t $SESSION
[launcher] tail log:
  kubectl exec -n $NAMESPACE $POD -c $CONTAINER -- bash -lc 'tail -f $(remote_quote "$log_file")'
EOF
