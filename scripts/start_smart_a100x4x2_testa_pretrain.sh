#!/usr/bin/env bash
# Start SMART branch pretrain on existing testa/testaa A100 x4 pods.
#
# The launcher only uses kubectl exec/cp and tmux inside already-running pods.
# It never creates, deletes, or restarts pods.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

shq() {
  printf "%q" "$1"
}

NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
PODS="${PODS:-testa testaa}"
BRANCH="${BRANCH:-SMART}"
REPO_URL="${REPO_URL:-https://github.com/seulbinHwang/catk.git}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_branch_a100x4x2_pretrain}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_RAW_cache}"
LOG_DIR="${LOG_DIR:-/mnt/nuplan/projects/catk/logs/tmux_smart_a100x4x2}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${TASK_NAME:-smart_pretrain_a100x4x2_smart_raw_fast_rmm_${RUN_STAMP}}"
SESSION="${SESSION:-catk-smart-a100x4x2-pretrain}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29531}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
EXPERIMENT="${EXPERIMENT:-pre_bc_a100x4x2}"
ACTION="${ACTION:-fit}"
REPLACE="${REPLACE:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
CKPT_PATH="${CKPT_PATH:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
WANDB_MODE="${WANDB_MODE:-online}"

PROJECT_ROOT_Q="$(shq "$PROJECT_ROOT")"
CACHE_ROOT_Q="$(shq "$CACHE_ROOT")"
TASK_NAME_Q="$(shq "$TASK_NAME")"
EXPERIMENT_Q="$(shq "$EXPERIMENT")"
ACTION_Q="$(shq "$ACTION")"
MASTER_PORT_Q="$(shq "$MASTER_PORT")"
TRAIN_BATCH_SIZE_Q="$(shq "$TRAIN_BATCH_SIZE")"
VAL_BATCH_SIZE_Q="$(shq "$VAL_BATCH_SIZE")"
TEST_BATCH_SIZE_Q="$(shq "$TEST_BATCH_SIZE")"
LIMIT_TRAIN_BATCHES_Q="$(shq "$LIMIT_TRAIN_BATCHES")"
LIMIT_VAL_BATCHES_Q="$(shq "$LIMIT_VAL_BATCHES")"
MAX_EPOCHS_Q="$(shq "$MAX_EPOCHS")"
CKPT_PATH_Q="$(shq "$CKPT_PATH")"
EXTRA_HYDRA_OVERRIDES_Q="$(shq "$EXTRA_HYDRA_OVERRIDES")"
WANDB_MODE_Q="$(shq "$WANDB_MODE")"
BRANCH_Q="$(shq "$BRANCH")"
REPO_URL_Q="$(shq "$REPO_URL")"
LOG_DIR_Q="$(shq "$LOG_DIR")"
SESSION_Q="$(shq "$SESSION")"

read -r -a POD_ARRAY <<< "$PODS"
if [[ "${#POD_ARRAY[@]}" -lt 2 ]]; then
  echo "[launcher] PODS must contain two pods, got: ${PODS}" >&2
  exit 1
fi
NNODES="${#POD_ARRAY[@]}"
MASTER_POD="${POD_ARRAY[0]}"
MASTER_ADDR="$(kubectl get pod "$MASTER_POD" -n "$NAMESPACE" -o jsonpath='{.status.podIP}')"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "[launcher] failed to resolve master pod IP for ${MASTER_POD}" >&2
  exit 1
fi
MASTER_ADDR_Q="$(shq "$MASTER_ADDR")"
NNODES_Q="$(shq "$NNODES")"
NPROC_PER_NODE_Q="$(shq "$NPROC_PER_NODE")"

REMOTE_SCRIPT="/tmp/smart_a100x4x2_pretrain_node.sh"
REMOTE_SCRIPT_Q="$(shq "$REMOTE_SCRIPT")"

prepare_pod() {
  local pod="$1"
  echo "[launcher] preparing ${pod}"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
    set -Eeuo pipefail
    if [[ ! -d ${PROJECT_ROOT_Q}/.git ]]; then
      rm -rf ${PROJECT_ROOT_Q}
      git clone ${REPO_URL_Q} ${PROJECT_ROOT_Q}
    fi
    cd ${PROJECT_ROOT_Q}
    git fetch origin --prune
    git checkout -B ${BRANCH_Q} origin/${BRANCH}
    git reset --hard origin/${BRANCH}
    for d in training validation testing validation_tfrecords_splitted; do
      test -d ${CACHE_ROOT_Q}/\$d
    done
    mkdir -p ${LOG_DIR_Q}/${TASK_NAME_Q}
  "
  kubectl cp -n "$NAMESPACE" -c "$CONTAINER" \
    "$REPO_ROOT/scripts/smart_a100x4x2_pretrain_node.sh" \
    "$pod:$REMOTE_SCRIPT"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- chmod +x "$REMOTE_SCRIPT"
}

start_pod() {
  local pod="$1"
  local rank="$2"
  local tmux_log="${LOG_DIR}/${TASK_NAME}/${pod}.tmux.log"
  local remote_run="/tmp/smart_a100x4x2_run_${TASK_NAME}_${pod}.sh"
  local local_run
  local_run="$(mktemp)"
  cat > "$local_run" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export PROJECT_ROOT=${PROJECT_ROOT_Q}
export CACHE_ROOT=${CACHE_ROOT_Q}
export TASK_NAME=${TASK_NAME_Q}
export EXPERIMENT=${EXPERIMENT_Q}
export ACTION=${ACTION_Q}
export NNODES=${NNODES_Q}
export NPROC_PER_NODE=${NPROC_PER_NODE_Q}
export NODE_RANK=$(shq "$rank")
export MASTER_ADDR=${MASTER_ADDR_Q}
export MASTER_PORT=${MASTER_PORT_Q}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE_Q}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE_Q}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE_Q}
export LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES_Q}
export LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES_Q}
export MAX_EPOCHS=${MAX_EPOCHS_Q}
export CKPT_PATH=${CKPT_PATH_Q}
export EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES_Q}
export WANDB_MODE=${WANDB_MODE_Q}
bash ${REMOTE_SCRIPT_Q} 2>&1 | tee $(shq "$tmux_log")
EOF
  chmod +x "$local_run"
  kubectl cp -n "$NAMESPACE" -c "$CONTAINER" "$local_run" "$pod:$remote_run"
  rm -f "$local_run"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- chmod +x "$remote_run"

  if [[ "$REPLACE" == "1" ]]; then
    kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- \
      tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
  echo "[launcher] starting ${pod} rank=${rank}"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
    set -Eeuo pipefail
    mkdir -p ${LOG_DIR_Q}/${TASK_NAME_Q}
    tmux new-session -d -s ${SESSION_Q} bash $(shq "$remote_run")
  "
}

echo "[launcher] task=${TASK_NAME}"
echo "[launcher] pods=${PODS}"
echo "[launcher] branch=${BRANCH}"
echo "[launcher] project_root=${PROJECT_ROOT}"
echo "[launcher] cache_root=${CACHE_ROOT}"
echo "[launcher] master=${MASTER_POD} ${MASTER_ADDR}:${MASTER_PORT}"
echo "[launcher] batch train/val/test=${TRAIN_BATCH_SIZE}/${VAL_BATCH_SIZE}/${TEST_BATCH_SIZE}"

for pod in "${POD_ARRAY[@]}"; do
  prepare_pod "$pod"
done

for i in "${!POD_ARRAY[@]}"; do
  start_pod "${POD_ARRAY[$i]}" "$i"
done

cat <<EOF
[launcher] started.

Attach:
  kubectl exec -it -n ${NAMESPACE} ${MASTER_POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}

Logs:
  ${LOG_DIR}/${TASK_NAME}/${MASTER_POD}.tmux.log

Stop training session only:
  for pod in ${PODS}; do kubectl exec -n ${NAMESPACE} \$pod -c ${CONTAINER} -- tmux send-keys -t ${SESSION} C-c; done
EOF
