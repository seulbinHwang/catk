#!/usr/bin/env bash
# Start SMART branch pretrain on existing testa/testaa A100 x4 pods.
#
# The launcher only uses kubectl exec/cp and tmux inside already-running pods.
# It never creates, deletes, or restarts pods.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

REMOTE_SCRIPT="/tmp/smart_a100x4x2_pretrain_node.sh"

prepare_pod() {
  local pod="$1"
  echo "[launcher] preparing ${pod}"
  kubectl exec -n "$NAMESPACE" "$pod" -c "$CONTAINER" -- bash -lc "
    set -Eeuo pipefail
    if [[ ! -d ${PROJECT_ROOT@Q}/.git ]]; then
      rm -rf ${PROJECT_ROOT@Q}
      git clone ${REPO_URL@Q} ${PROJECT_ROOT@Q}
    fi
    cd ${PROJECT_ROOT@Q}
    git fetch origin --prune
    git checkout -B ${BRANCH@Q} origin/${BRANCH}
    git reset --hard origin/${BRANCH}
    for d in training validation testing validation_tfrecords_splitted; do
      test -d ${CACHE_ROOT@Q}/\$d
    done
    mkdir -p ${LOG_DIR@Q}/${TASK_NAME@Q}
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
export PROJECT_ROOT=${PROJECT_ROOT@Q}
export CACHE_ROOT=${CACHE_ROOT@Q}
export TASK_NAME=${TASK_NAME@Q}
export EXPERIMENT=${EXPERIMENT@Q}
export ACTION=${ACTION@Q}
export NNODES=${NNODES@Q}
export NPROC_PER_NODE=${NPROC_PER_NODE@Q}
export NODE_RANK=${rank@Q}
export MASTER_ADDR=${MASTER_ADDR@Q}
export MASTER_PORT=${MASTER_PORT@Q}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE@Q}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE@Q}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE@Q}
export LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES@Q}
export LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES@Q}
export MAX_EPOCHS=${MAX_EPOCHS@Q}
export CKPT_PATH=${CKPT_PATH@Q}
export EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES@Q}
export WANDB_MODE=${WANDB_MODE@Q}
bash ${REMOTE_SCRIPT@Q} 2>&1 | tee ${tmux_log@Q}
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
    mkdir -p ${LOG_DIR@Q}/${TASK_NAME@Q}
    tmux new-session -d -s ${SESSION@Q} bash ${remote_run@Q}
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
