#!/usr/bin/env bash
# Run A100x7 execution-context pretrain on the existing testas pod with an
# in-pod OOM retry supervisor. This script is intended to be executed inside
# the testas container from /mnt/nuplan/projects/catk.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
BRANCH="${BRANCH:-semi_control_rolling_fd}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/nuplan/projects/catk/logs}"
EXPERIMENT="${EXPERIMENT:-pre_bc_flow_control_h100x4x2_execctx_balanced}"
TASK_NAME="${TASK_NAME:-flow_control_space_pretrain_a100x7_testas_execctx_prefix_balanced_lr6e-4_bs17_remote_oomretry}"
SESSION="${SESSION:-catk-control-pretrain-a100x7-testas-execctx-balanced-bs17-remote-retry}"
INITIAL_BS="${INITIAL_BS:-17}"
MIN_BS="${MIN_BS:-1}"
OOM_STEP="${OOM_STEP:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-32}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-32}"
METADATA_CACHE="${METADATA_CACHE:-${REMOTE_LOG_DIR}/dataset_metadata/womd_training_memory_balance_v1.pt}"
METADATA_NUM_WORKERS="${METADATA_NUM_WORKERS:-8}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
CATK_REMOTE_PYTHON="${CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
REPLACE=0
STOP=0
PREBUILD_METADATA=0

usage() {
  cat <<'USAGE'
Usage: launch_pre_bc_flow_control_a100x7_testas_remote_oom_retry.sh [options]

Options:
  --replace                 Kill the existing tmux session with the same name first.
  --stop                    Stop this script's tmux session and matching train processes.
  --prebuild-metadata       Build memory-balanced sampler metadata before launching.
  --initial-bs N            Starting per-GPU train batch size. Default: 17.
  --min-bs N                Minimum per-GPU train batch size. Default: 1.
  --task-name NAME          Hydra/W&B task name.
  --session NAME            tmux session name.
  --branch NAME             Git branch to checkout/pull before launch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replace) REPLACE=1; shift ;;
    --stop) STOP=1; shift ;;
    --prebuild-metadata) PREBUILD_METADATA=1; shift ;;
    --initial-bs) INITIAL_BS="$2"; shift 2 ;;
    --min-bs) MIN_BS="$2"; shift 2 ;;
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[launcher] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if (( INITIAL_BS < 1 || MIN_BS < 1 || OOM_STEP < 1 )); then
  echo "[launcher] INITIAL_BS, MIN_BS, and OOM_STEP must be positive." >&2
  exit 2
fi
if (( MIN_BS > INITIAL_BS )); then
  echo "[launcher] MIN_BS must be <= INITIAL_BS." >&2
  exit 2
fi

SAFE_TASK_NAME="${TASK_NAME//\//_}"
RUN_ROOT="${REMOTE_LOG_DIR%/}/tmux_a100x7_testas_pretrain/${SAFE_TASK_NAME}"
TMUX_LOG="${RUN_ROOT}/$(hostname).tmux.log"
STATUS_FILE="${RUN_ROOT}/$(hostname).supervisor_status"
ENV_FILE="${RUN_ROOT}/supervisor.env"
SUPERVISOR_FILE="${RUN_ROOT}/supervisor.sh"
MONITOR_FILE="${RUN_ROOT}/monitor.sh"

stop_existing() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "[launcher] stopped tmux session $SESSION"
  else
    echo "[launcher] tmux session not found: $SESSION"
  fi

  mapfile -t pids < <(
    pgrep -f "src.run .*task_name=${TASK_NAME}|torchrun .*task_name=${TASK_NAME}" 2>/dev/null || true
  )
  if (( ${#pids[@]} > 0 )); then
    echo "[launcher] terminating train processes for $TASK_NAME: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 10
    mapfile -t pids < <(
      pgrep -f "src.run .*task_name=${TASK_NAME}|torchrun .*task_name=${TASK_NAME}" 2>/dev/null || true
    )
    if (( ${#pids[@]} > 0 )); then
      echo "[launcher] force killing train processes for $TASK_NAME: ${pids[*]}"
      kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
  fi
}

if (( STOP )); then
  stop_existing
  exit 0
fi

cd "$PROJECT_ROOT"
git config --global --add safe.directory "$PROJECT_ROOT" || true
git fetch origin --prune "+${BRANCH}:refs/remotes/origin/${BRANCH}"
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "origin/${BRANCH}"
fi
git pull --ff-only origin "$BRANCH"

if (( PREBUILD_METADATA )); then
  "$CATK_REMOTE_PYTHON" tools/build_memory_balance_metadata.py \
    --raw-dir "${CACHE_ROOT%/}/training" \
    --cache-path "$METADATA_CACHE" \
    --num-workers "$METADATA_NUM_WORKERS"
fi
if [[ ! -f "$METADATA_CACHE" ]]; then
  echo "[launcher] missing metadata cache: $METADATA_CACHE" >&2
  echo "[launcher] rerun with --prebuild-metadata or set METADATA_CACHE." >&2
  exit 2
fi

if (( REPLACE )); then
  stop_existing
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launcher] tmux session already exists: $SESSION" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"
cat > "$ENV_FILE" <<CATK_ENV
export PROJECT_ROOT='$PROJECT_ROOT'
export CACHE_ROOT='$CACHE_ROOT'
export REMOTE_LOG_DIR='$REMOTE_LOG_DIR'
export EXPERIMENT='$EXPERIMENT'
export TASK_NAME='$TASK_NAME'
export INITIAL_BS='$INITIAL_BS'
export MIN_BS='$MIN_BS'
export OOM_STEP='$OOM_STEP'
export NPROC_PER_NODE='$NPROC_PER_NODE'
export LEARNING_RATE='$LEARNING_RATE'
export VAL_BATCH_SIZE='$VAL_BATCH_SIZE'
export N_ROLLOUT_CLOSED_VAL='$N_ROLLOUT_CLOSED_VAL'
export CHECK_VAL_EVERY_N_EPOCH='$CHECK_VAL_EVERY_N_EPOCH'
export METADATA_CACHE='$METADATA_CACHE'
export CATK_REMOTE_PYTHON='$CATK_REMOTE_PYTHON'
export LIMIT_TRAIN_BATCHES='$LIMIT_TRAIN_BATCHES'
export LIMIT_VAL_BATCHES='$LIMIT_VAL_BATCHES'
export MAX_EPOCHS='$MAX_EPOCHS'
export EXTRA_HYDRA_OVERRIDES='$EXTRA_HYDRA_OVERRIDES'
export RUN_ROOT='$RUN_ROOT'
export STATUS_FILE='$STATUS_FILE'
CATK_ENV

cat > "$SUPERVISOR_FILE" <<'CATK_SUPERVISOR'
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/supervisor.env"

export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"

if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate "${CATK_CONDA_ENV:-catk}" 2>/dev/null || true
fi

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
: > "$STATUS_FILE"

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

find_latest_ckpt() {
  { ls -t "${REMOTE_LOG_DIR%/}/${TASK_NAME}"/runs/*/checkpoints/epoch_last.ckpt 2>/dev/null || true
    ls -t "${REMOTE_LOG_DIR%/}/${TASK_NAME}"/runs/*/checkpoints/last.ckpt 2>/dev/null || true
  } | head -1
}

cleanup_task_processes() {
  mapfile -t pids < <(
    pgrep -f "src.run .*task_name=${TASK_NAME}|torchrun .*task_name=${TASK_NAME}" 2>/dev/null || true
  )
  if (( ${#pids[@]} > 0 )); then
    echo "[remote-retry] terminating leftover task processes: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 10
    mapfile -t pids < <(
      pgrep -f "src.run .*task_name=${TASK_NAME}|torchrun .*task_name=${TASK_NAME}" 2>/dev/null || true
    )
    if (( ${#pids[@]} > 0 )); then
      echo "[remote-retry] force killing leftover task processes: ${pids[*]}"
      kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
  fi
}

bs="$INITIAL_BS"
attempt=0
while (( bs >= MIN_BS )); do
  attempt=$((attempt + 1))
  attempt_log="${RUN_ROOT}/attempt_${attempt}_bs${bs}.log"
  ckpt_path="$(find_latest_ckpt || true)"

  overrides=(
    "experiment=${EXPERIMENT}"
    "action=fit"
    "trainer=ddp"
    "trainer.devices=${NPROC_PER_NODE}"
    "trainer.num_nodes=1"
    "trainer.enable_progress_bar=true"
    "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
    "trainer.use_distributed_sampler=false"
    "paths.cache_root=${CACHE_ROOT}"
    "paths.log_dir=${REMOTE_LOG_DIR}"
    "task_name=${TASK_NAME}"
    "data.train_batch_size=${bs}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.train_memory_balanced_batches=true"
    "data.train_memory_balance_metadata_cache=${METADATA_CACHE}"
    "data.train_memory_balance_build_on_missing=false"
    "model.model_config.lr=${LEARNING_RATE}"
    "model.model_config.n_rollout_closed_val=${N_ROLLOUT_CLOSED_VAL}"
  )
  [[ -n "$ckpt_path" ]] && overrides+=("ckpt_path=${ckpt_path}")
  [[ -n "$LIMIT_TRAIN_BATCHES" ]] && overrides+=("trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
  [[ -n "$LIMIT_VAL_BATCHES" ]] && overrides+=("trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
  [[ -n "$MAX_EPOCHS" ]] && overrides+=("trainer.max_epochs=${MAX_EPOCHS}")
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    # shellcheck disable=SC2206
    extra_overrides=( $EXTRA_HYDRA_OVERRIDES )
    overrides+=("${extra_overrides[@]}")
  fi

  echo
  echo "[remote-retry] attempt=${attempt} bs=${bs} started=$(date '+%F %T')"
  [[ -n "$ckpt_path" ]] && echo "[remote-retry] resume ckpt=${ckpt_path}"

  set +e
  torchrun --standalone --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
    -m src.run "${overrides[@]}" 2>&1 | tee -a "$attempt_log"
  status="${PIPESTATUS[0]}"
  set -e
  echo "$status" > "$STATUS_FILE"
  echo "[remote-retry] attempt=${attempt} bs=${bs} exited status=${status} at $(date '+%F %T')"

  if [[ "$status" == "0" ]]; then
    echo "[remote-retry] training completed successfully at bs=${bs}"
    exit 0
  fi

  cleanup_task_processes
  if grep -Eiq "$OOM_REGEX" "$attempt_log"; then
    next_bs=$((bs - OOM_STEP))
    echo "[remote-retry] CUDA OOM detected at bs=${bs}; next bs=${next_bs}"
    if (( next_bs < MIN_BS )); then
      echo "[remote-retry] next bs is below MIN_BS=${MIN_BS}; stopping"
      exit 1
    fi
    bs="$next_bs"
    continue
  fi

  echo "[remote-retry] non-OOM failure; see ${attempt_log}"
  exit "$status"
done

echo "[remote-retry] exhausted batch sizes down to MIN_BS=${MIN_BS}"
exit 1
CATK_SUPERVISOR
chmod +x "$SUPERVISOR_FILE"

cat > "$MONITOR_FILE" <<'CATK_MONITOR'
#!/usr/bin/env bash
set +e
source "$(dirname "$0")/supervisor.env"
while true; do
  echo
  echo "[monitor] $(date '+%F %T') pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep "${MONITOR_INTERVAL:-30}"
done
CATK_MONITOR
chmod +x "$MONITOR_FILE"

: > "$TMUX_LOG"
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "$SUPERVISOR_FILE"
tmux pipe-pane -t "$SESSION" -o "cat >> '$TMUX_LOG'"
tmux split-window -v -l 12 -t "$SESSION" "$MONITOR_FILE"
tmux select-pane -t "$SESSION"

echo "[launcher] started remote OOM-retry tmux session: $SESSION"
echo "[launcher] task_name: $TASK_NAME"
echo "[launcher] initial_bs: $INITIAL_BS, min_bs: $MIN_BS, oom_step: $OOM_STEP"
echo "[launcher] tmux log: $TMUX_LOG"
