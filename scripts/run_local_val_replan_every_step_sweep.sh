#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CACHE_ROOT="${CACHE_ROOT:-/media/user/E/dataset/womd_v1_3/SMART_cache}"
CKPT_PATH="${CKPT_PATH:-/media/user/E/projects/catk/wandb_model/flow_semi_continuous_pretrain_h1006/epoch_last.ckpt}"
CATK_PYTHON="${CATK_PYTHON:-/media/user/E/miniforge/envs/catk/bin/python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${REPO_ROOT}/logs/local_val_replan_every_step_sweep/${RUN_STAMP}}"
SUMMARY_LOG="${SWEEP_LOG_DIR}/summary.log"
LOCK_FILE="${REPO_ROOT}/logs/local_val_replan_every_step_sweep/.run_local_val_replan_every_step_sweep.lock"
SWEEP_ROOT="${REPO_ROOT}/logs/local_val_replan_every_step_sweep"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_COMMIT_SHORT="$(git rev-parse --short HEAD)"

. "${SCRIPT_DIR}/_wandb_env.sh"
require_wandb_env

if [[ ! -x "${CATK_PYTHON}" ]]; then
  echo "catk python not found: ${CATK_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

mkdir -p "${SWEEP_LOG_DIR}"
mkdir -p "$(dirname "${LOCK_FILE}")"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another run_local_val_replan_every_step_sweep.sh process is already running: ${LOCK_FILE}" >&2
  exit 1
fi

COMMON_ARGS=(
  -m src.run
  experiment=local_val_flow
  trainer=default
  trainer.accelerator=gpu
  trainer.devices=1
  trainer.strategy=auto
  "paths.cache_root=${CACHE_ROOT}"
  "ckpt_path=${CKPT_PATH}"
)

{
  echo "Sweep type: local_val_flow replan_every_step comparison"
  echo "Git commit: ${GIT_COMMIT}"
  echo "Git short: ${GIT_COMMIT_SHORT}"
  echo "Cache root: ${CACHE_ROOT}"
  echo "Checkpoint: ${CKPT_PATH}"
  echo "Python: ${CATK_PYTHON}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "WANDB_ENTITY: ${WANDB_ENTITY_VALUE}"
  echo "WANDB_PROJECT: ${WANDB_PROJECT_VALUE}"
  echo "Baseline: current smart_flow.yaml default + local_val_flow runtime settings"
  echo
} | tee -a "${SUMMARY_LOG}"

find_previous_success_log() {
  local task_name="$1"
  find "${SWEEP_ROOT}" -mindepth 2 -maxdepth 2 -type f -name "${task_name}.log" ! -path "${SWEEP_LOG_DIR}/*" -print 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if grep -q "Exit code: 0" "${candidate}"; then
          printf '%s\n' "${candidate}"
          break
        fi
      done
}

run_if_needed() {
  local task_name="$1"
  shift
  local previous_success_log
  previous_success_log="$(find_previous_success_log "${task_name}" || true)"

  if [[ -n "${previous_success_log}" ]]; then
    echo "Skipping ${task_name}: previously completed successfully at ${previous_success_log}" \
      | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  run_experiment "${task_name}" "$@"
}

run_experiment() {
  local task_name="$1"
  shift
  local task_log="${SWEEP_LOG_DIR}/${task_name}.log"
  local start_ts
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  local overrides=("$@")
  echo
  echo "============================================================"
  echo "Running ${task_name}"
  echo "Log file: ${task_log}"
  printf 'Overrides:'
  for arg in "${overrides[@]}"; do
    printf ' %q' "${arg}"
  done
  echo
  echo "============================================================"

  {
    echo "============================================================"
    echo "Start: ${start_ts}"
    echo "Task: ${task_name}"
    echo "Git short: ${GIT_COMMIT_SHORT}"
    echo "Log file: ${task_log}"
    printf 'Overrides:'
    for arg in "${overrides[@]}"; do
      printf ' %q' "${arg}"
    done
    echo
    echo "============================================================"
  } | tee -a "${SUMMARY_LOG}" "${task_log}"

  set +e
  env \
    -u PYTHONPATH \
    -u LD_LIBRARY_PATH \
    -u EXP_PATH \
    -u ISAACSIM_DOCKER_DIR \
    -u ISAAC_PATH \
    -u CARB_APP_PATH \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    WANDB_API_KEY="${WANDB_API_KEY}" \
    WANDB_ENTITY="${WANDB_ENTITY_VALUE}" \
    WANDB_PROJECT="${WANDB_PROJECT_VALUE}" \
    WANDB_BASE_URL="${WANDB_BASE_URL}" \
    "${CATK_PYTHON}" \
    "${COMMON_ARGS[@]}" \
    "task_name=${task_name}" \
    "${overrides[@]}" 2>&1 \
    | awk '
        /tensorflow\/compiler\/xla\/stream_executor\/cuda\/cuda_driver\.cc:1330/ { next }
        /CUDA_ERROR_NOT_INITIALIZED: initialization error/ { next }
        { print }
      ' \
    | tee -a "${task_log}"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  local end_ts
  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo
    echo "End: ${end_ts}"
    echo "Task: ${task_name}"
    echo "Exit code: ${cmd_status}"
    echo
  } | tee -a "${SUMMARY_LOG}" "${task_log}"

  if [[ ${cmd_status} -ne 0 ]]; then
    echo "Experiment failed: ${task_name}" | tee -a "${SUMMARY_LOG}"
    exit "${cmd_status}"
  fi
}

run_if_needed "flow_local_val_replan_true"

run_if_needed \
  "flow_local_val_replan_false" \
  "model.model_config.decoder.lqr_commit.replan_every_step=False"

echo
echo "All 2 replan_every_step experiments finished."
echo "Sweep logs: ${SWEEP_LOG_DIR}"
