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
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${REPO_ROOT}/logs/local_val_lqr_ablation_sweep/${RUN_STAMP}}"
SUMMARY_LOG="${SWEEP_LOG_DIR}/summary.log"
LOCK_FILE="${REPO_ROOT}/logs/local_val_lqr_ablation_sweep/.run_local_val_lqr_ablation_sweep.lock"
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
  echo "another run_local_val_lqr_ablation_sweep.sh process is already running: ${LOCK_FILE}" >&2
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
  echo "Sweep type: local_val_flow lqr ablation comparison"
  echo "Git commit: ${GIT_COMMIT}"
  echo "Git short: ${GIT_COMMIT_SHORT}"
  echo "Cache root: ${CACHE_ROOT}"
  echo "Checkpoint: ${CKPT_PATH}"
  echo "Python: ${CATK_PYTHON}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "WANDB_ENTITY: ${WANDB_ENTITY_VALUE}"
  echo "WANDB_PROJECT: ${WANDB_PROJECT_VALUE}"
  echo "Baseline: configs/model/smart_flow.yaml + local_val_flow runtime settings"
  echo
} | tee -a "${SUMMARY_LOG}"

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

run_experiment \
  "flow_local_val_no_smooth" \
  "model.model_config.decoder.lqr_commit.velocity_smooth_lambda=0.0" \
  "model.model_config.decoder.lqr_commit.curvature_smooth_lambda=0.0"

run_experiment \
  "flow_local_val_lqr_penalty" \
  "model.model_config.decoder.lqr_commit.longitudinal_r=1.0" \
  "model.model_config.decoder.lqr_commit.lateral_q_kappa=0.1" \
  "model.model_config.decoder.lqr_commit.lateral_r=1.0"

run_experiment \
  "flow_local_val_tau_delay" \
  "model.model_config.decoder.lqr_commit.accel_tau_s=0.2" \
  "model.model_config.decoder.lqr_commit.curvature_tau_s=0.05"

run_experiment \
  "flow_local_val_no_clip" \
  "model.model_config.decoder.lqr_commit.clip_longitudinal_command=False" \
  "model.model_config.decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=False"

run_experiment \
  "flow_local_val_horizon3" \
  "model.model_config.decoder.lqr_commit.horizon_steps=3"

run_experiment \
  "flow_local_val_horizon4" \
  "model.model_config.decoder.lqr_commit.horizon_steps=4"

run_experiment \
  "flow_local_val_horizon7" \
  "model.model_config.decoder.lqr_commit.horizon_steps=7"

echo
echo "All 7 LQR ablation experiments finished."
echo "Sweep logs: ${SWEEP_LOG_DIR}"
