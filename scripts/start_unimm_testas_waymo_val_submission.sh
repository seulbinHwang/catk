#!/usr/bin/env bash
set -Eeuo pipefail

# Generate and upload a Waymo Sim Agents validation submission for the UniMM
# Anchor-Based-4s model on the existing testas A100x7 pod.
#
# This script only runs commands inside an existing pod. It does not create,
# delete, or restart Kubernetes pods.

NAMESPACE="${NAMESPACE:-p-pnc}"
POD="${POD:-testas}"
CONTAINER="${CONTAINER:-main}"
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_unimm_testas_waymo_submission_topk768}"
REPO_URL="${REPO_URL:-https://github.com/seulbinHwang/catk.git}"
GIT_REF="${GIT_REF:-origin/UniMM}"
SESSION="${SESSION:-catk-unimm-testas-topk768-waymo-val-submission}"
TASK_NAME="${TASK_NAME:-unimm_lhzndj5b_v24_topk768_waymo_val_submission_testas_$(date +%Y%m%d_%H%M%S)}"
CKPT_PATH="${CKPT_PATH:-/tmp/unimm_lhzndj5b_epoch_last_v24/epoch_last.ckpt}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
LOG_ROOT="${LOG_ROOT:-/mnt/nuplan/projects/catk/logs/tmux_unimm_testas_waymo_submission/${TASK_NAME}}"
WAYMO_STORAGE_STATE_PATH="${WAYMO_STORAGE_STATE_PATH:-/mnt/nuplan/projects/catk/secrets/waymo/waymo_storage_state.json}"
DEVICES="${DEVICES:-7}"
VAL_BATCH_FALLBACKS="${VAL_BATCH_FALLBACKS:-80 72 64}"
INFERENCE_TOP_K="${INFERENCE_TOP_K:-768}"
WAYMO_UPLOAD_TIMEOUT_MS="${WAYMO_UPLOAD_TIMEOUT_MS:-7200000}"
POLL_SUBMISSION_STATUS="${POLL_SUBMISSION_STATUS:-false}"
REPLACE="${REPLACE:-false}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"

shq() {
  printf "%q" "$1"
}

run_kubectl() {
  "${KUBECTL_BIN}" "$@"
}

exec_in_pod() {
  run_kubectl exec -n "${NAMESPACE}" "${POD}" -c "${CONTAINER}" -- bash -lc "$1"
}

prepare_checkout() {
  local project_root_q repo_url_q git_ref_q
  project_root_q="$(shq "${PROJECT_ROOT}")"
  repo_url_q="$(shq "${REPO_URL}")"
  git_ref_q="$(shq "${GIT_REF}")"
  exec_in_pod "$(cat <<EOF
set -Eeuo pipefail
if [ ! -d ${project_root_q}/.git ]; then
  rm -rf ${project_root_q}
  git clone ${repo_url_q} ${project_root_q}
fi
cd ${project_root_q}
git config --global --add safe.directory ${project_root_q} || true
git fetch origin --prune
git checkout -B UniMM ${git_ref_q}
git reset --hard ${git_ref_q}
git clean -fdx
git rev-parse --short HEAD
EOF
)"
}

waymo_preflight() {
  local project_root_q log_root_py storage_state_py
  project_root_q="$(shq "${PROJECT_ROOT}")"
  log_root_py="$(LOG_ROOT_VALUE="${LOG_ROOT}" python3 - <<'PY'
import os
print(repr(os.environ["LOG_ROOT_VALUE"]))
PY
)"
  storage_state_py="$(WAYMO_STORAGE_STATE_VALUE="${WAYMO_STORAGE_STATE_PATH}" python3 - <<'PY'
import os
print(repr(os.environ["WAYMO_STORAGE_STATE_VALUE"]))
PY
)"
  exec_in_pod "$(cat <<EOF
set -Eeuo pipefail
cd ${project_root_q}
source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
conda activate catk
python - <<'PY'
from pathlib import Path

from src.utils.waymo_submission import (
    _SIM_AGENTS_CHALLENGE_NAME,
    _WaymoSubmissionRuntime,
    _WaymoSubmissionUploader,
)

storage_state = Path(${storage_state_py})
if not storage_state.is_file():
    raise FileNotFoundError(f"missing Waymo storage state: {storage_state}")

debug_root = Path(${log_root_py}) / "preflight"
debug_root.mkdir(parents=True, exist_ok=True)
dummy_archive = debug_root / "waymo_upload_preflight.tar.gz"
dummy_archive.write_bytes(b"preflight")

runtime = _WaymoSubmissionRuntime(
    archive_path=dummy_archive,
    storage_state_path=storage_state,
    output_dir=debug_root,
    challenge_url="https://waymo.com/open/challenges/2025/sim-agents/",
    submissions_url="https://waymo.com/open/challenges/submissions/",
    evaluation_set="validation",
    browser_name="chromium",
    browser_channel=None,
    browser_executable_path=None,
    headless=True,
    chromium_sandbox=False,
    navigation_timeout_ms=120000,
    upload_timeout_ms=120000,
    post_submit_wait_ms=0,
    poll_submission_status=False,
    poll_timeout_seconds=0,
    poll_interval_seconds=60,
    save_debug_artifacts=True,
    method_name="UniMM-Anchor-Based-4s-topk768",
)
uploader = _WaymoSubmissionUploader(runtime=runtime)
session = uploader._build_http_session()
challenge_response = session.get(runtime.challenge_url, timeout=120)
uploader._ensure_signed_in_http(challenge_response)
upload_metadata_response = session.get(
    "https://waymo.com/open/api/createUploadUrl.json",
    params={
        "challenge": _SIM_AGENTS_CHALLENGE_NAME,
        "submissionType": "VALIDATION",
        "filename": dummy_archive.name,
        "contentLength": str(dummy_archive.stat().st_size),
    },
    headers={"Referer": runtime.challenge_url},
    timeout=120,
)
uploader._raise_for_status(upload_metadata_response, "prepare the Waymo upload")
metadata = upload_metadata_response.json()
if not metadata.get("success") or not metadata.get("uploadUrl") or not metadata.get("contentType"):
    raise RuntimeError(f"Waymo upload preflight failed: {metadata!r}")
print("WAYMO_PREFLIGHT_OK")
PY
EOF
)"
}

write_remote_runner() {
  local log_root_q
  local project_root_q task_name_q ckpt_path_q cache_root_q log_root_remote_q storage_state_q devices_q fallbacks_q topk_q timeout_q poll_q
  log_root_q="$(shq "${LOG_ROOT}")"
  project_root_q="$(shq "${PROJECT_ROOT}")"
  task_name_q="$(shq "${TASK_NAME}")"
  ckpt_path_q="$(shq "${CKPT_PATH}")"
  cache_root_q="$(shq "${CACHE_ROOT}")"
  log_root_remote_q="$(shq "${LOG_ROOT}")"
  storage_state_q="$(shq "${WAYMO_STORAGE_STATE_PATH}")"
  devices_q="$(shq "${DEVICES}")"
  fallbacks_q="$(shq "${VAL_BATCH_FALLBACKS}")"
  topk_q="$(shq "${INFERENCE_TOP_K}")"
  timeout_q="$(shq "${WAYMO_UPLOAD_TIMEOUT_MS}")"
  poll_q="$(shq "${POLL_SUBMISSION_STATUS}")"
  exec_in_pod "$(cat <<EOF
set -Eeuo pipefail
mkdir -p ${log_root_q}
cat > ${log_root_q}/run_submission.sh <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${project_root_q}
TASK_NAME=${task_name_q}
CKPT_PATH=${ckpt_path_q}
CACHE_ROOT=${cache_root_q}
LOG_ROOT=${log_root_remote_q}
WAYMO_STORAGE_STATE_PATH=${storage_state_q}
DEVICES=${devices_q}
VAL_BATCH_FALLBACKS=${fallbacks_q}
INFERENCE_TOP_K=${topk_q}
WAYMO_UPLOAD_TIMEOUT_MS=${timeout_q}
POLL_SUBMISSION_STATUS=${poll_q}

source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
conda activate catk

cd "\${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL=1

echo "[UNIMM-SUBMIT] start \$(date '+%F %T %Z')"
echo "[UNIMM-SUBMIT] project_root=\${PROJECT_ROOT}"
echo "[UNIMM-SUBMIT] commit=\$(git rev-parse --short HEAD)"
echo "[UNIMM-SUBMIT] ckpt=\${CKPT_PATH}"
echo "[UNIMM-SUBMIT] top_k=\${INFERENCE_TOP_K}"
echo "[UNIMM-SUBMIT] val_batch_fallbacks=\${VAL_BATCH_FALLBACKS}"

attempt=0
for VAL_BS in \${VAL_BATCH_FALLBACKS}; do
  attempt=\$((attempt + 1))
  RUN_DIR="\${LOG_ROOT}/attempt_\$(printf '%03d' "\${attempt}")_bs\${VAL_BS}"
  ATTEMPT_TASK="\${TASK_NAME}_bs\${VAL_BS}"
  MASTER_PORT=\$((29870 + attempt))
  mkdir -p "\${RUN_DIR}"
  echo "[UNIMM-SUBMIT] attempt=\${attempt} val_batch_size=\${VAL_BS} start \$(date '+%F %T %Z')"
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  torchrun --standalone --master_port="\${MASTER_PORT}" --nproc_per_node="\${DEVICES}" -m src.run \
    experiment=unimm_anchor_based_4s \
    trainer=ddp \
    action=validate \
    ckpt_path="\${CKPT_PATH}" \
    paths.cache_root="\${CACHE_ROOT}" \
    hydra.run.dir="\${RUN_DIR}" \
    task_name="\${ATTEMPT_TASK}" \
    logger.wandb.offline=false \
    logger.wandb.log_model=false \
    logger.wandb.job_type=waymo_validation_submission \
    callbacks.model_checkpoint.save_top_k=0 \
    trainer.devices="\${DEVICES}" \
    trainer.limit_val_batches=1.0 \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16-mixed \
    data.val_batch_size="\${VAL_BS}" \
    data.num_workers=4 \
    data.persistent_workers=false \
    model.model_config.val_open_loop=false \
    model.model_config.val_closed_loop=true \
    model.model_config.inference_top_k="\${INFERENCE_TOP_K}" \
    model.model_config.n_vis_batch=0 \
    model.model_config.n_vis_scenario=0 \
    model.model_config.n_vis_rollout=0 \
    model.model_config.sim_agents_submission.is_active=true \
    'model.model_config.sim_agents_submission.method_name=UniMM-Anchor-Based-4s-topk768' \
    'model.model_config.sim_agents_submission.authors=["Seulbin Hwang","Kiyoung Om"]' \
    'model.model_config.sim_agents_submission.affiliation="NAVER LABS"' \
    'model.model_config.sim_agents_submission.description="UniMM lhzndj5b:v24 validation submission with inference_top_k=768."' \
    'model.model_config.sim_agents_submission.method_link="https://github.com/seulbinHwang/catk/tree/UniMM"' \
    'model.model_config.sim_agents_submission.account_name=h.sb@naverlabs.com' \
    'model.model_config.sim_agents_submission.num_model_parameters=7M' \
    waymo_submission.enabled=true \
    waymo_submission.submit_validate=true \
    waymo_submission.submit_test=false \
    waymo_submission.evaluation_set=validation \
    waymo_submission.storage_state_path="\${WAYMO_STORAGE_STATE_PATH}" \
    waymo_submission.poll_submission_status="\${POLL_SUBMISSION_STATUS}" \
    waymo_submission.upload_timeout_ms="\${WAYMO_UPLOAD_TIMEOUT_MS}" \
    2>&1 | tee "\${RUN_DIR}/submission.log"
  status="\${PIPESTATUS[0]}"
  set -e
  if [ "\${status}" -eq 0 ]; then
    echo "[UNIMM-SUBMIT] SUCCESS attempt=\${attempt} val_batch_size=\${VAL_BS} run_dir=\${RUN_DIR}"
    exit 0
  fi
  if grep -Eiq 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "\${RUN_DIR}/submission.log"; then
    echo "[UNIMM-SUBMIT] OOM attempt=\${attempt} val_batch_size=\${VAL_BS}; trying next fallback"
    continue
  fi
  echo "[UNIMM-SUBMIT] FAILED non-OOM status=\${status} attempt=\${attempt} val_batch_size=\${VAL_BS}"
  exit "\${status}"
done

echo "[UNIMM-SUBMIT] FAILED all val batch fallbacks exhausted"
exit 1
REMOTE
chmod +x ${log_root_q}/run_submission.sh
EOF
)"
}

launch_tmux() {
  local session_q log_root_q replace_q
  session_q="$(shq "${SESSION}")"
  log_root_q="$(shq "${LOG_ROOT}")"
  replace_q="$(shq "${REPLACE}")"
  exec_in_pod "$(cat <<EOF
set -Eeuo pipefail
if tmux has-session -t ${session_q} 2>/dev/null; then
  if [[ ${replace_q} == "true" || ${replace_q} == "1" ]]; then
    tmux kill-session -t ${session_q}
  else
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 3
  fi
fi
mkdir -p ${log_root_q}
tmux new-session -d -s ${session_q} "bash ${log_root_q}/run_submission.sh |& tee ${log_root_q}/tmux.log"
tmux ls | grep ${session_q}
EOF
)"
}

echo "[launcher] preparing UniMM checkout on ${POD}:${PROJECT_ROOT}"
prepare_checkout

if [[ "${RUN_PREFLIGHT}" == "true" || "${RUN_PREFLIGHT}" == "1" ]]; then
  echo "[launcher] running Waymo HTTP upload preflight"
  waymo_preflight
fi

echo "[launcher] writing remote runner"
write_remote_runner

echo "[launcher] launching tmux session ${SESSION}"
launch_tmux

echo "[launcher] launched. Attach with:"
echo "  kubectl exec -n ${NAMESPACE} ${POD} -c ${CONTAINER} -- tmux attach -t ${SESSION}"
echo "  log: ${LOG_ROOT}/tmux.log"
