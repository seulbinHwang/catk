#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

WANDB_ARTIFACT="${WANDB_ARTIFACT:-jksg01019-naver-labs/SMART-FLOW/epoch-last-48v4vo86:v61}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/wandb/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/epoch-last-48v4vo86_v61}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/epoch_last.ckpt}"
WANDB_DOWNLOAD_DIR="${WANDB_DOWNLOAD_DIR:-${CKPT_DIR}/artifact_download}"
CACHE_ROOT="${CACHE_ROOT:-/media/user/E/dataset/womd_v1_3/SMART_cache}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
TASK_NAME="${TASK_NAME:-flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8_wosac_val_submit}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
PRECISION="${PRECISION:-bf16-mixed}"

if [[ "${SYNC_SEMI_CONTROL:-1}" == "1" ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "[wosac-val-submit] tracked working tree is dirty; refusing to auto-sync semi_control." >&2
    echo "[wosac-val-submit] Commit/stash tracked changes or run with SYNC_SEMI_CONTROL=0." >&2
    exit 2
  fi
  git fetch origin semi_control
  git checkout semi_control
  git pull --ff-only origin semi_control
fi

if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "[wosac-val-submit] CACHE_ROOT does not exist: ${CACHE_ROOT}" >&2
  echo "[wosac-val-submit] Set CACHE_ROOT to the WOMD SMART cache root." >&2
  exit 3
fi

if [[ -f "${SCRIPT_DIR}/_activate_conda.sh" ]]; then
  # shellcheck source=/dev/null
  . "${SCRIPT_DIR}/_activate_conda.sh"
fi

mkdir -p "${CKPT_DIR}" "${WANDB_DOWNLOAD_DIR}" "${LOG_DIR}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[wosac-val-submit] downloading W&B artifact: ${WANDB_ARTIFACT}"
  export WANDB_ARTIFACT CKPT_PATH WANDB_DOWNLOAD_DIR
  python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_ARTIFACT"]
target_ckpt = Path(os.environ["CKPT_PATH"]).expanduser().resolve()
download_root = Path(os.environ["WANDB_DOWNLOAD_DIR"]).expanduser().resolve()

try:
    import wandb
except Exception as exc:
    print(f"ERROR: failed to import wandb: {exc}", file=sys.stderr)
    sys.exit(10)

target_ckpt.parent.mkdir(parents=True, exist_ok=True)
download_root.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact_dir = Path(artifact.download(root=download_root)).resolve()

candidates = []
preferred = artifact_dir / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred.as_posix())
candidates.extend(glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))

if not candidates:
    print(f"ERROR: no checkpoint file found in W&B artifact dir: {artifact_dir}", file=sys.stderr)
    sys.exit(11)

source = Path(candidates[0]).resolve()
if source != target_ckpt:
    shutil.copy2(source, target_ckpt)
print(f"[wosac-val-submit] checkpoint ready: {target_ckpt}")
PY
else
  echo "[wosac-val-submit] using cached checkpoint: ${CKPT_PATH}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cuda_ids <<< "${CUDA_VISIBLE_DEVICES}"
  DEFAULT_NPROC="${#_cuda_ids[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  DEFAULT_NPROC="1"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[wosac-val-submit] NPROC_PER_NODE must be >= 1, got ${NPROC_PER_NODE}" >&2
  exit 4
fi

COMMON_ARGS=(
  -m src.run
  experiment=sim_agents_sub_flow
  action=validate
  paths.cache_root="${CACHE_ROOT}"
  paths.log_dir="${LOG_DIR}"
  ckpt_path="${CKPT_PATH}"
  task_name="${TASK_NAME}"
  trainer.limit_val_batches=1.0
  trainer.precision="${PRECISION}"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val=32
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=true
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=true
  model.model_config.token_processor.control_round_trip_max_position_error_m=2.0
  model.model_config.sim_agents_submission.method_name="SMART-control-v100x47-prefix-rt2"
  "model.model_config.sim_agents_submission.authors=[Seulbin Hwang,Kiyoung Om]"
  model.model_config.sim_agents_submission.affiliation=NaverLabs
  model.model_config.sim_agents_submission.description="Control-space Flow Matching V100x47 prefix-valid round-trip 2.0 validation submission."
  model.model_config.sim_agents_submission.method_link="not available yet"
  model.model_config.sim_agents_submission.account_name="h.sb@naverlabs.com"
  waymo_submission.enabled=true
  waymo_submission.submit_validate=true
  waymo_submission.submit_test=false
  waymo_submission.evaluation_set=validation
  waymo_submission.poll_submission_status=false
)

echo "[wosac-val-submit] repo:      ${REPO_ROOT}"
echo "[wosac-val-submit] commit:    $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"
echo "[wosac-val-submit] artifact:  ${WANDB_ARTIFACT}"
echo "[wosac-val-submit] ckpt:      ${CKPT_PATH}"
echo "[wosac-val-submit] cache:     ${CACHE_ROOT}"
echo "[wosac-val-submit] task:      ${TASK_NAME}"
echo "[wosac-val-submit] nproc:     ${NPROC_PER_NODE}"
echo "[wosac-val-submit] precision: ${PRECISION}"

if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  python "${COMMON_ARGS[@]}" trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto
else
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${COMMON_ARGS[@]}" trainer=ddp trainer.devices="${NPROC_PER_NODE}"
fi
