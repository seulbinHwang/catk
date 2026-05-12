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
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
WAYMO_SUBMISSION_ENABLED="${WAYMO_SUBMISSION_ENABLED:-true}"
WAYMO_STORAGE_STATE_PATH="${WAYMO_STORAGE_STATE_PATH:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-}"
OOM_RETRY="${OOM_RETRY:-1}"
VAL_BATCH_SIZE_STEP="${VAL_BATCH_SIZE_STEP:-2}"
MIN_VAL_BATCH_SIZE="${MIN_VAL_BATCH_SIZE:-2}"
OOM_RETRY_SLEEP_SEC="${OOM_RETRY_SLEEP_SEC:-10}"
OOM_REGEX="${OOM_REGEX:-OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY}"

if [[ "${SYNC_SEMI_CONTROL:-1}" == "1" ]]; then
  TARGET_BRANCH="${SEMI_CONTROL_BRANCH:-semi_control}"
  CURRENT_BRANCH="$(git branch --show-current || true)"
  TRACKED_DIRTY="$(git status --porcelain --untracked-files=no)"

  git fetch origin "${TARGET_BRANCH}"

  if [[ -n "${TRACKED_DIRTY}" ]]; then
    if [[ "${CURRENT_BRANCH}" != "${TARGET_BRANCH}" ]]; then
      echo "[wosac-val-submit] tracked working tree is dirty; refusing to auto-checkout ${TARGET_BRANCH}." >&2
      echo "[wosac-val-submit] Current branch: ${CURRENT_BRANCH:-detached HEAD}" >&2
      echo "[wosac-val-submit] Commit/stash tracked changes or run with SYNC_SEMI_CONTROL=0." >&2
      exit 2
    fi

    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/${TARGET_BRANCH}")" ]]; then
      echo "[wosac-val-submit] tracked working tree is dirty; already on ${TARGET_BRANCH}, skipping auto-pull." >&2
      echo "[wosac-val-submit] Local HEAD differs from origin/${TARGET_BRANCH}; commit/stash changes before auto-syncing." >&2
    else
      echo "[wosac-val-submit] tracked working tree is dirty; already on ${TARGET_BRANCH}, skipping auto-sync." >&2
    fi
  else
    git checkout "${TARGET_BRANCH}"
    git pull --ff-only origin "${TARGET_BRANCH}"
  fi
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

if os.environ.get("WANDB_API_KEY"):
    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    except Exception as exc:
        print(f"ERROR: failed to login to W&B with WANDB_API_KEY: {exc}", file=sys.stderr)
        sys.exit(12)

api = wandb.Api()
expected_entity = artifact_name.split("/", 1)[0] if "/" in artifact_name else ""
try:
    viewer = api.viewer
    username = getattr(viewer, "username", "") or ""
    entity = getattr(viewer, "entity", "") or ""
    teams = [getattr(team, "name", str(team)) for team in getattr(viewer, "teams", [])]
    print(
        "[wosac-val-submit] W&B viewer: "
        f"username={username or 'unknown'} entity={entity or 'unknown'} "
        f"teams={teams}"
    )
    identities = {value for value in [username, entity, *teams] if value}
    if expected_entity and expected_entity not in identities:
        print(
            "[wosac-val-submit] WARNING: current W&B credentials do not list "
            f"artifact entity '{expected_entity}'. Artifact access may fail.",
            file=sys.stderr,
        )
except Exception as exc:
    print(f"[wosac-val-submit] WARNING: failed to inspect W&B viewer: {exc}", file=sys.stderr)

try:
    artifact = api.artifact(artifact_name)
except Exception as exc:
    print(f"ERROR: failed to access W&B artifact: {artifact_name}", file=sys.stderr)
    print(f"ERROR: {exc}", file=sys.stderr)
    print(
        "Log in with a W&B API key for a user that can access "
        f"'{expected_entity}', then rerun this script.",
        file=sys.stderr,
    )
    print("Example: conda run -n catk wandb login --relogin", file=sys.stderr)
    print("Or export WANDB_API_KEY before running this script.", file=sys.stderr)
    sys.exit(13)
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

waymo_submission_is_enabled() {
  case "$(printf '%s' "${WAYMO_SUBMISSION_ENABLED}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

TMP_WAYMO_STORAGE_STATE_PATH=""
cleanup_tmp_waymo_storage_state() {
  if [[ -n "${TMP_WAYMO_STORAGE_STATE_PATH}" ]]; then
    rm -f "${TMP_WAYMO_STORAGE_STATE_PATH}"
    rmdir "$(dirname "${TMP_WAYMO_STORAGE_STATE_PATH}")" 2>/dev/null || true
  fi
}
trap cleanup_tmp_waymo_storage_state EXIT

DEFAULT_WAYMO_STORAGE_STATE_PATH="${REPO_ROOT}/secrets/waymo/waymo_storage_state.json"
if waymo_submission_is_enabled \
  && [[ -z "${WAYMO_STORAGE_STATE_PATH}" ]] \
  && [[ ! -f "${DEFAULT_WAYMO_STORAGE_STATE_PATH}" ]]; then
  TMP_WAYMO_STORAGE_STATE_PATH="${TMPDIR:-/tmp}/catk_waymo_submission/manual-state-$USER-$$/waymo_storage_state.json"
  WAYMO_STORAGE_STATE_PATH="${TMP_WAYMO_STORAGE_STATE_PATH}"
  export WAYMO_STORAGE_STATE_PATH
  python -c '
import json
import os
import sys
from pathlib import Path

out_path = Path(sys.argv[1]).expanduser().resolve()
print("", file=sys.stderr)
print("Waymo auto submission could not find:", file=sys.stderr)
print(f"  {Path(sys.argv[2]).expanduser().resolve()}", file=sys.stderr)
print("", file=sys.stderr)
print("Paste the full contents of waymo_storage_state.json once now.", file=sys.stderr)
print("This script will reuse it for all OOM retry attempts. Press Ctrl-D to abort.", file=sys.stderr)
print("", file=sys.stderr)

try:
    input_stream = open("/dev/tty", "r", encoding="utf-8")
except OSError:
    input_stream = sys.stdin

collected = []
while True:
    line = input_stream.readline()
    if line == "":
        raise RuntimeError("No complete Waymo storage state JSON was provided.")
    collected.append(line)
    text = "".join(collected).strip()
    if not text:
        continue
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        continue
    if not isinstance(payload, dict):
        raise ValueError("Waymo storage state must be a JSON object.")
    if not isinstance(payload.get("cookies"), list):
        raise ValueError("Waymo storage state JSON must contain a cookies list.")
    origins = payload.get("origins")
    if origins is not None and not isinstance(origins, list):
        raise ValueError("Waymo storage state origins must be a list when present.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[wosac-val-submit] temporary Waymo storage state ready: {out_path}", file=sys.stderr)
    break
' "${TMP_WAYMO_STORAGE_STATE_PATH}" "${DEFAULT_WAYMO_STORAGE_STATE_PATH}"
fi

COMMON_ARGS=(
  -m src.run
  experiment=sim_agents_sub_flow
  action=validate
  paths.cache_root="${CACHE_ROOT}"
  paths.log_dir="${LOG_DIR}"
  ckpt_path="${CKPT_PATH}"
  task_name="${TASK_NAME}"
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}"
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
  waymo_submission.enabled="${WAYMO_SUBMISSION_ENABLED}"
  waymo_submission.submit_validate=true
  waymo_submission.submit_test=false
  waymo_submission.evaluation_set=validation
  waymo_submission.poll_submission_status=false
)

if [[ -n "${WAYMO_STORAGE_STATE_PATH}" ]]; then
  COMMON_ARGS+=(waymo_submission.storage_state_path="${WAYMO_STORAGE_STATE_PATH}")
fi
if [[ -n "${DATA_NUM_WORKERS}" ]]; then
  COMMON_ARGS+=(data.num_workers="${DATA_NUM_WORKERS}")
fi
if [[ -n "${PREFETCH_FACTOR}" ]]; then
  COMMON_ARGS+=(data.prefetch_factor="${PREFETCH_FACTOR}")
fi
if [[ -n "${SCORER_SCENE_NUM}" ]]; then
  COMMON_ARGS+=(model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}")
fi

echo "[wosac-val-submit] repo:      ${REPO_ROOT}"
echo "[wosac-val-submit] commit:    $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"
echo "[wosac-val-submit] conda:     ${CONDA_DEFAULT_ENV:-unknown}"
echo "[wosac-val-submit] python:    $(command -v python)"
echo "[wosac-val-submit] artifact:  ${WANDB_ARTIFACT}"
echo "[wosac-val-submit] ckpt:      ${CKPT_PATH}"
echo "[wosac-val-submit] cache:     ${CACHE_ROOT}"
echo "[wosac-val-submit] task:      ${TASK_NAME}"
echo "[wosac-val-submit] nproc:     ${NPROC_PER_NODE}"
echo "[wosac-val-submit] precision: ${PRECISION}"
echo "[wosac-val-submit] limit_val_batches: ${LIMIT_VAL_BATCHES}"
echo "[wosac-val-submit] waymo_submission.enabled: ${WAYMO_SUBMISSION_ENABLED}"
if [[ -n "${WAYMO_STORAGE_STATE_PATH}" ]]; then
  echo "[wosac-val-submit] waymo_storage_state: ${WAYMO_STORAGE_STATE_PATH}"
fi
if [[ -n "${DATA_NUM_WORKERS}" ]]; then
  echo "[wosac-val-submit] workers:   ${DATA_NUM_WORKERS}"
fi
if [[ -n "${PREFETCH_FACTOR}" ]]; then
  echo "[wosac-val-submit] prefetch:  ${PREFETCH_FACTOR}"
fi
if [[ -n "${SCORER_SCENE_NUM}" ]]; then
  echo "[wosac-val-submit] scorer_scene_num: ${SCORER_SCENE_NUM}"
fi
echo "[wosac-val-submit] oom_retry: ${OOM_RETRY} step=${VAL_BATCH_SIZE_STEP} min_val_bs=${MIN_VAL_BATCH_SIZE}"

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

START_VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
if ! is_positive_int "${START_VAL_BATCH_SIZE}"; then
  echo "[wosac-val-submit] VAL_BATCH_SIZE must be a positive integer, got ${START_VAL_BATCH_SIZE}" >&2
  exit 5
fi
if ! is_positive_int "${VAL_BATCH_SIZE_STEP}"; then
  echo "[wosac-val-submit] VAL_BATCH_SIZE_STEP must be a positive integer, got ${VAL_BATCH_SIZE_STEP}" >&2
  exit 5
fi
if ! is_positive_int "${MIN_VAL_BATCH_SIZE}"; then
  echo "[wosac-val-submit] MIN_VAL_BATCH_SIZE must be a positive integer, got ${MIN_VAL_BATCH_SIZE}" >&2
  exit 5
fi
if [[ "${START_VAL_BATCH_SIZE}" -lt "${MIN_VAL_BATCH_SIZE}" ]]; then
  echo "[wosac-val-submit] VAL_BATCH_SIZE must be >= MIN_VAL_BATCH_SIZE (${MIN_VAL_BATCH_SIZE}), got ${START_VAL_BATCH_SIZE}" >&2
  exit 5
fi

ATTEMPT_LOG_DIR="${LOG_DIR}/${TASK_NAME}/retry_attempts"
mkdir -p "${ATTEMPT_LOG_DIR}"

run_attempt() {
  local attempt="$1"
  local val_bs="$2"
  local attempt_id
  local attempt_output_dir
  local attempt_log
  local status

  attempt_id="$(date +%Y-%m-%d_%H-%M-%S)-pid$$-try$(printf '%02d' "${attempt}")-valbs${val_bs}"
  attempt_output_dir="${LOG_DIR}/${TASK_NAME}/runs/${attempt_id}"
  attempt_log="${ATTEMPT_LOG_DIR}/${attempt_id}.log"
  mkdir -p "${attempt_output_dir}" "${ATTEMPT_LOG_DIR}"

  echo "[wosac-val-submit] attempt ${attempt}: val_batch_size=${val_bs}"
  echo "[wosac-val-submit] attempt ${attempt}: output_dir=${attempt_output_dir}"
  echo "[wosac-val-submit] attempt ${attempt}: log=${attempt_log}"

  local attempt_args=(
    "${COMMON_ARGS[@]}"
    data.val_batch_size="${val_bs}"
    hydra.run.dir="${attempt_output_dir}"
  )

  set +e
  if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
    python "${attempt_args[@]}" trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto 2>&1 | tee "${attempt_log}"
    status="${PIPESTATUS[0]}"
  else
    python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${attempt_args[@]}" trainer=ddp trainer.devices="${NPROC_PER_NODE}" 2>&1 | tee "${attempt_log}"
    status="${PIPESTATUS[0]}"
  fi
  set -e

  LAST_ATTEMPT_LOG="${attempt_log}"
  LAST_ATTEMPT_OUTPUT_DIR="${attempt_output_dir}"
  return "${status}"
}

attempt=1
current_val_bs="${START_VAL_BATCH_SIZE}"
LAST_ATTEMPT_LOG=""
LAST_ATTEMPT_OUTPUT_DIR=""

while true; do
  set +e
  run_attempt "${attempt}" "${current_val_bs}"
  status="$?"
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "[wosac-val-submit] SUCCESS: val_batch_size=${current_val_bs}"
    echo "[wosac-val-submit] final_output_dir=${LAST_ATTEMPT_OUTPUT_DIR}"
    exit 0
  fi

  if [[ "${OOM_RETRY}" != "1" ]]; then
    echo "[wosac-val-submit] attempt ${attempt} failed with status ${status}; OOM_RETRY=${OOM_RETRY}, not retrying." >&2
    exit "${status}"
  fi

  if [[ -z "${LAST_ATTEMPT_LOG}" ]] || ! grep -Eqi "${OOM_REGEX}" "${LAST_ATTEMPT_LOG}"; then
    echo "[wosac-val-submit] attempt ${attempt} failed with status ${status}, but no OOM marker was found; not retrying." >&2
    exit "${status}"
  fi

  next_val_bs=$(( current_val_bs - VAL_BATCH_SIZE_STEP ))
  if [[ "${next_val_bs}" -lt "${MIN_VAL_BATCH_SIZE}" ]]; then
    echo "[wosac-val-submit] OOM detected, but next val_batch_size ${next_val_bs} is below MIN_VAL_BATCH_SIZE=${MIN_VAL_BATCH_SIZE}." >&2
    echo "[wosac-val-submit] last_attempt_log=${LAST_ATTEMPT_LOG}" >&2
    exit "${status}"
  fi

  echo "[wosac-val-submit] OOM detected in attempt ${attempt}; retrying from scratch with val_batch_size ${current_val_bs} -> ${next_val_bs}."
  echo "[wosac-val-submit] failed attempt output remains isolated at: ${LAST_ATTEMPT_OUTPUT_DIR}"
  sleep "${OOM_RETRY_SLEEP_SEC}"
  current_val_bs="${next_val_bs}"
  attempt=$(( attempt + 1 ))
done
