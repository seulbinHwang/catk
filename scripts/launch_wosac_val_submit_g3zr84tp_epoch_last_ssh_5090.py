#!/usr/bin/env python3
"""Launch full validation-set WOSAC submission for g3zr84tp epoch-last.

This launcher runs from a machine with SSH access to ``user@10.60.188.78``.
It verifies and downloads the W&B artifact
``jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64`` on the RTX 5090
host, then starts ``sim_agents_sub_flow`` validation export with Waymo auto
submission enabled inside the existing ``hsb-rl-train`` tmux session.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys


DEFAULT_SSH_HOST = "user@10.60.188.78"
DEFAULT_REMOTE_PROJECT_ROOT = "/media/user/E/projects/catk"
DEFAULT_REMOTE_CACHE_ROOT = "/media/user/E/dataset/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/media/user/D/catk_wosac_val_submit_logs"
DEFAULT_REMOTE_CKPT_PATH = (
    "/media/user/E/projects/catk/checkpoints/from_wandb/"
    "flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "epoch-last-g3zr84tp_v64/epoch_last.ckpt"
)
DEFAULT_REMOTE_DOWNLOAD_DIR = (
    "/media/user/E/projects/catk/checkpoints/from_wandb/"
    "flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "epoch-last-g3zr84tp_v64/artifact"
)
DEFAULT_WANDB_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64"
DEFAULT_EXPECTED_ARTIFACT_EPOCH = 64
DEFAULT_EXPECTED_GLOBAL_STEP = 149888
DEFAULT_COMPLETED_EPOCHS = 64
DEFAULT_STEPS_PER_EPOCH = 2342
DEFAULT_TMUX_SESSION = "hsb-rl-train"
DEFAULT_WINDOW_NAME = "wosac-submit-g3zr84tp-v64"
DEFAULT_TASK_NAME = (
    "flow_semi_continuous_pretrain_h100x4x2_bs26_"
    "epoch_last_g3zr84tp_v64_wosac_val_submit"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "semi_control"
    branch = result.stdout.strip()
    return branch if branch else "semi_control"


def run(command: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
    if dry_run:
        print("+ " + " ".join(shq(part) for part in command))
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout is not None else ""


def run_ssh(args: argparse.Namespace, script: str, *, capture: bool = False) -> str:
    return run(
        ["ssh", args.ssh_host, "bash -lc " + shq(script)],
        capture=capture,
        dry_run=args.dry_run,
    )


def remote_conda_prefix(args: argparse.Namespace) -> str:
    return f"""
set -Eeuo pipefail
cd {shq(args.remote_project_root)}
if [[ -f scripts/_activate_conda.sh ]]; then
  # shellcheck source=/dev/null
  . scripts/_activate_conda.sh
elif [[ -f /media/user/E/miniforge/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  . /media/user/E/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
"""


def prepare_checkpoint_on_remote(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"[launcher] verify W&B artifact: {args.wandb_artifact}")
        print(f"[launcher] expected metadata epoch: {args.expected_artifact_epoch}")
        print(f"[launcher] expected global_step: {args.expected_global_step}")
        print(f"[launcher] remote checkpoint: {args.remote_ckpt_path}")
        return

    script = (
        remote_conda_prefix(args)
        + f"""
export WANDB_ARTIFACT={shq(args.wandb_artifact)}
export REMOTE_CKPT_PATH={shq(args.remote_ckpt_path)}
export REMOTE_DOWNLOAD_DIR={shq(args.remote_download_dir)}
export EXPECTED_ARTIFACT_EPOCH={shq(args.expected_artifact_epoch)}
export EXPECTED_GLOBAL_STEP={shq(args.expected_global_step)}
export COMPLETED_EPOCHS={shq(args.completed_epochs)}
export STEPS_PER_EPOCH={shq(args.steps_per_epoch)}
python - <<'PY'
from pathlib import Path
import glob
import json
import os
import shutil
import sys

import wandb

artifact_name = os.environ["WANDB_ARTIFACT"]
target = Path(os.environ["REMOTE_CKPT_PATH"]).expanduser().resolve()
download_dir = Path(os.environ["REMOTE_DOWNLOAD_DIR"]).expanduser().resolve()
expected_epoch = int(os.environ["EXPECTED_ARTIFACT_EPOCH"])
expected_global_step = int(os.environ["EXPECTED_GLOBAL_STEP"])
completed_epochs = int(os.environ["COMPLETED_EPOCHS"])
steps_per_epoch = int(os.environ["STEPS_PER_EPOCH"])

artifact = wandb.Api().artifact(artifact_name)
metadata = dict(artifact.metadata or {{}})
files = list(artifact.files())
ckpt_files = [file for file in files if file.name.endswith("epoch_last.ckpt")]
if len(ckpt_files) != 1:
    raise SystemExit(f"expected exactly one epoch_last.ckpt in {{artifact_name}}, got {{len(ckpt_files)}}")
ckpt_file = ckpt_files[0]

artifact_epoch = int(metadata.get("epoch", -1))
global_step = int(metadata.get("global_step", -1))
expected_from_epochs = completed_epochs * steps_per_epoch
if artifact_epoch != expected_epoch:
    raise SystemExit(f"artifact epoch mismatch: expected {{expected_epoch}}, got {{artifact_epoch}}")
if global_step != expected_global_step:
    raise SystemExit(f"artifact global_step mismatch: expected {{expected_global_step}}, got {{global_step}}")
if global_step != expected_from_epochs:
    raise SystemExit(
        f"artifact completed epoch check failed: global_step={{global_step}}, "
        f"expected={{expected_from_epochs}}"
    )

summary = {{
    "artifact": artifact_name,
    "version": artifact.version,
    "aliases": artifact.aliases,
    "created_at": str(artifact.created_at),
    "size": artifact.size,
    "file": ckpt_file.name,
    "file_size": ckpt_file.size,
    "metadata": metadata,
    "verified_completed_epochs": completed_epochs,
}}
print("[launcher] verified W&B artifact:")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

if target.is_file() and target.stat().st_size == ckpt_file.size:
    print(f"[launcher] using cached checkpoint: {{target}}")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] downloading W&B artifact to {{download_dir}}")
    artifact_dir = Path(artifact.download(root=download_dir)).resolve()
    candidates = []
    preferred = artifact_dir / "epoch_last.ckpt"
    if preferred.is_file():
        candidates.append(preferred)
    candidates.extend(
        Path(path) for path in glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True)
    )
    candidates.extend(
        Path(path) for path in glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True)
    )
    candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    if not candidates:
        raise SystemExit(f"no checkpoint file found after artifact download: {{artifact_dir}}")
    source = candidates[0]
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)
    print(f"[launcher] checkpoint ready: {{target}} ({{target.stat().st_size}} bytes)")

if target.stat().st_size != ckpt_file.size:
    raise SystemExit(
        f"checkpoint size mismatch: target={{target.stat().st_size}}, artifact_file={{ckpt_file.size}}"
    )

try:
    import torch
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    print(
        "[launcher] checkpoint content:",
        json.dumps(
            {{"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
except Exception as exc:
    print(f"[launcher] checkpoint content inspection skipped: {{exc}}", file=sys.stderr)
PY
"""
    )
    run_ssh(args, script)


def render_worker_script(args: argparse.Namespace) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export WANDB_ENTITY="${{WANDB_ENTITY:-jksg01019-naver-labs}}"
export WANDB_PROJECT="${{WANDB_PROJECT:-SMART-FLOW}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export CUDA_VISIBLE_DEVICES={shq(args.cuda_visible_devices)}

cd {shq(args.remote_project_root)}
if [[ -f scripts/_activate_conda.sh ]]; then
  # shellcheck source=/dev/null
  . scripts/_activate_conda.sh
elif [[ -f /media/user/E/miniforge/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  . /media/user/E/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi

TASK_NAME={shq(args.task_name)}
CACHE_ROOT={shq(args.remote_cache_root)}
CATK_LOG_DIR={shq(args.remote_log_dir)}
CKPT_PATH={shq(args.remote_ckpt_path)}
VAL_BATCH_SIZE={shq(args.val_batch_size)}
MIN_VAL_BATCH_SIZE={shq(args.min_val_batch_size)}
VAL_BATCH_SIZE_STEP={shq(args.val_batch_size_step)}
NPROC_PER_NODE={shq(args.nproc_per_node)}
PRECISION={shq(args.precision)}
DATA_NUM_WORKERS={shq(args.num_workers)}
PREFETCH_FACTOR={shq(args.prefetch_factor)}
N_ROLLOUT_CLOSED_VAL={shq(args.n_rollout_closed_val)}
WAYMO_SUBMISSION_ENABLED={shq(str(args.waymo_submission_enabled).lower())}
WAYMO_STORAGE_STATE_PATH={shq(args.waymo_storage_state_path)}
OOM_REGEX={shq(args.oom_regex)}

echo "[wosac-submit-g3zr84tp] host=$(hostname) task=${{TASK_NAME}}"
echo "[wosac-submit-g3zr84tp] started at $(date '+%F %T')"
echo "[wosac-submit-g3zr84tp] repo=$(pwd)"
echo "[wosac-submit-g3zr84tp] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[wosac-submit-g3zr84tp] cache=${{CACHE_ROOT}}"
echo "[wosac-submit-g3zr84tp] ckpt=${{CKPT_PATH}}"
echo "[wosac-submit-g3zr84tp] val_batch_size=${{VAL_BATCH_SIZE}} nproc=${{NPROC_PER_NODE}} precision=${{PRECISION}}"
echo "[wosac-submit-g3zr84tp] waymo_submission.enabled=${{WAYMO_SUBMISSION_ENABLED}}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[wosac-submit-g3zr84tp] checkpoint does not exist: $CKPT_PATH" >&2
  exec bash
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[wosac-submit-g3zr84tp] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  exec bash
fi

COMMON_ARGS=(
  -m src.run
  experiment=sim_agents_sub_flow
  action=validate
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$CATK_LOG_DIR"
  ckpt_path="$CKPT_PATH"
  task_name="$TASK_NAME"
  trainer.limit_val_batches=1.0
  trainer.precision="$PRECISION"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val="$N_ROLLOUT_CLOSED_VAL"
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=false
  model.model_config.decoder.use_kinematic_control_flow=false
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=false
  model.model_config.decoder.use_stop_motion=true
  model.model_config.sim_agents_submission.method_name="SMART-flow-g3zr84tp"
  "model.model_config.sim_agents_submission.authors=[Seulbin Hwang,Kiyoung Om]"
  model.model_config.sim_agents_submission.affiliation=NaverLabs
  model.model_config.sim_agents_submission.description="Pose-space Flow Matching H100x4x2 bs26 epoch-last validation submission."
  model.model_config.sim_agents_submission.method_link="not available yet"
  model.model_config.sim_agents_submission.account_name="h.sb@naverlabs.com"
  waymo_submission.enabled="$WAYMO_SUBMISSION_ENABLED"
  waymo_submission.submit_validate=true
  waymo_submission.submit_test=false
  waymo_submission.evaluation_set=validation
  waymo_submission.poll_submission_status=false
  logger.wandb.name="$TASK_NAME"
  logger.wandb.group=wosac_validation_submission
  "logger.wandb.tags=[wosac_submission,g3zr84tp,v64,epoch_last,rtx5090,pose_space]"
)

if [[ -n "$WAYMO_STORAGE_STATE_PATH" ]]; then
  COMMON_ARGS+=(waymo_submission.storage_state_path="$WAYMO_STORAGE_STATE_PATH")
fi
if [[ -n "$DATA_NUM_WORKERS" ]]; then
  COMMON_ARGS+=(data.num_workers="$DATA_NUM_WORKERS")
fi
if [[ -n "$PREFETCH_FACTOR" ]]; then
  COMMON_ARGS+=(data.prefetch_factor="$PREFETCH_FACTOR")
fi

is_positive_int() {{
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}}

if ! is_positive_int "$VAL_BATCH_SIZE" || ! is_positive_int "$MIN_VAL_BATCH_SIZE" || ! is_positive_int "$VAL_BATCH_SIZE_STEP"; then
  echo "[wosac-submit-g3zr84tp] invalid val batch settings" >&2
  exec bash
fi

ATTEMPT_LOG_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/retry_attempts"
mkdir -p "$ATTEMPT_LOG_DIR"

attempt=1
current_val_bs="$VAL_BATCH_SIZE"
while true; do
  RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')-try$(printf '%02d' "$attempt")-valbs${{current_val_bs}}"
  OUTPUT_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
  ATTEMPT_LOG="${{ATTEMPT_LOG_DIR}}/${{RUN_ID}}.log"
  mkdir -p "$OUTPUT_DIR" "$ATTEMPT_LOG_DIR"

  echo
  echo "[wosac-submit-g3zr84tp] attempt $attempt val_batch_size=$current_val_bs"
  echo "[wosac-submit-g3zr84tp] output_dir=$OUTPUT_DIR"
  echo "[wosac-submit-g3zr84tp] log=$ATTEMPT_LOG"

  ATTEMPT_ARGS=(
    "${{COMMON_ARGS[@]}}"
    data.val_batch_size="$current_val_bs"
    hydra.run.dir="$OUTPUT_DIR"
  )

  set +e
  if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    python "${{ATTEMPT_ARGS[@]}}" trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto 2>&1 | tee "$ATTEMPT_LOG"
    status="${{PIPESTATUS[0]}}"
  else
    python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${{ATTEMPT_ARGS[@]}}" trainer=ddp trainer.devices="$NPROC_PER_NODE" 2>&1 | tee "$ATTEMPT_LOG"
    status="${{PIPESTATUS[0]}}"
  fi
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "[wosac-submit-g3zr84tp] SUCCESS val_batch_size=$current_val_bs"
    echo "[wosac-submit-g3zr84tp] final_output_dir=$OUTPUT_DIR"
    break
  fi

  if ! grep -Eqi "$OOM_REGEX" "$ATTEMPT_LOG"; then
    echo "[wosac-submit-g3zr84tp] failed with status $status and no OOM marker; not retrying." >&2
    break
  fi

  next_val_bs=$(( current_val_bs - VAL_BATCH_SIZE_STEP ))
  if [[ "$next_val_bs" -lt "$MIN_VAL_BATCH_SIZE" ]]; then
    echo "[wosac-submit-g3zr84tp] OOM but next val_batch_size $next_val_bs is below minimum $MIN_VAL_BATCH_SIZE." >&2
    break
  fi
  echo "[wosac-submit-g3zr84tp] OOM detected; retrying val_batch_size $current_val_bs -> $next_val_bs."
  current_val_bs="$next_val_bs"
  attempt=$(( attempt + 1 ))
done

echo
echo "[wosac-submit-g3zr84tp] finished at $(date '+%F %T')"
echo "[wosac-submit-g3zr84tp] leaving shell open for inspection"
exec bash
"""


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} host=$(hostname)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_remote_start(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    run_root = f"{args.remote_log_dir.rstrip('/')}/tmux_wosac_val_submit/{safe_task}"
    worker_file = f"{run_root}/worker.sh"
    monitor_file = f"{run_root}/monitor.sh"
    tmux_log = f"{run_root}/tmux.log"

    sync_block = ""
    if args.pull:
        sync_block = f"""
git config --global --add safe.directory {shq(args.remote_project_root)} || true
git fetch origin {shq(args.branch)}
TRACKED_DIRTY="$(git status --porcelain --untracked-files=no)"
CURRENT_BRANCH="$(git branch --show-current || true)"
if [[ -n "$TRACKED_DIRTY" && "$CURRENT_BRANCH" != {shq(args.branch)} ]]; then
  echo "[launcher] tracked working tree is dirty on $CURRENT_BRANCH; refusing to checkout {args.branch}" >&2
  exit 4
fi
if [[ -z "$TRACKED_DIRTY" ]]; then
  git checkout {shq(args.branch)}
  git pull --ff-only origin {shq(args.branch)}
else
  echo "[launcher] tracked working tree is dirty; staying on $CURRENT_BRANCH and skipping pull."
fi
"""

    replace_block = ""
    if args.replace:
        replace_block = f"""
while IFS=: read -r window_id window_name; do
  if [[ "$window_name" == {shq(args.window_name)} ]]; then
    tmux kill-window -t {shq(args.tmux_session)}:"$window_id" || true
  fi
done < <(tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_index}}:#{{window_name}}' 2>/dev/null || true)
"""
    else:
        replace_block = f"""
if tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_name}}' 2>/dev/null | grep -Fx {shq(args.window_name)} >/dev/null; then
  echo "[launcher] tmux window already exists: {args.tmux_session}:{args.window_name}" >&2
  exit 3
fi
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{render_monitor_script(args.monitor_interval, args.task_name).rstrip()}
CATK_MONITOR
chmod +x {shq(monitor_file)}
tmux split-window -v -l 10 -t {shq(args.tmux_session)}:{shq(args.window_name)} {shq(monitor_file)}
tmux select-pane -t {shq(args.tmux_session)}:{shq(args.window_name)}.0
"""

    return f"""set -Eeuo pipefail
if [[ ! -d {shq(args.remote_project_root)}/.git ]]; then
  echo "[launcher] remote project root is not a git checkout: {args.remote_project_root}" >&2
  exit 2
fi
cd {shq(args.remote_project_root)}
{sync_block}
if ! tmux has-session -t {shq(args.tmux_session)} 2>/dev/null; then
  tmux new-session -d -s {shq(args.tmux_session)} -c {shq(args.remote_project_root)}
fi
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(args).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-window -t {shq(args.tmux_session)} -n {shq(args.window_name)} -c {shq(args.remote_project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.tmux_session)}:{shq(args.window_name)} -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.tmux_session}:{args.window_name}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_remote_stop(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
if ! tmux has-session -t {shq(args.tmux_session)} 2>/dev/null; then
  echo "[launcher] tmux session not found: {args.tmux_session}"
  exit 0
fi
found=0
while IFS=: read -r window_id window_name; do
  if [[ "$window_name" == {shq(args.window_name)} ]]; then
    tmux kill-window -t {shq(args.tmux_session)}:"$window_id" || true
    found=1
  fi
done < <(tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_index}}:#{{window_name}}')
if (( found )); then
  echo "[launcher] stopped tmux window {args.tmux_session}:{args.window_name}"
else
  echo "[launcher] tmux window not found: {args.tmux_session}:{args.window_name}"
fi
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download g3zr84tp:v64 epoch_last.ckpt and launch full validation-set "
            "Waymo/WOSAC auto submission on user@10.60.188.78."
        )
    )
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT)
    parser.add_argument("--remote-cache-root", default=DEFAULT_REMOTE_CACHE_ROOT)
    parser.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    parser.add_argument("--remote-ckpt-path", default=DEFAULT_REMOTE_CKPT_PATH)
    parser.add_argument("--remote-download-dir", default=DEFAULT_REMOTE_DOWNLOAD_DIR)
    parser.add_argument("--wandb-artifact", default=DEFAULT_WANDB_ARTIFACT)
    parser.add_argument("--expected-artifact-epoch", type=int, default=DEFAULT_EXPECTED_ARTIFACT_EPOCH)
    parser.add_argument("--expected-global-step", type=int, default=DEFAULT_EXPECTED_GLOBAL_STEP)
    parser.add_argument("--completed-epochs", type=int, default=DEFAULT_COMPLETED_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--branch", default=current_branch())
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--min-val-batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size-step", type=int, default=2)
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--waymo-submission-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--waymo-storage-state-path",
        default="/media/user/E/projects/catk/secrets/waymo/waymo_storage_state.json",
    )
    parser.add_argument(
        "--oom-regex",
        default=(
            "OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|"
            "torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY"
        ),
    )
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.min_val_batch_size < 1:
        parser.error("--min-val-batch-size must be >= 1")
    if args.val_batch_size_step < 1:
        parser.error("--val-batch-size-step must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        run_ssh(args, render_remote_stop(args))
        return

    print(f"[launcher] ssh_host:       {args.ssh_host}")
    print(f"[launcher] remote repo:    {args.remote_project_root}")
    print(f"[launcher] branch:         {args.branch}")
    print(f"[launcher] tmux target:    {args.tmux_session}:{args.window_name}")
    print(f"[launcher] artifact:       {args.wandb_artifact}")
    print(f"[launcher] remote ckpt:    {args.remote_ckpt_path}")
    print(f"[launcher] task_name:      {args.task_name}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")

    prepare_checkpoint_on_remote(args)
    run_ssh(args, render_remote_start(args))

    print("\nAttach command:")
    print(f"ssh -t {shq(args.ssh_host)} {shq('tmux attach -t ' + args.tmux_session)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[launcher] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
