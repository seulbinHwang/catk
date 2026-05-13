#!/usr/bin/env python3
"""Launch g3zr84tp:v31 Fast WOSAC validation on the RTX 5090 host.

The checkpoint is downloaded from the W&B artifact
``jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v31`` after verifying
that it matches the expected 32 completed epochs. Then a new window is opened
inside the existing ``hsb-rl-train`` tmux session on the SSH host.
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
DEFAULT_REMOTE_LOG_DIR = "/media/user/D/catk_fast_wosac_logs"
DEFAULT_REMOTE_CKPT_PATH = (
    "/media/user/E/projects/catk/checkpoints/from_wandb/"
    "flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "epoch-last-g3zr84tp_v31/epoch_last.ckpt"
)
DEFAULT_REMOTE_DOWNLOAD_DIR = (
    "/media/user/E/projects/catk/checkpoints/from_wandb/"
    "flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "epoch-last-g3zr84tp_v31/artifact"
)
DEFAULT_WANDB_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v31"
DEFAULT_EXPECTED_ARTIFACT_EPOCH = 32
DEFAULT_EXPECTED_GLOBAL_STEP = 74944
DEFAULT_EXPECTED_COMPLETED_EPOCHS = 32
DEFAULT_STEPS_PER_EPOCH = 2342
DEFAULT_BRANCH = "semi_control"
DEFAULT_TMUX_SESSION = "hsb-rl-train"
DEFAULT_WINDOW_NAME = "fast-wosac-g3zr84tp-v31"
DEFAULT_TASK_NAME = (
    "flow_semi_continuous_pretrain_h100x4x2_bs26_"
    "epoch031_v31_fast_wosac_val1680"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(
    command: list[str],
    *,
    capture: bool = False,
    dry_run: bool = False,
    check: bool = True,
) -> str:
    if dry_run:
        print("+ " + " ".join(shq(part) for part in command))
        return ""
    result = subprocess.run(
        command,
        check=check,
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


def default_limit_val_batches(args: argparse.Namespace) -> int:
    per_rank_scenes = math.ceil(args.scorer_scene_num / args.nproc_per_node)
    return max(1, math.ceil(per_rank_scenes / args.val_batch_size))


def remote_python_prefix(args: argparse.Namespace) -> str:
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
export WANDB_ARTIFACT={shq(args.wandb_artifact)}
export REMOTE_CKPT_PATH={shq(args.remote_ckpt_path)}
export REMOTE_DOWNLOAD_DIR={shq(args.remote_download_dir)}
export EXPECTED_ARTIFACT_EPOCH={shq(args.expected_artifact_epoch)}
export EXPECTED_GLOBAL_STEP={shq(args.expected_global_step)}
export EXPECTED_COMPLETED_EPOCHS={shq(args.expected_completed_epochs)}
export STEPS_PER_EPOCH={shq(args.steps_per_epoch)}
"""


def prepare_checkpoint_on_remote(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"[launcher] verify W&B artifact: {args.wandb_artifact}")
        print(f"[launcher] expected metadata epoch: {args.expected_artifact_epoch}")
        print(f"[launcher] expected global_step: {args.expected_global_step}")
        print(
            "[launcher] expected completed epochs check: "
            f"global_step == {args.expected_completed_epochs} * {args.steps_per_epoch}"
        )
        print(f"[launcher] remote checkpoint: {args.remote_ckpt_path}")
        return

    script = (
        remote_python_prefix(args)
        + r'''
python - <<'PY'
from pathlib import Path
import glob
import json
import os
import shutil
import sys

import wandb

artifact_name = os.environ["WANDB_ARTIFACT"]
target = Path(os.environ["REMOTE_CKPT_PATH"])
download_dir = Path(os.environ["REMOTE_DOWNLOAD_DIR"])
expected_artifact_epoch = int(os.environ["EXPECTED_ARTIFACT_EPOCH"])
expected_global_step = int(os.environ["EXPECTED_GLOBAL_STEP"])
expected_completed_epochs = int(os.environ["EXPECTED_COMPLETED_EPOCHS"])
steps_per_epoch = int(os.environ["STEPS_PER_EPOCH"])

api = wandb.Api()
artifact = api.artifact(artifact_name)
metadata = dict(artifact.metadata or {})
files = list(artifact.files())
ckpt_files = [file for file in files if file.name.endswith("epoch_last.ckpt")]
if not ckpt_files:
    raise SystemExit(f"artifact has no epoch_last.ckpt: {artifact_name}")
ckpt_file = ckpt_files[0]

artifact_epoch = int(metadata.get("epoch", -1))
global_step = int(metadata.get("global_step", -1))
if artifact_epoch != expected_artifact_epoch:
    raise SystemExit(
        f"artifact epoch mismatch: expected {expected_artifact_epoch}, got {artifact_epoch}"
    )
if global_step != expected_global_step:
    raise SystemExit(
        f"artifact global_step mismatch: expected {expected_global_step}, got {global_step}"
    )
if global_step != expected_completed_epochs * steps_per_epoch:
    raise SystemExit(
        "artifact does not match expected completed epoch count: "
        f"global_step={global_step}, "
        f"expected={expected_completed_epochs * steps_per_epoch}"
    )

summary = {
    "artifact": artifact_name,
    "version": artifact.version,
    "created_at": str(artifact.created_at),
    "size": artifact.size,
    "file": ckpt_file.name,
    "file_size": ckpt_file.size,
    "metadata": metadata,
    "verified_completed_epochs": expected_completed_epochs,
}
print("[launcher] verified W&B artifact before download:")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

if target.is_file() and target.stat().st_size == ckpt_file.size:
    print(f"[launcher] remote checkpoint already exists with expected size: {target}")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] downloading W&B artifact to {download_dir}")
    artifact_dir = Path(artifact.download(root=download_dir)).resolve()
    candidates = []
    preferred = artifact_dir / "epoch_last.ckpt"
    if preferred.is_file():
        candidates.append(preferred)
    candidates.extend(Path(path) for path in glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True))
    candidates.extend(Path(path) for path in glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True))
    candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    if not candidates:
        raise SystemExit(f"no checkpoint file found after artifact download: {artifact_dir}")
    source = candidates[0]
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)
    print(f"[launcher] checkpoint ready: {target} ({target.stat().st_size} bytes)")

if target.stat().st_size != ckpt_file.size:
    raise SystemExit(
        f"checkpoint size mismatch: target={target.stat().st_size}, artifact_file={ckpt_file.size}"
    )

try:
    import torch
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    print(
        "[launcher] checkpoint content:",
        json.dumps(
            {
                "epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
except Exception as exc:
    print(f"[launcher] checkpoint content inspection skipped: {exc}", file=sys.stderr)
PY
'''
    )
    run_ssh(args, script)


def render_worker_script(args: argparse.Namespace) -> str:
    limit_val_batches = (
        default_limit_val_batches(args)
        if args.limit_val_batches == "auto"
        else args.limit_val_batches
    )
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
LIMIT_VAL_BATCHES={shq(limit_val_batches)}
SCORER_SCENE_NUM={shq(args.scorer_scene_num)}
NPROC_PER_NODE={shq(args.nproc_per_node)}
N_ROLLOUT_CLOSED_VAL={shq(args.n_rollout_closed_val)}
PRECISION={shq(args.precision)}
DATA_NUM_WORKERS={shq(args.num_workers)}
PREFETCH_FACTOR={shq(args.prefetch_factor)}

echo "[fast-wosac-g3zr84tp-v31] host=$(hostname) task=${{TASK_NAME}}"
echo "[fast-wosac-g3zr84tp-v31] started at $(date '+%F %T')"
echo "[fast-wosac-g3zr84tp-v31] repo=$(pwd)"
echo "[fast-wosac-g3zr84tp-v31] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[fast-wosac-g3zr84tp-v31] python=$(command -v python 2>/dev/null)"
echo "[fast-wosac-g3zr84tp-v31] cache=${{CACHE_ROOT}}"
echo "[fast-wosac-g3zr84tp-v31] ckpt=${{CKPT_PATH}}"
echo "[fast-wosac-g3zr84tp-v31] scorer_scene_num=${{SCORER_SCENE_NUM}} val_batch_size=${{VAL_BATCH_SIZE}} limit_val_batches=${{LIMIT_VAL_BATCHES}}"
echo "[fast-wosac-g3zr84tp-v31] n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} nproc=${{NPROC_PER_NODE}} precision=${{PRECISION}}"
echo

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[fast-wosac-g3zr84tp-v31] checkpoint does not exist: $CKPT_PATH" >&2
  exec bash
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[fast-wosac-g3zr84tp-v31] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  exec bash
fi
if ! command -v python >/dev/null 2>&1; then
  echo "[fast-wosac-g3zr84tp-v31] python is not available after conda activation" >&2
  exec bash
fi

RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
OUTPUT_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
mkdir -p "$OUTPUT_DIR"

ARGS=(
  -m src.run
  experiment=local_val_flow
  action=validate
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$CATK_LOG_DIR"
  ckpt_path="$CKPT_PATH"
  task_name="$TASK_NAME"
  hydra.run.dir="$OUTPUT_DIR"
  trainer=ddp
  trainer.devices="$NPROC_PER_NODE"
  trainer.num_nodes=1
  trainer.precision="$PRECISION"
  trainer.limit_val_batches="$LIMIT_VAL_BATCHES"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.num_workers="$DATA_NUM_WORKERS"
  data.prefetch_factor="$PREFETCH_FACTOR"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val="$N_ROLLOUT_CLOSED_VAL"
  model.model_config.scorer_scene_num="$SCORER_SCENE_NUM"
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=false
  model.model_config.decoder.use_kinematic_control_flow=false
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=false
  model.model_config.decoder.use_stop_motion=true
  model.model_config.sim_agents_submission.is_active=false
  logger.wandb.name="$TASK_NAME"
  logger.wandb.group=fast_wosac_validation
  "logger.wandb.tags=[fast_wosac,g3zr84tp,v31,epoch031,rtx5090,pose_space]"
)

echo "[fast-wosac-g3zr84tp-v31] output_dir=$OUTPUT_DIR"
echo "[fast-wosac-g3zr84tp-v31] command: python -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE ${{ARGS[*]}}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${{ARGS[@]}}"
status=$?

echo
echo "[fast-wosac-g3zr84tp-v31] exited with status $status at $(date '+%F %T')"
echo "[fast-wosac-g3zr84tp-v31] output_dir=$OUTPUT_DIR"
echo "[fast-wosac-g3zr84tp-v31] leaving shell open for inspection"
exec bash
"""


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} host=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_remote_start(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    run_root = f"{args.remote_log_dir.rstrip('/')}/tmux_fast_wosac_val/{safe_task}"
    worker_file = f"{run_root}/worker.sh"
    monitor_file = f"{run_root}/monitor.sh"
    tmux_log = f"{run_root}/tmux.log"

    pull_block = ""
    if args.pull:
        pull_block = f"""
git config --global --add safe.directory {shq(args.remote_project_root)} || true
git fetch origin {shq(args.branch)}
git checkout {shq(args.branch)}
git pull --ff-only origin {shq(args.branch)}
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
{pull_block}
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
            "Verify and download W&B artifact epoch-last-g3zr84tp:v31, then "
            "launch Fast WOSAC validation on user@10.60.188.78 inside hsb-rl-train."
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
    parser.add_argument("--expected-completed-epochs", type=int, default=DEFAULT_EXPECTED_COMPLETED_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--limit-val-batches", default="auto")
    parser.add_argument("--scorer-scene-num", type=int, default=1680)
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--skip-ckpt-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.scorer_scene_num < 1:
        parser.error("--scorer-scene-num must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.limit_val_batches != "auto":
        try:
            parsed_limit = float(args.limit_val_batches)
        except ValueError:
            parser.error("--limit-val-batches must be 'auto' or a numeric value")
        if parsed_limit <= 0:
            parser.error("--limit-val-batches must be positive")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if args.expected_completed_epochs < 1:
        parser.error("--expected-completed-epochs must be >= 1")
    if args.steps_per_epoch < 1:
        parser.error("--steps-per-epoch must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        run_ssh(args, render_remote_stop(args))
        return

    limit_val_batches = (
        default_limit_val_batches(args)
        if args.limit_val_batches == "auto"
        else args.limit_val_batches
    )
    estimated_scenes = (
        int(limit_val_batches) * args.val_batch_size * args.nproc_per_node
        if isinstance(limit_val_batches, int) or str(limit_val_batches).isdigit()
        else "unknown"
    )

    print(f"[launcher] ssh_host:       {args.ssh_host}")
    print(f"[launcher] remote repo:    {args.remote_project_root}")
    print(f"[launcher] branch:         {args.branch}")
    print(f"[launcher] tmux target:    {args.tmux_session}:{args.window_name}")
    print(f"[launcher] W&B artifact:   {args.wandb_artifact}")
    print(f"[launcher] expected completed epochs: {args.expected_completed_epochs}")
    print(f"[launcher] expected artifact epoch metadata: {args.expected_artifact_epoch}")
    print(f"[launcher] expected global_step: {args.expected_global_step}")
    print(f"[launcher] remote ckpt:    {args.remote_ckpt_path}")
    print(f"[launcher] task_name:      {args.task_name}")
    print(f"[launcher] scorer_scene_num: {args.scorer_scene_num}")
    print(f"[launcher] limit_val_batches: {limit_val_batches} (estimated scenes={estimated_scenes})")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")

    if not args.skip_ckpt_download:
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
