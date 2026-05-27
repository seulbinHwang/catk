#!/usr/bin/env python3
"""Launch H100x6 hsb-npc-training-1 train+val Flow pretrain.

This launcher targets the existing ``hsb-npc-training-1`` pod only. It does not
create, delete, or restart the pod; it only starts/replaces one tmux session and
the matching training processes inside that pod.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "hsb-npc-training-1"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_stable_w_val"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_PYTHON = "/mnt/nuplan/miniforge/envs/catk/bin/python"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_"
    "train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18"
)
DEFAULT_SESSION = "catk-control-pretrain-h100x6-hsb1-prefix-default-noslip-train-plus-validation"
DEFAULT_METADATA_CACHE = (
    "dataset_metadata/womd_training_validation_memory_balance_h100x6_hsb1.pt"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
    command = ["kubectl", *args]
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


def remote_script_quote(script: str) -> str:
    return shq(script)


def render_stop_script(session: str, task_name: str) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
fi
TASK_NAME_TO_STOP={shq(task_name)}
mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
"""


def render_tmux_run_script(
    *,
    args: argparse.Namespace,
    metadata_cache_path: str,
    run_root: str,
    env_file: str,
) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1
export CATK_REMOTE_PYTHON={shq(args.remote_python)}

cd {shq(args.project_root)}
set -a
source {shq(env_file)}
set +a

echo "[tmux-run] pod=$(hostname) task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
echo "[tmux-run] attach survives after exit; press Ctrl-b d to detach"
echo "[tmux-run] metadata cache: {metadata_cache_path}"
echo

mkdir -p "$RUN_ROOT"
torch_status_file="$RUN_ROOT/$(hostname).torchrun_status"
torch_pgid_file="$RUN_ROOT/$(hostname).torchrun_pgid"
rm -f "$torch_status_file" "$torch_pgid_file"

task_process_pids() {{
  pgrep -f "task_name=${{TASK_NAME}}" 2>/dev/null | while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
      echo "$pid"
    fi
  done
}}

terminate_task_processes() {{
  local reason="${{1:-cleanup}}"
  local pids=()
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} == 0 )); then
    return 0
  fi
  echo "[tmux-run] terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 15
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[tmux-run] force killing task processes for $reason: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
}}

terminate_process_group() {{
  local pgid="$1"
  if [[ -z "$pgid" || "$pgid" == "0" ]]; then
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 20
  kill -KILL -- "-$pgid" 2>/dev/null || true
}}

terminate_attempt_processes() {{
  local pgid=""
  pgid="$(cat "$torch_pgid_file" 2>/dev/null || true)"
  terminate_process_group "$pgid"
  terminate_task_processes "$1"
}}

terminate_task_processes "pre-run stale cleanup"
(
  set +e
  setsid bash -c 'pgid_file="$1"; shift; echo "$$" > "$pgid_file"; exec "$@"' \\
    bash "$torch_pgid_file" bash scripts/h100x4_multinode_pretrain.sh
  echo "$?" > "$torch_status_file"
) &
runner_pid=$!

wait "$runner_pid"
status="$(cat "$torch_status_file" 2>/dev/null || echo 1)"
if [[ "$status" != "0" ]]; then
  terminate_attempt_processes "post-run cleanup status=$status"
fi

echo
echo "[tmux-run] exited with status $status at $(date '+%F %T')"
echo "[tmux-run] leaving shell open for inspection"
exec bash
"""


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_start_script(args: argparse.Namespace, metadata_cache_path: str) -> str:
    safe_task = args.task_name.replace("/", "_")
    run_root = f"{args.remote_log_dir.rstrip('/')}/tmux_h100x6_hsb1_pretrain/{safe_task}"
    env_file = f"{run_root}/{args.pod}.env"
    run_file = f"{run_root}/{args.pod}_run.sh"
    monitor_file = f"{run_root}/{args.pod}_monitor.sh"
    log_file = f"{run_root}/{args.pod}.tmux.log"
    extra_hydra_overrides = " ".join(
        part
        for part in (
            "trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy",
            "trainer.strategy.cluster_environment=null",
            "trainer.check_val_every_n_epoch=16",
            f"data.train_memory_balance_metadata_cache={metadata_cache_path}",
            args.extra_hydra_overrides.strip(),
        )
        if part
    )
    optional_env = {
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
    }
    env_lines = [
        f"export CACHE_ROOT={shq(args.cache_root)}",
        "export NNODES=1",
        f"export NPROC_PER_NODE={int(args.gpus)}",
        "export NODE_RANK=0",
        "export MASTER_ADDR=127.0.0.1",
        f"export MASTER_PORT={shq(args.master_port)}",
        f"export TASK_NAME={shq(args.task_name)}",
        f"export CATK_EXPERIMENT={shq(args.experiment)}",
        "export CATK_ACTION=fit",
        f"export LOG_DIR={shq(args.remote_log_dir)}",
        f"export RUN_ROOT={shq(run_root)}",
        f"export TRAIN_BATCH_SIZE={int(args.train_batch_size)}",
        f"export VAL_BATCH_SIZE={int(args.val_batch_size)}",
        f"export CATK_LR={shq(args.learning_rate)}",
        f"export CATK_HYDRA_OVERRIDES={shq(extra_hydra_overrides)}",
    ]
    env_lines.extend(
        f"export {name}={shq(value)}"
        for name, value in optional_env.items()
        if value not in (None, "")
    )
    env_text = "\n".join(env_lines) + "\n"

    pull_block = ""
    if args.git_ref:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune {shq(args.branch)}
git checkout -f {shq(args.git_ref)}
"""
    else:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(args.branch)}
if git show-ref --verify --quiet {shq(f"refs/heads/{args.branch}")}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq(f"origin/{args.branch}")}
fi
git pull --ff-only origin {shq(args.branch)}
"""

    preflight_block = ""
    if not args.skip_memory_metadata_preflight:
        raw_training = f"{args.cache_root.rstrip('/')}/training"
        raw_validation = f"{args.cache_root.rstrip('/')}/validation"
        force_arg = " --force" if args.force_memory_metadata_rebuild else ""
        preflight_block = f"""
mkdir -p "$(dirname {shq(metadata_cache_path)})"
test -d {shq(raw_training)}
test -d {shq(raw_validation)}
{shq(args.remote_python)} tools/build_memory_balance_metadata.py \\
  --raw-dir {shq(raw_training)} \\
  --raw-dir {shq(raw_validation)} \\
  --cache-path {shq(metadata_cache_path)} \\
  --num-workers {int(args.memory_metadata_num_workers)}{force_arg}
"""

    replace_block = ""
    if args.replace:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
"""
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  echo "[launcher] attach with: tmux attach -t {args.session}" >&2
  exit 3
fi
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_text = render_monitor_script(args.monitor_interval, args.task_name)
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{monitor_text.rstrip()}
CATK_MONITOR
chmod +x {shq(monitor_file)}
tmux split-window -v -l 12 -t {shq(args.session)} {shq(monitor_file)}
tmux select-pane -t {shq(args.session)}
"""

    run_text = render_tmux_run_script(
        args=args,
        metadata_cache_path=metadata_cache_path,
        run_root=run_root,
        env_file=env_file,
    )
    return f"""set -Eeuo pipefail
if [ ! -d {shq(args.project_root)}/.git ]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
{preflight_block}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(env_file)} <<'CATK_ENV'
{env_text.rstrip()}
CATK_ENV
cat > {shq(run_file)} <<'CATK_RUN'
{run_text.rstrip()}
CATK_RUN
chmod +x {shq(run_file)}
: > {shq(log_file)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq(f"cat >> {shq(log_file)}")}
{monitor_block}
echo "[launcher] started tmux session {args.session} on pod {args.pod}"
echo "[launcher] cache root: {args.cache_root}"
echo "[launcher] train raw dirs: {args.cache_root.rstrip()}/training + {args.cache_root.rstrip()}/validation"
echo "[launcher] metadata cache: {metadata_cache_path}"
echo "[launcher] tmux log: {log_file}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch semi_control_stable_w_val train+validation Flow pretrain "
            "on hsb-npc-training-1 H100x6."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--pod", default=os.environ.get("POD", DEFAULT_POD))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or DEFAULT_BRANCH)
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--remote-python", default=os.environ.get("CATK_REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--gpus", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=18)
    parser.add_argument("--val-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--master-port", default="29651")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="128")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--memory-metadata-cache-path", default="")
    parser.add_argument("--memory-metadata-num-workers", type=int, default=8)
    parser.add_argument("--force-memory-metadata-rebuild", action="store_true")
    parser.add_argument("--skip-memory-metadata-preflight", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    args = parser.parse_args()

    if args.gpus < 1:
        parser.error("--gpus must be >= 1")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.memory_metadata_num_workers < 1:
        parser.error("--memory-metadata-num-workers must be >= 1")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    metadata_cache_path = args.memory_metadata_cache_path or (
        f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE}"
    )

    script = (
        render_stop_script(args.session, args.task_name)
        if args.stop
        else render_start_script(args, metadata_cache_path)
    )
    if args.dry_run:
        print("[dry-run] command:")
        print(
            "kubectl "
            + " ".join(
                shq(part)
                for part in [
                    "exec",
                    "-n",
                    args.namespace,
                    args.pod,
                    "-c",
                    args.container,
                    "--",
                    "bash",
                    "-lc",
                    script,
                ]
            )
        )
        return 0

    run_kubectl(
        [
            "exec",
            "-n",
            args.namespace,
            args.pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            script,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
