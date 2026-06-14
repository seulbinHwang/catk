#!/usr/bin/env python3
"""Launch goal-free RLFTSim fine-tuning on the existing testas A100x7 pod."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "testas"
DEFAULT_BRANCH = "main"
DEFAULT_PROJECT_ROOT = "/tmp/catk_rlftsim_testas_a100x7"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_NPROC_PER_NODE = "7"
DEFAULT_EXPERIMENT = "rlftsim"
DEFAULT_ACTION = "rlftsim_finetune"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def split_extra_hydra_overrides(overrides: str) -> list[str]:
    if not overrides.strip():
        return []
    try:
        return shlex.split(overrides)
    except ValueError as exc:
        raise ValueError(f"--extra-hydra-overrides is not shell-parseable: {exc}") from exc


def render_env_file(*, args: argparse.Namespace, task_name: str, run_id: str) -> str:
    lines = [
        export_line("CACHE_ROOT", args.cache_root),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("TRAINER_DEVICES", args.nproc_per_node),
        export_line("MASTER_ADDR", "127.0.0.1"),
        export_line("MASTER_PORT", args.master_port),
        export_line("TASK_NAME", task_name),
        export_line("CATK_EXPERIMENT", args.experiment),
        export_line("CATK_ACTION", args.action),
        export_line("LOG_DIR", args.log_dir),
        export_line("CATK_RUN_ID", run_id),
        export_line("TRAIN_BATCH_SIZE", args.train_batch_size),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("MAX_EPOCHS", args.max_epochs),
        export_line("CATK_LR", args.learning_rate),
        export_line("CATK_CKPT_PATH", args.ckpt_path),
        export_line("CATK_HYDRA_OVERRIDES", args.extra_hydra_overrides),
        export_line("CATK_ATTENTION_GRAPH_FP32", args.graph_attn_fp32),
        export_line("WANDB_GROUP", args.wandb_group),
    ]
    optional_env = {
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "LIMIT_TEST_BATCHES": args.limit_test_batches,
    }
    for name, value in optional_env.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
    return "\n".join(lines) + "\n"


def render_run_script(project_root: str, env_file: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

echo "[tmux-run] pod=$(hostname) task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
echo "[tmux-run] attach survives after exit; press Ctrl-b d to detach"
echo

bash scripts/smart_rlftsim_testas_a100x7_finetune.sh
status=$?
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
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_start_command(*, args: argparse.Namespace, task_name: str, run_id: str) -> str:
    safe_task = task_name.replace("/", "_")
    run_root = f"{args.log_dir.rstrip('/')}/tmux_smart_rlftsim_testas_a100x7/{safe_task}"
    env_file = f"{run_root}/{args.pod}.env"
    run_file = f"{run_root}/{args.pod}_run.sh"
    monitor_file = f"{run_root}/{args.pod}_monitor.sh"
    log_file = f"{run_root}/{args.pod}.tmux.log"
    pipe_command = f"cat >> {shq(log_file)}"
    env_text = render_env_file(args=args, task_name=task_name, run_id=run_id)
    run_text = render_run_script(args.project_root, env_file)
    monitor_text = render_monitor_script(args.monitor_interval, task_name)

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

    pull_block = ""
    if args.git_ref:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune
git checkout --detach {shq(args.git_ref)}
"""
    elif args.pull:
        fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(fetch_refspec)}
git checkout -B {shq(args.branch)} {shq(f"origin/{args.branch}")}
git pull --ff-only origin {shq(args.branch)}
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{monitor_text.rstrip()}
CATK_MONITOR
chmod +x {shq(monitor_file)}
tmux split-window -v -l 12 -t {shq(args.session)} {shq(monitor_file)}
tmux select-pane -t {shq(args.session)}
"""

    return f"""set -Eeuo pipefail
if [ ! -d {shq(args.project_root)}/.git ]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
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
tmux pipe-pane -t {shq(args.session)} -o {shq(pipe_command)}
{monitor_block}
echo "[launcher] started tmux session {args.session} on pod {args.pod}"
echo "[launcher] cache root: {args.cache_root}"
echo "[launcher] tmux log: {log_file}"
"""


def render_stop_command(session: str, task_name: str) -> str:
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


def exec_in_pod(args: argparse.Namespace, script: str) -> None:
    cmd = [
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
    if args.dry_run:
        print("kubectl " + " ".join(shq(part) for part in cmd))
        return
    run_kubectl(cmd)


def prepare_project_root(args: argparse.Namespace) -> None:
    script = f"""
set -Eeuo pipefail
root={shq(args.project_root)}
repo={shq(args.repo_url)}
branch={shq(args.branch)}
mkdir -p "$(dirname "$root")"
if [ ! -d "$root/.git" ]; then
  git clone "$repo" "$root"
fi
cd "$root"
git config --global --add safe.directory "$root" || true
git fetch origin --prune
git checkout -B "$branch" "origin/$branch"
git status --short --branch
git rev-parse --short HEAD
"""
    exec_in_pod(args, script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start RLFTSim fine-tuning in tmux on the testas A100x7 pod.",
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--pod", default=os.environ.get("POD", DEFAULT_POD))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument("--repo-url", default=os.environ.get("REPO_URL", "https://github.com/seulbinHwang/catk.git"))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--no-prepare", dest="prepare", action="store_false")
    parser.set_defaults(prepare=True)
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--action",
        choices=["rlftsim_finetune", "validate", "test"],
        default=os.environ.get("CATK_ACTION", DEFAULT_ACTION),
    )
    parser.add_argument("--experiment", default=os.environ.get("CATK_EXPERIMENT", DEFAULT_EXPERIMENT))
    parser.add_argument("--ckpt-path", default=os.environ.get("CKPT_PATH", ""))
    parser.add_argument("--task-name", default="")
    parser.add_argument("--run-id", default=os.environ.get("CATK_RUN_ID", ""))
    parser.add_argument("--session", default=os.environ.get("SESSION", "catk-smart-rlftsim-testas-a100x7"))
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "29571"))
    parser.add_argument("--nproc-per-node", default=os.environ.get("NPROC_PER_NODE", DEFAULT_NPROC_PER_NODE))
    parser.add_argument("--log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_LOG_DIR))
    parser.add_argument("--train-batch-size", default=os.environ.get("TRAIN_BATCH_SIZE", "8"))
    parser.add_argument("--val-batch-size", default=os.environ.get("VAL_BATCH_SIZE", "8"))
    parser.add_argument("--test-batch-size", default=os.environ.get("TEST_BATCH_SIZE", "8"))
    parser.add_argument("--limit-train-batches", default=os.environ.get("LIMIT_TRAIN_BATCHES", ""))
    parser.add_argument("--limit-val-batches", default=os.environ.get("LIMIT_VAL_BATCHES", ""))
    parser.add_argument("--limit-test-batches", default=os.environ.get("LIMIT_TEST_BATCHES", ""))
    parser.add_argument("--max-epochs", default=os.environ.get("MAX_EPOCHS", "1"))
    parser.add_argument("--learning-rate", default=os.environ.get("LEARNING_RATE", "3e-6"))
    parser.add_argument("--graph-attn-fp32", default=os.environ.get("CATK_ATTENTION_GRAPH_FP32", "1"), choices=["0", "1"])
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", "smart_rlftsim_testas_a100x7"))
    parser.add_argument("--extra-hydra-overrides", default=os.environ.get("EXTRA_HYDRA_OVERRIDES", ""))
    parser.add_argument("--monitor-interval", type=int, default=int(os.environ.get("MONITOR_INTERVAL", "30")))
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.stop and not args.ckpt_path:
        parser.error("--ckpt-path or CKPT_PATH is required")
    if not args.nproc_per_node.isdigit() or int(args.nproc_per_node) != 7:
        parser.error("--nproc-per-node must be 7 for the testas A100x7 preset")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if args.extra_hydra_overrides:
        split_extra_hydra_overrides(args.extra_hydra_overrides)
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.task_name = f"smart_rlftsim_testas_a100x7_{stamp}"
    if not args.run_id:
        args.run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session, args.task_name))
        return

    if args.prepare:
        prepare_project_root(args)

    print(f"[launcher] pod:       {args.pod}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] run_id:    {args.run_id}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] cache root: {args.cache_root}")
    print(f"[launcher] ckpt_path:  {args.ckpt_path}")
    print(f"[launcher] GPUs:       {args.nproc_per_node}")

    script = render_start_command(args=args, task_name=args.task_name, run_id=args.run_id)
    exec_in_pod(args, script)

    print("\nAttach command:")
    print(
        "  kubectl exec -it "
        f"-n {args.namespace} {args.pod} -c {args.container} -- "
        f"tmux attach -t {args.session}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
