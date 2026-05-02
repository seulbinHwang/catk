#!/usr/bin/env python3
"""Launch CAT-K DDP on static pods with different GPU counts per pod.

The regular static launcher uses one torchrun per physical pod and therefore
requires the same nproc_per_node on every pod. This launcher treats each GPU as
one logical 1-GPU node: e.g. pods with 8 and 7 GPUs become a 15-node
torchrun job with nproc_per_node=1. It is a pragmatic recovery path for MLX
static pods when only a 7-GPU V100 box is available next to an 8-GPU box.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_BRANCH = "semi_continuous_track_loss"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout else ""


def pod_ip(namespace: str, pod: str) -> str:
    return run_kubectl(
        ["get", "pod", pod, "-n", namespace, "-o", "jsonpath={.status.podIP}"],
        capture=True,
    )


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def render_env_file(
    *,
    args: argparse.Namespace,
    world_size: int,
    master_addr: str,
    task_name: str,
    run_dir: str,
) -> str:
    extra_overrides = args.extra_hydra_overrides.strip()
    forced_overrides = f"hydra.run.dir={run_dir}"
    if extra_overrides:
        extra_overrides = f"{extra_overrides} {forced_overrides}"
    else:
        extra_overrides = forced_overrides

    lines = [
        export_line("CACHE_ROOT", args.cache_root),
        export_line("PRETRAIN_CKPT", args.pretrain_ckpt),
        export_line("NNODES", world_size),
        export_line("NPROC_PER_NODE", 1),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("TASK_NAME", task_name),
        export_line("CATK_EXPERIMENT", args.experiment),
        export_line("CATK_ACTION", args.action),
        export_line("CATK_LR", args.learning_rate),
        export_line("LOG_DIR", args.log_dir),
        export_line("CATK_HYDRA_OVERRIDES", extra_overrides),
    ]
    optional_env = {
        "CATK_CKPT_PATH": args.ckpt_path,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "SOFT_LIMIT_RATIO": args.soft_limit_ratio,
        "TOPK_VIOLATION_K": args.topk_violation_k,
        "BACKPROP_LAST_K": args.backprop_last_k,
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "ACCUMULATE_GRAD_BATCHES": args.accumulate_grad_batches,
    }
    for name, value in optional_env.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
    return "\n".join(lines) + "\n"


def render_run_script(
    *,
    project_root: str,
    env_file: str,
    rank_start: int,
    local_gpu_count: int,
    pod: str,
    rank_log_dir: str,
) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

mkdir -p {shq(rank_log_dir)}
echo "[hetero-run] pod=$(hostname) task=${{TASK_NAME}}"
echo "[hetero-run] logical ranks {rank_start}..$(({rank_start} + {local_gpu_count} - 1))"
echo "[hetero-run] world_size=${{NNODES}} master=${{MASTER_ADDR}}:${{MASTER_PORT}}"
echo "[hetero-run] started at $(date '+%F %T')"
echo

pids=()
cleanup() {{
  for pid in "${{pids[@]}}"; do
    kill "$pid" 2>/dev/null || true
  done
}}
trap cleanup INT TERM

for local_gpu in $(seq 0 $(({local_gpu_count} - 1))); do
  logical_rank=$(({rank_start} + local_gpu))
  rank_log={shq(rank_log_dir)}/rank${{logical_rank}}.log
  (
    set +e
    export CUDA_VISIBLE_DEVICES="$local_gpu"
    export NODE_RANK="$logical_rank"
    export LOCAL_RANK=0
    echo "[rank $logical_rank] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash scripts/mlx_finetune_draft_flow_v100x8_multinode.sh 2>&1
    code=$?
    echo "[rank $logical_rank] exited with status $code at $(date '+%F %T')"
    exit "$code"
  ) > >(sed -u "s/^/[{pod} r${{logical_rank}} g${{local_gpu}}] /" | tee -a "$rank_log") 2>&1 &
  pids+=("$!")
done

overall=0
for pid in "${{pids[@]}}"; do
  wait "$pid" || overall=$?
done
trap - INT TERM

echo
echo "[hetero-run] exited with status $overall at $(date '+%F %T')"
echo "[hetero-run] leaving shell open for inspection"
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


def render_start_command(
    *,
    args: argparse.Namespace,
    pod: str,
    rank_start: int,
    local_gpu_count: int,
    master_addr: str,
    task_name: str,
    run_dir: str,
    world_size: int,
) -> str:
    safe_task = task_name.replace("/", "_")
    run_root = f"{args.log_dir.rstrip('/')}/tmux_hetero_static_multinode/{safe_task}"
    env_file = f"{run_root}/{pod}.env"
    run_file = f"{run_root}/{pod}_run.sh"
    monitor_file = f"{run_root}/{pod}_monitor.sh"
    rank_log_dir = f"{run_root}/{pod}_rank_logs"
    log_file = f"{run_root}/{pod}.tmux.log"
    pipe_command = f"cat >> {shq(log_file)}"
    env_text = render_env_file(
        args=args,
        world_size=world_size,
        master_addr=master_addr,
        task_name=task_name,
        run_dir=run_dir,
    )
    run_text = render_run_script(
        project_root=args.project_root,
        env_file=env_file,
        rank_start=rank_start,
        local_gpu_count=local_gpu_count,
        pod=pod,
        rank_log_dir=rank_log_dir,
    )
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
  exit 3
fi
"""

    pull_block = ""
    if args.pull:
        fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq('refs/heads/' + args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq('origin/' + args.branch)}
fi
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
tmux select-pane -t {shq(args.session)}:0.0
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
tmux pipe-pane -t {shq(args.session)}:0.0 -o {shq(pipe_command)}
{monitor_block}
echo "[launcher] started hetero tmux session {args.session} on pod {pod}"
echo "[launcher] world_size={world_size}; local_gpus={local_gpu_count}; rank_start={rank_start}"
echo "[launcher] tmux log: {log_file}"
"""


def render_stop_command(session: str) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
fi
"""


def exec_in_pod(namespace: str, container: str, pod: str, script: str, *, dry_run: bool) -> None:
    cmd = [
        "exec",
        "-n",
        namespace,
        pod,
        "-c",
        container,
        "--",
        "bash",
        "-lc",
        script,
    ]
    if dry_run:
        print("kubectl " + " ".join(shq(part) for part in cmd))
        return
    run_kubectl(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start heterogeneous static multi-node fine-tuning in tmux.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", required=True)
    parser.add_argument("--nproc-per-pod", nargs="+", type=int, required=True)
    parser.add_argument("--container", default="main")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--pretrain-ckpt", default="")
    parser.add_argument("--action", choices=["finetune", "fit"], default="finetune")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--experiment", default="finetune_draft_flow_v100x8")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--session", default="catk-draft-hetero")
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29537")
    parser.add_argument("--learning-rate", default="2e-4")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--soft-limit-ratio", default="")
    parser.add_argument("--topk-violation-k", default="")
    parser.add_argument("--backprop-last-k", default="")
    parser.add_argument("--train-batch-size", default="")
    parser.add_argument("--accumulate-grad-batches", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(args.nproc_per_pod):
        parser.error("--pods and --nproc-per-pod must have the same length")
    if any(count < 1 for count in args.nproc_per_pod):
        parser.error("all --nproc-per-pod values must be >= 1")
    if not args.pretrain_ckpt and not args.ckpt_path and not args.stop:
        parser.error("--pretrain-ckpt or --ckpt-path is required unless --stop is set")
    if args.action == "fit" and not args.ckpt_path and not args.stop:
        parser.error("--ckpt-path is required when --action=fit")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        world_size = sum(args.nproc_per_pod)
        args.task_name = f"catk_draft_hetero_v100x{world_size}_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        for pod in args.pods:
            exec_in_pod(
                args.namespace,
                args.container,
                pod,
                render_stop_command(args.session),
                dry_run=args.dry_run,
            )
        return

    world_size = sum(args.nproc_per_pod)
    master_addr = args.master_addr or pod_ip(args.namespace, args.pods[0])
    run_stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"{args.log_dir.rstrip('/')}/{args.task_name}/runs/{run_stamp}"

    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] world_size: {world_size}")
    print(f"[launcher] run_dir:    {run_dir}")

    rank_start = 0
    for pod, local_gpu_count in zip(args.pods, args.nproc_per_pod):
        script = render_start_command(
            args=args,
            pod=pod,
            rank_start=rank_start,
            local_gpu_count=local_gpu_count,
            master_addr=master_addr,
            task_name=args.task_name,
            run_dir=run_dir,
            world_size=world_size,
        )
        exec_in_pod(args.namespace, args.container, pod, script, dry_run=args.dry_run)
        rank_start += local_gpu_count

    print("\nAttach commands:")
    for pod in args.pods:
        print(
            "  kubectl exec -it "
            f"-n {args.namespace} {pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
