#!/usr/bin/env python3
"""Launch SMART NTP pretrain on hsb-npc-training(H100x4) + wo-pvc(H100x2).

This launcher never creates, deletes, or restarts pods. It only runs
``kubectl exec`` against already-running pods and starts/kills tmux sessions
inside them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["hsb-npc-training", "wo-pvc"]
DEFAULT_BRANCH = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_CACHE_ROOT_BY_POD = {
    "hsb-npc-training": "/workspace/womd_v1_3/SMART_cache",
    "wo-pvc": "/workspace/womd_v1_3/SMART_cache",
}
DEFAULT_GPU_COUNT_BY_POD = {
    "hsb-npc-training": 4,
    "wo-pvc": 2,
}
STRICT_EXPERIMENT = "pre_bc_a100x4x2"
MAX_TRAIN_BATCH_SIZE = 24
STRICT_ACCUMULATE_GRAD_BATCHES = "1"
DEFAULT_EXTRA_HYDRA_OVERRIDES = (
    "trainer.strategy._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy "
    "+trainer.strategy.cluster_environment._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment"
)


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


def pod_ip(namespace: str, pod: str) -> str:
    return run_kubectl(
        [
            "get",
            "pod",
            pod,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.podIP}",
        ],
        capture=True,
    )


def pod_gpu_count(namespace: str, container: str, pod: str) -> int:
    output = run_kubectl(
        [
            "exec",
            "-n",
            namespace,
            pod,
            "-c",
            container,
            "--",
            "bash",
            "-lc",
            "nvidia-smi -L 2>/dev/null | wc -l",
        ],
        capture=True,
    )
    count = int(output.strip())
    if count < 1:
        raise RuntimeError(f"no GPUs found in pod {pod}")
    return count


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def parse_pod_cache_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--pod-cache-root entries must use POD=PATH, "
                f"but got: {value!r}"
            )
        pod, path = value.split("=", 1)
        pod = pod.strip()
        path = path.strip()
        if not pod or not path:
            raise ValueError(
                "--pod-cache-root entries must include both POD and PATH, "
                f"but got: {value!r}"
            )
        roots[pod] = path
    return roots


def split_extra_hydra_overrides(overrides: str) -> list[str]:
    if not overrides.strip():
        return []
    try:
        return shlex.split(overrides)
    except ValueError as exc:
        raise ValueError(f"--extra-hydra-overrides is not shell-parseable: {exc}") from exc


def validate_nproc_per_node(value: str) -> str:
    if value in {"auto", "gpu"}:
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--nproc-per-node must be a positive integer or one of: auto, gpu"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--nproc-per-node must be >= 1")
    return value


def validate_strict_pretrain(args: argparse.Namespace) -> None:
    if args.stop:
        return
    if args.action != "fit" or args.experiment != STRICT_EXPERIMENT:
        return

    if args.train_batch_size:
        try:
            train_batch_size = int(args.train_batch_size)
        except ValueError as exc:
            raise ValueError(
                f"--train-batch-size must be a positive integer, got "
                f"{args.train_batch_size!r}."
            ) from exc
        if train_batch_size < 1 or train_batch_size > MAX_TRAIN_BATCH_SIZE:
            raise ValueError(
                f"{STRICT_EXPERIMENT} must use train_batch_size in "
                f"[1, {MAX_TRAIN_BATCH_SIZE}], but got "
                f"--train-batch-size {args.train_batch_size!r}."
            )
    if (
        args.accumulate_grad_batches
        and args.accumulate_grad_batches != STRICT_ACCUMULATE_GRAD_BATCHES
    ):
        raise ValueError(
            f"{STRICT_EXPERIMENT} must use accumulate_grad_batches="
            f"{STRICT_ACCUMULATE_GRAD_BATCHES}, but got "
            f"--accumulate-grad-batches {args.accumulate_grad_batches!r}."
        )

    for override in split_extra_hydra_overrides(args.extra_hydra_overrides):
        if override.startswith("data.train_batch_size="):
            value = override.split("=", 1)[1]
            try:
                train_batch_size = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"{STRICT_EXPERIMENT} requires integer data.train_batch_size, "
                    f"but got override {override!r}."
                ) from exc
            if train_batch_size < 1 or train_batch_size > MAX_TRAIN_BATCH_SIZE:
                raise ValueError(
                    f"{STRICT_EXPERIMENT} must use data.train_batch_size in "
                    f"[1, {MAX_TRAIN_BATCH_SIZE}], but got override {override!r}."
                )
        elif override.startswith("trainer.accumulate_grad_batches="):
            value = override.split("=", 1)[1]
            if value != STRICT_ACCUMULATE_GRAD_BATCHES:
                raise ValueError(
                    f"{STRICT_EXPERIMENT} must use trainer.accumulate_grad_batches="
                    f"{STRICT_ACCUMULATE_GRAD_BATCHES}, but got override {override!r}."
                )


def cache_root_for_pod(args: argparse.Namespace, pod: str) -> str:
    if pod in args.pod_cache_root_map:
        return args.pod_cache_root_map[pod]
    if args.cache_root:
        return args.cache_root
    return DEFAULT_CACHE_ROOT_BY_POD.get(pod, DEFAULT_CACHE_ROOT)


def combine_hydra_overrides(user_overrides: str) -> str:
    parts = split_extra_hydra_overrides(DEFAULT_EXTRA_HYDRA_OVERRIDES)
    parts.extend(split_extra_hydra_overrides(user_overrides))
    return " ".join(shq(part) for part in parts)


def render_env_file(
    *,
    args: argparse.Namespace,
    cache_root: str,
    rank: int,
    master_addr: str,
    task_name: str,
    local_world_size: int,
    manual_rank_offset: int,
    manual_world_size: int,
) -> str:
    lines = [
        export_line("CACHE_ROOT", cache_root),
        export_line("NNODES", len(args.pods)),
        export_line("NPROC_PER_NODE", local_world_size),
        export_line("NODE_RANK", rank),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("MANUAL_RANK_OFFSET", manual_rank_offset),
        export_line("MANUAL_WORLD_SIZE", manual_world_size),
        export_line("TRAINER_DEVICES", local_world_size),
        export_line("TASK_NAME", task_name),
        export_line("CATK_EXPERIMENT", args.experiment),
        export_line("CATK_ACTION", args.action),
        export_line("LOG_DIR", args.log_dir),
    ]
    optional_env = {
        "CATK_CKPT_PATH": args.ckpt_path,
        "CATK_AUTO_RESUME": "true" if args.auto_resume else "",
        "CATK_RESUME_TASK_NAME": args.resume_task_name,
        "CATK_RESUME_CHECKPOINT_NAME": args.resume_checkpoint_name,
        "CATK_RESUME_REQUIRE_CHECKPOINT": "false"
        if args.allow_missing_resume_checkpoint
        else "",
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "TEST_BATCH_SIZE": args.test_batch_size,
        "ACCUMULATE_GRAD_BATCHES": args.accumulate_grad_batches,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "CATK_LR": args.learning_rate,
        "CATK_HYDRA_OVERRIDES": combine_hydra_overrides(args.extra_hydra_overrides),
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

echo "[tmux-run] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
echo "[tmux-run] attach survives after exit; press Ctrl-b d to detach"
echo

bash scripts/smart_ntp_h100x4_h100x2_pretrain.sh
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
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_start_command(
    *,
    args: argparse.Namespace,
    pod: str,
    rank: int,
    master_addr: str,
    task_name: str,
    local_world_size: int,
    manual_rank_offset: int,
    manual_world_size: int,
) -> str:
    safe_task = task_name.replace("/", "_")
    run_root = f"{args.log_dir.rstrip('/')}/tmux_smart_ntp_h100x4_h100x2/{safe_task}"
    env_file = f"{run_root}/{pod}.env"
    run_file = f"{run_root}/{pod}_run.sh"
    monitor_file = f"{run_root}/{pod}_monitor.sh"
    log_file = f"{run_root}/{pod}.tmux.log"
    pipe_command = f"cat >> {shq(log_file)}"
    cache_root = cache_root_for_pod(args, pod)
    env_text = render_env_file(
        args=args,
        cache_root=cache_root,
        rank=rank,
        master_addr=master_addr,
        task_name=task_name,
        local_world_size=local_world_size,
        manual_rank_offset=manual_rank_offset,
        manual_world_size=manual_world_size,
    )
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
        branch_ref = f"refs/heads/{args.branch}"
        origin_ref = f"origin/{args.branch}"
        fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq(branch_ref)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq(origin_ref)}
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
echo "[launcher] started tmux session {args.session} on pod {pod}"
echo "[launcher] local_world_size: {local_world_size}"
echo "[launcher] rank_offset: {manual_rank_offset}"
echo "[launcher] cache root: {cache_root}"
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


def exec_in_pod(
    namespace: str,
    container: str,
    pod: str,
    script: str,
    *,
    dry_run: bool,
) -> None:
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
        description="Start SMART NTP H100x4+H100x2 pretrain in tmux on existing pods.",
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", " ".join(DEFAULT_PODS)).split(),
    )
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT),
    )
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument(
        "--git-ref",
        default="",
        help="Checkout this exact git ref/SHA on every pod before launch.",
    )
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default="")
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help="Override CACHE_ROOT for one pod. Can be repeated.",
    )
    parser.add_argument("--action", choices=["fit", "validate", "test"], default="fit")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="For action=fit, resume from the newest task-local epoch_last.ckpt.",
    )
    parser.add_argument(
        "--resume-task-name",
        default="",
        help="Task name to search when --auto-resume is set. Defaults to --task-name.",
    )
    parser.add_argument("--resume-checkpoint-name", default="epoch_last.ckpt")
    parser.add_argument(
        "--allow-missing-resume-checkpoint",
        action="store_true",
        help="With --auto-resume, start from scratch if no checkpoint is found.",
    )
    parser.add_argument("--experiment", default=STRICT_EXPERIMENT)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--session", default="catk-smart-ntp-h100x4-h100x2")
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29531")
    parser.add_argument("--nproc-per-node", type=validate_nproc_per_node, default="gpu")
    parser.add_argument("--log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_LOG_DIR))
    parser.add_argument("--train-batch-size", default="")
    parser.add_argument("--val-batch-size", default="")
    parser.add_argument("--test-batch-size", default="")
    parser.add_argument("--accumulate-grad-batches", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--learning-rate", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended to the run.",
    )
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        args.pod_cache_root_map = parse_pod_cache_roots(args.pod_cache_root)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_strict_pretrain(args)
    except ValueError as exc:
        parser.error(str(exc))

    if len(args.pods) != 2 and not args.stop:
        parser.error("this preset expects exactly two pods: H100x4 + H100x2")
    if args.action in {"validate", "test"} and not args.ckpt_path and not args.stop:
        parser.error(f"--ckpt-path is required when --action={args.action}")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.task_name = f"smart_ntp_pretrain_h100x4_h100x2_{stamp}"
    return args


def resolve_local_world_sizes(args: argparse.Namespace) -> dict[str, int]:
    local_world_sizes: dict[str, int] = {}
    for pod in args.pods:
        if args.dry_run and args.nproc_per_node in {"auto", "gpu"}:
            local_size = DEFAULT_GPU_COUNT_BY_POD.get(pod, 1)
        elif args.nproc_per_node in {"auto", "gpu"}:
            local_size = pod_gpu_count(args.namespace, args.container, pod)
        else:
            local_size = int(args.nproc_per_node)
        if local_size < 1:
            raise RuntimeError(f"invalid local world size for {pod}: {local_size}")
        local_world_sizes[pod] = local_size
    return local_world_sizes


def main() -> None:
    args = parse_args()
    if args.stop:
        for pod in args.pods:
            exec_in_pod(
                args.namespace,
                args.container,
                pod,
                render_stop_command(args.session, args.task_name),
                dry_run=args.dry_run,
            )
        return

    master_addr = args.master_addr or (
        "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    )
    local_world_sizes = resolve_local_world_sizes(args)
    rank_offsets: dict[str, int] = {}
    offset = 0
    for pod in args.pods:
        rank_offsets[pod] = offset
        offset += local_world_sizes[pod]
    manual_world_size = offset

    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print("[launcher] manual rank layout:")
    for pod in args.pods:
        print(
            f"  {pod}: local_world_size={local_world_sizes[pod]} "
            f"rank_offset={rank_offsets[pod]} cache={cache_root_for_pod(args, pod)}"
        )
    print(f"[launcher] manual world_size: {manual_world_size}")

    for rank, pod in enumerate(args.pods):
        script = render_start_command(
            args=args,
            pod=pod,
            rank=rank,
            master_addr=master_addr,
            task_name=args.task_name,
            local_world_size=local_world_sizes[pod],
            manual_rank_offset=rank_offsets[pod],
            manual_world_size=manual_world_size,
        )
        exec_in_pod(args.namespace, args.container, pod, script, dry_run=args.dry_run)

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
