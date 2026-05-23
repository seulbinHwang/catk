#!/usr/bin/env python3
"""Launch A100x7 execution-context pretrain on the existing testas pod.

This launcher targets one already-running ``testas`` pod with seven A100 GPUs.
It never creates, deletes, or restarts the pod. It prepares/verifies the
memory-balance metadata cache, starts a tmux training session, and lowers
``train_batch_size`` by one on CUDA OOM before resuming from the latest
checkpoint.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_POD = "testas"
DEFAULT_BRANCH = "semi_control_rolling_fd"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4x2_execctx_balanced"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_a100x7_testas_"
    "execctx_prefix_balanced_lr6e-4_bs16_oomretry"
)
DEFAULT_SESSION = "catk-control-pretrain-a100x7-testas-execctx-balanced-bs16-retry"
DEFAULT_METADATA_CACHE_RELATIVE = "dataset_metadata/womd_training_memory_balance_v1.pt"

OOM_REGEX = (
    r"OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|"
    r"cuda runtime error.*out of memory|torch\.OutOfMemoryError|"
    r"CUDA_ERROR_OUT_OF_MEMORY"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: argparse.Namespace, script: str, *, capture: bool = False) -> str:
    command = [
        "kubectl",
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
        print(" ".join(shq(part) for part in command))
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture else ""


def metadata_cache_path(args: argparse.Namespace) -> str:
    if args.metadata_cache_path:
        return args.metadata_cache_path
    return f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE_RELATIVE}"


def safe_task_name(task_name: str) -> str:
    return task_name.replace("/", "_")


def remote_run_root(args: argparse.Namespace) -> str:
    return (
        f"{args.remote_log_dir.rstrip('/')}/"
        f"tmux_a100x7_testas_pretrain/{safe_task_name(args.task_name)}"
    )


def remote_tmux_log(args: argparse.Namespace) -> str:
    return f"{remote_run_root(args)}/{args.pod}.tmux.log"


def remote_status_file(args: argparse.Namespace) -> str:
    return f"{remote_run_root(args)}/{args.pod}.torchrun_status"


def local_retry_log_dir(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "logs" / "_a100x7_testas_pretrain_oom_retry" / args.task_name


def remote_git_prepare_script(args: argparse.Namespace) -> str:
    if args.no_pull:
        return f"git config --global --add safe.directory {shq(args.project_root)} || true"
    if args.git_ref:
        fetch_refspec = f"+{args.branch}:refs/remotes/origin/{args.branch}"
        return " && ".join(
            [
                f"git config --global --add safe.directory {shq(args.project_root)} || true",
                f"git update-ref -d {shq(f'refs/remotes/origin/{args.branch}')} || true",
                f"git fetch origin --prune {shq(fetch_refspec)}",
                f"git checkout -f {shq(args.git_ref)}",
            ]
        )
    branch_ref = f"refs/heads/{args.branch}"
    origin_ref = f"origin/{args.branch}"
    fetch_refspec = f"+{args.branch}:refs/remotes/origin/{args.branch}"
    return " && ".join(
        [
            f"git config --global --add safe.directory {shq(args.project_root)} || true",
            f"git update-ref -d {shq(f'refs/remotes/origin/{args.branch}')} || true",
            f"git fetch origin --prune {shq(fetch_refspec)}",
            (
                f"if git show-ref --verify --quiet {shq(branch_ref)}; then "
                f"git checkout {shq(args.branch)}; "
                f"else git checkout -b {shq(args.branch)} {shq(origin_ref)}; fi"
            ),
            f"git pull --ff-only origin {shq(args.branch)}",
        ]
    )


def prebuild_metadata(args: argparse.Namespace) -> None:
    command = [
        "cd",
        shq(args.project_root),
        "&&",
        remote_git_prepare_script(args),
        "&&",
        'CATK_REMOTE_PYTHON="${CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}"',
        "&&",
        '"$CATK_REMOTE_PYTHON"',
        "tools/build_memory_balance_metadata.py",
        "--raw-dir",
        shq(f"{args.cache_root.rstrip('/')}/training"),
        "--cache-path",
        shq(metadata_cache_path(args)),
        "--num-workers",
        shq(args.metadata_num_workers),
    ]
    if args.force_metadata:
        command.append("--force")
    run_kubectl(args, " ".join(str(part) for part in command))


def verify_metadata_cache(args: argparse.Namespace) -> None:
    path = metadata_cache_path(args)
    script = (
        f"if [[ ! -f {shq(path)} ]]; then "
        f"echo {shq('[metadata-check] missing memory-balance metadata cache: ' + path)} >&2; "
        "echo '[metadata-check] rerun with --prebuild-metadata, or pass --metadata-cache-path.' >&2; "
        "exit 2; "
        "fi"
    )
    run_kubectl(args, script)


def hydra_overrides(args: argparse.Namespace, train_batch_size: int, ckpt_path: str) -> list[str]:
    overrides = [
        f"experiment={args.experiment}",
        "action=fit",
        "trainer=ddp",
        f"trainer.devices={args.nproc_per_node}",
        "trainer.num_nodes=1",
        "trainer.enable_progress_bar=true",
        f"trainer.check_val_every_n_epoch={args.check_val_every_n_epoch}",
        "trainer.use_distributed_sampler=false",
        f"paths.cache_root={args.cache_root}",
        f"paths.log_dir={args.remote_log_dir}",
        f"task_name={args.task_name}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        "data.train_memory_balanced_batches=true",
        f"data.train_memory_balance_metadata_cache={metadata_cache_path(args)}",
        "data.train_memory_balance_build_on_missing=false",
        f"model.model_config.lr={args.learning_rate}",
        f"model.model_config.n_rollout_closed_val={args.n_rollout_closed_val}",
    ]
    if ckpt_path:
        overrides.append(f"ckpt_path={ckpt_path}")
    if args.limit_train_batches:
        overrides.append(f"trainer.limit_train_batches={args.limit_train_batches}")
    if args.limit_val_batches:
        overrides.append(f"trainer.limit_val_batches={args.limit_val_batches}")
    if args.max_epochs:
        overrides.append(f"trainer.max_epochs={args.max_epochs}")
    if args.extra_hydra_overrides:
        overrides.extend(shlex.split(args.extra_hydra_overrides))
    return overrides


def render_monitor_script(interval: int) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_run_script(args: argparse.Namespace, env_file: str, train_batch_size: int, ckpt_path: str) -> str:
    override_args = " ".join(shq(item) for item in hydra_overrides(args, train_batch_size, ckpt_path))
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR="${{HYDRA_FULL_ERROR:-1}}"
export TF_CPP_MIN_LOG_LEVEL="${{TF_CPP_MIN_LOG_LEVEL:-2}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-eth0}}"
export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${{NCCL_SOCKET_FAMILY:-AF_INET}}"
export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
export NCCL_NVLS_ENABLE="${{NCCL_NVLS_ENABLE:-0}}"
export NCCL_CUMEM_ENABLE="${{NCCL_CUMEM_ENABLE:-0}}"
export CATK_ATTENTION_GRAPH_FP32="${{CATK_ATTENTION_GRAPH_FP32:-1}}"
export CATK_REMOTE_PYTHON="${{CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}}"

cd {shq(args.project_root)}
set -a
source {shq(env_file)}
set +a

if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate "${{CATK_CONDA_ENV:-catk}}" 2>/dev/null || true
fi

echo "[tmux-run] pod=$(hostname) task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
echo "[tmux-run] experiment=${{CATK_EXPERIMENT}} bs=${{TRAIN_BATCH_SIZE}} nproc=${{NPROC_PER_NODE}}"
echo "[tmux-run] attach survives after exit; press Ctrl-b d to detach"
echo

mkdir -p "$RUN_ROOT"
torch_status_file="$RUN_ROOT/$(hostname).torchrun_status"
torch_pgid_file="$RUN_ROOT/$(hostname).torchrun_pgid"
rm -f "$torch_status_file" "$torch_pgid_file"

terminate_process_group() {{
  local pgid="$1"
  if [[ -z "$pgid" || "$pgid" == "0" ]]; then
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep "${{REMOTE_KILL_GRACE_SEC:-20}}"
  kill -KILL -- "-$pgid" 2>/dev/null || true
}}

(
  set +e
  setsid bash -c 'pgid_file="$1"; shift; echo "$$" > "$pgid_file"; exec "$@"' \\
    bash "$torch_pgid_file" torchrun --standalone --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \\
      -m src.run {override_args}
  echo "$?" > "$torch_status_file"
) &
runner_pid=$!

wait "$runner_pid"
status="$(cat "$torch_status_file" 2>/dev/null || echo 1)"
if [[ "$status" != "0" ]]; then
  pgid="$(cat "$torch_pgid_file" 2>/dev/null || true)"
  terminate_process_group "$pgid"
fi

echo
echo "[tmux-run] exited with status $status at $(date '+%F %T')"
echo "[tmux-run] leaving shell open for inspection"
exec bash
"""


def render_start_command(args: argparse.Namespace, train_batch_size: int, ckpt_path: str) -> str:
    run_root = remote_run_root(args)
    env_file = f"{run_root}/{args.pod}.env"
    run_file = f"{run_root}/{args.pod}_run.sh"
    monitor_file = f"{run_root}/{args.pod}_monitor.sh"
    log_file = remote_tmux_log(args)
    pipe_command = f"cat >> {shq(log_file)}"
    env_text = "\n".join(
        [
            f"export CACHE_ROOT={shq(args.cache_root)}",
            f"export TASK_NAME={shq(args.task_name)}",
            f"export CATK_EXPERIMENT={shq(args.experiment)}",
            f"export LOG_DIR={shq(args.remote_log_dir)}",
            f"export RUN_ROOT={shq(run_root)}",
            f"export NPROC_PER_NODE={shq(args.nproc_per_node)}",
            f"export TRAIN_BATCH_SIZE={shq(train_batch_size)}",
            f"export VAL_BATCH_SIZE={shq(args.val_batch_size)}",
        ]
    ) + "\n"
    run_text = render_run_script(args, env_file, train_batch_size, ckpt_path)
    monitor_text = render_monitor_script(args.monitor_interval)

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
{remote_git_prepare_script(args)}
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


def render_stop_command(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
  echo "[launcher] stopped tmux session {args.session}"
else
  echo "[launcher] tmux session not found: {args.session}"
fi
TASK_NAME_TO_STOP={shq(args.task_name)}
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


def stop_attempt(args: argparse.Namespace) -> None:
    run_kubectl(args, render_stop_command(args))


def find_latest_epoch_last_ckpt(args: argparse.Namespace) -> str:
    runs_dir = f"{args.remote_log_dir.rstrip('/')}/{args.task_name}/runs"
    command = (
        f"{{ ls -t {shq(runs_dir)}/*/checkpoints/epoch_last.ckpt 2>/dev/null; "
        f"ls -t {shq(runs_dir)}/*/checkpoints/last.ckpt 2>/dev/null; }} | head -1"
    )
    try:
        return run_kubectl(args, command, capture=True).replace("\r", "")
    except subprocess.CalledProcessError:
        return ""


def remote_log_contains_oom(args: argparse.Namespace) -> bool:
    command = f"grep -Eq {shq(OOM_REGEX)} {shq(remote_tmux_log(args))} 2>/dev/null"
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            args.pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def remote_status(args: argparse.Namespace) -> str:
    command = f"cat {shq(remote_status_file(args))} 2>/dev/null | tail -1"
    try:
        status = run_kubectl(args, command, capture=True).replace("\r", "")
    except subprocess.CalledProcessError:
        return ""
    return status if re.fullmatch(r"[0-9]+", status) else ""


def copy_remote_log(args: argparse.Namespace, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = run_kubectl(args, f"cat {shq(remote_tmux_log(args))}", capture=True)
    except subprocess.CalledProcessError as exc:
        body = f"warning: failed to copy remote log: {exc}\n"
    destination.write_text(body + "\n", encoding="utf-8")


def start_attempt(args: argparse.Namespace, train_batch_size: int, ckpt_path: str) -> None:
    print(f"[launcher] pod: {args.pod}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session: {args.session}")
    print(f"[launcher] bs: {train_batch_size}")
    if ckpt_path:
        print(f"[launcher] resume ckpt: {ckpt_path}")
    run_kubectl(args, render_start_command(args, train_batch_size, ckpt_path))


def wait_for_attempt(args: argparse.Namespace) -> tuple[str, bool]:
    while True:
        if remote_log_contains_oom(args):
            print("[launcher] OOM marker observed; stopping tmux session before retry.")
            stop_attempt(args)
            return "1", True
        status = remote_status(args)
        if status:
            return status, False
        print(
            "[launcher] waiting; attach: "
            f"kubectl exec -it -n {args.namespace} {args.pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )
        time.sleep(args.poll_interval)


def launch_with_retry(args: argparse.Namespace) -> int:
    bs = args.initial_bs
    attempt = 0
    oom_attempt_count = 0
    log_dir = local_retry_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)

    while bs >= args.min_bs:
        attempt += 1
        latest_ckpt = find_latest_epoch_last_ckpt(args)
        print(f"[launcher] attempt #{attempt}: bs={bs}")
        if latest_ckpt:
            print(f"[launcher] latest checkpoint: {latest_ckpt}")
        stop_attempt(args)
        start_attempt(args, bs, latest_ckpt)
        status, oom_seen = wait_for_attempt(args)
        attempt_log = log_dir / f"attempt_{attempt:03d}_bs{bs}.log"
        copy_remote_log(args, attempt_log)

        if status == "0":
            print(f"[launcher] training completed successfully at bs={bs}.")
            return 0

        if oom_seen or re.search(OOM_REGEX, attempt_log.read_text(encoding="utf-8", errors="ignore")):
            oom_attempt_count += 1
            if args.max_oom_attempts > 0 and oom_attempt_count >= args.max_oom_attempts:
                print(
                    "[launcher] OOM detected and max OOM attempts reached; "
                    f"last bs={bs}."
                )
                return 1
            new_bs = bs - args.oom_step
            print(f"[launcher] OOM detected at bs={bs}; lowering to bs={new_bs}.")
            if new_bs < args.min_bs:
                print(f"[launcher] next bs={new_bs} is below min bs={args.min_bs}.")
                return 1
            bs = new_bs
            continue

        print(f"[launcher] non-OOM failure status={status}; see {attempt_log}.")
        return int(status)

    print(f"[launcher] reached min bs={args.min_bs} without success.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch A100x7 execution-context pretrain on the existing testas pod."
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument("--pod", default=os.environ.get("POD", DEFAULT_POD))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--initial-bs", type=int, default=16)
    parser.add_argument("--oom-step", type=int, default=1)
    parser.add_argument("--min-bs", type=int, default=1)
    parser.add_argument("--max-oom-attempts", type=int, default=0)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--nproc-per-node", type=int, default=7)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="16")
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=32)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--metadata-cache-path", default=os.environ.get("MEMORY_BALANCE_METADATA_CACHE", ""))
    parser.add_argument("--metadata-num-workers", type=int, default=8)
    parser.add_argument("--prebuild-metadata", action="store_true")
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--monitor-interval", type=int, default=30)
    args = parser.parse_args()

    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.min_bs > args.initial_bs:
        parser.error("--min-bs must be <= --initial-bs")
    if args.max_oom_attempts < 0:
        parser.error("--max-oom-attempts must be >= 0")
    if args.poll_interval < 1:
        parser.error("--poll-interval must be >= 1")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.check_val_every_n_epoch < 1:
        parser.error("--check-val-every-n-epoch must be >= 1")
    if args.metadata_num_workers < 1:
        parser.error("--metadata-num-workers must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if args.stop:
        stop_attempt(args)
        return 0

    if args.prebuild_metadata:
        prebuild_metadata(args)
    else:
        verify_metadata_cache(args)

    if args.dry_run:
        print("[dry-run] metadata verified/prebuilt; launch command would run here.")
        print(f"[dry-run] metadata cache: {metadata_cache_path(args)}")
        print(f"[dry-run] remote log: {remote_tmux_log(args)}")
        return 0

    return launch_with_retry(args)


if __name__ == "__main__":
    raise SystemExit(main())
