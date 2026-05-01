#!/usr/bin/env python3
"""Run static multi-node CAT-K fine-tuning with adaptive train_batch_size."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["testv", "testvv"]
DEFAULT_CONTAINER = "main"
DEFAULT_BRANCH = "semi_continuous_track_loss"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_PRETRAIN_CKPT = (
    "/mnt/nuplan/projects/catk/checkpoints/"
    "flow_semi_continuous_pretrain_all_target_h1006/"
    "4pxhrpv8_v70_e64_step259776/epoch_last.ckpt"
)
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|"
    r"CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|"
    r"DefaultCPUAllocator: can't allocate memory|Killed|signal 9|SIGKILL",
    re.IGNORECASE,
)
EXIT_RE = re.compile(r"\[tmux-run\] exited with status (?P<status>\d+)")
PROGRESS_RE = re.compile(r"Epoch \d+:.*")


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout else ""


def kubectl_exec(namespace: str, pod: str, container: str, script: str, *, check: bool = True) -> str:
    return run(
        [
            "kubectl",
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
        ],
        capture=True,
        check=check,
    )


def log(message: str) -> None:
    print(f"[{dt.datetime.now():%F %T}] {message}", flush=True)


def tmux_log_path(log_dir: str, task_name: str, pod: str) -> str:
    safe_task = task_name.replace("/", "_")
    return f"{log_dir.rstrip('/')}/tmux_static_multinode/{safe_task}/{pod}.tmux.log"


def read_remote_log(args: argparse.Namespace, pod: str, *, tail_lines: int = 220) -> str:
    path = tmux_log_path(args.log_dir, args.task_name, pod)
    return kubectl_exec(
        args.namespace,
        pod,
        args.container,
        f"test -f {shq(path)} && tail -n {int(tail_lines)} {shq(path)} || true",
        check=False,
    )


def archive_attempt_logs(args: argparse.Namespace, attempt: int, batch_size: int) -> None:
    archive_dir = Path(args.local_log_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for pod in args.pods:
        text = read_remote_log(args, pod, tail_lines=args.archive_tail_lines)
        out_path = archive_dir / f"{args.task_name}_attempt{attempt}_bs{batch_size}_{pod}.log"
        out_path.write_text(text, encoding="utf-8")
        log(f"archived {pod} tmux log: {out_path}")


def latest_progress(text: str) -> str:
    matches = PROGRESS_RE.findall(text.replace("\r", "\n"))
    return matches[-1] if matches else ""


def session_exists(args: argparse.Namespace, pod: str) -> bool:
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            pod,
            "-c",
            args.container,
            "--",
            "tmux",
            "has-session",
            "-t",
            args.session,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def classify_logs(logs: dict[str, str]) -> tuple[str, str]:
    all_text = "\n".join(logs.values())
    oom = bool(OOM_RE.search(all_text))
    statuses = [int(match.group("status")) for match in EXIT_RE.finditer(all_text)]
    if not statuses:
        return ("running_oom_seen" if oom else "running", "")
    if len(statuses) < len(logs) and all(status == 0 for status in statuses):
        return ("running_oom_seen" if oom else "running", "")
    if len(statuses) >= len(logs) and all(status == 0 for status in statuses):
        return "success", "all observed tmux runners exited with status 0"
    if oom:
        return "oom", "OOM pattern found in tmux logs"
    return "failed", f"non-zero tmux exit status observed: {statuses}"


def latest_epoch_checkpoint(args: argparse.Namespace) -> str:
    find_script = f"""
root={shq(args.log_dir.rstrip('/') + '/' + args.task_name + '/runs')}
if [ -d "$root" ]; then
  find "$root" -name epoch_last.ckpt -printf '%T@ %p\\n' 2>/dev/null \\
    | sort -rn | head -1 | cut -d' ' -f2-
fi
"""
    return kubectl_exec(args.namespace, args.pods[0], args.container, find_script, check=False).strip()


def stop_session(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/launch_mlx_static_pods_tmux.py",
        "--namespace",
        args.namespace,
        "--pods",
        *args.pods,
        "--container",
        args.container,
        "--session",
        args.session,
        "--stop",
    ]
    run(cmd, check=False)


def wait_for_gpu_release(args: argparse.Namespace) -> None:
    log("waiting for GPU memory to be released")
    for attempt in range(1, args.gpu_release_checks + 1):
        totals: list[int] = []
        for pod in args.pods:
            text = kubectl_exec(
                args.namespace,
                pod,
                args.container,
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "
                "| awk '{s += $1} END {print s + 0}'",
                check=False,
            )
            totals.append(int(text.strip() or "0"))
        if all(total <= args.gpu_release_mib for total in totals):
            log(f"GPU memory released: {dict(zip(args.pods, totals))}")
            return
        log(f"GPU memory still in use {dict(zip(args.pods, totals))}; check {attempt}/{args.gpu_release_checks}")
        time.sleep(args.gpu_release_sleep)
    log("WARNING: GPU memory did not fully release before next attempt")


def launch_attempt(args: argparse.Namespace, *, batch_size: int, action: str, ckpt_path: str) -> None:
    cmd = [
        sys.executable,
        "scripts/launch_mlx_static_pods_tmux.py",
        "--namespace",
        args.namespace,
        "--pods",
        *args.pods,
        "--container",
        args.container,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--cache-root",
        args.cache_root,
        "--pretrain-ckpt",
        args.pretrain_ckpt,
        "--action",
        action,
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--master-port",
        args.master_port,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--learning-rate",
        args.learning_rate,
        "--log-dir",
        args.log_dir,
        "--soft-limit-ratio",
        args.soft_limit_ratio,
        "--train-batch-size",
        str(batch_size),
        "--accumulate-grad-batches",
        str(args.accumulate_grad_batches),
        "--replace",
    ]
    if action == "fit":
        cmd.extend(["--ckpt-path", ckpt_path])
    if args.limit_train_batches:
        cmd.extend(["--limit-train-batches", args.limit_train_batches])
    if args.limit_val_batches:
        cmd.extend(["--limit-val-batches", args.limit_val_batches])
    if args.extra_hydra_overrides:
        cmd.extend(["--extra-hydra-overrides", args.extra_hydra_overrides])
    if args.no_pull:
        cmd.append("--no-pull")
    log("launching attempt command:")
    print("  " + " ".join(shq(part) for part in cmd), flush=True)
    run(cmd)


def monitor_attempt(args: argparse.Namespace, *, attempt: int, batch_size: int) -> str:
    last_heartbeat = 0.0
    while True:
        logs = {pod: read_remote_log(args, pod) for pod in args.pods}
        state, detail = classify_logs(logs)
        if state in {"running", "running_oom_seen"} and not any(
            session_exists(args, pod) for pod in args.pods
        ):
            state = "failed"
            detail = "tmux sessions disappeared before runner exit status was logged"
        if state in {"success", "oom", "failed"}:
            if state == "failed" and args.failure_grace_seconds > 0:
                time.sleep(args.failure_grace_seconds)
                logs = {pod: read_remote_log(args, pod) for pod in args.pods}
                state, detail = classify_logs(logs)
            if state in {"success", "oom", "failed"}:
                log(f"attempt {attempt} bs={batch_size} ended as {state}: {detail}")
                archive_attempt_logs(args, attempt, batch_size)
                return state
        now = time.monotonic()
        if now - last_heartbeat >= args.heartbeat_interval:
            progress = latest_progress(logs.get(args.pods[0], ""))
            suffix = f" | {progress}" if progress else ""
            if state == "running_oom_seen":
                suffix += " | OOM pattern seen; waiting for tmux runner exit"
            log(f"attempt {attempt} bs={batch_size} still running{suffix}")
            last_heartbeat = now
        time.sleep(args.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive train_batch_size sweep for existing MLX static multi-node pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--experiment", default="finetune_draft_flow_v100x8")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--session", default="catk-draft-bs-sweep")
    parser.add_argument("--master-port", default="29517")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--learning-rate", default="2e-4")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--soft-limit-ratio", default="0.8")
    parser.add_argument("--start-batch-size", type=int, default=36)
    parser.add_argument("--batch-step", type=int, default=8)
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--heartbeat-interval", type=int, default=120)
    parser.add_argument("--failure-grace-seconds", type=int, default=30)
    parser.add_argument("--gpu-release-mib", type=int, default=2000)
    parser.add_argument("--gpu-release-checks", type=int, default=12)
    parser.add_argument("--gpu-release-sleep", type=int, default=15)
    parser.add_argument("--archive-tail-lines", type=int, default=4000)
    parser.add_argument("--local-log-dir", default="/tmp/catk_v100x8x2_bs_sweep")
    args = parser.parse_args()

    if len(args.pods) < 2:
        parser.error("--pods must contain at least two pods")
    if args.start_batch_size < args.min_batch_size:
        parser.error("--start-batch-size must be >= --min-batch-size")
    if args.batch_step < 1:
        parser.error("--batch-step must be >= 1")
    if args.accumulate_grad_batches < 1:
        parser.error("--accumulate-grad-batches must be >= 1")
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ratio = str(args.soft_limit_ratio).replace(".", "p")
        args.task_name = f"catk_draft_v100x8x{len(args.pods)}_soft_limit_ratio_{ratio}_bs_sweep_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    log("starting adaptive static multi-node train_batch_size sweep")
    log(f"task_name: {args.task_name}")
    log(f"pods: {args.pods}")
    log(
        "batch sweep: "
        f"{args.start_batch_size} -> ... -> {args.min_batch_size} "
        f"(step {args.batch_step}), accumulate_grad_batches={args.accumulate_grad_batches}"
    )
    stop_session(args)
    wait_for_gpu_release(args)

    batch_size = args.start_batch_size
    attempt = 0
    while batch_size >= args.min_batch_size:
        attempt += 1
        resume_ckpt = latest_epoch_checkpoint(args)
        if resume_ckpt:
            action = "fit"
            ckpt_path = resume_ckpt
            resume_note = f"resume from {resume_ckpt}"
        else:
            action = "finetune"
            ckpt_path = args.pretrain_ckpt
            resume_note = "fresh finetune from pretrained checkpoint"

        log("=" * 72)
        log(f"attempt {attempt}: train_batch_size={batch_size}")
        log(f"action={action}; {resume_note}")
        launch_attempt(args, batch_size=batch_size, action=action, ckpt_path=ckpt_path)
        state = monitor_attempt(args, attempt=attempt, batch_size=batch_size)
        if state == "success":
            log(f"sweep finished successfully at train_batch_size={batch_size}")
            return
        if state != "oom":
            raise SystemExit(f"attempt {attempt} failed with non-OOM error")

        stop_session(args)
        wait_for_gpu_release(args)
        batch_size -= args.batch_step
        if batch_size >= args.min_batch_size:
            log(f"retrying after OOM with train_batch_size={batch_size}")

    raise SystemExit(
        f"exhausted batch sizes down to {args.min_batch_size}; no successful run"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
