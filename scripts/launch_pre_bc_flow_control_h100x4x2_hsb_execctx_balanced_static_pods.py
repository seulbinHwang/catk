#!/usr/bin/env python3
"""Launch the semi_control_rolling execution-context pretrain on H100x4x2.

This launcher targets the already-running ``hsb-npc-training`` and
``hsb-npc-training-2`` pods. It does not create, delete, or restart pods. It
only prepares the optional memory-balance metadata cache and starts/replaces
the configured tmux training session inside the existing pods.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py"
)

DEFAULT_PODS = ("hsb-npc-training", "hsb-npc-training-2")
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4x2_execctx_balanced"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x4x2_execctx_prefix_balanced_lr6e-4_bs20"
)
DEFAULT_SESSION = "catk-control-pretrain-h100x4x2-execctx-balanced"
DEFAULT_METADATA_CACHE_RELATIVE = "dataset_metadata/womd_training_memory_balance_v1.pt"
DEFAULT_CACHE_ROOT_BY_POD = {
    "hsb-npc-training": "/mnt/nuplan/womd_v1_3/SMART_cache",
    "hsb-npc-training-2": "/workspace/womd_v1_3/SMART_cache",
}


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_pod_cache_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"--pod-cache-root must use POD=PATH, got {value!r}"
            )
        pod, path = value.split("=", 1)
        pod = pod.strip()
        path = path.strip()
        if not pod or not path:
            raise argparse.ArgumentTypeError(
                f"--pod-cache-root must include both POD and PATH, got {value!r}"
            )
        roots[pod] = path
    return roots


def cache_root_for_pod(args: argparse.Namespace, pod: str) -> str:
    if pod in args.pod_cache_root_map:
        return args.pod_cache_root_map[pod]
    if args.cache_root:
        return args.cache_root
    return DEFAULT_CACHE_ROOT_BY_POD.get(pod, "/workspace/womd_v1_3/SMART_cache")


def metadata_cache_path(args: argparse.Namespace) -> str:
    if args.metadata_cache_path:
        return args.metadata_cache_path
    return f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE_RELATIVE}"


def training_extra_hydra_overrides(args: argparse.Namespace) -> str:
    overrides: list[str] = []
    if args.extra_hydra_overrides:
        overrides.append(args.extra_hydra_overrides)
    if args.metadata_cache_path:
        overrides.append(
            f"data.train_memory_balance_metadata_cache={args.metadata_cache_path}"
        )
    return " ".join(overrides)


def remote_git_prepare_script(args: argparse.Namespace) -> str:
    branch_ref = f"refs/heads/{args.branch}"
    origin_ref = f"origin/{args.branch}"
    fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
    if args.git_ref:
        return " && ".join(
            [
                f"git config --global --add safe.directory {shq(args.project_root)} || true",
                f"git update-ref -d {shq(f'refs/remotes/origin/{args.branch}')} || true",
                f"git fetch origin --prune {shq('+' + fetch_refspec)}",
                f"git checkout -f {shq(args.git_ref)}",
            ]
        )
    return " && ".join(
        [
            f"git config --global --add safe.directory {shq(args.project_root)} || true",
            f"git fetch origin {shq(fetch_refspec)}",
            (
                f"if git show-ref --verify --quiet {shq(branch_ref)}; then "
                f"git checkout {shq(args.branch)}; "
                f"else git checkout -b {shq(args.branch)} {shq(origin_ref)}; fi"
            ),
            f"git pull --ff-only origin {shq(args.branch)}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch H100x4x2 execution-context-aligned control-space pretrain "
            "on hsb-npc-training and hsb-npc-training-2."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", " ".join(DEFAULT_PODS)).split(),
    )
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or "semi_control_rolling")
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("CATK_GIT_REF", ""),
        help="Exact git ref/SHA to checkout on every pod instead of the branch head.",
    )
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--initial-bs", type=int, default=20)
    parser.add_argument("--oom-step", type=int, default=0)
    parser.add_argument(
        "--max-oom-attempts",
        type=int,
        default=3,
        help=(
            "Stop after this many OOM attempts. The default keeps "
            "train_batch_size=20 for three OOM attempts, then exits cleanly."
        ),
    )
    parser.add_argument("--min-bs", type=int, default=20)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29571")
    parser.add_argument("--checkpoint-sync-port", default="29572")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="16")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended after the preset.",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("CACHE_ROOT", ""),
        help="Use one CACHE_ROOT for every pod. Pod-specific defaults are used when omitted.",
    )
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help="Override CACHE_ROOT for one pod. Can be repeated.",
    )
    parser.add_argument(
        "--metadata-cache-path",
        default=os.environ.get("MEMORY_BALANCE_METADATA_CACHE", ""),
        help=(
            "Absolute metadata cache path. Defaults to "
            "$REMOTE_LOG_DIR/dataset_metadata/womd_training_memory_balance_v1.pt."
        ),
    )
    parser.add_argument("--metadata-num-workers", type=int, default=8)
    parser.add_argument(
        "--prebuild-metadata",
        action="store_true",
        help="Build the memory-balance metadata cache on each pod before launch.",
    )
    parser.add_argument(
        "--force-metadata",
        action="store_true",
        help="Pass --force to the metadata prebuild tool to remove stale locks first.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} H100x4 pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 0:
        parser.error("--oom-step must be >= 0")
    if args.max_oom_attempts < 0:
        parser.error("--max-oom-attempts must be >= 0")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.metadata_num_workers < 1:
        parser.error("--metadata-num-workers must be >= 1")
    try:
        args.pod_cache_root_map = parse_pod_cache_roots(args.pod_cache_root)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def run_pod_command(args: argparse.Namespace, pod: str, script: str) -> int:
    command = [
        "kubectl",
        "exec",
        "-n",
        args.namespace,
        pod,
        "-c",
        args.container,
        "--",
        "bash",
        "-lc",
        script,
    ]
    if args.dry_run:
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


def prebuild_metadata(args: argparse.Namespace) -> int:
    cache_path = metadata_cache_path(args)
    for pod in args.pods:
        cache_root = cache_root_for_pod(args, pod)
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
            shq(f"{cache_root.rstrip('/')}/training"),
            "--cache-path",
            shq(cache_path),
            "--num-workers",
            shq(args.metadata_num_workers),
        ]
        if args.force_metadata:
            command.append("--force")
        script = " ".join(str(part) for part in command)
        status = run_pod_command(args, pod, script)
        if status != 0:
            return status
    return 0


def verify_metadata_cache(args: argparse.Namespace) -> int:
    cache_path = metadata_cache_path(args)
    for pod in args.pods:
        script = (
            f"if [[ ! -f {shq(cache_path)} ]]; then "
            f"echo {shq('[metadata-check] missing memory-balance metadata cache: ' + cache_path)} >&2; "
            "echo '[metadata-check] rerun with --prebuild-metadata, or pass --metadata-cache-path to an existing cache.' >&2; "
            "exit 2; "
            "fi"
        )
        status = run_pod_command(args, pod, script)
        if status != 0:
            return status
    return 0


def main() -> int:
    args = parse_args()
    if args.prebuild_metadata and not args.stop:
        status = prebuild_metadata(args)
        if status != 0:
            return status
    elif not args.stop:
        status = verify_metadata_cache(args)
        if status != 0:
            return status

    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pods",
        *args.pods,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--remote-log-dir",
        args.remote_log_dir,
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--initial-bs",
        str(args.initial_bs),
        "--oom-step",
        str(args.oom_step),
        "--max-oom-attempts",
        str(args.max_oom_attempts),
        "--min-bs",
        str(args.min_bs),
        "--poll-interval",
        str(args.poll_interval),
        "--master-port",
        str(args.master_port),
        "--checkpoint-sync-port",
        str(args.checkpoint_sync_port),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--learning-rate",
        str(args.learning_rate),
        "--val-batch-size",
        str(args.val_batch_size),
    ]
    if args.git_ref:
        command.extend(["--git-ref", args.git_ref])
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", args.limit_train_batches])
    if args.limit_val_batches:
        command.extend(["--limit-val-batches", args.limit_val_batches])
    if args.max_epochs:
        command.extend(["--max-epochs", args.max_epochs])
    extra_hydra_overrides = training_extra_hydra_overrides(args)
    if extra_hydra_overrides:
        command.extend(["--extra-hydra-overrides", extra_hydra_overrides])
    if args.cache_root:
        for pod in args.pods:
            command.extend(["--pod-cache-root", f"{pod}={args.cache_root}"])
    for mapping in args.pod_cache_root:
        command.extend(["--pod-cache-root", mapping])
    if args.replace:
        command.append("--replace")
    if args.stop:
        command.append("--stop")
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
