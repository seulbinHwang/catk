#!/usr/bin/env python3
"""Launch the V100x47 prefix-valid default no-slip control pretrain.

This is the static V100-fleet counterpart of the A100x4x2 prefix-default-no-slip run.

It reuses the generic V100x47 launcher for heterogeneous pod GPU counts,
manual rank offsets, checkpoint sync, and OOM restart. Unlike the fallback
launchers, this wrapper keeps train_batch_size unchanged after OOM and only
resumes from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_pre_bc_flow_control_v100x47_static_pods.py")

DEFAULT_PODS = (
    "sv",
    "svv",
    "svvv",
    "svvvv",
    "testsv",
    "testsvv",
    "testsvvv",
    "testsvvvv",
    "fv",
    "fvv",
    "fvvv",
    "fvvvv",
    "fvvvvv",
)
DEFAULT_EXPERIMENT = "pre_bc_flow_control_v100x47_prefix_default_noslip"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_"
    "execctx_lr6e-4_bs4"
)
DEFAULT_SESSION = "catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix"


def shq(value: object) -> str:
    return shlex.quote(str(value))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch V100x47 prefix-valid kinematic control-space pretrain "
            "with default vehicle/cyclist no-slip point ratios."
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
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or "semi_control_stable")
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("CATK_GIT_REF", ""),
        help="Exact git ref/SHA to checkout on every pod instead of the branch head.",
    )
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument(
        "--nproc-per-node",
        type=validate_nproc_per_node,
        default="gpu",
        help="Use 'gpu' to spawn one worker per visible GPU on each pod.",
    )
    parser.add_argument("--initial-bs", type=int, default=4)
    parser.add_argument(
        "--oom-step",
        type=int,
        default=0,
        help=(
            "Batch-size decrement after CUDA OOM. The default 0 keeps "
            "train_batch_size unchanged and only resumes from the latest checkpoint."
        ),
    )
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29561")
    parser.add_argument("--checkpoint-sync-port", default="29562")
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="4")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended to the preset.",
    )
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help="Override CACHE_ROOT for one pod. Can be repeated.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} V100 pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 0:
        parser.error("--oom-step must be >= 0")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    return args


def main() -> int:
    args = parse_args()
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
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--initial-bs",
        str(args.initial_bs),
        "--oom-step",
        str(args.oom_step),
        "--min-bs",
        str(args.min_bs),
        "--poll-interval",
        str(args.poll_interval),
        "--master-port",
        str(args.master_port),
        "--checkpoint-sync-port",
        str(args.checkpoint_sync_port),
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
    if args.extra_hydra_overrides:
        command.extend(["--extra-hydra-overrides", args.extra_hydra_overrides])
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
