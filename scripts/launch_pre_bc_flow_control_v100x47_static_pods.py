#!/usr/bin/env python3
"""Launch V100 control-space pretrain on the static 47-GPU pod fleet.

This wrapper intentionally does not create, delete, or restart pods. It only
starts/replaces the configured tmux session and task processes inside already
running pods.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


RETRY_SCRIPT = Path(__file__).with_name("h100x4_multinode_pretrain_with_oom_retry.sh")
BASE_LAUNCHER = Path(__file__).with_name("launch_h100x4_multinode_pretrain_tmux.py")

DEFAULT_PODS = (
    "testsv",
    "testsvv",
    "testsvvv",
    "testsvvvv",
    "sv",
    "svv",
    "svvv",
    "svvvv",
    "fv",
    "fvv",
    "fvvv",
    "fvvvv",
    "fvvvvv",
)
DEFAULT_EXTRA_HYDRA_OVERRIDES = (
    "trainer.strategy._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy "
    "trainer.strategy.cluster_environment._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment"
)


def current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "semi_control"
    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else "semi_control"


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
            "Launch kinematic control-space prefix-valid V100 pretrain on "
            "testsv*/sv*/fv* static pods."
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
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default="pre_bc_flow_control_v100x47")
    parser.add_argument(
        "--task-name",
        default="flow_control_space_pretrain_v100x47_prefix_roundtrip05_stable_lr6e-4_bs4",
    )
    parser.add_argument("--session", default="catk-control-pretrain-v100x47")
    parser.add_argument(
        "--nproc-per-node",
        type=validate_nproc_per_node,
        default="gpu",
        help=(
            "Use 'gpu' to spawn one worker per visible GPU on each pod. This "
            "uses 4 ranks on V100x4 pods and 3 ranks on V100x3 pods."
        ),
    )
    parser.add_argument("--initial-bs", type=int, default=4)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29531")
    parser.add_argument("--checkpoint-sync-port", default="29532")
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="4")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended to every attempt.",
    )
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help=(
            "Override CACHE_ROOT for one pod. Can be repeated. If omitted, "
            "the generic launcher defaults to /workspace/womd_v1_3/SMART_cache "
            "for these V100 pods."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Accepted for consistency; retry launcher already replaces the tmux session each attempt.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the configured tmux session/task processes on the pods without starting training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} V100 pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    return args


def run_stop(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
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
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--stop",
    ]
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


def main() -> int:
    args = parse_args()
    if args.stop:
        return run_stop(args)

    env = os.environ.copy()
    env.update(
        {
            "NAMESPACE": args.namespace,
            "CONTAINER": args.container,
            "PODS": " ".join(args.pods),
            "PROJECT_ROOT": args.project_root,
            "BRANCH": args.branch,
            "TASK_NAME": args.task_name,
            "SESSION": args.session,
            "EXPERIMENT": args.experiment,
            "REMOTE_LOG_DIR": args.remote_log_dir,
            "MASTER_PORT": str(args.master_port),
            "CHECKPOINT_SYNC_PORT": str(args.checkpoint_sync_port),
            "NPROC_PER_NODE": str(args.nproc_per_node),
            "MANUAL_RANK_OFFSETS": "1",
            "INITIAL_BS": str(args.initial_bs),
            "OOM_STEP": str(args.oom_step),
            "MIN_BS": str(args.min_bs),
            "POLL_INTERVAL": str(args.poll_interval),
            "LEARNING_RATE": str(args.learning_rate),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
        }
    )
    extra_hydra_overrides = " ".join(
        part for part in (DEFAULT_EXTRA_HYDRA_OVERRIDES, args.extra_hydra_overrides) if part
    )
    optional_env = {
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "EXTRA_HYDRA_OVERRIDES": extra_hydra_overrides,
    }
    env.update({name: value for name, value in optional_env.items() if value})
    if args.pod_cache_root:
        env["POD_CACHE_ROOTS"] = " ".join(args.pod_cache_root)

    command = ["bash", str(RETRY_SCRIPT)]
    if args.dry_run:
        print("[dry-run] environment:")
        for name in sorted(
            [
                "NAMESPACE",
                "CONTAINER",
                "PODS",
                "PROJECT_ROOT",
                "BRANCH",
                "TASK_NAME",
                "SESSION",
                "EXPERIMENT",
                "REMOTE_LOG_DIR",
                "MASTER_PORT",
                "CHECKPOINT_SYNC_PORT",
                "NPROC_PER_NODE",
                "MANUAL_RANK_OFFSETS",
                "INITIAL_BS",
                "OOM_STEP",
                "MIN_BS",
                "POLL_INTERVAL",
                "LEARNING_RATE",
                "VAL_BATCH_SIZE",
                "LIMIT_TRAIN_BATCHES",
                "LIMIT_VAL_BATCHES",
                "MAX_EPOCHS",
                "EXTRA_HYDRA_OVERRIDES",
                "POD_CACHE_ROOTS",
            ]
        ):
            if name in env:
                print(f"  {name}={shq(env[name])}")
        print("[dry-run] command:")
        print("  " + " ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
