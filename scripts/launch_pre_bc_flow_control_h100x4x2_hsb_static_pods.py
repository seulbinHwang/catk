#!/usr/bin/env python3
"""Launch control-space Flow Matching pretrain on hsb-npc-training{,2}.

This is a thin wrapper around ``h100x4_multinode_pretrain_with_oom_retry.sh``.
The underlying script starts tmux sessions inside already-running pods, lowers
``data.train_batch_size`` by ``OOM_STEP`` on CUDA OOM, and resumes from the
latest rank-0 ``epoch_last.ckpt`` after synchronizing it to peer pods.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch H100x4x2 kinematic control-space pretrain on "
            "hsb-npc-training and hsb-npc-training2."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", "hsb-npc-training hsb-npc-training2").split(),
    )
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or current_branch())
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default="pre_bc_flow_control_2x4_h100")
    parser.add_argument(
        "--task-name",
        default="flow_control_space_pretrain_h100x4x2_fullvalid_roundtrip05_lr6e-4_bs26",
    )
    parser.add_argument("--session", default="catk-control-pretrain-h100x4x2")
    parser.add_argument("--initial-bs", type=int, default=26)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29511")
    parser.add_argument("--checkpoint-sync-port", default="29512")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--val-batch-size", default="")
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
            "the base launcher uses its hsb-npc-training defaults."
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
    if len(args.pods) != 2 and not args.stop:
        parser.error("this preset expects exactly two H100x4 pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
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
            "INITIAL_BS": str(args.initial_bs),
            "OOM_STEP": str(args.oom_step),
            "MIN_BS": str(args.min_bs),
            "POLL_INTERVAL": str(args.poll_interval),
        }
    )
    optional_env = {
        "LEARNING_RATE": args.learning_rate,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "EXTRA_HYDRA_OVERRIDES": args.extra_hydra_overrides,
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
                "INITIAL_BS",
                "OOM_STEP",
                "MIN_BS",
                "POLL_INTERVAL",
                "POD_CACHE_ROOTS",
                "LEARNING_RATE",
                "VAL_BATCH_SIZE",
                "LIMIT_TRAIN_BATCHES",
                "LIMIT_VAL_BATCHES",
                "MAX_EPOCHS",
                "EXTRA_HYDRA_OVERRIDES",
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
