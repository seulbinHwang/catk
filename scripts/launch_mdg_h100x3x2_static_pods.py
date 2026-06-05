#!/usr/bin/env python3
"""Launch semi_mdg pretrain on hsb-npc-training-3-1/3-2.

This wrapper uses the existing multinode tmux + OOM retry launcher. It only
sets the MDG experiment defaults and the H100x3x2 pod layout.
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
        return "semi_mdg"
    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else "semi_mdg"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 2-node H100x3 semi_mdg pretrain.")
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", "hsb-npc-training-3-1 hsb-npc-training-3-2").split(),
    )
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or current_branch())
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default="mdg_pretrain_h100x3x2")
    parser.add_argument("--task-name", default="semi_mdg_pretrain_h100x3x2")
    parser.add_argument("--session", default="catk-semi-mdg-h100x3x2")
    parser.add_argument("--initial-bs", type=int, default=20)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29531")
    parser.add_argument("--checkpoint-sync-port", default="29532")
    parser.add_argument("--nproc-per-node", type=int, default=3)
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--val-batch-size", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help="Override CACHE_ROOT for one pod. Repeat as needed.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.pods) != 2 and not args.stop:
        parser.error("this preset expects exactly two pods")
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
    for name, value in {
        "LEARNING_RATE": args.learning_rate,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "EXTRA_HYDRA_OVERRIDES": args.extra_hydra_overrides,
    }.items():
        if value:
            env[name] = value
    if args.pod_cache_root:
        env["POD_CACHE_ROOTS"] = " ".join(args.pod_cache_root)

    command = ["bash", str(RETRY_SCRIPT)]
    if args.dry_run:
        for name in sorted(env):
            if name in {
                "PODS",
                "BRANCH",
                "TASK_NAME",
                "SESSION",
                "EXPERIMENT",
                "INITIAL_BS",
                "NPROC_PER_NODE",
                "EXTRA_HYDRA_OVERRIDES",
            }:
                print(f"{name}={shq(env[name])}")
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
