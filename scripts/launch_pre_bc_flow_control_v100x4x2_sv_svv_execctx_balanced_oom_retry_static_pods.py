#!/usr/bin/env python3
"""Launch V100x4x2 execution-context pretrain on sv + svv.

This wrapper targets the already-running ``sv`` and ``svv`` V100x4 pods. It
does not create, delete, or restart pods. It prepares or verifies the
memory-balance metadata cache, then starts the shared tmux OOM-retry launcher.

The key hardware adaptation from the H100x6 4+2 launcher is that this run uses
8 homogeneous V100 ranks with fp16 mixed precision and a smaller per-rank batch.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import launch_pre_bc_flow_control_h100x6_hsb2_wo1_execctx_balanced_static_pods as h100x6


RETRY_SCRIPT = Path(__file__).with_name("h100x4_multinode_pretrain_with_oom_retry.sh")
BASE_LAUNCHER = Path(__file__).with_name("launch_h100x4_multinode_pretrain_tmux.py")

DEFAULT_PODS = ("sv", "svv")
DEFAULT_EXPERIMENT = "pre_bc_flow_control_v100x4x2_execctx_balanced"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_v100x4x2_sv_svv_"
    "execctx_prefix_balanced_lr2e-4_bs4_oomretry"
)
DEFAULT_SESSION = "catk-control-pretrain-v100x4x2-sv-svv-execctx-balanced"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch semi_control_rolling_gan V100x4x2 pretrain on sv and svv "
            "with train_batch_size OOM fallback."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", " ".join(DEFAULT_PODS)).split(),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"),
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("CATK_BRANCH") or "semi_control_rolling_gan",
    )
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("CATK_GIT_REF", ""),
        help="Exact git ref/SHA to checkout on every pod instead of the branch head.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not git fetch/pull during metadata prebuild. Retry attempts still pull unless --git-ref is used.",
    )
    parser.add_argument(
        "--remote-log-dir",
        default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"),
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--initial-bs", type=int, default=4)
    parser.add_argument("--oom-step", type=int, default=1)
    parser.add_argument("--min-bs", type=int, default=1)
    parser.add_argument(
        "--max-oom-attempts",
        type=int,
        default=0,
        help="0 means keep retrying until --min-bs is crossed.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29640")
    parser.add_argument("--checkpoint-sync-port", default="29641")
    parser.add_argument("--nproc-per-node", default="gpu", choices=("gpu", "auto"))
    parser.add_argument("--learning-rate", default="2e-4")
    parser.add_argument("--val-batch-size", default="4")
    parser.add_argument(
        "--n-rollout-closed-val",
        type=int,
        default=16,
        help="Closed-loop validation rollouts. Keep 16 by default for V100 32GB.",
    )
    parser.add_argument(
        "--nccl-algo",
        default=os.environ.get("NCCL_ALGO", "Ring"),
        help="NCCL_ALGO exported in each remote tmux run.",
    )
    parser.add_argument(
        "--nccl-proto",
        default=os.environ.get("NCCL_PROTO", "Simple"),
        help="NCCL_PROTO exported in each remote tmux run.",
    )
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=16)
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended before pinned overrides.",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("CACHE_ROOT", ""),
        help="Use one CACHE_ROOT for every pod. Defaults to /workspace/womd_v1_3/SMART_cache.",
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
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Accepted for consistency. Each retry attempt always replaces the tmux session.",
    )
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} pods")
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
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.metadata_num_workers < 1:
        parser.error("--metadata-num-workers must be >= 1")
    if args.check_val_every_n_epoch < 1:
        parser.error("--check-val-every-n-epoch must be >= 1")
    h100x6.validate_extra_hydra_overrides(parser, args.extra_hydra_overrides)
    try:
        args.pod_cache_root_map = h100x6.parse_pod_cache_roots(args.pod_cache_root)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def stop_command(args: argparse.Namespace) -> list[str]:
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
        "--log-dir",
        args.remote_log_dir,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--stop",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def pod_cache_roots_env(args: argparse.Namespace) -> str:
    return " ".join(
        f"{pod}={h100x6.cache_root_for_pod(args, pod)}"
        for pod in args.pods
    )


def remote_env_overrides(args: argparse.Namespace) -> str:
    return " ".join(
        item
        for item in (
            "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
            f"NCCL_ALGO={args.nccl_algo}" if args.nccl_algo else "",
            f"NCCL_PROTO={args.nccl_proto}" if args.nccl_proto else "",
        )
        if item
    )


def training_extra_hydra_overrides(args: argparse.Namespace) -> str:
    overrides = [h100x6.training_extra_hydra_overrides(args)]
    if str(args.limit_val_batches).strip() in {"0", "0.0"}:
        overrides.append(
            "model.model_config.scorer_scene_num=0 "
            "model.model_config.val_open_loop=false "
            "model.model_config.val_closed_loop=false"
        )
    return " ".join(part for part in overrides if part)


def retry_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NAMESPACE": args.namespace,
            "CONTAINER": args.container,
            "PODS": " ".join(args.pods),
            "PROJECT_ROOT": args.project_root,
            "BRANCH": args.branch,
            "GIT_REF": args.git_ref,
            "TASK_NAME": args.task_name,
            "SESSION": args.session,
            "EXPERIMENT": args.experiment,
            "REMOTE_LOG_DIR": args.remote_log_dir,
            "POD_CACHE_ROOTS": pod_cache_roots_env(args),
            "MASTER_PORT": str(args.master_port),
            "CHECKPOINT_SYNC_PORT": str(args.checkpoint_sync_port),
            "NPROC_PER_NODE": args.nproc_per_node,
            "MANUAL_RANK_OFFSETS": "1",
            "INITIAL_BS": str(args.initial_bs),
            "OOM_STEP": str(args.oom_step),
            "MIN_BS": str(args.min_bs),
            "MAX_OOM_ATTEMPTS": str(args.max_oom_attempts),
            "POLL_INTERVAL": str(args.poll_interval),
            "LEARNING_RATE": str(args.learning_rate),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
            "REMOTE_ENV_OVERRIDES": remote_env_overrides(args),
            "LIMIT_TRAIN_BATCHES": str(args.limit_train_batches),
            "LIMIT_VAL_BATCHES": str(args.limit_val_batches),
            "MAX_EPOCHS": str(args.max_epochs),
            "EXTRA_HYDRA_OVERRIDES": training_extra_hydra_overrides(args),
        }
    )
    return env


def print_retry_command(args: argparse.Namespace) -> None:
    env = retry_environment(args)
    keys = [
        "NAMESPACE",
        "CONTAINER",
        "PODS",
        "PROJECT_ROOT",
        "BRANCH",
        "GIT_REF",
        "TASK_NAME",
        "SESSION",
        "EXPERIMENT",
        "REMOTE_LOG_DIR",
        "POD_CACHE_ROOTS",
        "MASTER_PORT",
        "CHECKPOINT_SYNC_PORT",
        "NPROC_PER_NODE",
        "MANUAL_RANK_OFFSETS",
        "INITIAL_BS",
        "OOM_STEP",
        "MIN_BS",
        "MAX_OOM_ATTEMPTS",
        "POLL_INTERVAL",
        "LEARNING_RATE",
        "VAL_BATCH_SIZE",
        "REMOTE_ENV_OVERRIDES",
        "LIMIT_TRAIN_BATCHES",
        "LIMIT_VAL_BATCHES",
        "MAX_EPOCHS",
        "EXTRA_HYDRA_OVERRIDES",
    ]
    print(
        " ".join(f"{key}={shq(env[key])}" for key in keys)
        + " "
        + shq(str(RETRY_SCRIPT))
    )


def main() -> int:
    args = parse_args()
    if args.stop:
        command = stop_command(args)
        if args.dry_run:
            print(" ".join(shq(part) for part in command))
            return 0
        return subprocess.call(command)

    if args.prebuild_metadata:
        status = h100x6.prebuild_metadata(args)
        if status != 0:
            return status
    else:
        status = h100x6.verify_metadata_cache(args)
        if status != 0:
            return status

    if args.dry_run:
        print_retry_command(args)
        return 0
    return subprocess.call([str(RETRY_SCRIPT)], env=retry_environment(args))


if __name__ == "__main__":
    raise SystemExit(main())
