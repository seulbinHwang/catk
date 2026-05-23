#!/usr/bin/env python3
"""Launch H100 4+2 prefix-valid default no-slip control pretrain.

Targets existing pods only:

* hsb-npc-training: 4 visible H100 GPUs
* wo-pvc-2:         2 visible H100 GPUs

The launcher uses the generic heterogeneous-rank retry wrapper, so it never
creates, deletes, or restarts pods. It starts/replaces only the configured tmux
session and task processes inside those pods.
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

DEFAULT_PODS = ("hsb-npc-training", "wo-pvc-2")
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_"
    "tailprefix_roundtrip05_lr6e-4_bs17"
)
DEFAULT_SESSION = "catk-control-pretrain-h100x4-h100x2-prefix-default-noslip"
DEFAULT_METADATA_CACHE = (
    "dataset_metadata/womd_training_memory_balance_h100x6_hsb_wo_pvc2.pt"
)
DEFAULT_EXTRA_HYDRA_OVERRIDES = (
    "trainer.strategy._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy "
    "trainer.strategy.cluster_environment._target_="
    "src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch semi_control_stable H100x4 + H100x2 prefix-valid default "
            "no-slip control-space pretrain."
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
    parser.add_argument("--remote-python", default=os.environ.get("CATK_REMOTE_PYTHON", "/mnt/nuplan/miniforge/envs/catk/bin/python"))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--initial-bs", type=int, default=17)
    parser.add_argument("--oom-step", type=int, default=1)
    parser.add_argument("--min-bs", type=int, default=12)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29631")
    parser.add_argument("--checkpoint-sync-port", default="29632")
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="12")
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
        help="Override CACHE_ROOT for one pod. Defaults both pods to /workspace/womd_v1_3/SMART_cache.",
    )
    parser.add_argument(
        "--skip-memory-metadata-preflight",
        action="store_true",
        help="Skip the default per-pod memory-balanced metadata build/validation.",
    )
    parser.add_argument(
        "--memory-metadata-cache-path",
        default="",
        help=(
            "Remote metadata cache path. Defaults to "
            "REMOTE_LOG_DIR/dataset_metadata/womd_training_memory_balance_h100x6_hsb_wo_pvc2.pt."
        ),
    )
    parser.add_argument("--memory-metadata-num-workers", type=int, default=8)
    parser.add_argument("--force-memory-metadata-rebuild", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 0:
        parser.error("--oom-step must be >= 0")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.memory_metadata_num_workers < 1:
        parser.error("--memory-metadata-num-workers must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if args.stop:
        return run_stop(args)

    metadata_cache_path = args.memory_metadata_cache_path or (
        f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE}"
    )
    pod_cache_roots = args.pod_cache_root or [
        "hsb-npc-training=/workspace/womd_v1_3/SMART_cache",
        "wo-pvc-2=/workspace/womd_v1_3/SMART_cache",
    ]
    extra_hydra_overrides = " ".join(
        part
        for part in (
            DEFAULT_EXTRA_HYDRA_OVERRIDES,
            f"data.train_memory_balance_metadata_cache={metadata_cache_path}",
            args.extra_hydra_overrides.strip(),
        )
        if part
    )

    env = os.environ.copy()
    env.update(
        {
            "NAMESPACE": args.namespace,
            "CONTAINER": args.container,
            "PODS": " ".join(args.pods),
            "PROJECT_ROOT": args.project_root,
            "BRANCH": args.branch,
            "GIT_REF": args.git_ref,
            "CATK_REMOTE_PYTHON": args.remote_python,
            "TASK_NAME": args.task_name,
            "SESSION": args.session,
            "EXPERIMENT": args.experiment,
            "REMOTE_LOG_DIR": args.remote_log_dir,
            "MASTER_PORT": str(args.master_port),
            "CHECKPOINT_SYNC_PORT": str(args.checkpoint_sync_port),
            "NPROC_PER_NODE": "gpu",
            "MANUAL_RANK_OFFSETS": "1",
            "INITIAL_BS": str(args.initial_bs),
            "OOM_STEP": str(args.oom_step),
            "MIN_BS": str(args.min_bs),
            "POLL_INTERVAL": str(args.poll_interval),
            "LEARNING_RATE": str(args.learning_rate),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
            "EXTRA_HYDRA_OVERRIDES": extra_hydra_overrides,
            "POD_CACHE_ROOTS": " ".join(pod_cache_roots),
        }
    )
    optional_env = {
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
    }
    env.update({name: value for name, value in optional_env.items() if value})
    if not args.skip_memory_metadata_preflight:
        env.update(
            {
                "MEMORY_BALANCE_PREFLIGHT": "1",
                "MEMORY_BALANCE_METADATA_CACHE": metadata_cache_path,
                "MEMORY_BALANCE_METADATA_NUM_WORKERS": str(args.memory_metadata_num_workers),
                "MEMORY_BALANCE_METADATA_FORCE_REBUILD": "1"
                if args.force_memory_metadata_rebuild
                else "0",
            }
        )

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
                "GIT_REF",
                "CATK_REMOTE_PYTHON",
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
                "MEMORY_BALANCE_PREFLIGHT",
                "MEMORY_BALANCE_METADATA_CACHE",
                "MEMORY_BALANCE_METADATA_NUM_WORKERS",
                "MEMORY_BALANCE_METADATA_FORCE_REBUILD",
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
