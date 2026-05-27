#!/usr/bin/env python3
"""Launch H100 4+2 prefix-valid holonomic control pretrain.

This is the ablation counterpart of
``launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py``.
It keeps the same pods, config, batch size, learning rate, prefix-valid target
selection, and round-trip threshold, but forces
``model.model_config.token_processor.use_holonomic_model_only=true``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py"
)
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x4_h100x2_holonomic_"
    "tailprefix_roundtrip05_lr6e-4_bs18"
)
DEFAULT_SESSION = "catk-control-pretrain-h100x4-h100x2-holonomic"
HOLONOMIC_OVERRIDE = "model.model_config.token_processor.use_holonomic_model_only=true"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch semi_control_holonomic H100x4 + H100x2 prefix-valid "
            "holonomic control-space pretrain."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", "hsb-npc-training wo-pvc-2").split(),
    )
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or "semi_control_holonomic")
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--remote-python", default=os.environ.get("CATK_REMOTE_PYTHON", "/mnt/nuplan/miniforge/envs/catk/bin/python"))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--initial-bs", type=int, default=18)
    parser.add_argument("--oom-step", type=int, default=1)
    parser.add_argument("--min-bs", type=int, default=12)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29641")
    parser.add_argument("--checkpoint-sync-port", default="29642")
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="12")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help=(
            "Additional Hydra overrides. The holonomic override is appended last "
            "so this launcher always runs use_holonomic_model_only=true."
        ),
    )
    parser.add_argument("--pod-cache-root", action="append", default=[], metavar="POD=PATH")
    parser.add_argument("--skip-memory-metadata-preflight", action="store_true")
    parser.add_argument("--memory-metadata-cache-path", default="")
    parser.add_argument("--memory-metadata-num-workers", type=int, default=8)
    parser.add_argument("--force-memory-metadata-rebuild", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_hydra_overrides = " ".join(
        part for part in (args.extra_hydra_overrides.strip(), HOLONOMIC_OVERRIDE) if part
    )
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
        "--remote-python",
        args.remote_python,
        "--remote-log-dir",
        args.remote_log_dir,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
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
        "--memory-metadata-num-workers",
        str(args.memory_metadata_num_workers),
        "--extra-hydra-overrides",
        extra_hydra_overrides,
    ]
    if args.git_ref:
        command.extend(["--git-ref", args.git_ref])
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", args.limit_train_batches])
    if args.limit_val_batches:
        command.extend(["--limit-val-batches", args.limit_val_batches])
    if args.max_epochs:
        command.extend(["--max-epochs", args.max_epochs])
    if args.memory_metadata_cache_path:
        command.extend(["--memory-metadata-cache-path", args.memory_metadata_cache_path])
    for pod_cache_root in args.pod_cache_root:
        command.extend(["--pod-cache-root", pod_cache_root])
    if args.skip_memory_metadata_preflight:
        command.append("--skip-memory-metadata-preflight")
    if args.force_memory_metadata_rebuild:
        command.append("--force-memory-metadata-rebuild")
    if args.replace:
        command.append("--replace")
    if args.stop:
        command.append("--stop")
    if args.dry_run:
        command.append("--dry-run")
        print("[dry-run] delegated command:")
        print("  " + " ".join(shq(part) for part in command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
