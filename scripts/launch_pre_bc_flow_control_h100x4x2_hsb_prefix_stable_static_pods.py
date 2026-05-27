#!/usr/bin/env python3
"""Launch H100x4x2 control-space pretrain with the V100x47 stable logic.

This launcher targets the already-running ``hsb-npc-training`` and
``hsb-npc-training2`` pods. It reuses the H100x4x2 retry launcher so the
hardware-sensitive settings stay H100-friendly, while explicitly matching the
V100x47 stable experiment's control-space semantics:

* use_prefix_valid_future_loss_mask=true
* use_kinematic_control_flow=true
* use_holonomic_model_only=true
* use_rolling_supervision=true
* control_round_trip_max_position_error_m=0.5

The script does not create, delete, or restart pods. It only starts/replaces the
configured tmux session and task processes inside the existing pods.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py")

DEFAULT_PODS = ("hsb-npc-training", "hsb-npc-training2")
DEFAULT_EXTRA_HYDRA_OVERRIDES = (
    "model.model_config.token_processor.use_prefix_valid_future_loss_mask=true",
    "model.model_config.token_processor.use_kinematic_control_flow=true",
    "model.model_config.token_processor.use_holonomic_model_only=true",
    "model.model_config.token_processor.use_rolling_supervision=true",
    "model.model_config.token_processor.control_round_trip_max_position_error_m=0.5",
    "model.model_config.decoder.flow_window_steps=20",
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch H100x4x2 prefix-valid control-space pretrain on "
            "hsb-npc-training and hsb-npc-training2."
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
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or "semi_control_stable_w_val")
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"))
    parser.add_argument("--experiment", default="pre_bc_flow_control_2x4_h100")
    parser.add_argument(
        "--task-name",
        default="flow_control_space_pretrain_h100x4x2_holonomic_prefix_roundtrip05_stable_lr6e-4_bs26",
    )
    parser.add_argument("--session", default="catk-control-pretrain-h100x4x2-holonomic-prefix")
    parser.add_argument("--initial-bs", type=int, default=26)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=20)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--master-port", default="29541")
    parser.add_argument("--checkpoint-sync-port", default="29542")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="16")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended after the preset overrides.",
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
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} H100x4 pods")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    extra_overrides = " ".join(
        part
        for part in (
            " ".join(DEFAULT_EXTRA_HYDRA_OVERRIDES),
            args.extra_hydra_overrides.strip(),
        )
        if part
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
        "--extra-hydra-overrides",
        extra_overrides,
    ]
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", args.limit_train_batches])
    if args.limit_val_batches:
        command.extend(["--limit-val-batches", args.limit_val_batches])
    if args.max_epochs:
        command.extend(["--max-epochs", args.max_epochs])
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
