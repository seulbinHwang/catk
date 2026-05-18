#!/usr/bin/env python3
"""Launch the latest-code V100x47 default no-slip tail-prefix Flow pretrain.

This wrapper intentionally reuses the production V100x47 prefix/default/no-slip
launcher and only changes the default task/session names so a latest
``semi_control_stable`` run does not accidentally resume or mix with the older
W&B run:

    flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs4
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py"
)

DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_"
    "roundtrip05_lr6e-4_bs4_stable_latest"
)
DEFAULT_SESSION = (
    "catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix-stable-latest"
)
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_METADATA_CACHE_BASENAME = (
    "womd_training_memory_balance_v1_stable_latest.pt"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the latest semi_control_stable V100x47 tail-prefix Flow "
            "pretrain. Unknown arguments are passed through to the base launcher."
        )
    )
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or "semi_control_stable")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    return parser.parse_known_args()


def has_passthrough_option(args: list[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def passthrough_option_value(args: list[str], option: str, default: str) -> str:
    prefix = option + "="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == option and index + 1 < len(args):
            return args[index + 1]
    return default


def main() -> int:
    args, passthrough = parse_args()
    if (
        "--stop" not in passthrough
        and not has_passthrough_option(passthrough, "--memory-metadata-cache-path")
    ):
        remote_log_dir = passthrough_option_value(
            passthrough,
            "--remote-log-dir",
            os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR),
        )
        passthrough = [
            *passthrough,
            "--memory-metadata-cache-path",
            str(
                Path(remote_log_dir.rstrip("/"))
                / "dataset_metadata"
                / DEFAULT_METADATA_CACHE_BASENAME
            ),
        ]
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--branch",
        args.branch,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        *passthrough,
    ]
    if "--dry-run" in passthrough:
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
