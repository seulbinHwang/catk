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


def main() -> int:
    args, passthrough = parse_args()
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
