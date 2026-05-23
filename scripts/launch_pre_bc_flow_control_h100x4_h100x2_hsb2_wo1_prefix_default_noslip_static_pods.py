#!/usr/bin/env python3
"""Launch H100 4+2 stable control pretrain on hsb-npc-training-2 + wo-pvc-1.

This is a thin preset wrapper around
``launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py``.
It keeps the same semi_control_stable experiment/config, but targets the
hsb-npc-training-2 and wo-pvc-1 static pods and starts OOM retry from
``train_batch_size=18`` with one-batch decrements.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py"
)


DEFAULT_ARGS = [
    "--pods",
    "hsb-npc-training-2",
    "wo-pvc-1",
    "--task-name",
    "flow_control_space_pretrain_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_lr6e-4_bs18",
    "--session",
    "catk-control-pretrain-h100x4-h100x2-hsb2-wo1-prefix-default-noslip",
    "--initial-bs",
    "18",
    "--oom-step",
    "1",
    "--min-bs",
    "12",
    "--pod-cache-root",
    "hsb-npc-training-2=/workspace/womd_v1_3/SMART_cache",
    "--pod-cache-root",
    "wo-pvc-1=/workspace/womd_v1_3/SMART_cache",
    "--memory-metadata-cache-path",
    "/mnt/nuplan/projects/catk/logs/dataset_metadata/womd_training_memory_balance_h100x6_hsb2_wo1.pt",
]


def shq(value: object) -> str:
    return shlex.quote(str(value))


def main(argv: list[str]) -> int:
    command = [sys.executable, str(BASE_LAUNCHER), *DEFAULT_ARGS, *argv]
    if "--dry-run-wrapper" in argv:
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
