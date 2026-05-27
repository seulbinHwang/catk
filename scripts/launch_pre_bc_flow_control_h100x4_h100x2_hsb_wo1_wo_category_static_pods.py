#!/usr/bin/env python3
"""Launch the wo-category ablation pretrain on hsb-npc-training + wo-pvc-1.

This is a thin preset wrapper around
``launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py``.
It keeps the same H100 4+2 control-space pretrain recipe used by
``semi_control_stable``, but checks out ``semi_control_stable_wo_category`` so
new caches merge crosswalk / speed_bump / driveway into the old crosswalk-style
map category space.
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
    "hsb-npc-training",
    "wo-pvc-1",
    "--branch",
    "semi_control_stable_wo_category",
    "--task-name",
    "flow_control_space_pretrain_h100x4_h100x2_wo_category_prefix_default_noslip_lr6e-4_bs20",
    "--session",
    "catk-control-pretrain-h100x4-h100x2-wo-category",
    "--initial-bs",
    "20",
    "--oom-step",
    "1",
    "--min-bs",
    "12",
    "--pod-cache-root",
    "hsb-npc-training=/workspace/womd_v1_3/SMART_cache",
    "--pod-cache-root",
    "wo-pvc-1=/workspace/womd_v1_3/SMART_cache",
    "--memory-metadata-cache-path",
    "/mnt/nuplan/projects/catk/logs/dataset_metadata/womd_training_memory_balance_h100x6_hsb_wo1_wo_category.pt",
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
