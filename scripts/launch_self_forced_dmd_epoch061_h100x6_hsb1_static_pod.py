#!/usr/bin/env python3
"""Launch DMD self-forcing from the x5f9g0ce epoch061 Flow checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_self_forced_dmd_h100x6_hsb1_static_pod.py",
)

DEFAULT_ARGS = [
    "--wandb-pretrain-artifact",
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
    "--pretrain-ckpt",
    (
        "/workspace/flow_self_forced_dmd_h100x6_hsb1_pretrain_epoch061_x5f9g0ce/"
        "v57/epoch_061.ckpt"
    ),
    "--pretrain-download-dir",
    (
        "/workspace/flow_self_forced_dmd_h100x6_hsb1_pretrain_epoch061_x5f9g0ce/"
        "v57/artifact"
    ),
    "--task-name",
    (
        "flow_self_forced_dmd_h100x6_hsb1_epoch061_x5f9g0ce_activecontrol_"
        "sample16_backprop8_lr1e-6_bs18_frac010_ep16_oomretry"
    ),
    "--session",
    "catk-self-forced-dmd-epoch061-h100x6-hsb1",
]


def main() -> int:
    command = [sys.executable, str(BASE_LAUNCHER), *DEFAULT_ARGS, *sys.argv[1:]]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
