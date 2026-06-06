#!/usr/bin/env python3
"""Launch DMD self-forced Flow fine-tuning on the `testaa` A100x4 pod.

This is the `testaa` default wrapper for
`launch_self_forced_dmd_a100x4_testa_static_pod.py`. The underlying launcher
and training recipe stay identical; only pod/session/task/checkpoint paths are
changed so single-pod `testa` and `testaa` jobs do not collide.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULT_TESTAA_ARGS = [
    "--pod",
    "testaa",
    "--task-name",
    (
        "flow_self_forced_dmd_a100x4_testaa_epoch061_x5f9g0ce_activecontrol_"
        "sample16_backprop8_lr1e-6_bs160_frac025_ep16_middle_oomretry"
    ),
    "--session",
    "catk-self-forced-dmd-a100x4-testaa",
    "--pretrain-ckpt",
    (
        "/workspace/flow_self_forced_dmd_a100x4_testaa_pretrain_epoch061_x5f9g0ce/"
        "v57/epoch_061.ckpt"
    ),
    "--pretrain-download-dir",
    (
        "/workspace/flow_self_forced_dmd_a100x4_testaa_pretrain_epoch061_x5f9g0ce/"
        "v57/artifact"
    ),
]


def main() -> None:
    target = Path(__file__).with_name("launch_self_forced_dmd_a100x4_testa_static_pod.py")
    sys.argv = [str(target), *DEFAULT_TESTAA_ARGS, *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
