#!/usr/bin/env python3
"""Launch the semi_control_decoder_last 2s global step-refiner pretrain on seven.

This wrapper reuses the existing single-pod H100 open-loop pretrain launcher and
only changes the pod shape, branch, task/session names, and hardware defaults.
The delegated launcher handles CUDA OOM retry by resuming from the latest
Lightning checkpoint while lowering ``data.train_batch_size``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_h100x8_fmsf3_a343315f_gelu_static_pod.py"
)

DEFAULT_ARGS = [
    "--pod",
    "seven",
    "--branch",
    "semi_control_decoder_last",
    "--cuda-visible-devices",
    "0,1,2,3,4,5,6",
    "--nproc-per-node",
    "7",
    "--initial-bs",
    "18",
    "--oom-step",
    "2",
    "--min-bs",
    "12",
    "--learning-rate",
    "6.5e-4",
    "--lr-warmup-steps",
    "5",
    "--check-val-every-n-epoch",
    "8",
    "--limit-val-batches",
    "0.1",
    "--train-memory-balanced-batches",
    "true",
    "--task-name",
    "flow_open_loop_pretrain_global2s_refiner_a1b277f_h100x7_seven_bs18_lr6p5e-4_warm5_val8_membal",
    "--session",
    "catk-pretrain-global2s-refiner-h100x7-seven",
]


def main() -> int:
    command = [sys.executable, str(BASE_LAUNCHER), *DEFAULT_ARGS, *sys.argv[1:]]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
