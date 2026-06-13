#!/usr/bin/env python3
"""Resume the seven global-2s Flow pretrain from the epoch-20 checkpoint.

This wrapper is intentionally identical to the original seven/global-2s
pretrain launcher except that it passes an explicit ``ckpt_path`` for the first
attempt. Lightning restores optimizer, scheduler, epoch, and global step from
that checkpoint, so the resumed run should continue at epoch 21 with the same
training semantics as an uninterrupted run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_h100x8_fmsf3_a343315f_gelu_static_pod.py"
)

TASK_NAME = (
    "flow_open_loop_pretrain_global2s_refiner_a1b277f_"
    "h100x7_seven_bs18_lr6p5e-4_warm5_val8_membal"
)
EPOCH20_CKPT = (
    "/mnt/nuplan/projects/catk/logs/"
    f"{TASK_NAME}/runs/2026-06-13_02-44-30/checkpoints/epoch_last.ckpt"
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
    TASK_NAME,
    "--session",
    "catk-pretrain-global2s-refiner-h100x7-seven-resume-epoch20",
    "--resume-ckpt-path",
    EPOCH20_CKPT,
]


def main() -> int:
    command = [sys.executable, str(BASE_LAUNCHER), *DEFAULT_ARGS, *sys.argv[1:]]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
