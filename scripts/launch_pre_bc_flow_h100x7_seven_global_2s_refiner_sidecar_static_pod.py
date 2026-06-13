#!/usr/bin/env python3
"""Launch the seven H100x7 global-2s refiner pretrain with sidecar fast path.

This wrapper reuses the sidecar-enabled single-pod Flow pretrain launcher and
only changes the pod shape, branch, task/session names, and hardware defaults.
The delegated launcher prebuilds deterministic Flow target sidecars, preloads
them in DataLoader workers, disables train-only open-loop metrics, and retries
CUDA OOM by resuming from the latest Lightning checkpoint at a lower batch size.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_h100x8_fmsf6_sidecar_static_pod.py"
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
    "flow_open_loop_pretrain_global2s_refiner_sidecar_metricoff_h100x7_seven_bs18_lr6p5e-4_warm5_val8_membal",
    "--session",
    "catk-pretrain-global2s-refiner-sidecar-h100x7-seven",
]


def main() -> int:
    command = [sys.executable, str(BASE_LAUNCHER), *DEFAULT_ARGS, *sys.argv[1:]]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
