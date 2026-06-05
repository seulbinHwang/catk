#!/usr/bin/env python3
"""Launch the epoch-116 H100x6 Waymo validation iid-noise1000 preset.

This wrapper reuses the guarded epoch-61 validation launcher while pinning the
checkpoint and pure iid Gaussian inference preset selected for validation
leaderboard submission. It never creates, deletes, or restarts pods.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


BASE_LAUNCHER = "launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py"
DEFAULT_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-mqfq3u39:v121"
DEFAULT_EPOCH = "116"
DEFAULT_TASK_NAME = (
    "flow_agents_7m_waymo_val_epoch116_mqfq3u39_h100x6_hsb1_"
    "sample16_euler_iid_iidgaussian_noise1000"
)
DEFAULT_DESCRIPTION = (
    "flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_"
    "train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18_"
    "116_false_stratified_false_1.0"
)
DEFAULT_SESSION = (
    "catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-"
    "iid-iidgaussian-noise1000"
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_launcher = script_dir / BASE_LAUNCHER
    command = [
        sys.executable,
        str(base_launcher),
        "--artifact",
        DEFAULT_ARTIFACT,
        "--epoch",
        DEFAULT_EPOCH,
        "--task-name",
        DEFAULT_TASK_NAME,
        "--description",
        DEFAULT_DESCRIPTION,
        "--session",
        DEFAULT_SESSION,
        "--antithetic-pairs",
        "false",
        "--stratified-gaussian-noise",
        "false",
        "--noise-scale",
        "1.0",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
