#!/usr/bin/env python3
"""Launch mixed H100/A100 FW30 prefix-valid pretrain on static pods.

This wrapper only supplies safe defaults to the generic tmux launcher. It does
not create, delete, or restart pods.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_h100x4_multinode_pretrain_tmux.py")

DEFAULT_PODS = ("wo-pvc-800", "testa", "testaa")
DEFAULT_TASK_NAME = "flow_pretrain_prefix_valid_fw30_maskaware_mixed_h100x4_a100x4x2_bs26"
DEFAULT_SESSION = "catk-pretrain-mixed-h100-a100-prefix-fw30"
DEFAULT_EXPERIMENT = "pre_bc_flow_mixed_h100x4_a100x4x2_prefix_valid"
DEFAULT_BRANCH = "self_forcing_anchor_new"
DEFAULT_LR = "5.0e-4"
DEFAULT_TRAIN_BS = "26"

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "model.model_config.decoder.flow_window_steps=30",
        "model.model_config.token_processor.use_prefix_valid_future_loss_mask=true",
    ]
)


def split_wrapper_args(argv: list[str]) -> tuple[list[str], str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-hydra-overrides", default="")
    known, remaining = parser.parse_known_args(argv)
    extra_overrides = " ".join(
        part for part in (DEFAULT_EXTRA_OVERRIDES, known.extra_hydra_overrides) if part
    )
    return remaining, extra_overrides


def main() -> int:
    passthrough_args, extra_overrides = split_wrapper_args(sys.argv[1:])
    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--pods",
        *DEFAULT_PODS,
        "--branch",
        DEFAULT_BRANCH,
        "--experiment",
        DEFAULT_EXPERIMENT,
        "--task-name",
        DEFAULT_TASK_NAME,
        "--session",
        DEFAULT_SESSION,
        "--train-batch-size",
        DEFAULT_TRAIN_BS,
        "--learning-rate",
        DEFAULT_LR,
        "--extra-hydra-overrides",
        extra_overrides,
        *passthrough_args,
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
