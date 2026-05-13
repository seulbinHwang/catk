#!/usr/bin/env python3
"""Launch prefix-valid FW30 fine-tuning on the static testa/testaa A100 pods.

This wrapper delegates the multi-node orchestration to
``launch_self_forced_a100x4x2_static_pods.py`` so it inherits the existing
OOM retry, checkpoint sync, and W&B artifact download behavior.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_self_forced_a100x4x2_static_pods.py")

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "model.model_config.decoder.flow_window_steps=30",
        "model.model_config.token_processor.use_prefix_valid_future_loss_mask=true",
        "model.model_config.finetune.enabled=false",
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
        "--branch",
        "self_forcing_anchor_new",
        "--experiment",
        "finetune_flow_prefix_valid_a100_4x2",
        "--pretrain-ckpt",
        "/workspace/fw_30_pretrain/epoch_last.ckpt",
        "--wandb-pretrain-artifact",
        "jksg01019-naver-labs/SMART-FLOW/epoch-last-swkp98ig:v64",
        "--pretrain-download-dir",
        "/workspace/fw_30_pretrain/artifact",
        "--task-name",
        "flow_prefix_valid_finetune_a100x4x2_fw30_bs26",
        "--session",
        "catk-prefix-valid-a100x4x2-fw30",
        "--initial-bs",
        "26",
        "--oom-step",
        "2",
        "--min-bs",
        "2",
        "--val-batch-size",
        "8",
        "--test-batch-size",
        "8",
        "--precision",
        "bf16-mixed",
        "--learning-rate",
        "1.0e-4",
        "--scorer-scene-num",
        "1680",
        "--extra-hydra-overrides",
        extra_overrides,
        *passthrough_args,
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
