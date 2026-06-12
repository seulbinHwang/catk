#!/usr/bin/env python3
"""Launch decoder-agent-attention open-loop Flow pretrain on fm-sf-6 H100x8.

This is a thin preset wrapper around the generic single-pod H100 launcher.  It
keeps the same CUDA-OOM retry/resume behavior while pinning the defaults to the
semi_control_decoder branch and the fm-sf-6 pod.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_h100x8_fmsf3_a343315f_gelu_static_pod.py"
)

DEFAULT_TASK_NAME = (
    "flow_open_loop_pretrain_decoder_agent_attention_effect_"
    "h100x8_fmsf6_bs20to10_lr6e-4_warm4_val8_membal"
)
DEFAULT_SESSION = "catk-pretrain-decoder-agent-attention-h100x8-fmsf6"
DEFAULT_WANDB_GROUP = "decoder_agent_attention_pretrain"


def main() -> None:
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-sp-labs-reai-training",
        "--pod",
        "fm-sf-6",
        "--container",
        "main",
        "--branch",
        "semi_control_decoder",
        "--experiment",
        "pre_bc_flow_2x4_h100",
        "--task-name",
        DEFAULT_TASK_NAME,
        "--session",
        DEFAULT_SESSION,
        "--initial-bs",
        "20",
        "--oom-step",
        "2",
        "--min-bs",
        "10",
        "--val-batch-size",
        "16",
        "--test-batch-size",
        "16",
        "--max-epochs",
        "64",
        "--check-val-every-n-epoch",
        "8",
        "--limit-val-batches",
        "0.1",
        "--learning-rate",
        "6e-4",
        "--lr-warmup-steps",
        "4",
        "--train-memory-balanced-batches",
        "true",
        "--extra-hydra-overrides",
        (
            "+trainer.use_distributed_sampler=false "
            f"logger.wandb.name={DEFAULT_TASK_NAME} "
            f"logger.wandb.group={DEFAULT_WANDB_GROUP}"
        ),
    ]
    command.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
