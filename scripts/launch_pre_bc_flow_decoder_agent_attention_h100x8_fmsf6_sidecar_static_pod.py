#!/usr/bin/env python3
"""Launch decoder-agent-attention Flow pretrain on fm-sf-6 with sidecars.

This preset mirrors the non-sidecar fm-sf-6 decoder pretrain launcher, but uses
the sidecar-aware H100x8 launcher so deterministic Flow targets are prebuilt and
loaded through the DataLoader fast path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name(
    "launch_pre_bc_flow_h100x8_fmsf5_sidecar_static_pod.py"
)

DEFAULT_TASK_NAME = (
    "flow_open_loop_pretrain_decoder_agent_attention_sidecar_metricoff_"
    "h100x8_fmsf6_bs20to10_lr6e-4_warm4_val8_membal"
)
DEFAULT_SESSION = "catk-pretrain-decoder-agent-attention-sidecar-h100x8-fmsf6"
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
        "--use-distributed-sampler",
        "false",
        "--train-memory-balanced-batches",
        "true",
        "--train-open-loop-metrics",
        "false",
        "--skip-empty-open-loop-optimizer-guard",
        "true",
        "--sidecar-prebuild",
        "true",
        "--extra-hydra-overrides",
        (
            f"logger.wandb.name={DEFAULT_TASK_NAME} "
            f"logger.wandb.group={DEFAULT_WANDB_GROUP}"
        ),
    ]
    command.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
