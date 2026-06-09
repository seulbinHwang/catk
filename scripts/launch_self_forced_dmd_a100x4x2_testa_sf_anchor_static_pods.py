#!/usr/bin/env python3
"""Launch multi-anchor DMD self-forcing on testa/testaa A100x4x2 pods.

This wrapper is the semi_control_sf_anchor sibling of
``launch_self_forced_dmd_a100x4x2_testa_static_pods.py``. It keeps the generic
2-node orchestration in ``launch_self_forced_a100x4x2_static_pods.py`` and pins
the multi-anchor defaults introduced by the sf-anchor branch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_self_forced_a100x4x2_static_pods.py")

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "model.model_config.val_open_loop=false",
        "model.model_config.decoder.detach_train_metric_clean=true",
        "model.model_config.self_forced.distribution_matching_objective=dmd",
        "model.model_config.self_forced.detach_block_transition=false",
        "model.model_config.self_forced.rollout_anchor_stride=2",
        "model.model_config.self_forced.clean_dmd_normalizer_eps=0.05",
        "model.model_config.self_forced.clean_dmd_tau_low=0.02",
        "model.model_config.self_forced.clean_dmd_tau_high=0.98",
        "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch",
        "model.model_config.self_forced.sampling.random_terminal_step.policy=all",
        "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=16",
        "model.model_config.self_forced.sampling.backprop_last_k=8",
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
        "semi_control_sf_anchor",
        "--experiment",
        "self_forced_npfm_a100x4x2",
        "--wandb-pretrain-artifact",
        "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
        "--pretrain-ckpt",
        (
            "/workspace/flow_self_forced_dmd_a100x4x2_testa_pretrain_epoch061_x5f9g0ce/"
            "v57/epoch_061.ckpt"
        ),
        "--pretrain-download-dir",
        (
            "/workspace/flow_self_forced_dmd_a100x4x2_testa_pretrain_epoch061_x5f9g0ce/"
            "v57/artifact"
        ),
        "--task-name",
        (
            "flow_self_forced_dmd_a100x4x2_testa_sfanchor_stride2_epoch061_x5f9g0ce_"
            "activecontrol_sample16_backprop8_lr1e-6_bs24_frac025_ep16_middle_oomretry"
        ),
        "--session",
        "catk-self-forced-dmd-a100x4x2-testa-sfanchor-stride2",
        "--initial-bs",
        "24",
        "--oom-step",
        "1",
        "--min-bs",
        "4",
        "--val-batch-size",
        "8",
        "--test-batch-size",
        "8",
        "--precision",
        "bf16-mixed",
        "--learning-rate",
        "1.0e-6",
        "--generated-estimator-learning-rate",
        "1.0e-6",
        "--scorer-scene-num",
        "1680",
        "--estimator-warmup-epochs",
        "1",
        "--self-forced-use-stop-motion",
        "false",
        "--decoder-use-stop-motion",
        "false",
        "--train-epoch-sample-fraction",
        "0.25",
        "--train-memory-balanced-batches",
        "true",
        "--max-epochs",
        "16",
        "--limit-val-batches",
        "0.1",
        "--extra-hydra-overrides",
        extra_overrides,
        *passthrough_args,
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
