#!/usr/bin/env python3
"""Launch stride-1 multi-anchor DMD self-forcing on the fm-sf-1 H100x8 pod."""

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
        "model.model_config.self_forced.project_dmd_to_pose_space=false",
        "model.model_config.self_forced.dmd_use_stable_scale_filter=true",
        "model.model_config.self_forced.dmd_stable_scale_scope=agent",
        "model.model_config.self_forced.rollout_anchor_stride=1",
        "model.model_config.self_forced.clean_dmd_normalizer_eps=0.05",
        "model.model_config.self_forced.clean_dmd_tau_low=0.02",
        "model.model_config.self_forced.clean_dmd_tau_high=0.98",
        "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch",
        "model.model_config.self_forced.sampling.random_terminal_step.policy=all",
        "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=16",
        "model.model_config.self_forced.sampling.backprop_last_k=8",
        "data.train_epoch_sample_fraction_shuffle_flag=false",
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
        "--namespace",
        "p-sp-labs-reai-training",
        "--branch",
        "semi_control_sf_anchor",
        "--pods",
        "fm-sf-1",
        "--nproc-per-node",
        "8",
        "--experiment",
        "self_forced_npfm_h100_6",
        "--wandb-pretrain-artifact",
        "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
        "--pretrain-ckpt",
        (
            "/workspace/flow_self_forced_dmd_h100x8_fmsf1_pretrain_epoch061_x5f9g0ce/"
            "v57/epoch_061.ckpt"
        ),
        "--pretrain-download-dir",
        (
            "/workspace/flow_self_forced_dmd_h100x8_fmsf1_pretrain_epoch061_x5f9g0ce/"
            "v57/artifact"
        ),
        "--task-name",
        (
            "flow_self_forced_dmd_h100x8_fmsf1_sfanchor_stride1_epoch061_x5f9g0ce_"
            "activecontrol_sample16_backprop8_lr5e-5_bs8to6_frac025_ep6_warm2_middle_val1_agent_oomretry"
        ),
        "--session",
        "catk-self-forced-dmd-h100x8-fmsf1-sfanchor-stride1-lr5e5",
        "--initial-bs",
        "8",
        "--oom-step",
        "2",
        "--min-bs",
        "6",
        "--val-batch-size",
        "8",
        "--test-batch-size",
        "8",
        "--precision",
        "bf16-mixed",
        "--learning-rate",
        "5.0e-5",
        "--generated-estimator-learning-rate",
        "5.0e-5",
        "--scorer-scene-num",
        "1680",
        "--estimator-warmup-epochs",
        "2",
        "--estimator-warmup-bank-artifact",
        "generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr5e-5:latest",
        "--estimator-warmup-bank-artifact-name",
        "generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr5e-5",
        "--self-forced-use-stop-motion",
        "false",
        "--decoder-use-stop-motion",
        "false",
        "--train-epoch-sample-fraction",
        "0.25",
        "--train-memory-balanced-batches",
        "true",
        "--max-epochs",
        "8",
        "--check-val-every-n-epoch",
        "1",
        "--limit-val-batches",
        "0.1",
        "--extra-hydra-overrides",
        extra_overrides,
        *passthrough_args,
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
