#!/usr/bin/env python3
"""Launch closed-loop SF-anchor fine-tuning on fm-sf-quarter-* H100x2x4 pods."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_self_forced_a100x4x2_static_pods.py")


def default_extra_overrides(rollout_anchor_stride: int) -> str:
    return " ".join(
        [
            "model.model_config.val_open_loop=false",
            "model.model_config.decoder.detach_train_metric_clean=true",
            "model.model_config.self_forced.distribution_matching_objective=dmd",
            "model.model_config.self_forced.detach_block_transition=false",
            "model.model_config.self_forced.project_dmd_to_pose_space=false",
            "model.model_config.self_forced.dmd_use_stable_scale_filter=true",
            "model.model_config.self_forced.dmd_stable_scale_scope=agent",
            f"model.model_config.self_forced.rollout_anchor_stride={rollout_anchor_stride}",
            "model.model_config.self_forced.closed_loop_sf_global_max_step=4",
            "model.model_config.self_forced.closed_loop_sf_local_max_step=4",
            "model.model_config.self_forced.update_open_loop_teacher_when_roll=false",
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


def split_wrapper_args(argv: list[str]) -> tuple[list[str], str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--closed-loop-see-all", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rollout-anchor-stride", type=int, default=2)
    known, remaining = parser.parse_known_args(argv)
    if known.rollout_anchor_stride < 1:
        parser.error("--rollout-anchor-stride must be >= 1")

    wrapper_overrides = []
    if known.closed_loop_see_all is not None:
        wrapper_overrides.append(
            f"model.model_config.self_forced.closed_loop_see_all={str(known.closed_loop_see_all).lower()}"
        )
    extra_overrides = " ".join(
        part
        for part in (
            default_extra_overrides(known.rollout_anchor_stride),
            " ".join(wrapper_overrides),
            known.extra_hydra_overrides,
        )
        if part
    )
    return remaining, extra_overrides, known.rollout_anchor_stride


def main() -> int:
    passthrough_args, extra_overrides, rollout_anchor_stride = split_wrapper_args(sys.argv[1:])
    pod_label = "h100x2x4_quarter"
    lr_tag = "lr7e-5"
    pretrain_root = (
        f"/workspace/flow_closed_loop_self_forced_{pod_label}_fmsf4_pretrain_epoch061_x5f9g0ce/v57"
    )
    task_name = (
        f"flow_closed_loop_self_forced_dmd_{pod_label}_fmsf4_sfanchor_stride{rollout_anchor_stride}_"
        f"epoch061_x5f9g0ce_activecontrol_sample16_backprop8_{lr_tag}_bs4to2_frac025_"
        "ep2_warm0_global4_local4"
    )
    bank_name = f"generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-{lr_tag}"

    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-sp-labs-reai-training",
        "--branch",
        "closed_loop_sf_anchor",
        "--pods",
        "fm-sf-quarter-1",
        "fm-sf-quarter-2",
        "fm-sf-quarter-3",
        "fm-sf-quarter-4",
        "--nproc-per-node",
        "2",
        "--experiment",
        "self_forced_npfm_h100_6",
        "--wandb-pretrain-artifact",
        "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
        "--pretrain-ckpt",
        f"{pretrain_root}/epoch_061.ckpt",
        "--pretrain-download-dir",
        f"{pretrain_root}/artifact",
        "--task-name",
        task_name,
        "--session",
        f"catk-closed-loop-sf-h100x2x4-quarter-fmsf4-stride{rollout_anchor_stride}",
        "--initial-bs",
        "4",
        "--oom-step",
        "1",
        "--min-bs",
        "2",
        "--val-batch-size",
        "8",
        "--test-batch-size",
        "8",
        "--precision",
        "bf16-mixed",
        "--learning-rate",
        "7.0e-5",
        "--generated-estimator-learning-rate",
        "7.0e-5",
        "--scorer-scene-num",
        "1680",
        "--estimator-warmup-epochs",
        "0",
        "--estimator-warmup-bank-artifact",
        f"{bank_name}:latest",
        "--estimator-warmup-bank-artifact-name",
        bank_name,
        "--self-forced-use-stop-motion",
        "false",
        "--decoder-use-stop-motion",
        "false",
        "--train-epoch-sample-fraction",
        "0.25",
        "--train-memory-balanced-batches",
        "true",
        "--max-epochs",
        "2",
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
