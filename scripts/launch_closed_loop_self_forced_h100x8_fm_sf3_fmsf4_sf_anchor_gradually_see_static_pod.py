#!/usr/bin/env python3
"""Launch the fm-sf-3 H100x8 gradually-see variant of the quarter SF-anchor run."""

from __future__ import annotations

import argparse
import subprocess
import sys

from launch_closed_loop_self_forced_h100x2x4_quarter_fmsf4_sf_anchor_static_pods import (
    BASE_LAUNCHER,
    INITIAL_STAGE_CHECKPOINT_ARTIFACT,
    INITIAL_STAGE_CHECKPOINT_EPOCH,
    INITIAL_STAGE_CHECKPOINT_GLOBAL_STEP,
    INITIAL_STAGE_CHECKPOINT_PATH,
    INITIAL_STAGE_CHECKPOINT_DOWNLOAD_DIR,
    INITIAL_STAGE_CHECKPOINT_SHA256,
    default_extra_overrides,
)


def split_wrapper_args(argv: list[str]) -> tuple[list[str], str, int, bool, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--closed-loop-sf-global-max-step", type=int, default=4)
    parser.add_argument("--rollout-anchor-stride", type=int, default=4)
    parser.add_argument(
        "--skip-initial-stage-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    known, remaining = parser.parse_known_args(argv)
    if known.rollout_anchor_stride < 1:
        parser.error("--rollout-anchor-stride must be >= 1")
    if known.closed_loop_sf_global_max_step < 1:
        parser.error("--closed-loop-sf-global-max-step must be >= 1")

    variant_overrides = " ".join(
        [
            "model.model_config.self_forced.closed_loop_sf_global_max_step="
            f"{known.closed_loop_sf_global_max_step}",
            "model.model_config.self_forced.closed_loop_see_all=true",
            "model.model_config.self_forced.gradually_see=true",
        ]
    )
    extra_overrides = " ".join(
        part
        for part in (
            default_extra_overrides(
                known.rollout_anchor_stride,
                skip_initial_stage_checkpoint=known.skip_initial_stage_checkpoint,
            ),
            variant_overrides,
            known.extra_hydra_overrides,
        )
        if part
    )
    return (
        remaining,
        extra_overrides,
        known.rollout_anchor_stride,
        known.skip_initial_stage_checkpoint,
        known.closed_loop_sf_global_max_step,
    )


def main() -> int:
    (
        passthrough_args,
        extra_overrides,
        rollout_anchor_stride,
        skip_initial_stage_checkpoint,
        closed_loop_sf_global_max_step,
    ) = split_wrapper_args(sys.argv[1:])
    pod_label = "h100x8_fm_sf3"
    lr_tag = "lr5e-5"
    if skip_initial_stage_checkpoint:
        checkpoint_artifact = INITIAL_STAGE_CHECKPOINT_ARTIFACT
        checkpoint_path = INITIAL_STAGE_CHECKPOINT_PATH
        checkpoint_download_dir = INITIAL_STAGE_CHECKPOINT_DOWNLOAD_DIR
        initial_action = "fit"
        checkpoint_tag = "resume81s5s6k3_epoch5"
        expected_sha256 = INITIAL_STAGE_CHECKPOINT_SHA256
        expected_epoch = INITIAL_STAGE_CHECKPOINT_EPOCH
        expected_global_step = INITIAL_STAGE_CHECKPOINT_GLOBAL_STEP
    else:
        checkpoint_artifact = "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57"
        checkpoint_path = (
            "/workspace/flow_closed_loop_self_forced_h100x8_fmsf4_pretrain_epoch061_x5f9g0ce/"
            "v57/epoch_061.ckpt"
        )
        checkpoint_download_dir = (
            "/workspace/flow_closed_loop_self_forced_h100x8_fmsf4_pretrain_epoch061_x5f9g0ce/"
            "v57/artifact"
        )
        initial_action = "finetune"
        checkpoint_tag = "epoch061_x5f9g0ce"
        expected_sha256 = ""
        expected_epoch = ""
        expected_global_step = ""

    task_name = (
        f"flow_closed_loop_self_forced_dmd_{pod_label}_fmsf4_sfanchor_seeall_gradual_"
        f"stride{rollout_anchor_stride}_{checkpoint_tag}_activecontrol_sample16_backprop8_"
        f"{lr_tag}_bs24to4step4_frac025_ep6_warm0_global{closed_loop_sf_global_max_step}_local4"
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
        "fm-sf-3",
        "--nproc-per-node",
        "8",
        "--experiment",
        "self_forced_npfm_h100_6",
        "--wandb-pretrain-artifact",
        checkpoint_artifact,
        "--pretrain-ckpt",
        checkpoint_path,
        "--pretrain-download-dir",
        checkpoint_download_dir,
        "--initial-action",
        initial_action,
        "--pretrain-expected-sha256",
        expected_sha256,
        "--pretrain-expected-epoch",
        expected_epoch,
        "--pretrain-expected-global-step",
        expected_global_step,
        "--task-name",
        task_name,
        "--session",
        f"catk-closed-loop-sf-h100x8-fm-sf3-fmsf4-gradual-stride{rollout_anchor_stride}",
        "--initial-bs",
        "24",
        "--oom-step",
        "4",
        "--min-bs",
        "4",
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
        "--lr-cosine-final-ratio",
        "0.1",
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
