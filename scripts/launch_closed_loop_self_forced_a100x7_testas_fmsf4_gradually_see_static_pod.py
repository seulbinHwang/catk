#!/usr/bin/env python3
"""Launch the fm-sf-4 closed-loop see-all run variant on testas with gradually_see=true."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_self_forced_a100x4x2_static_pods.py")
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/closed_loop_gradually_see_resume/"
    "fmsf2_epoch005_81s5s6k3/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/closed_loop_gradually_see_resume/"
    "fmsf2_epoch005_81s5s6k3/artifact"
)

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
        "model.model_config.self_forced.closed_loop_sf_global_max_step=4",
        "model.model_config.self_forced.closed_loop_sf_local_max_step=4",
        "model.model_config.self_forced.closed_loop_see_all=true",
        "model.model_config.self_forced.gradually_see=true",
        "model.model_config.self_forced.update_open_loop_teacher_when_roll=false",
        "model.model_config.self_forced.clean_dmd_normalizer_eps=0.05",
        "model.model_config.self_forced.clean_dmd_tau_low=0.02",
        "model.model_config.self_forced.clean_dmd_tau_high=0.98",
        "model.model_config.self_forced.sampling.sample_steps=16",
        "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch",
        "model.model_config.self_forced.sampling.random_terminal_step.policy=all",
        "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=16",
        "model.model_config.self_forced.sampling.backprop_last_k=8",
        "data.train_epoch_sample_fraction_shuffle_flag=false",
    ]
)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = (
        "flow_closed_loop_self_forced_dmd_a100x7_testas_fmsf4_sfanchor_"
        "seeall_graduallysee_from_fmsf2_epoch005_81s5s6k3_"
        "lr5e-5_bs3to2_step1_frac025_reqep4_to20_warm0_global4_local4_"
        f"{timestamp}"
    )
    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-pnc",
        "--branch",
        "self_forcing_closed_loop",
        "--pods",
        "testas",
        "--nproc-per-node",
        "7",
        "--project-root",
        "/tmp/catk_self_forcing_closed_loop_testas",
        "--experiment",
        "self_forced_npfm_h100_6",
        "--wandb-pretrain-artifact",
        "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
        "--pretrain-ckpt",
        DEFAULT_PRETRAIN_CKPT,
        "--pretrain-download-dir",
        DEFAULT_PRETRAIN_DOWNLOAD_DIR,
        "--task-name",
        task_name,
        "--session",
        "catk-clsf-testas-gradually-see-g4",
        "--initial-action",
        "fit",
        "--initial-bs",
        "3",
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
        "5.0e-5",
        "--generated-estimator-learning-rate",
        "5.0e-5",
        "--lr-cosine-final-ratio",
        "1.0",
        "--scorer-scene-num",
        "1680",
        "--estimator-warmup-epochs",
        "0",
        "--no-estimator-warmup-bank",
        "--self-forced-use-stop-motion",
        "false",
        "--decoder-use-stop-motion",
        "false",
        "--train-epoch-sample-fraction",
        "0.25",
        "--train-memory-balanced-batches",
        "true",
        "--max-epochs",
        "4",
        "--check-val-every-n-epoch",
        "1",
        "--limit-val-batches",
        "0.1",
        "--extra-hydra-overrides",
        DEFAULT_EXTRA_OVERRIDES,
        *sys.argv[1:],
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
