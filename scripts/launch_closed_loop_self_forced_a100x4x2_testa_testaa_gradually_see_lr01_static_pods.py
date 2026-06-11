#!/usr/bin/env python3
"""Launch the fm-sf-5 gradually-see recipe on testa+testaa with LR final ratio 0.1."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_self_forced_a100x4x2_static_pods.py")
DEFAULT_RESUME_CKPT = (
    "/workspace/closed_loop_gradually_see_resume/"
    "a100x4x2_lrdecay001_resume_e5/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/closed_loop_gradually_see_resume/"
    "a100x4x2_lrdecay001_resume_e5/artifact"
)
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_SESSION = "catk-clsf-a100x4x2-testa-testaa-graduallysee-maxcl10-lrdecay01"

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "trainer.max_closed_loop_epochs=10",
        "model.model_config.val_open_loop=false",
        "model.model_config.self_forced.use_distribution_matching_loss=true",
        "model.model_config.self_forced.distribution_matching_objective=dmd",
        "model.model_config.self_forced.use_anchor_flow_matching_loss=false",
        "model.model_config.self_forced.generated_estimator_bank_target_warmup_epochs=0",
        "model.model_config.self_forced.generated_estimator_bank_loaded_warmup_epochs=0",
        "model.model_config.self_forced.closed_loop_sf_global_max_step=4",
        "model.model_config.self_forced.closed_loop_sf_local_max_step=4",
        "model.model_config.self_forced.closed_loop_see_all=true",
        "model.model_config.self_forced.gradually_see=true",
        "model.model_config.self_forced.update_open_loop_teacher_when_roll=false",
        "data.num_workers=4",
        "data.prefetch_factor=1",
        "data.train_epoch_sample_fraction_shuffle_flag=false",
        "model.model_config.self_forced.estimator_updates_per_step=5",
        "model.model_config.self_forced.project_dmd_to_pose_space=false",
        "model.model_config.self_forced.dmd_use_stable_scale_filter=true",
        "model.model_config.self_forced.dmd_stable_scale_scope=agent",
        "model.model_config.self_forced.dmd_use_teacher_alignment_filter=false",
        "model.model_config.self_forced.dmd_use_trust_region_filter=false",
        "model.model_config.self_forced.dmd_use_injection_ramp=false",
        "model.model_config.self_forced.detach_block_transition=false",
        "model.model_config.self_forced.sampling.sample_steps=16",
        "model.model_config.self_forced.sampling.sample_method=euler",
        "model.model_config.self_forced.sampling.noise_scale=1.0",
        "model.model_config.self_forced.sampling.backprop_last_k=8",
        "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch",
        "model.model_config.self_forced.sampling.random_terminal_step.policy=all",
        "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=16",
    ]
)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = (
        "flow_closed_loop_self_forced_a100x4x2_testa_testaa_g4_seeall_"
        "graduallysee_1e1dec_lrdecay01_resume_e5_maxcl10_"
        f"{timestamp}"
    )
    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-pnc",
        "--pods",
        "testa",
        "testaa",
        "--branch",
        "self_forcing_closed_loop",
        "--project-root",
        DEFAULT_PROJECT_ROOT,
        "--log-dir",
        DEFAULT_LOG_DIR,
        "--cache-root",
        "/workspace/womd_v1_3/SMART_cache",
        "--experiment",
        "self_forced_npfm_a100x4x2",
        "--pretrain-ckpt",
        DEFAULT_RESUME_CKPT,
        "--pretrain-download-dir",
        DEFAULT_PRETRAIN_DOWNLOAD_DIR,
        "--task-name",
        task_name,
        "--session",
        DEFAULT_SESSION,
        "--initial-action",
        "fit",
        "--nproc-per-node",
        "4",
        "--initial-bs",
        "72",
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
        "7e-5",
        "--generated-estimator-learning-rate",
        "7e-5",
        "--lr-cosine-final-ratio",
        "0.1",
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
        "5",
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
