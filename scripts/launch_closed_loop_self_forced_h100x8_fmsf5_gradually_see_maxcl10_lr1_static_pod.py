#!/usr/bin/env python3
"""Launch the fm-sf-5 gradually-see maxcl10 run with LR final ratio 1.0."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_closed_loop_self_forced_h100x8_fmsf3_static_pod.py")
DEFAULT_RESUME_CKPT = (
    "/workspace/closed_loop_gradually_see_resume/"
    "a100x4x2_lrdecay001_resume_e5/epoch_last.ckpt"
)
DEFAULT_PROJECT_ROOT = "/tmp/catk_self_forcing_closed_loop_fmsf5_maxcl10"
DEFAULT_LOG_DIR = "/tmp/catk_self_forcing_closed_loop_logs"
DEFAULT_SESSION = "catk-clsf-h100x8-fmsf5-graduallysee-maxcl10-lrdecay1"

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "trainer.num_nodes=1",
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
        "flow_closed_loop_self_forced_h100x8_fmsf5_g4_seeall_"
        "graduallysee_1e1dec_lrdecay1_resume_e5_maxcl10_"
        f"{timestamp}"
    )
    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-sp-labs-reai-training",
        "--pod",
        "fm-sf-5",
        "--branch",
        "self_forcing_closed_loop",
        "--pull",
        "--project-root",
        DEFAULT_PROJECT_ROOT,
        "--log-dir",
        DEFAULT_LOG_DIR,
        "--experiment",
        "self_forced_npfm_a100x4x2",
        "--action",
        "fit",
        "--ckpt-path",
        DEFAULT_RESUME_CKPT,
        "--pretrain-ckpt",
        DEFAULT_RESUME_CKPT,
        "--task-name",
        task_name,
        "--session",
        DEFAULT_SESSION,
        "--nproc-per-node",
        "8",
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
        "--learning-rate",
        "7e-5",
        "--generated-estimator-learning-rate",
        "7e-5",
        "--lr-cosine-final-ratio",
        "1.0",
        "--estimator-warmup-epochs",
        "0",
        "--max-epochs",
        "5",
        "--max-closed-loop-epochs",
        "10",
        "--check-val-every-n-epoch",
        "1",
        "--limit-val-batches",
        "0.1",
        "--train-epoch-sample-fraction",
        "0.25",
        "--train-memory-balanced-batches",
        "--extra-hydra-overrides",
        DEFAULT_EXTRA_OVERRIDES,
        *sys.argv[1:],
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
