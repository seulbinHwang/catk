#!/usr/bin/env python3
"""Launch the fm-sf-6 H100x8 see_all=false maxcl10 run with LR final ratio 0.1."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_closed_loop_self_forced_h100x8_fmsf3_static_pod.py")
DEFAULT_RESUME_CKPT = (
    "/workspace/closed_loop_see_all_resume/"
    "epoch-last-62ihisgm-v7/epoch_last_start_epoch5.ckpt"
)

DEFAULT_EXTRA_OVERRIDES = " ".join(
    [
        "model.model_config.scorer_scene_num=1680",
        "model.model_config.val_open_loop=false",
        "model.model_config.self_forced.generated_estimator_bank_target_warmup_epochs=0",
        "model.model_config.self_forced.generated_estimator_bank_loaded_warmup_epochs=0",
    ]
)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    task_name = (
        "flow_closed_loop_self_forced_h100x8_fmsf6_g4_seeallfalse_"
        "1e1dec_lrdecay01_maxcl10_resume_e5_"
        f"{timestamp}"
    )
    args = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        "p-sp-labs-reai-training",
        "--pod",
        "fm-sf-6",
        "--container",
        "main",
        "--project-root",
        "/tmp/catk_self_forcing_closed_loop_fmsf6_seeallfalse_maxcl10",
        "--branch",
        "self_forcing_closed_loop",
        "--pull",
        "--replace",
        "--session",
        "catk-closed-loop-sf-g4-seeallfalse-lrdecay01-maxcl10-fmsf6",
        "--task-name",
        task_name,
        "--experiment",
        "self_forced_npfm_h100_6",
        "--action",
        "fit",
        "--ckpt-path",
        DEFAULT_RESUME_CKPT,
        "--initial-bs",
        "72",
        "--oom-step",
        "2",
        "--min-bs",
        "2",
        "--learning-rate",
        "7e-5",
        "--generated-estimator-learning-rate",
        "7e-5",
        "--lr-cosine-final-ratio",
        "0.1",
        "--estimator-warmup-epochs",
        "0",
        "--max-epochs",
        "5",
        "--max-closed-loop-epochs",
        "10",
        "--check-val-every-n-epoch",
        "1",
        "--closed-loop-sf-global-max-step",
        "4",
        "--closed-loop-sf-local-max-step",
        "4",
        "--no-closed-loop-see-all",
        "--no-update-open-loop-teacher-when-roll",
        "--train-epoch-sample-fraction",
        "0.25",
        "--no-train-epoch-sample-fraction-shuffle-flag",
        "--train-memory-balanced-batches",
        "--limit-val-batches",
        "0.1",
        "--val-batch-size",
        "8",
        "--test-batch-size",
        "8",
        "--estimator-updates-per-step",
        "5",
        "--unfrozen-range",
        "middle",
        "--no-project-dmd-to-pose-space",
        "--dmd-use-stable-scale-filter",
        "--dmd-stable-scale-scope",
        "agent",
        "--no-dmd-use-teacher-alignment-filter",
        "--no-dmd-use-trust-region-filter",
        "--no-dmd-use-injection-ramp",
        "--no-detach-block-transition",
        "--sample-steps",
        "16",
        "--sample-method",
        "euler",
        "--noise-scale",
        "1.0",
        "--backprop-last-k",
        "8",
        "--random-terminal-scope",
        "global_batch",
        "--random-terminal-policy",
        "all",
        "--min-executed-steps",
        "16",
        "--extra-hydra-overrides",
        DEFAULT_EXTRA_OVERRIDES,
        *sys.argv[1:],
    ]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
