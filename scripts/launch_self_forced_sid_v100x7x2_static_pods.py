#!/usr/bin/env python3
"""Launch SiD self-forced V100x7x2 static multi-node training on existing pods."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_self_forced_v100x4x4_static_pods as base


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["testv", "testvv"]
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_sid_v100x7x2"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-sjan8kmh:v32"
)
DEFAULT_PRETRAIN_CKPT = (
    "/mnt/nuplan/projects/catk/downloads/wandb_ckpts/"
    "flow_semi_continuous_finetune_inv_euler_32_a100x4/"
    "epoch-last-sjan8kmh_v32/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/mnt/nuplan/projects/catk/downloads/wandb_ckpts/"
    "flow_semi_continuous_finetune_inv_euler_32_a100x4/"
    "epoch-last-sjan8kmh_v32/artifact"
)
DEFAULT_TASK_NAME = (
    "flow_self_forced_sid_v100x7x2_"
    "unfrozen_except_map_encoder_estimator_warmup_1_bs4"
)
DEFAULT_SESSION = "catk-sf-sid-v100x7x2-exceptmap-warmup1"

_base_render_worker_script = base.render_worker_script


def render_worker_script(env_file: str) -> str:
    """Reuse the battle-tested V100 launcher body with SID/V100x7x2 log labels."""
    return _base_render_worker_script(env_file).replace(
        "[self-forced-v100x4x4]",
        "[self-forced-sid-v100x7x2]",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SiD self-forced V100x7x2 training on existing static pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--pretrain-ckpt",
        default=DEFAULT_PRETRAIN_CKPT,
        help=(
            "Local checkpoint path. If missing, the launcher downloads "
            "WANDB_PRETRAIN_ARTIFACT here before training."
        ),
    )
    parser.add_argument("--wandb-pretrain-artifact", default=DEFAULT_WANDB_PRETRAIN_ARTIFACT)
    parser.add_argument("--pretrain-download-dir", default=DEFAULT_PRETRAIN_DOWNLOAD_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29547")
    parser.add_argument(
        "--retry-sync-port",
        default="29548",
        help="Rank-0 pod HTTP port used to collect retry status from all pods.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=7)
    parser.add_argument("--initial-bs", type=int, default=4)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--scorer-scene-num", type=int, default=280)
    parser.add_argument("--unfrozen-range", default="except_map_encoder")
    parser.add_argument("--estimator-warmup-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--train-epoch-sample-fraction", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if len(args.pods) != 2:
        parser.error("--pods must contain exactly two pods for the V100x7x2 preset")
    if args.nproc_per_node != 7:
        parser.error("--nproc-per-node must be 7 for the V100x7x2 preset")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if not args.pretrain_ckpt:
        parser.error("--pretrain-ckpt must not be empty unless --stop is set")
    if not args.wandb_pretrain_artifact:
        parser.error("--wandb-pretrain-artifact must not be empty unless --stop is set")
    if not args.pretrain_download_dir:
        parser.error("--pretrain-download-dir must not be empty unless --stop is set")
    return args


def main() -> None:
    base.parse_args = parse_args
    base.render_worker_script = render_worker_script
    base.main()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
