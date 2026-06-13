#!/usr/bin/env python3
"""Resume the fm-sf-6 decoder-agent-attention pretrain from epoch 23.

This launcher starts a new tmux session on the existing fm-sf-6 pod and resumes
Lightning state from the source task's latest ``epoch_last.ckpt``. It keeps the
training hyperparameters identical to the original open-loop pretrain run; only
the task/W&B names are changed so the verification resume is isolated.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = (
    Path(__file__).resolve().parent
    / "launch_pre_bc_flow_h100x8_fmsf3_a343315f_gelu_static_pod.py"
)

DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-6"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_decoder"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_2x4_h100"
SOURCE_TASK_NAME = (
    "flow_open_loop_pretrain_decoder_agent_attention_effect_h100x8_"
    "fmsf6_bs20to10_lr6e-4_warm4_val8_membal"
)
DEFAULT_TASK_NAME = f"{SOURCE_TASK_NAME}_resume_epoch24_from_epoch23"
DEFAULT_SESSION = "catk-pretrain-decoder-agent-attention-h100x8-fmsf6-resume-e23"
DEFAULT_WANDB_GROUP = "decoder_agent_attention_pretrain"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_command(command: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
    if dry_run:
        print("+ " + " ".join(shq(part) for part in command))
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout is not None else ""


def kubectl_exec(args: argparse.Namespace, remote_command: str, *, capture: bool = False) -> str:
    return run_command(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            args.pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            remote_command,
        ],
        capture=capture,
        dry_run=args.dry_run,
    )


def resolve_resume_ckpt(args: argparse.Namespace) -> str:
    if args.resume_ckpt:
        return args.resume_ckpt

    runs_dir = (
        f"{args.log_dir.rstrip('/')}/{args.source_task_name.replace('/', '_')}"
        "/runs"
    )
    remote = (
        "set -euo pipefail; "
        f"find {shq(runs_dir)} -path '*/checkpoints/epoch_last.ckpt' "
        "-type f -printf '%T@ %p\\n' 2>/dev/null | "
        "sort -nr | head -1 | cut -d' ' -f2-"
    )
    ckpt = kubectl_exec(args, remote, capture=True).strip()
    if not ckpt:
        raise SystemExit(
            "No epoch_last.ckpt found for source task on the pod: "
            f"{args.source_task_name}"
        )
    return ckpt


def extra_overrides(args: argparse.Namespace) -> str:
    parts = [
        "+trainer.use_distributed_sampler=false",
        f"logger.wandb.name={args.task_name}",
        f"logger.wandb.group={args.wandb_group}",
        "logger.wandb.job_type=resume_epoch24_verify",
    ]
    if args.extra_hydra_overrides:
        parts.append(args.extra_hydra_overrides)
    return " ".join(parts)


def build_base_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        args.namespace,
        "--pod",
        args.pod,
        "--container",
        args.container,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--cache-root",
        args.cache_root,
        "--log-dir",
        args.log_dir,
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--initial-bs",
        str(args.initial_bs),
        "--oom-step",
        str(args.oom_step),
        "--min-bs",
        str(args.min_bs),
        "--val-batch-size",
        str(args.val_batch_size),
        "--test-batch-size",
        str(args.test_batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--check-val-every-n-epoch",
        str(args.check_val_every_n_epoch),
        "--limit-val-batches",
        str(args.limit_val_batches),
        "--learning-rate",
        str(args.learning_rate),
        "--lr-warmup-steps",
        str(args.lr_warmup_steps),
        "--train-memory-balanced-batches",
        str(args.train_memory_balanced_batches),
        "--max-non-oom-retries",
        str(args.max_non_oom_retries),
        "--extra-hydra-overrides",
        extra_overrides(args),
    ]
    if args.replace:
        command.append("--replace")
    if args.no_pull:
        command.append("--no-pull")
    if args.no_monitor_pane:
        command.append("--no-monitor-pane")
    if args.dry_run:
        command.append("--dry-run")
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", args.limit_train_batches])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the fm-sf-6 decoder-agent-attention open-loop pretrain from "
            "the source task epoch_last.ckpt."
        )
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source-task-name", default=SOURCE_TASK_NAME)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--initial-bs", type=int, default=20)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=10)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=64)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=8)
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--lr-warmup-steps", default="4")
    parser.add_argument("--train-memory-balanced-batches", default="true")
    parser.add_argument("--max-non-oom-retries", type=int, default=3)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.nproc_per_node < 1:
        raise SystemExit("--nproc-per-node must be >= 1")
    if args.initial_bs < 1:
        raise SystemExit("--initial-bs must be >= 1")
    if args.oom_step < 1:
        raise SystemExit("--oom-step must be >= 1")
    if args.min_bs < 1:
        raise SystemExit("--min-bs must be >= 1")
    if args.initial_bs < args.min_bs:
        raise SystemExit("--initial-bs must be >= --min-bs")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    command = build_base_command(args)
    if args.stop:
        command.append("--stop")
    else:
        resume_ckpt = resolve_resume_ckpt(args)
        print(f"[resume-decoder-attention] resume checkpoint: {resume_ckpt}")
        command.extend(["--ckpt-path", resume_ckpt])

    run_command(command)


if __name__ == "__main__":
    main()
