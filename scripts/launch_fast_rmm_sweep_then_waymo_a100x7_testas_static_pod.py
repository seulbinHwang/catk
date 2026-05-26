#!/usr/bin/env python3
"""Run Fast-RMM epoch sweep, then launch Waymo submission from the best epoch."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "testas"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_SWEEP_NAME = "fast_rmm_epoch_sweep_a100x7_testas"
DEFAULT_SWEEP_GROUP = "fast_rmm_epoch_sweep_a100x7_testas_rmm_only_bs16"
DEFAULT_SUBMISSION_TASK = "flow_control_waymo_val_best_rmm_a100x7_testas"
DEFAULT_SUBMISSION_GROUP = "waymo_submission_best_rmm_a100x7_testas"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(command: list[str], *, capture: bool = False) -> str:
    print("+ " + " ".join(shq(part) for part in command), flush=True)
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout is not None else ""


def kubectl_exec(args: argparse.Namespace, script: str, *, capture: bool = False) -> str:
    return run(
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
            script,
        ],
        capture=capture,
    )


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def sweep_summary_path(args: argparse.Namespace) -> str:
    return f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}/epoch_sweep_summary.txt"


def launch_sweep(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        script_path("launch_fast_rmm_epoch_sweep_a100x7_testas_static_pod.py"),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pod",
        args.pod,
        "--branch",
        args.branch,
        "--remote-log-dir",
        args.remote_log_dir,
        "--artifact-prefix",
        args.artifact_prefix,
        "--epoch-metadata-values",
        args.epoch_metadata_values,
        "--sweep-name",
        args.sweep_name,
        "--wandb-group",
        args.sweep_wandb_group,
        "--val-batch-size",
        str(args.sweep_val_batch_size),
        "--limit-val-batches",
        str(args.sweep_limit_val_batches),
        "--scorer-scene-num",
        str(args.scorer_scene_num),
        "--n-rollout-closed-val",
        str(args.n_rollout_closed_val),
    ]
    if args.epoch_versions:
        command.extend(["--epoch-versions", args.epoch_versions])
    if args.replace:
        command.append("--replace")
    if args.dry_run:
        command.append("--dry-run")
    run(command)


def launch_submission(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        script_path("launch_waymo_submission_from_best_a100x7_testas_static_pod.py"),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pod",
        args.pod,
        "--branch",
        args.branch,
        "--remote-log-dir",
        args.remote_log_dir,
        "--sweep-name",
        args.sweep_name,
        "--task-name",
        args.submission_task_name,
        "--wandb-group",
        args.submission_wandb_group,
        "--val-batch-size",
        str(args.submission_val_batch_size),
        "--n-rollout-closed-val",
        str(args.n_rollout_closed_val),
        "--evaluation-set",
        args.evaluation_set,
        "--waymo-storage-state-path",
        args.waymo_storage_state_path,
    ]
    if args.replace:
        command.append("--replace")
    if args.dry_run:
        command.append("--dry-run")
    run(command)


def stop(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            script_path("launch_fast_rmm_epoch_sweep_a100x7_testas_static_pod.py"),
            "--namespace",
            args.namespace,
            "--container",
            args.container,
            "--pod",
            args.pod,
            "--sweep-name",
            args.sweep_name,
            "--stop",
        ]
    )
    run(
        [
            sys.executable,
            script_path("launch_waymo_submission_from_best_a100x7_testas_static_pod.py"),
            "--namespace",
            args.namespace,
            "--container",
            args.container,
            "--pod",
            args.pod,
            "--task-name",
            args.submission_task_name,
            "--stop",
        ]
    )


def wait_for_sweep_summary(args: argparse.Namespace) -> str:
    summary = sweep_summary_path(args)
    deadline = time.time() + args.sweep_timeout_seconds
    last_status = ""
    while time.time() < deadline:
        try:
            return kubectl_exec(args, f"test -s {shq(summary)} && cat {shq(summary)}", capture=True)
        except subprocess.CalledProcessError:
            try:
                last_status = kubectl_exec(
                    args,
                    f"tail -n 20 {shq(args.remote_log_dir.rstrip() + '/' + args.sweep_name + '/' + args.pod + '.status')} 2>/dev/null || true",
                    capture=True,
                )
            except subprocess.CalledProcessError:
                last_status = ""
            time.sleep(args.poll_interval_seconds)
    raise SystemExit(
        "Fast-RMM sweep summary did not appear before timeout. Last status:\n"
        + last_status
    )


def parse_best(summary_text: str) -> tuple[int, float | None, float | None, float | None]:
    match = re.search(
        r"BEST_BY_RMM\s+epoch=(\d+)\s+RMM=([0-9.eE+-]+)\s+CPD=([0-9.eE+-]+|NA)\s+CES=([0-9.eE+-]+|NA)",
        summary_text,
    )
    if not match:
        raise SystemExit("BEST_BY_RMM row not found in sweep summary:\n" + summary_text)

    def maybe_float(value: str) -> float | None:
        return None if value == "NA" else float(value)

    return (
        int(match.group(1)),
        maybe_float(match.group(2)),
        maybe_float(match.group(3)),
        maybe_float(match.group(4)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Fast-RMM epoch sweep on testas A100x7, wait for the "
            "BEST_BY_RMM summary, then launch Waymo validation submission from "
            "that best checkpoint."
        )
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--branch", default="semi_control_rolling_fd")
    parser.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    parser.add_argument("--artifact-prefix", default="jksg01019-naver-labs/SMART-FLOW/epoch-last-kngl2eq8")
    parser.add_argument("--epoch-metadata-values", default="57-64")
    parser.add_argument("--epoch-versions", default="")
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--sweep-wandb-group", default=DEFAULT_SWEEP_GROUP)
    parser.add_argument("--submission-task-name", default=DEFAULT_SUBMISSION_TASK)
    parser.add_argument("--submission-wandb-group", default=DEFAULT_SUBMISSION_GROUP)
    parser.add_argument("--sweep-val-batch-size", type=int, default=16)
    parser.add_argument("--sweep-limit-val-batches", default="auto")
    parser.add_argument("--submission-val-batch-size", type=int, default=16)
    parser.add_argument("--scorer-scene-num", type=int, default=1680)
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--evaluation-set", default="validation")
    parser.add_argument(
        "--waymo-storage-state-path",
        default="/mnt/nuplan/projects/catk/secrets/waymo/waymo_storage_state.json",
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--sweep-timeout-seconds", type=int, default=14400)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop:
        stop(args)
        return

    launch_sweep(args)
    if args.dry_run:
        if not args.skip_submission:
            launch_submission(args)
        print("[pipeline] dry-run only; not waiting for sweep summary.")
        return

    summary_text = wait_for_sweep_summary(args)
    epoch, rmm, cpd, ces = parse_best(summary_text)
    print(
        "[pipeline] BEST_BY_RMM "
        f"epoch={epoch} RMM={rmm} CPD={cpd} CES={ces}",
        flush=True,
    )

    if args.skip_submission:
        print("[pipeline] --skip-submission set; Waymo submission not launched.")
        return

    launch_submission(args)
    print("[pipeline] Waymo submission launcher started from best Fast-RMM checkpoint.")


if __name__ == "__main__":
    main()
