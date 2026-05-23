#!/usr/bin/env python3
"""Launch execution-context pretrain on hsb-npc-training-2 + wo-pvc-1.

This launcher targets the already-running ``hsb-npc-training-2`` H100x4 pod
and ``wo-pvc-1`` H100x2 pod. It does not create, delete, or restart pods. It
only prepares the optional memory-balance metadata cache and starts/replaces
the configured tmux training session inside the existing pods.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_h100x4_multinode_pretrain_tmux.py")

DEFAULT_PODS = ("hsb-npc-training-2", "wo-pvc-1")
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4x2_execctx_balanced"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x6_hsb2_wo1_"
    "execctx_prefix_balanced_lr6e-4_bs18"
)
DEFAULT_SESSION = "catk-control-pretrain-h100x6-hsb2-wo1-execctx-balanced"
DEFAULT_METADATA_CACHE_RELATIVE = "dataset_metadata/womd_training_memory_balance_v1.pt"
DEFAULT_CACHE_ROOT_BY_POD = {
    "hsb-npc-training-2": "/workspace/womd_v1_3/SMART_cache",
    "wo-pvc-1": "/workspace/womd_v1_3/SMART_cache",
}
HETEROGENEOUS_STRATEGY_OVERRIDES = (
    "trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy",
    "trainer.strategy.find_unused_parameters=true",
    "+trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment",
)
PINNED_BOOLEAN_OVERRIDES = {
    "data.train_memory_balanced_batches": "true",
    "trainer.use_distributed_sampler": "false",
}


def shq(value: object) -> str:
    return shlex.quote(str(value))


def parse_pod_cache_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"--pod-cache-root must use POD=PATH, got {value!r}"
            )
        pod, path = value.split("=", 1)
        pod = pod.strip()
        path = path.strip()
        if not pod or not path:
            raise argparse.ArgumentTypeError(
                f"--pod-cache-root must include both POD and PATH, got {value!r}"
            )
        roots[pod] = path
    return roots


def cache_root_for_pod(args: argparse.Namespace, pod: str) -> str:
    if pod in args.pod_cache_root_map:
        return args.pod_cache_root_map[pod]
    if args.cache_root:
        return args.cache_root
    return DEFAULT_CACHE_ROOT_BY_POD.get(pod, "/workspace/womd_v1_3/SMART_cache")


def metadata_cache_path(args: argparse.Namespace) -> str:
    if args.metadata_cache_path:
        return args.metadata_cache_path
    return f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE_RELATIVE}"


def hydra_override_key_value(token: str) -> tuple[str, str] | None:
    if "=" not in token:
        return None
    key, value = token.split("=", 1)
    return key.lstrip("+").strip(), value.strip().lower()


def validate_extra_hydra_overrides(parser: argparse.ArgumentParser, overrides: str) -> None:
    if not overrides:
        return
    try:
        tokens = shlex.split(overrides)
    except ValueError as exc:
        parser.error(f"--extra-hydra-overrides could not be parsed: {exc}")
    for token in tokens:
        parsed = hydra_override_key_value(token)
        if parsed is None:
            continue
        key, value = parsed
        if key in PINNED_BOOLEAN_OVERRIDES and value != PINNED_BOOLEAN_OVERRIDES[key]:
            parser.error(
                f"{key}={value} is unsafe for the heterogeneous H100x6 launcher. "
                f"This launcher pins {key}={PINNED_BOOLEAN_OVERRIDES[key]}."
            )
        if key == "trainer.check_val_every_n_epoch":
            parser.error(
                "Use --check-val-every-n-epoch instead of overriding "
                "trainer.check_val_every_n_epoch through --extra-hydra-overrides."
            )


def training_extra_hydra_overrides(args: argparse.Namespace) -> str:
    overrides: list[str] = []
    if args.extra_hydra_overrides:
        overrides.append(args.extra_hydra_overrides)
    overrides.extend(
        [
            "data.train_memory_balanced_batches=true",
            f"data.train_memory_balance_metadata_cache={metadata_cache_path(args)}",
            "data.train_memory_balance_build_on_missing=false",
            "trainer.use_distributed_sampler=false",
            f"trainer.check_val_every_n_epoch={args.check_val_every_n_epoch}",
            f"model.model_config.n_rollout_closed_val={args.n_rollout_closed_val}",
            *HETEROGENEOUS_STRATEGY_OVERRIDES,
        ]
    )
    return " ".join(overrides)


def remote_git_prepare_script(args: argparse.Namespace) -> str:
    branch_ref = f"refs/heads/{args.branch}"
    origin_ref = f"origin/{args.branch}"
    fetch_refspec = f"+{args.branch}:refs/remotes/origin/{args.branch}"
    if args.git_ref:
        return " && ".join(
            [
                f"git config --global --add safe.directory {shq(args.project_root)} || true",
                f"git update-ref -d {shq(f'refs/remotes/origin/{args.branch}')} || true",
                f"git fetch origin --prune {shq(fetch_refspec)}",
                f"git checkout -f {shq(args.git_ref)}",
            ]
        )
    if args.no_pull:
        return f"git config --global --add safe.directory {shq(args.project_root)} || true"
    return " && ".join(
        [
            f"git config --global --add safe.directory {shq(args.project_root)} || true",
            f"git update-ref -d {shq(f'refs/remotes/origin/{args.branch}')} || true",
            f"git fetch origin --prune {shq(fetch_refspec)}",
            (
                f"if git show-ref --verify --quiet {shq(branch_ref)}; then "
                f"git checkout {shq(args.branch)}; "
                f"else git checkout -b {shq(args.branch)} {shq(origin_ref)}; fi"
            ),
            f"git pull --ff-only origin {shq(args.branch)}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch 6-H100 execution-context-aligned control-space pretrain on "
            "hsb-npc-training-2 and wo-pvc-1."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "p-pnc"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument(
        "--pods",
        nargs="+",
        default=os.environ.get("PODS", " ".join(DEFAULT_PODS)).split(),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", "/mnt/nuplan/projects/catk"),
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("CATK_BRANCH") or "semi_control_rolling",
    )
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("CATK_GIT_REF", ""),
        help="Exact git ref/SHA to checkout on every pod instead of the branch head.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not git fetch/pull before metadata prebuild or launch.",
    )
    parser.add_argument(
        "--remote-log-dir",
        default=os.environ.get("REMOTE_LOG_DIR", "/mnt/nuplan/projects/catk/logs"),
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--train-batch-size", type=int, default=18)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--val-batch-size", default="16")
    parser.add_argument(
        "--nccl-algo",
        default=os.environ.get("NCCL_ALGO", "Ring"),
        help="NCCL_ALGO exported in each remote tmux run. Default avoids 4+2 default NCCL hangs.",
    )
    parser.add_argument(
        "--nccl-proto",
        default=os.environ.get("NCCL_PROTO", "Simple"),
        help="NCCL_PROTO exported in each remote tmux run. Default avoids 4+2 default NCCL hangs.",
    )
    parser.add_argument(
        "--n-rollout-closed-val",
        type=int,
        default=32,
        help=(
            "Number of closed-loop rollouts per scenario during validation. "
            "The H100x6 run uses 32 by default for a lower-variance fit-time Fast RMM score."
        ),
    )
    parser.add_argument("--master-port", default="29620")
    parser.add_argument("--checkpoint-sync-port", default="29621")
    parser.add_argument("--nproc-per-node", default="gpu", choices=("gpu", "auto"))
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=16,
        help=(
            "Validation cadence for the H100x6 run. Default runs four fit-time "
            "validations during the 64-epoch pretrain."
        ),
    )
    parser.add_argument(
        "--extra-hydra-overrides",
        default="",
        help="Additional space-separated Hydra overrides appended after the preset.",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("CACHE_ROOT", ""),
        help="Use one CACHE_ROOT for every pod. Pod-specific defaults are used when omitted.",
    )
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help="Override CACHE_ROOT for one pod. Can be repeated.",
    )
    parser.add_argument(
        "--metadata-cache-path",
        default=os.environ.get("MEMORY_BALANCE_METADATA_CACHE", ""),
        help=(
            "Absolute metadata cache path. Defaults to "
            "$REMOTE_LOG_DIR/dataset_metadata/womd_training_memory_balance_v1.pt."
        ),
    )
    parser.add_argument("--metadata-num-workers", type=int, default=8)
    parser.add_argument(
        "--prebuild-metadata",
        action="store_true",
        help="Build the memory-balance metadata cache on each pod before launch.",
    )
    parser.add_argument(
        "--force-metadata",
        action="store_true",
        help="Pass --force to the metadata prebuild tool to remove stale locks first.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != len(DEFAULT_PODS) and not args.stop:
        parser.error(f"this preset expects exactly {len(DEFAULT_PODS)} pods")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.metadata_num_workers < 1:
        parser.error("--metadata-num-workers must be >= 1")
    if args.check_val_every_n_epoch < 1:
        parser.error("--check-val-every-n-epoch must be >= 1")
    validate_extra_hydra_overrides(parser, args.extra_hydra_overrides)
    try:
        args.pod_cache_root_map = parse_pod_cache_roots(args.pod_cache_root)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def run_pod_command(args: argparse.Namespace, pod: str, script: str) -> int:
    command = [
        "kubectl",
        "exec",
        "-n",
        args.namespace,
        pod,
        "-c",
        args.container,
        "--",
        "bash",
        "-lc",
        script,
    ]
    if args.dry_run:
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


def prebuild_metadata(args: argparse.Namespace) -> int:
    cache_path = metadata_cache_path(args)
    for pod in args.pods:
        cache_root = cache_root_for_pod(args, pod)
        command = [
            "cd",
            shq(args.project_root),
            "&&",
            remote_git_prepare_script(args),
            "&&",
            'CATK_REMOTE_PYTHON="${CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}"',
            "&&",
            '"$CATK_REMOTE_PYTHON"',
            "tools/build_memory_balance_metadata.py",
            "--raw-dir",
            shq(f"{cache_root.rstrip('/')}/training"),
            "--cache-path",
            shq(cache_path),
            "--num-workers",
            shq(args.metadata_num_workers),
        ]
        if args.force_metadata:
            command.append("--force")
        status = run_pod_command(args, pod, " ".join(str(part) for part in command))
        if status != 0:
            return status
    return 0


def verify_metadata_cache(args: argparse.Namespace) -> int:
    cache_path = metadata_cache_path(args)
    for pod in args.pods:
        script = (
            f"if [[ ! -f {shq(cache_path)} ]]; then "
            f"echo {shq('[metadata-check] missing memory-balance metadata cache: ' + cache_path)} >&2; "
            "echo '[metadata-check] rerun with --prebuild-metadata, or pass --metadata-cache-path to an existing cache.' >&2; "
            "exit 2; "
            "fi"
        )
        status = run_pod_command(args, pod, script)
        if status != 0:
            return status
    return 0


def base_launcher_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pods",
        *args.pods,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--log-dir",
        args.remote_log_dir,
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--train-batch-size",
        str(args.train_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--val-batch-size",
        str(args.val_batch_size),
        "--master-port",
        str(args.master_port),
        "--checkpoint-sync-port",
        str(args.checkpoint_sync_port),
        "--manual-rank-offsets",
        "--nproc-per-node",
        args.nproc_per_node,
        "--extra-hydra-overrides",
        training_extra_hydra_overrides(args),
    ]
    if args.nccl_algo:
        command.extend(["--remote-env", f"NCCL_ALGO={args.nccl_algo}"])
    if args.nccl_proto:
        command.extend(["--remote-env", f"NCCL_PROTO={args.nccl_proto}"])
    if args.git_ref:
        command.extend(["--git-ref", args.git_ref])
    if args.no_pull:
        command.append("--no-pull")
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", args.limit_train_batches])
    if args.limit_val_batches:
        command.extend(["--limit-val-batches", args.limit_val_batches])
    if args.max_epochs:
        command.extend(["--max-epochs", args.max_epochs])
    if args.cache_root:
        command.extend(["--cache-root", args.cache_root])
    for mapping in args.pod_cache_root:
        command.extend(["--pod-cache-root", mapping])
    if args.replace:
        command.append("--replace")
    if args.stop:
        command.append("--stop")
    return command


def main() -> int:
    args = parse_args()
    if args.prebuild_metadata and not args.stop:
        status = prebuild_metadata(args)
        if status != 0:
            return status
    elif not args.stop:
        status = verify_metadata_cache(args)
        if status != 0:
            return status

    command = base_launcher_command(args)
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(shq(part) for part in command))
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
