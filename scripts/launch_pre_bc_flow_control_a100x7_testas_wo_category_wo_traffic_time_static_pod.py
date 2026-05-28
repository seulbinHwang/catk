#!/usr/bin/env python3
"""Launch holonomic wo-category + wo-traffic-time ablation pretrain on testas A100x7.

This launcher targets one existing pod only. It never creates, deletes, or
restarts pods; it starts/stops only one tmux session and task-specific training
processes inside ``testas``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "testas"
DEFAULT_BRANCH = "semi_control_stable_wo_category_wo_traffic_time"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_a100x7_wo_category_wo_traffic_time_"
    "holonomic_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs16"
)
DEFAULT_SESSION = "catk-control-pretrain-a100x7-wo-category-wo-traffic-time-holonomic"
DEFAULT_METADATA_CACHE = (
    "dataset_metadata/"
    "womd_training_memory_balance_a100x7_testas_wo_category_wo_traffic_time.pt"
)
DEFAULT_REMOTE_PYTHON = "/mnt/nuplan/miniforge/envs/catk/bin/python"
DEFAULT_HOLONOMIC_OVERRIDE = "model.model_config.token_processor.use_holonomic_model_only=true"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
    command = ["kubectl", *args]
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


def kubectl_bash(
    *,
    namespace: str,
    container: str,
    pod: str,
    script: str,
    capture: bool = False,
    dry_run: bool = False,
) -> str:
    return run_kubectl(
        [
            "exec",
            "-n",
            namespace,
            pod,
            "-c",
            container,
            "--",
            "bash",
            "-lc",
            script,
        ],
        capture=capture,
        dry_run=dry_run,
    )


def pod_gpu_count(args: argparse.Namespace) -> int:
    if args.dry_run:
        return args.nproc_per_node
    output = kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script="nvidia-smi -L 2>/dev/null | wc -l",
        capture=True,
    )
    count = int(output.strip())
    if count < 1:
        raise RuntimeError(f"no visible GPUs found in pod {args.pod}")
    return count


def metadata_cache_path(args: argparse.Namespace) -> str:
    return args.memory_metadata_cache_path or (
        f"{args.remote_log_dir.rstrip('/')}/{DEFAULT_METADATA_CACHE}"
    )


def sync_project(args: argparse.Namespace) -> None:
    branch_q = shq(args.branch)
    git_ref_q = shq(args.git_ref)
    project_root_q = shq(args.project_root)
    if args.git_ref:
        git_cmd = f"git fetch origin {branch_q} && git checkout {git_ref_q}"
    else:
        git_cmd = (
            f"git fetch origin {branch_q} && "
            f"{{ git checkout {branch_q} 2>/dev/null || git checkout -B {branch_q} origin/{branch_q}; }} && "
            f"git pull --ff-only origin {branch_q}"
        )
    kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script=f"set -euo pipefail\ncd {project_root_q}\n{git_cmd}",
        dry_run=args.dry_run,
    )


def stop_session(args: argparse.Namespace) -> None:
    session_q = shq(args.session)
    if args.task_name:
        # Avoid matching the kubectl/bash command that is currently executing
        # this cleanup script.
        task_pattern = f"[{args.task_name[0]}]{args.task_name[1:]}"
    else:
        task_pattern = ""
    task_q = shq(task_pattern)
    script = f"""
set -euo pipefail
if tmux has-session -t {session_q} 2>/dev/null; then
  tmux send-keys -t {session_q} C-c || true
  sleep 3
  tmux kill-session -t {session_q} || true
fi
if [[ -n {task_q} ]]; then
  pkill -TERM -f {task_q} 2>/dev/null || true
fi
sleep 3
if [[ -n {task_q} ]]; then
  pkill -KILL -f {task_q} 2>/dev/null || true
fi
"""
    kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script=script,
        dry_run=args.dry_run,
    )


def prebuild_metadata(args: argparse.Namespace) -> None:
    metadata = metadata_cache_path(args)
    force_arg = "--force" if args.force_memory_metadata_rebuild else ""
    project_root_q = shq(args.project_root)
    raw_dir_q = shq(f"{args.cache_root.rstrip('/')}/training")
    metadata_q = shq(metadata)
    python_q = shq(args.remote_python)
    workers_q = shq(args.memory_metadata_num_workers)
    script = f"""
set -euo pipefail
cd {project_root_q}
mkdir -p "$(dirname {metadata_q})"
test -d {raw_dir_q}
{python_q} tools/build_memory_balance_metadata.py \
  --raw-dir {raw_dir_q} \
  --cache-path {metadata_q} \
  --num-workers {workers_q} \
  {force_arg}
"""
    kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script=script,
        dry_run=args.dry_run,
    )


def validate_metadata(args: argparse.Namespace) -> None:
    metadata = metadata_cache_path(args)
    metadata_q = shq(metadata)
    script = f"""
set -euo pipefail
test -s {metadata_q} || {{
  echo "missing memory-balanced metadata cache: {metadata}" >&2
  echo "rerun with --prebuild-metadata or pass --memory-metadata-cache-path" >&2
  exit 4
}}
"""
    kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script=script,
        dry_run=args.dry_run,
    )


def render_remote_train_script(args: argparse.Namespace, *, gpu_count: int) -> str:
    metadata = metadata_cache_path(args)
    run_root = (
        f"{args.remote_log_dir.rstrip('/')}/tmux_a100x7_single_pod_pretrain/"
        f"{args.task_name}"
    )
    run_log = f"{run_root}/train.log"
    extra_overrides = " ".join(
        part
        for part in (
            "trainer.check_val_every_n_epoch=16",
            f"data.train_memory_balance_metadata_cache={metadata}",
            "data.train_memory_balance_build_on_missing=false",
            args.extra_hydra_overrides.strip(),
            DEFAULT_HOLONOMIC_OVERRIDE,
        )
        if part
    )
    optional_exports: list[str] = []
    if args.limit_train_batches:
        optional_exports.append(f"export LIMIT_TRAIN_BATCHES={shq(args.limit_train_batches)}")
    if args.limit_val_batches:
        optional_exports.append(f"export LIMIT_VAL_BATCHES={shq(args.limit_val_batches)}")
    if args.max_epochs:
        optional_exports.append(f"export MAX_EPOCHS={shq(args.max_epochs)}")

    return f"""
set -Eeuo pipefail
export TERM="${{TERM:-xterm-256color}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-eth0}}"
export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${{NCCL_SOCKET_FAMILY:-AF_INET}}"
export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
export NCCL_NVLS_ENABLE="${{NCCL_NVLS_ENABLE:-0}}"
export NCCL_CUMEM_ENABLE="${{NCCL_CUMEM_ENABLE:-0}}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${{TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}}"
export CATK_ATTENTION_GRAPH_FP32="${{CATK_ATTENTION_GRAPH_FP32:-1}}"
export WANDB_ENTITY="${{WANDB_ENTITY:-jksg01019-naver-labs}}"
export WANDB_PROJECT="${{WANDB_PROJECT:-SMART-FLOW}}"

PROJECT_ROOT={shq(args.project_root)}
RUN_ROOT={shq(run_root)}
RUN_LOG={shq(run_log)}
mkdir -p "$RUN_ROOT"
cd "$PROJECT_ROOT"

echo "[$(date '+%F %T')] A100x7 holonomic wo-category + wo-traffic-time pretrain start" | tee -a "$RUN_LOG"
echo "branch={args.branch} commit=$(git rev-parse --short HEAD)" | tee -a "$RUN_LOG"
echo "pod={args.pod} gpu_count={gpu_count} train_batch_size={args.train_batch_size} val_batch_size={args.val_batch_size}" | tee -a "$RUN_LOG"
echo "holonomic_override={DEFAULT_HOLONOMIC_OVERRIDE}" | tee -a "$RUN_LOG"
echo "metadata={metadata}" | tee -a "$RUN_LOG"

export CATK_EXPERIMENT={shq(args.experiment)}
export CATK_ACTION=fit
export TASK_NAME={shq(args.task_name)}
export LOG_DIR={shq(args.remote_log_dir)}
export CACHE_ROOT={shq(args.cache_root)}
export NNODES=1
export NPROC_PER_NODE={gpu_count}
export TRAINER_DEVICES={gpu_count}
export RDZV_ID={shq(args.task_name)}
export RDZV_ENDPOINT={shq('127.0.0.1:' + str(args.master_port))}
export MASTER_PORT={shq(args.master_port)}
export TRAIN_BATCH_SIZE={shq(args.train_batch_size)}
export VAL_BATCH_SIZE={shq(args.val_batch_size)}
export CATK_LR={shq(args.learning_rate)}
export CATK_HYDRA_OVERRIDES={shq(extra_overrides)}
{chr(10).join(optional_exports)}

bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee -a "$RUN_LOG"
"""


def start_session(args: argparse.Namespace, *, gpu_count: int) -> None:
    session_q = shq(args.session)
    remote_script = render_remote_train_script(args, gpu_count=gpu_count)
    run_root = (
        f"{args.remote_log_dir.rstrip('/')}/tmux_a100x7_single_pod_pretrain/"
        f"{args.task_name}"
    )
    run_root_q = shq(run_root)
    remote_script_path = f"{run_root}/launcher.sh"
    remote_script_path_q = shq(remote_script_path)
    command = f"""
set -euo pipefail
if tmux has-session -t {session_q} 2>/dev/null; then
  echo "tmux session already exists: {args.session}" >&2
  exit 3
fi
mkdir -p {run_root_q}
cat > {remote_script_path_q} <<'CATK_REMOTE_SCRIPT'
{remote_script}
CATK_REMOTE_SCRIPT
chmod +x {remote_script_path_q}
tmux new-session -d -s {session_q} "bash {remote_script_path_q}"
tmux ls | grep -F {session_q}
"""
    kubectl_bash(
        namespace=args.namespace,
        container=args.container,
        pod=args.pod,
        script=command,
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch semi_control_stable_wo_category_wo_traffic_time pretrain on "
            "the existing testas A100x7 pod with use_holonomic_model_only=true."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--pod", default=os.environ.get("POD", DEFAULT_POD))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--remote-python", default=os.environ.get("CATK_REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--nproc-per-node", type=int, default=7, help="Expected GPU count used in dry-run output.")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", default="6e-4")
    parser.add_argument("--master-port", default="29871")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--memory-metadata-cache-path", default="")
    parser.add_argument("--memory-metadata-num-workers", type=int, default=8)
    parser.add_argument("--prebuild-metadata", action="store_true")
    parser.add_argument("--force-memory-metadata-rebuild", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.memory_metadata_num_workers < 1:
        parser.error("--memory-metadata-num-workers must be >= 1")
    return args


def main() -> int:
    args = parse_args()

    if args.stop:
        stop_session(args)
        return 0

    gpu_count = pod_gpu_count(args)
    if gpu_count != args.nproc_per_node:
        print(
            f"[info] pod {args.pod} reports {gpu_count} GPUs; using actual count instead of "
            f"--nproc-per-node={args.nproc_per_node}."
        )

    if args.replace:
        stop_session(args)
    sync_project(args)
    if args.prebuild_metadata:
        prebuild_metadata(args)
    else:
        validate_metadata(args)
    start_session(args, gpu_count=gpu_count)

    if args.dry_run:
        print("Dry-run complete; no tmux session was started.")
    else:
        print(f"Started tmux session {args.session!r} on pod {args.pod!r}.")
        print(
            "Attach with: "
            f"kubectl exec -it -n {args.namespace} {args.pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
