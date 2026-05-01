#!/usr/bin/env python3
"""Render an MLX Kubeflow PyTorchJob for CAT-K V100x8 multi-node fine-tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_IMAGE = "labs-ad2flow.n3r.reg.navercorp.com/mlx_exp/pnc_traffic_model:20260121"
DEFAULT_REPO_URL = "https://github.com/seulbinHwang/catk.git"
DEFAULT_BRANCH = "semi_continuous_track_loss"
DEFAULT_ZONE = "private-v100-naverlabs-0"
DEFAULT_PRETRAIN_CKPT = (
    "/mnt/nuplan/projects/catk/checkpoints/"
    "flow_semi_continuous_pretrain_all_target_h1006/run_4pxhrpv8_v70/epoch_last.ckpt"
)


def q(value: object) -> str:
    """Return a YAML-safe double-quoted scalar."""
    return json.dumps(str(value), ensure_ascii=True)


def indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else prefix for line in text.splitlines())


def render_env_value(name: str, value: str, spaces: int = 16) -> str:
    prefix = " " * spaces
    return f"{prefix}- name: {name}\n{prefix}  value: {q(value)}"


def render_manifest(args: argparse.Namespace) -> str:
    image_pull_secret = ""
    if args.image_pull_secret:
        image_pull_secret = f"""
          imagePullSecrets:
            - name: {args.image_pull_secret}
"""

    shm_size = ""
    if args.shm_size:
        shm_size = f"\n            sizeLimit: {q(args.shm_size)}"

    rdma_request = ""
    rdma_limit = ""
    ipc_lock = ""
    if args.rdma_resource:
        rdma_request = f"\n                  {args.rdma_resource}: {q(1)}"
        rdma_limit = f"\n                  {args.rdma_resource}: {q(1)}"
        ipc_lock = """
                capabilities:
                  add: ["IPC_LOCK"]"""

    env_lines = [
        render_env_value("LANG", "ko_KR.utf8"),
        render_env_value("LANGUAGE", "ko_KR:ko"),
        render_env_value("LC_ALL", "ko_KR.utf8"),
        render_env_value("LC_CTYPE", "ko_KR.utf8"),
        render_env_value("TERM", "xterm-256color"),
        render_env_value("PROJECT_ROOT", args.project_root),
        render_env_value("HOME", args.home),
        render_env_value(
            "PATH",
            "/mnt/nuplan/miniforge/envs/catk/bin:/mnt/nuplan/miniforge/bin:"
            "/mnt/nuplan/tools:/mnt/nuplan/home/.local/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        ),
        render_env_value("CATK_REPO_URL", args.repo_url),
        render_env_value("CATK_BRANCH", args.branch),
        render_env_value("CACHE_ROOT", args.cache_root),
        render_env_value("PRETRAIN_CKPT", args.pretrain_ckpt),
        render_env_value("TASK_NAME", args.task_name),
        render_env_value("CATK_LR", args.learning_rate),
        render_env_value("NONFINITE_FM_LOSS_POLICY", args.nonfinite_fm_loss_policy),
        render_env_value("CATK_INSTALL_REQUIREMENTS", args.install_requirements),
        render_env_value("NUBES_GATEWAY_ADDRESS", args.nubes_gateway),
        render_env_value("WANDB_ENTITY", args.wandb_entity),
        render_env_value("WANDB_PROJECT", args.wandb_project),
        render_env_value("WANDB_MODE", args.wandb_mode),
    ]
    optional_env = {
        "LOG_DIR": args.log_dir,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "SOFT_LIMIT_RATIO": args.soft_limit_ratio,
        "TOPK_VIOLATION_K": args.topk_violation_k,
        "BACKPROP_LAST_K": args.backprop_last_k,
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "ACCUMULATE_GRAD_BATCHES": args.accumulate_grad_batches,
        "CATK_HYDRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional_env.items():
        if value not in (None, ""):
            env_lines.append(render_env_value(name, str(value)))

    if args.wandb_secret_name:
        env_lines.append(
            "                - name: WANDB_API_KEY\n"
            "                  valueFrom:\n"
            "                    secretKeyRef:\n"
            f"                      name: {args.wandb_secret_name}\n"
            f"                      key: {args.wandb_secret_key}"
        )
    env_block = "\n".join(env_lines)

    setup_miniforge = r"""set -Eeuo pipefail
umask 002

MINIFORGE_DIR=/mnt/nuplan/miniforge
if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "[init] miniforge already installed - skip"
else
  echo "[init] installing Miniforge..."
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
  rm -f /tmp/miniforge.sh
fi

export PATH="$MINIFORGE_DIR/bin:$PATH"
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate base
if conda env list | awk '{print $1}' | grep -qx catk; then
  echo "[init] catk env already exists - skip"
else
  echo "[init] creating catk conda environment (python 3.11)..."
  conda create -y -n catk python=3.11
fi
conda activate catk
python -m pip install --no-cache-dir wandb || echo "[init] wandb install failed - continuing"
conda info --envs
"""

    worker_command = r"""set -Eeuo pipefail
umask 002

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nuplan/projects/catk}"
export HOME="${HOME:-/mnt/nuplan/home}"
export CATK_BRANCH="${CATK_BRANCH:-semi_continuous_track_loss}"
export CATK_REPO_URL="${CATK_REPO_URL:-https://github.com/seulbinHwang/catk.git}"
export PATH="/mnt/nuplan/miniforge/envs/catk/bin:/mnt/nuplan/miniforge/bin:/mnt/nuplan/tools:/mnt/nuplan/home/.local/bin:$PATH"

mkdir -p "$HOME" "$(dirname "$PROJECT_ROOT")"
if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  echo "[worker] cloning CAT-K branch $CATK_BRANCH"
  rm -rf "$PROJECT_ROOT"
  git clone --branch "$CATK_BRANCH" "$CATK_REPO_URL" "$PROJECT_ROOT"
fi

cd "$PROJECT_ROOT"
git config --global --add safe.directory "$PROJECT_ROOT" || true
git fetch origin "$CATK_BRANCH"
git checkout -B "$CATK_BRANCH" "origin/$CATK_BRANCH"

bash scripts/mlx_finetune_draft_flow_v100x8_multinode.sh
"""

    return f"""apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {args.job_name}
  namespace: {args.namespace}
spec:
  elasticPolicy:
    rdzvId: {args.job_name}
    rdzvBackend: c10d
    minReplicas: {args.workers}
    maxReplicas: {args.workers}
    nProcPerNode: {args.gpus_per_worker}
  runPolicy:
    cleanPodPolicy: {args.clean_pod_policy}
    ttlSecondsAfterFinished: {args.ttl_seconds_after_finished}
  pytorchReplicaSpecs:
    Worker:
      replicas: {args.workers}
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          terminationGracePeriodSeconds: 120
          restartPolicy: OnFailure
          nodeSelector:
            mlx.navercorp.com/zone: {args.zone}
{image_pull_secret}          securityContext:
            fsGroup: 1000
          initContainers:
            - name: setup-miniforge
              image: {args.image}
              imagePullPolicy: Always
              securityContext:
                allowPrivilegeEscalation: false
              resources:
                requests:
                  cpu: "4"
                  memory: "16Gi"
                limits:
                  cpu: "4"
                  memory: "16Gi"
              command: ["/bin/bash", "-lc"]
              args:
                - |
{indent(setup_miniforge, 18)}
              volumeMounts:
                - name: nuplan-storage
                  mountPath: /mnt/nuplan
          containers:
            - name: pytorch
              image: {args.image}
              imagePullPolicy: Always
              securityContext:
                allowPrivilegeEscalation: false{ipc_lock}
              env:
{env_block}
              command: ["/bin/bash", "-lc"]
              args:
                - |
{indent(worker_command, 18)}
              resources:
                requests:
                  memory: {q(args.memory_request)}
                  cpu: {q(args.cpu_request)}
                  nvidia.com/gpu: {q(args.gpus_per_worker)}{rdma_request}
                limits:
                  memory: {q(args.memory_limit)}
                  cpu: {q(args.cpu_limit)}
                  nvidia.com/gpu: {q(args.gpus_per_worker)}{rdma_limit}
              volumeMounts:
                - name: nuplan-storage
                  mountPath: /mnt/nuplan
                - name: dshm2
                  mountPath: /dev/shm
          volumes:
            - name: nuplan-storage
              emptyDir: {{}}
            - name: dshm2
              emptyDir:
                medium: Memory{shm_size}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an MLX PyTorchJob for CAT-K V100x8 multi-node fine-tuning.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Worker pod count N.")
    parser.add_argument("--gpus-per-worker", type=int, default=8)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--namespace", default="p-pnc")
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--image-pull-secret", default="pnc-secret")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--project-root", default="/mnt/nuplan/projects/catk")
    parser.add_argument("--home", default="/mnt/nuplan/home")
    parser.add_argument("--cache-root", default="/workspace/womd_v1_3/SMART_cache")
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--learning-rate",
        default="auto",
        help="Use 'auto' for 2e-4 * workers, or pass a concrete value such as 4e-4.",
    )
    parser.add_argument(
        "--nonfinite-fm-loss-policy",
        default="skip",
        choices=("raise", "skip"),
        help="Use 'skip' to zero-gradient only the rank/batch whose FM loss becomes non-finite.",
    )
    parser.add_argument("--install-requirements", default="auto", choices=("auto", "0", "1"))
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--soft-limit-ratio", default="")
    parser.add_argument("--topk-violation-k", default="")
    parser.add_argument("--backprop-last-k", default="")
    parser.add_argument("--train-batch-size", default="")
    parser.add_argument("--accumulate-grad-batches", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--log-dir", default="/mnt/nuplan/projects/catk/logs")
    parser.add_argument("--nubes-gateway", default="c.nubes.sto.navercorp.com:8000")
    parser.add_argument("--wandb-secret-name", default="wandb-secret")
    parser.add_argument("--wandb-secret-key", default="api-key")
    parser.add_argument("--wandb-entity", default="jksg01019-naver-labs")
    parser.add_argument("--wandb-project", default="SMART-FLOW")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--cpu-request", default="32")
    parser.add_argument("--cpu-limit", default="32")
    parser.add_argument("--memory-request", default="128Gi")
    parser.add_argument("--memory-limit", default="480Gi")
    parser.add_argument(
        "--shm-size",
        default="",
        help="Optional /dev/shm emptyDir sizeLimit. Empty means no explicit sizeLimit.",
    )
    parser.add_argument(
        "--rdma-resource",
        default="",
        help="Optional InfiniBand resource name, e.g. rdma/hca_shared_devices_a.",
    )
    parser.add_argument("--clean-pod-policy", default="None")
    parser.add_argument("--ttl-seconds-after-finished", type=int, default=1814400)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Run kubectl apply on the rendered manifest.")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.gpus_per_worker < 1:
        parser.error("--gpus-per-worker must be >= 1")
    if args.job_name is None:
        args.job_name = f"catk-draft-v100x8x{args.workers}"
    if args.task_name is None:
        args.task_name = args.job_name.replace("-", "_")
    return args


def main() -> None:
    args = parse_args()
    manifest = render_manifest(args)

    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(manifest, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(manifest)

    if args.apply:
        if output is None:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
                tmp.write(manifest)
                output = Path(tmp.name)
        subprocess.run(["kubectl", "apply", "-f", str(output)], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
