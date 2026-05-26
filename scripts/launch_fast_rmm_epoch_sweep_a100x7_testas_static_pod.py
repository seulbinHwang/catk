#!/usr/bin/env python3
"""Launch a Fast-RMM checkpoint sweep on one existing A100x7 pod.

The target pod is ``testas`` by default. The launcher never creates,
deletes, or restarts pods. It only starts/stops a dedicated tmux session inside
the existing pod and evaluates W&B ``epoch-last`` artifact versions with
closed-loop Fast-RMM validation.

Use this after the matching pretrain run has finished. Do not run it while the
training tmux session is still using the same GPUs.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "testas"
DEFAULT_BRANCH = "semi_control_rolling_fd"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4x2_execctx_balanced"
DEFAULT_EPOCH_METADATA_VALUES = "57-64"
DEFAULT_SESSION = "fast-rmm-epoch-sweep-a100x7-testas"
DEFAULT_SWEEP_NAME = "fast_rmm_epoch_sweep_a100x7_testas"
DEFAULT_WANDB_GROUP = "fast_rmm_epoch_sweep_a100x7_testas_rmm_only_bs16"
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29880"
DEFAULT_NPROC_PER_NODE = 7
DEFAULT_VAL_BATCH_SIZE = 16
DEFAULT_SCORER_SCENE_NUM = 1680
DEFAULT_N_ROLLOUT_CLOSED_VAL = 32


@dataclass(frozen=True)
class SweepEpoch:
    epoch: int
    artifact_version: str
    metadata_epoch: int | None = None


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


def parse_epoch_versions(value: str) -> list[SweepEpoch]:
    epochs: list[SweepEpoch] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            epoch_text, version = item.split("=", 1)
        elif ":" in item:
            epoch_text, version = item.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                "--epoch-versions entries must look like 56:v52 or 56=v52"
            )
        try:
            epoch = int(epoch_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid epoch: {epoch_text!r}") from exc
        version = version.strip()
        if not version:
            raise argparse.ArgumentTypeError(f"missing artifact version for epoch {epoch}")
        if not version.startswith("v"):
            version = "v" + version
        epochs.append(SweepEpoch(epoch=epoch, artifact_version=version))
    if not epochs:
        raise argparse.ArgumentTypeError("--epoch-versions must not be empty")
    return epochs


def parse_epoch_metadata_values(value: str) -> list[int]:
    epochs: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid epoch range: {item!r}"
                ) from exc
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending epoch range: {item!r}")
            epochs.extend(range(start, end + 1))
        else:
            try:
                epochs.append(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid epoch: {item!r}") from exc
    if not epochs:
        raise argparse.ArgumentTypeError("--epoch-metadata-values must not be empty")
    if any(epoch < 1 for epoch in epochs):
        raise argparse.ArgumentTypeError("metadata epochs must be >= 1")
    return epochs


def artifact_version_number(version: str) -> int:
    text = str(version).strip()
    if text.startswith("v"):
        text = text[1:]
    try:
        return int(text)
    except ValueError:
        return -1


def resolve_epoch_versions_from_wandb(
    *, artifact_prefix: str, metadata_epochs: Sequence[int]
) -> list[SweepEpoch]:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required to auto-resolve artifact versions. "
            "Install wandb or pass --epoch-versions explicitly."
        ) from exc

    collection_name = artifact_prefix.split(":", 1)[0]
    api = wandb.Api()
    try:
        collection = api.artifact_collection(type_name="model", name=collection_name)
        artifacts = list(collection.artifacts())
    except Exception as exc:
        raise SystemExit(
            f"failed to query W&B artifact collection: {collection_name}"
        ) from exc

    by_epoch: dict[int, object] = {}
    for artifact in artifacts:
        metadata = dict(getattr(artifact, "metadata", None) or {})
        raw_epoch = metadata.get("epoch")
        if raw_epoch is None:
            continue
        try:
            metadata_epoch = int(raw_epoch)
        except (TypeError, ValueError):
            continue
        previous = by_epoch.get(metadata_epoch)
        if previous is None or artifact_version_number(artifact.version) > artifact_version_number(
            previous.version
        ):
            by_epoch[metadata_epoch] = artifact

    missing = [epoch for epoch in metadata_epochs if epoch not in by_epoch]
    if missing:
        available = sorted(by_epoch)
        raise SystemExit(
            "missing W&B epoch-last artifact versions for metadata epoch(s) "
            f"{missing}. Available metadata epochs for {collection_name}: {available}. "
            "If the pretrain is still running, wait until those epochs are uploaded."
        )

    return [
        SweepEpoch(
            epoch=metadata_epoch - 1,
            metadata_epoch=metadata_epoch,
            artifact_version=by_epoch[metadata_epoch].version,
        )
        for metadata_epoch in metadata_epochs
    ]


def compute_limit_val_batches(
    *, scorer_scene_num: int, world_size: int, val_batch_size: int
) -> int:
    per_rank_scenes = math.ceil(scorer_scene_num / world_size)
    return max(1, math.ceil(per_rank_scenes / val_batch_size))


def render_epoch_artifact_map(epochs: list[SweepEpoch], artifact_prefix: str) -> str:
    return "\n".join(
        f"{item.epoch}={artifact_prefix}:{item.artifact_version}" for item in epochs
    )


def render_worker_script(
    *,
    args: argparse.Namespace,
    epoch_artifacts: str,
    limit_val_batches: int,
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    ckpt_dir = f"{run_root}/ckpts"
    run_log_dir = f"{run_root}/run_logs_rmm_only_bs{args.val_batch_size}"
    status_file = f"{run_root}/{args.pod}.status"
    epoch_csv = ",".join(str(item.epoch) for item in args.epoch_versions)
    tags = (
        f"[fast_rmm,epoch_sweep,a100x7,{args.branch},"
        f"{args.sweep_name},rmm_only,bs{args.val_batch_size}]"
    )
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail

export TERM="${{TERM:-xterm-256color}}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
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
export WANDB_ENTITY="${{WANDB_ENTITY:-jksg01019-naver-labs}}"
export WANDB_PROJECT="${{WANDB_PROJECT:-SMART-FLOW}}"

PROJECT_ROOT={shq(args.project_root)}
CACHE_ROOT={shq(args.cache_root)}
RUN_ROOT={shq(run_root)}
CKPT_DIR={shq(ckpt_dir)}
RUN_LOG_DIR={shq(run_log_dir)}
STATUS_FILE={shq(status_file)}
EXPERIMENT={shq(args.experiment)}
SWEEP_NAME={shq(args.sweep_name)}
WANDB_GROUP={shq(args.wandb_group)}
MASTER_ADDR={shq(args.master_addr)}
MASTER_PORT={shq(args.master_port)}
NPROC_PER_NODE={int(args.nproc_per_node)}
VAL_BATCH_SIZE={int(args.val_batch_size)}
LIMIT_VAL_BATCHES={int(limit_val_batches)}
N_ROLLOUT_CLOSED_VAL={int(args.n_rollout_closed_val)}
SCORER_SCENE_NUM={int(args.scorer_scene_num)}

mkdir -p "$CKPT_DIR" "$RUN_LOG_DIR" "$(dirname "$STATUS_FILE")"
touch "$STATUS_FILE"

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T')] sweep start pod={args.pod} epochs={epoch_csv} val_batch_size=$VAL_BATCH_SIZE" | tee -a "$STATUS_FILE"
echo "branch={args.branch} commit=$(git rev-parse --short HEAD)" | tee -a "$STATUS_FILE"

artifact_for_epoch() {{
  local epoch="$1"
  awk -F= -v epoch="$epoch" '$1 == epoch {{ print $2; exit }}' <<'CATK_EPOCH_ARTIFACTS'
{epoch_artifacts}
CATK_EPOCH_ARTIFACTS
}}

download_ckpt_if_needed() {{
  local epoch="$1"
  local artifact="$2"
  local padded
  padded="$(printf '%03d' "$epoch")"
  local dst="$CKPT_DIR/epoch_${{padded}}.ckpt"
  local lock_dir="$dst.lock"
  local waited=0

  if [[ -s "$dst" ]]; then
    echo "$dst"
    return 0
  fi

  if mkdir "$lock_dir" 2>/dev/null; then
    trap 'rm -rf "$lock_dir"' RETURN
    echo "[$(date '+%F %T')] downloading epoch=$epoch artifact=$artifact" | tee -a "$STATUS_FILE" >&2
    CATK_ARTIFACT="$artifact" CATK_DST="$dst" CATK_DOWNLOAD_ROOT="$CKPT_DIR/artifacts/epoch_${{padded}}" python - <<'PY' >&2
import glob
import os
import shutil
from pathlib import Path

import wandb

artifact_name = os.environ["CATK_ARTIFACT"]
dst = Path(os.environ["CATK_DST"])
download_root = Path(os.environ["CATK_DOWNLOAD_ROOT"])
download_root.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact_dir = Path(artifact.download(root=str(download_root)))
candidates = []
preferred = artifact_dir / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred)
candidates.extend(Path(p) for p in glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(Path(p) for p in glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True))
seen = set()
unique = []
for path in candidates:
    resolved = path.resolve()
    if resolved in seen or not path.is_file():
        continue
    seen.add(resolved)
    unique.append(path)
if not unique:
    raise SystemExit(
        "no checkpoint file found in artifact "
        + artifact_name
        + " under "
        + str(artifact_dir)
    )
tmp = dst.with_suffix(dst.suffix + ".tmp")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(unique[0], tmp)
tmp.replace(dst)
print("[epoch-sweep] wrote " + str(dst) + " from " + artifact_name)
PY
    rm -rf "$lock_dir"
    trap - RETURN
  else
    while [[ ! -s "$dst" ]]; do
      if (( waited >= 7200 )); then
        echo "timed out waiting for checkpoint: $dst" >&2
        return 1
      fi
      sleep 5
      waited=$(( waited + 5 ))
    done
  fi

  echo "$dst"
}}

run_one_epoch() {{
  local epoch="$1"
  local artifact
  artifact="$(artifact_for_epoch "$epoch")"
  if [[ -z "$artifact" ]]; then
    echo "missing artifact mapping for epoch $epoch" >&2
    return 2
  fi
  local ckpt
  ckpt="$(download_ckpt_if_needed "$epoch" "$artifact")"
  local padded
  padded="$(printf '%03d' "$epoch")"
  local task_name="${{SWEEP_NAME}}_epoch_${{padded}}_rmm_only_bs${{VAL_BATCH_SIZE}}"
  local task_dir="$RUN_LOG_DIR/$task_name"
  mkdir -p "$task_dir"

  echo "[$(date '+%F %T')] START epoch=$epoch task=$task_name ckpt=$ckpt port=$MASTER_PORT" | tee -a "$STATUS_FILE"

  export CACHE_ROOT
  export NNODES=1
  export NPROC_PER_NODE
  export TRAINER_DEVICES="$NPROC_PER_NODE"
  export NODE_RANK=0
  export MASTER_ADDR
  export MASTER_PORT
  export MANUAL_RANK_OFFSET=0
  export MANUAL_WORLD_SIZE="$NPROC_PER_NODE"
  export CATK_EXPERIMENT="$EXPERIMENT"
  export CATK_ACTION=validate
  export CATK_CKPT_PATH="$ckpt"
  export TASK_NAME="$task_name"
  export LOG_DIR="$task_dir"
  export VAL_BATCH_SIZE
  export LIMIT_VAL_BATCHES
  export CATK_HYDRA_OVERRIDES="trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment model.model_config.val_closed_loop=true model.model_config.val_open_loop=false model.model_config.n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} model.model_config.scorer_scene_num=${{SCORER_SCENE_NUM}} logger.wandb.group=${{WANDB_GROUP}} logger.wandb.job_type=fast_rmm_epoch_sweep logger.wandb.tags={tags} logger.wandb.log_model=false"

  set +e
  bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee "$task_dir/{args.pod}.log"
  local status="${{PIPESTATUS[0]}}"
  set -e
  if [[ "$status" != "0" ]]; then
    echo "[$(date '+%F %T')] FAIL epoch=$epoch status=$status" | tee -a "$STATUS_FILE"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE epoch=$epoch" | tee -a "$STATUS_FILE"
}}

IFS=',' read -r -a epochs <<< {shq(epoch_csv)}
for epoch in "${{epochs[@]}}"; do
  run_one_epoch "$epoch"
done

echo "[$(date '+%F %T')] sweep complete pod={args.pod}" | tee -a "$STATUS_FILE"
"""


def remote_git_prepare_script(args: argparse.Namespace) -> str:
    fetch_refspec = f"+{args.branch}:refs/remotes/origin/{args.branch}"
    if args.git_ref:
        return f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune {shq(fetch_refspec)}
git checkout -f {shq(args.git_ref)}
"""
    if args.no_pull:
        return f"git config --global --add safe.directory {shq(args.project_root)} || true"
    return f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq('refs/heads/' + args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq('origin/' + args.branch)}
fi
git pull --ff-only origin {shq(args.branch)}
"""


def render_start_command(
    *, args: argparse.Namespace, epoch_artifacts: str, limit_val_batches: int
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    script_path = f"{run_root}/{args.pod}_run_epoch_sweep.sh"
    worker = render_worker_script(
        args=args,
        epoch_artifacts=epoch_artifacts,
        limit_val_batches=limit_val_batches,
    )
    replace_block = (
        f"tmux kill-session -t {shq(args.session)} 2>/dev/null || true"
        if args.replace
        else f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  exit 3
fi
"""
    )
    return f"""set -Eeuo pipefail
if [[ ! -d {shq(args.project_root)}/.git ]]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{remote_git_prepare_script(args)}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(script_path)} <<'CATK_FAST_RMM_SWEEP'
{worker.rstrip()}
CATK_FAST_RMM_SWEEP
chmod +x {shq(script_path)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(script_path)}
echo "[launcher] started {args.session} on {args.pod}"
echo "[launcher] script: {script_path}"
"""


def render_stop_command(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
tmux kill-session -t {shq(args.session)} 2>/dev/null || true
mapfile -t pids < <(
  pgrep -f {shq(args.sweep_name)} 2>/dev/null | while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
      echo "$pid"
    fi
  done
)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating sweep processes: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(
    pgrep -f {shq(args.sweep_name)} 2>/dev/null | while read -r pid; do
      if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
        echo "$pid"
      fi
    done
  )
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing sweep processes: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
echo "[launcher] stopped session/processes for {args.sweep_name}"
"""


def exec_in_pod(args: argparse.Namespace, script: str) -> None:
    command = [
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
    ]
    run_kubectl(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate W&B epoch-last checkpoint artifact versions with "
            "closed-loop Fast RMM on one existing A100x7 pod."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--pod", default=os.environ.get("POD", DEFAULT_POD))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--artifact-prefix",
        default=os.environ.get("WANDB_EPOCH_LAST_ARTIFACT_PREFIX", ""),
        help=(
            "W&B artifact prefix, e.g. "
            "jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id>."
        ),
    )
    parser.add_argument(
        "--epoch-versions",
        type=parse_epoch_versions,
        default=None,
        help=(
            "Explicit comma-separated mapping like 56:v52,57:v53. "
            "When omitted, versions are resolved from W&B artifact metadata."
        ),
    )
    parser.add_argument(
        "--epoch-metadata-values",
        type=parse_epoch_metadata_values,
        default=parse_epoch_metadata_values(DEFAULT_EPOCH_METADATA_VALUES),
        help=(
            "Completed epoch numbers stored in W&B artifact metadata. "
            "Default 57-64 maps to zero-based labels 56-63."
        ),
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--master-addr", default=DEFAULT_MASTER_ADDR)
    parser.add_argument("--master-port", default=DEFAULT_MASTER_PORT)
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--limit-val-batches", default="auto")
    parser.add_argument("--scorer-scene-num", type=int, default=DEFAULT_SCORER_SCENE_NUM)
    parser.add_argument("--n-rollout-closed-val", type=int, default=DEFAULT_N_ROLLOUT_CLOSED_VAL)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.artifact_prefix and not args.stop:
        parser.error("--artifact-prefix is required unless --stop is set")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.limit_val_batches != "auto":
        try:
            parsed_limit = int(args.limit_val_batches)
        except ValueError as exc:
            raise SystemExit("--limit-val-batches must be 'auto' or a positive integer") from exc
        if parsed_limit < 1:
            parser.error("--limit-val-batches must be >= 1")
        args.limit_val_batches = parsed_limit
    if args.scorer_scene_num < 1:
        parser.error("--scorer-scene-num must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.epoch_versions is None and not args.stop:
        args.epoch_versions = resolve_epoch_versions_from_wandb(
            artifact_prefix=args.artifact_prefix,
            metadata_epochs=args.epoch_metadata_values,
        )
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args))
        return

    limit_val_batches = (
        compute_limit_val_batches(
            scorer_scene_num=args.scorer_scene_num,
            world_size=args.nproc_per_node,
            val_batch_size=args.val_batch_size,
        )
        if args.limit_val_batches == "auto"
        else int(args.limit_val_batches)
    )
    epoch_artifacts = render_epoch_artifact_map(args.epoch_versions, args.artifact_prefix)

    print(f"[launcher] pod: {args.pod}")
    print(f"[launcher] world_size: {args.nproc_per_node}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")
    print(f"[launcher] limit_val_batches: {limit_val_batches}")
    print(f"[launcher] scorer scenes: {args.val_batch_size * args.nproc_per_node * limit_val_batches}")
    print(f"[launcher] sweep: {args.sweep_name}")
    print(f"[launcher] wandb group: {args.wandb_group}")
    print("[launcher] epoch artifacts:")
    for item, line in zip(args.epoch_versions, epoch_artifacts.splitlines()):
        metadata_text = (
            f" metadata_epoch={item.metadata_epoch}" if item.metadata_epoch is not None else ""
        )
        print(f"  {line}{metadata_text}")

    exec_in_pod(
        args,
        render_start_command(
            args=args,
            epoch_artifacts=epoch_artifacts,
            limit_val_batches=limit_val_batches,
        ),
    )
    print("\nAttach command:")
    print(
        "  kubectl exec -it "
        f"-n {args.namespace} {args.pod} -c {args.container} -- "
        f"tmux attach -t {args.session}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
