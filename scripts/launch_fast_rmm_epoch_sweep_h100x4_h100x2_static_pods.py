#!/usr/bin/env python3
"""Launch a fast-RMM checkpoint sweep on existing H100 4+2 pods.

This launcher is intentionally conservative:

* it never creates, deletes, or restarts pods;
* it starts/stops only one tmux session inside the target pods;
* it evaluates closed-loop Fast WOSAC/RMM only, skipping open-loop validation;
* it downloads W&B ``epoch-last`` artifact versions to each pod before running.

Default values match the post-training sweep for run ``x5f9g0ce``:
epochs 56..63 from ``epoch-last-x5f9g0ce`` artifacts, val batch size 48, and
six validation batches, i.e. 48 * 6 ranks * 6 batches = 1728 scenes.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_PODS = ("hsb-npc-training", "wo-pvc-2")
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip"
DEFAULT_ARTIFACT_PREFIX = "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce"
DEFAULT_EPOCH_VERSIONS = (
    "56:v52,57:v53,58:v54,59:v55,60:v56,61:v57,62:v58,63:v60"
)
DEFAULT_SESSION = "fast-rmm-epoch-sweep-h100x4-h100x2"
DEFAULT_SWEEP_NAME = "fast_rmm_epoch_sweep_x5f9g0ce"
DEFAULT_WANDB_GROUP = "fast_rmm_epoch_sweep_x5f9g0ce_rmm_only_bs48"
DEFAULT_MASTER_PORT = "29860"
DEFAULT_VAL_BATCH_SIZE = 48
DEFAULT_SCORER_SCENE_NUM = 1680
DEFAULT_N_ROLLOUT_CLOSED_VAL = 32


@dataclass(frozen=True)
class PodLayout:
    pod: str
    node_rank: int
    local_world_size: int
    rank_offset: int


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


def pod_ip(namespace: str, pod: str, *, dry_run: bool) -> str:
    if dry_run:
        return "<MASTER_POD_IP>"
    return run_kubectl(
        [
            "get",
            "pod",
            pod,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.podIP}",
        ],
        capture=True,
    )


def pod_gpu_count(namespace: str, container: str, pod: str, *, dry_run: bool) -> int:
    if dry_run:
        return 4 if pod == DEFAULT_PODS[0] else 2
    output = run_kubectl(
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
            "nvidia-smi -L 2>/dev/null | wc -l",
        ],
        capture=True,
    )
    count = int(output.strip())
    if count < 1:
        raise RuntimeError(f"no GPUs found in pod {pod}")
    return count


def parse_epoch_versions(value: str) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
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
        pairs.append((epoch, version))
    if not pairs:
        raise argparse.ArgumentTypeError("--epoch-versions must not be empty")
    return pairs


def compute_limit_val_batches(
    *, scorer_scene_num: int, world_size: int, val_batch_size: int
) -> int:
    per_rank_scenes = math.ceil(scorer_scene_num / world_size)
    return max(1, math.ceil(per_rank_scenes / val_batch_size))


def render_epoch_artifact_map(pairs: list[tuple[int, str]], artifact_prefix: str) -> str:
    lines = []
    for epoch, version in pairs:
        lines.append(f"{epoch}={artifact_prefix}:{version}")
    return "\n".join(lines)


def render_worker_script(
    *,
    args: argparse.Namespace,
    layout: PodLayout,
    master_addr: str,
    world_size: int,
    epoch_artifacts: str,
    limit_val_batches: int,
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    ckpt_dir = f"{run_root}/ckpts"
    run_log_dir = f"{run_root}/run_logs_rmm_only_bs{args.val_batch_size}"
    status_file = f"{run_root}/{layout.pod}.status"
    tags = (
        f"[fast_rmm,epoch_sweep,h100x6,{args.branch},"
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
MASTER_ADDR={shq(master_addr)}
MASTER_PORT={shq(args.master_port)}
NODE_RANK={layout.node_rank}
NPROC_PER_NODE={layout.local_world_size}
MANUAL_RANK_OFFSET={layout.rank_offset}
MANUAL_WORLD_SIZE={world_size}
VAL_BATCH_SIZE={int(args.val_batch_size)}
LIMIT_VAL_BATCHES={int(limit_val_batches)}
N_ROLLOUT_CLOSED_VAL={int(args.n_rollout_closed_val)}
SCORER_SCENE_NUM={int(args.scorer_scene_num)}

mkdir -p "$CKPT_DIR" "$RUN_LOG_DIR" "$(dirname "$STATUS_FILE")"
touch "$STATUS_FILE"

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T')] sweep start pod={layout.pod} epochs={','.join(str(e) for e, _ in args.epoch_versions)} val_batch_size=$VAL_BATCH_SIZE" | tee -a "$STATUS_FILE"
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
  export NNODES=2
  export NPROC_PER_NODE
  export TRAINER_DEVICES="$NPROC_PER_NODE"
  export NODE_RANK
  export MASTER_ADDR
  export MASTER_PORT
  export MANUAL_RANK_OFFSET
  export MANUAL_WORLD_SIZE
  export CATK_EXPERIMENT="$EXPERIMENT"
  export CATK_ACTION=validate
  export CATK_CKPT_PATH="$ckpt"
  export TASK_NAME="$task_name"
  export LOG_DIR="$task_dir"
  export VAL_BATCH_SIZE
  export LIMIT_VAL_BATCHES
  export CATK_HYDRA_OVERRIDES="trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment model.model_config.val_closed_loop=true model.model_config.val_open_loop=false model.model_config.n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} model.model_config.scorer_scene_num=${{SCORER_SCENE_NUM}} logger.wandb.group=${{WANDB_GROUP}} logger.wandb.job_type=fast_rmm_epoch_sweep logger.wandb.tags={tags} logger.wandb.log_model=false"

  set +e
  bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee "$task_dir/{layout.pod}.log"
  local status="${{PIPESTATUS[0]}}"
  set -e
  if [[ "$status" != "0" ]]; then
    echo "[$(date '+%F %T')] FAIL epoch=$epoch status=$status" | tee -a "$STATUS_FILE"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE epoch=$epoch" | tee -a "$STATUS_FILE"
}}

IFS=',' read -r -a epochs <<< {shq(','.join(str(e) for e, _ in args.epoch_versions))}
for epoch in "${{epochs[@]}}"; do
  run_one_epoch "$epoch"
done

echo "[$(date '+%F %T')] sweep complete pod={layout.pod}" | tee -a "$STATUS_FILE"
"""


def render_start_command(
    *,
    args: argparse.Namespace,
    layout: PodLayout,
    master_addr: str,
    world_size: int,
    epoch_artifacts: str,
    limit_val_batches: int,
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    script_path = f"{run_root}/{layout.pod}_run_epoch_sweep.sh"
    worker = render_worker_script(
        args=args,
        layout=layout,
        master_addr=master_addr,
        world_size=world_size,
        epoch_artifacts=epoch_artifacts,
        limit_val_batches=limit_val_batches,
    )
    pull_block = ""
    if args.git_ref:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune {shq(args.branch + ':refs/remotes/origin/' + args.branch)}
git checkout -f {shq(args.git_ref)}
"""
    elif args.pull:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(args.branch + ':refs/remotes/origin/' + args.branch)}
if git show-ref --verify --quiet {shq('refs/heads/' + args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq('origin/' + args.branch)}
fi
git pull --ff-only origin {shq(args.branch)}
"""
    replace_block = ""
    if args.replace:
        replace_block = f"tmux kill-session -t {shq(args.session)} 2>/dev/null || true"
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  exit 3
fi
"""
    return f"""set -Eeuo pipefail
if [[ ! -d {shq(args.project_root)}/.git ]]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(script_path)} <<'CATK_FAST_RMM_SWEEP'
{worker.rstrip()}
CATK_FAST_RMM_SWEEP
chmod +x {shq(script_path)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(script_path)}
echo "[launcher] started {args.session} on {layout.pod}"
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


def exec_in_pod(
    *,
    namespace: str,
    container: str,
    pod: str,
    script: str,
    dry_run: bool,
) -> None:
    command = [
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
    ]
    run_kubectl(command, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple W&B epoch-last checkpoint artifact versions with "
            "closed-loop Fast RMM on existing H100x4+H100x2 pods."
        )
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--pods", nargs="+", default=list(DEFAULT_PODS))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--git-ref", default=os.environ.get("CATK_GIT_REF", ""))
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--remote-log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_REMOTE_LOG_DIR))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument(
        "--epoch-versions",
        type=parse_epoch_versions,
        default=parse_epoch_versions(DEFAULT_EPOCH_VERSIONS),
        help=(
            "Comma-separated mapping like 56:v52,57:v53. Versions are appended "
            "to --artifact-prefix."
        ),
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--master-port", default=DEFAULT_MASTER_PORT)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--limit-val-batches", default="auto")
    parser.add_argument("--scorer-scene-num", type=int, default=DEFAULT_SCORER_SCENE_NUM)
    parser.add_argument("--n-rollout-closed-val", type=int, default=DEFAULT_N_ROLLOUT_CLOSED_VAL)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.pods) != 2 and not args.stop:
        parser.error("this preset expects exactly two pods")
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
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        for pod in args.pods:
            exec_in_pod(
                namespace=args.namespace,
                container=args.container,
                pod=pod,
                script=render_stop_command(args),
                dry_run=args.dry_run,
            )
        return

    master_addr = pod_ip(args.namespace, args.pods[0], dry_run=args.dry_run)
    layouts: list[PodLayout] = []
    rank_offset = 0
    for node_rank, pod in enumerate(args.pods):
        local_world_size = pod_gpu_count(
            args.namespace, args.container, pod, dry_run=args.dry_run
        )
        layouts.append(
            PodLayout(
                pod=pod,
                node_rank=node_rank,
                local_world_size=local_world_size,
                rank_offset=rank_offset,
            )
        )
        rank_offset += local_world_size
    world_size = rank_offset
    limit_val_batches = (
        compute_limit_val_batches(
            scorer_scene_num=args.scorer_scene_num,
            world_size=world_size,
            val_batch_size=args.val_batch_size,
        )
        if args.limit_val_batches == "auto"
        else int(args.limit_val_batches)
    )
    epoch_artifacts = render_epoch_artifact_map(args.epoch_versions, args.artifact_prefix)

    print(f"[launcher] master: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] world_size: {world_size}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")
    print(f"[launcher] limit_val_batches: {limit_val_batches}")
    print(f"[launcher] scorer scenes: {args.val_batch_size * world_size * limit_val_batches}")
    print(f"[launcher] sweep: {args.sweep_name}")
    print(f"[launcher] wandb group: {args.wandb_group}")
    print("[launcher] epoch artifacts:")
    for line in epoch_artifacts.splitlines():
        print(f"  {line}")

    for layout in layouts:
        script = render_start_command(
            args=args,
            layout=layout,
            master_addr=master_addr,
            world_size=world_size,
            epoch_artifacts=epoch_artifacts,
            limit_val_batches=limit_val_batches,
        )
        exec_in_pod(
            namespace=args.namespace,
            container=args.container,
            pod=layout.pod,
            script=script,
            dry_run=args.dry_run,
        )

    print("\nAttach commands:")
    for pod in args.pods:
        print(
            "  kubectl exec -it "
            f"-n {args.namespace} {pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
