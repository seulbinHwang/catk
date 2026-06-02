#!/usr/bin/env python3
"""Launch a Fast-RMM midpoint flow sample-step sweep on existing testa/testaa A100 4+4 pods.

This launcher is intentionally conservative:

* it never creates, deletes, or restarts pods;
* it starts/stops only one tmux session inside the target pods;
* it evaluates closed-loop Fast WOSAC/RMM only, skipping open-loop validation;
* it downloads one fixed W&B ``epoch-last`` checkpoint artifact per pod;
* it fixes the Flow rollout solver method to ``midpoint``;
* it sweeps ``model.model_config.validation_rollout_sampling.sample_steps``
  while keeping the checkpoint and all other validation settings fixed.

Use this to measure how Fast-RMM changes as the number of Flow denoising steps
inside each closed-loop rollout changes under midpoint integration.
``n_rollout_closed_val`` stays fixed; it controls the number of rollout samples
per scene, not the denoising depth.
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
DEFAULT_PODS = ("testa", "testaa")
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_control_a100x4x2_prefix_default_noslip"
DEFAULT_ARTIFACT_PREFIX = "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce"
DEFAULT_EPOCH = 61
DEFAULT_ARTIFACT_VERSION = "v57"
DEFAULT_SAMPLE_STEPS = "2,4,6,8,10,12,16,20,24,28,32"
DEFAULT_FLOW_SOLVER_METHOD = "midpoint"
DEFAULT_SESSION = "fast-rmm-midpoint-sample-steps-sweep-a100x4x2-testa-testaa"
DEFAULT_SWEEP_NAME = "fast_rmm_midpoint_sample_steps_sweep_epoch061_x5f9g0ce_a100x4x2"
DEFAULT_WANDB_GROUP = "fast_rmm_midpoint_sample_steps_sweep_epoch061_x5f9g0ce_a100x4x2_bs42"
DEFAULT_MASTER_PORT = "29930"
DEFAULT_INTER_RUN_SLEEP_SEC = 15
DEFAULT_VAL_BATCH_SIZE = 42
DEFAULT_SCORER_SCENE_NUM = 1728
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
        return 4
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


def parse_sample_steps(value: str) -> list[int]:
    steps: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid sample step count: {item!r}") from exc
        if step < 1:
            raise argparse.ArgumentTypeError("sample step counts must be >= 1")
        steps.append(step)
    if not steps:
        raise argparse.ArgumentTypeError("--sample-steps must not be empty")
    return steps


def normalize_artifact_version(value: str) -> str:
    version = value.strip()
    if not version:
        raise argparse.ArgumentTypeError("artifact version must not be empty")
    return version if version.startswith("v") else "v" + version


def compute_limit_val_batches(
    *, scorer_scene_num: int, world_size: int, val_batch_size: int
) -> int:
    per_rank_scenes = math.ceil(scorer_scene_num / world_size)
    return max(1, math.ceil(per_rank_scenes / val_batch_size))


def render_epoch_artifact(artifact_prefix: str, artifact_version: str) -> str:
    return f"{artifact_prefix}:{artifact_version}"


def render_worker_script(
    *,
    args: argparse.Namespace,
    layout: PodLayout,
    master_addr: str,
    world_size: int,
    checkpoint_artifact: str,
    limit_val_batches: int,
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    ckpt_dir = f"{run_root}/ckpts"
    run_log_dir = f"{run_root}/run_logs_rmm_only_bs{args.val_batch_size}"
    status_file = f"{run_root}/{layout.pod}.status"
    tags = (
        f"[fast_rmm,midpoint_sample_steps_sweep,a100x4x2,{args.branch},"
        f"{args.sweep_name},epoch{args.epoch:03d},rmm_only,bs{args.val_batch_size}]"
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
BASE_MASTER_PORT={shq(args.master_port)}
INTER_RUN_SLEEP_SEC={int(args.inter_run_sleep_sec)}
NODE_RANK={layout.node_rank}
NPROC_PER_NODE={layout.local_world_size}
MANUAL_RANK_OFFSET={layout.rank_offset}
MANUAL_WORLD_SIZE={world_size}
VAL_BATCH_SIZE={int(args.val_batch_size)}
LIMIT_VAL_BATCHES={int(limit_val_batches)}
N_ROLLOUT_CLOSED_VAL={int(args.n_rollout_closed_val)}
SCORER_SCENE_NUM={int(args.scorer_scene_num)}
CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS="${{CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS:-50000}}"
CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS="${{CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS:-50000}}"
export CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS
FLOW_SOLVER_METHOD={shq(args.flow_solver_method)}

mkdir -p "$CKPT_DIR" "$RUN_LOG_DIR" "$(dirname "$STATUS_FILE")"
touch "$STATUS_FILE"

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T')] sweep start pod={layout.pod} epoch={args.epoch} solver=$FLOW_SOLVER_METHOD sample_steps={','.join(str(s) for s in args.sample_steps)} val_batch_size=$VAL_BATCH_SIZE n_rollout_closed_val=$N_ROLLOUT_CLOSED_VAL" | tee -a "$STATUS_FILE"
echo "branch={args.branch} commit=$(git rev-parse --short HEAD)" | tee -a "$STATUS_FILE"

CHECKPOINT_ARTIFACT={shq(checkpoint_artifact)}
CHECKPOINT_EPOCH={int(args.epoch)}

download_ckpt_if_needed() {{
  local epoch="$CHECKPOINT_EPOCH"
  local artifact="$CHECKPOINT_ARTIFACT"
  local dst="$CKPT_DIR/epoch_$(printf '%03d' "$epoch").ckpt"
  if [[ -f "$dst" ]]; then
    echo "$dst"
    return 0
  fi
  echo "[$(date '+%F %T')] downloading epoch=$epoch artifact=$artifact" | tee -a "$STATUS_FILE" >&2
  python - "$artifact" "$dst" <<'PY'
import sys
from pathlib import Path
import wandb

artifact_name, dst = sys.argv[1], Path(sys.argv[2])
run = wandb.init(project="SMART-FLOW", entity="jksg01019-naver-labs", job_type="download_epoch_last", mode="online")
artifact = run.use_artifact(artifact_name, type="model")
root = Path(artifact.download(root=str(dst.parent / (dst.stem + "_artifact"))))
candidates = list(root.rglob("*.ckpt"))
if not candidates:
    raise SystemExit(f"no .ckpt found in artifact {{artifact_name}}")
source = candidates[0]
dst.write_bytes(source.read_bytes())
run.finish()
PY

  echo "$dst"
}}

run_one_sample_steps() {{
  local sample_steps="$1"
  local sample_index="$2"
  local master_port="$((BASE_MASTER_PORT + sample_index))"
  local ckpt
  ckpt="$(download_ckpt_if_needed)"
  local padded_epoch padded_steps
  padded_epoch="$(printf '%03d' "$CHECKPOINT_EPOCH")"
  padded_steps="$(printf '%03d' "$sample_steps")"
  local task_name="${{SWEEP_NAME}}_epoch_${{padded_epoch}}_sample_steps_${{padded_steps}}_rmm_only_bs${{VAL_BATCH_SIZE}}"
  local task_dir="$RUN_LOG_DIR/$task_name"
  mkdir -p "$task_dir"

  echo "[$(date '+%F %T')] START epoch=$CHECKPOINT_EPOCH sample_steps=$sample_steps task=$task_name ckpt=$ckpt port=$master_port" | tee -a "$STATUS_FILE"

  export CACHE_ROOT
  export NNODES=2
  export NPROC_PER_NODE
  export TRAINER_DEVICES="$NPROC_PER_NODE"
  export NODE_RANK
  export MASTER_ADDR
  export MASTER_PORT="$master_port"
  export MANUAL_RANK_OFFSET
  export MANUAL_WORLD_SIZE
  export CATK_EXPERIMENT="$EXPERIMENT"
  export CATK_ACTION=validate
  export CATK_CKPT_PATH="$ckpt"
  export TASK_NAME="$task_name"
  export LOG_DIR="$task_dir"
  export VAL_BATCH_SIZE
  export LIMIT_VAL_BATCHES
  export N_ROLLOUT_CLOSED_VAL
  export SAMPLE_STEPS="$sample_steps"
  export CATK_HYDRA_OVERRIDES="++trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy ++trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment model.model_config.val_closed_loop=true model.model_config.val_open_loop=false model.model_config.n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} model.model_config.decoder.flow_solver_method=${{FLOW_SOLVER_METHOD}} model.model_config.validation_rollout_sampling.sample_method=${{FLOW_SOLVER_METHOD}} model.model_config.validation_rollout_sampling.sample_steps=${{SAMPLE_STEPS}} model.model_config.decoder.use_stop_motion=false model.model_config.self_forced.use_stop_motion=false model.model_config.scorer_scene_num=${{SCORER_SCENE_NUM}} logger.wandb.group=${{WANDB_GROUP}} logger.wandb.job_type=fast_rmm_midpoint_sample_steps_sweep logger.wandb.tags={tags} logger.wandb.log_model=false"

  set +e
  bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee "$task_dir/{layout.pod}.log"
  local status="${{PIPESTATUS[0]}}"
  set -e
  if [[ "$status" != "0" ]]; then
    echo "[$(date '+%F %T')] FAIL sample_steps=$sample_steps status=$status" | tee -a "$STATUS_FILE"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE sample_steps=$sample_steps" | tee -a "$STATUS_FILE"
  sleep "$INTER_RUN_SLEEP_SEC"
}}

IFS=',' read -r -a sample_steps_values <<< {shq(','.join(str(s) for s in args.sample_steps))}
for sample_index in "${{!sample_steps_values[@]}}"; do
  run_one_sample_steps "${{sample_steps_values[$sample_index]}}" "$sample_index"
done

echo "[$(date '+%F %T')] sweep complete pod={layout.pod}" | tee -a "$STATUS_FILE"
"""


def render_start_command(
    *,
    args: argparse.Namespace,
    layout: PodLayout,
    master_addr: str,
    world_size: int,
    checkpoint_artifact: str,
    limit_val_batches: int,
) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    script_path = f"{run_root}/{layout.pod}_run_midpoint_sample_steps_sweep.sh"
    worker = render_worker_script(
        args=args,
        layout=layout,
        master_addr=master_addr,
        world_size=world_size,
        checkpoint_artifact=checkpoint_artifact,
        limit_val_batches=limit_val_batches,
    )
    pull_block = ""
    if args.git_ref:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git update-ref -d {shq('refs/remotes/origin/' + args.branch)} || true
git fetch origin --prune {shq('+' + args.branch + ':refs/remotes/origin/' + args.branch)}
git checkout -f {shq(args.git_ref)}
"""
    elif args.pull:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git update-ref -d {shq('refs/remotes/origin/' + args.branch)} || true
git fetch origin --prune {shq('+' + args.branch + ':refs/remotes/origin/' + args.branch)}
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
            "Evaluate one fixed W&B epoch-last checkpoint artifact while sweeping "
            "model.model_config.validation_rollout_sampling.sample_steps with "
            "midpoint Flow rollout integration on "
            "existing testa/testaa A100x4x2 pods."
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
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument(
        "--artifact-version",
        type=normalize_artifact_version,
        default=normalize_artifact_version(DEFAULT_ARTIFACT_VERSION),
        help="W&B artifact version for the fixed checkpoint, e.g. v57.",
    )
    parser.add_argument(
        "--sample-steps",
        type=parse_sample_steps,
        default=parse_sample_steps(DEFAULT_SAMPLE_STEPS),
        help="Comma-separated Flow denoising sample_steps sweep, e.g. 2,4,8,16,32.",
    )
    parser.add_argument(
        "--flow-solver-method",
        default=DEFAULT_FLOW_SOLVER_METHOD,
        choices=("midpoint",),
        help=(
            "Fixed solver method for this launcher. The override is applied to both "
            "decoder.flow_solver_method and validation_rollout_sampling.sample_method."
        ),
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--master-port", default=DEFAULT_MASTER_PORT)
    parser.add_argument(
        "--inter-run-sleep-sec",
        type=int,
        default=int(os.environ.get("CATK_FAST_RMM_INTER_RUN_SLEEP_SEC", DEFAULT_INTER_RUN_SLEEP_SEC)),
        help=(
            "Seconds to wait after each sequential DDP validation. This gives NCCL/TCP "
            "rendezvous sockets time to settle before the next sample_steps run."
        ),
    )
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
    if args.epoch < 0:
        parser.error("--epoch must be >= 0")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.inter_run_sleep_sec < 0:
        parser.error("--inter-run-sleep-sec must be >= 0")
    try:
        int(args.master_port)
    except ValueError:
        parser.error("--master-port must be an integer because sequential sweep points use port offsets")
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
    checkpoint_artifact = render_epoch_artifact(args.artifact_prefix, args.artifact_version)

    print(f"[launcher] master: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(
        "[launcher] per-sample-step ports: "
        f"{args.master_port}..{int(args.master_port) + len(args.sample_steps) - 1}"
    )
    print(f"[launcher] world_size: {world_size}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")
    print(f"[launcher] limit_val_batches: {limit_val_batches}")
    print(f"[launcher] scorer scenes: {args.val_batch_size * world_size * limit_val_batches}")
    print(f"[launcher] sweep: {args.sweep_name}")
    print(f"[launcher] wandb group: {args.wandb_group}")
    print(f"[launcher] checkpoint: epoch {args.epoch} = {checkpoint_artifact}")
    print(f"[launcher] flow solver method: {args.flow_solver_method}")
    print(f"[launcher] sample steps: {','.join(str(s) for s in args.sample_steps)}")
    print(f"[launcher] n_rollout_closed_val: {args.n_rollout_closed_val}")

    for layout in layouts:
        script = render_start_command(
            args=args,
            layout=layout,
            master_addr=master_addr,
            world_size=world_size,
            checkpoint_artifact=checkpoint_artifact,
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
