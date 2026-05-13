#!/usr/bin/env python3
"""Launch Fast WOSAC validation for the g3zr84tp pose-space checkpoint on wo-pvc-800.

This launcher only starts or stops a tmux session inside the existing
``wo-pvc-800`` pod. It does not create, delete, or restart pods.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "wo-pvc-800"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_TASK_NAME = "flow_semi_continuous_pretrain_h100x4x2_bs26_g3zr84tp_fast_wosac_val1680"
DEFAULT_SESSION = "catk-fast-wosac-g3zr84tp-wo-pvc-800"
DEFAULT_WANDB_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64"
DEFAULT_CKPT_PATH = "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt"
DEFAULT_DOWNLOAD_DIR = "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/artifact"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_fast_wosac_val/{safe_task}"


def default_limit_val_batches(args: argparse.Namespace) -> int:
    per_rank_scenes = math.ceil(args.scorer_scene_num / args.nproc_per_node)
    return max(1, math.ceil(per_rank_scenes / args.val_batch_size))


def render_env(args: argparse.Namespace) -> str:
    limit_val_batches = (
        default_limit_val_batches(args)
        if args.limit_val_batches == "auto"
        else args.limit_val_batches
    )
    lines = [
        export_line("TASK_NAME", args.task_name),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("CKPT_PATH", args.ckpt_path),
        export_line("WANDB_ARTIFACT", args.wandb_artifact),
        export_line("WANDB_DOWNLOAD_DIR", args.download_dir),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("LIMIT_VAL_BATCHES", limit_val_batches),
        export_line("SCORER_SCENE_NUM", args.scorer_scene_num),
        export_line("N_ROLLOUT_CLOSED_VAL", args.n_rollout_closed_val),
        export_line("PRECISION", args.precision),
        export_line("DATA_NUM_WORKERS", args.num_workers),
        export_line("PREFETCH_FACTOR", args.prefetch_factor),
    ]
    if args.extra_hydra_overrides:
        lines.append(export_line("CATK_EXTRA_OVERRIDES", args.extra_hydra_overrides))
    return "\n".join(lines) + "\n"


def render_worker_script(project_root: str, env_file: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export WANDB_ENTITY="${{WANDB_ENTITY:-jksg01019-naver-labs}}"
export WANDB_PROJECT="${{WANDB_PROJECT:-SMART-FLOW}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

echo "[fast-wosac-g3zr84tp] pod=$(hostname) task=${{TASK_NAME}}"
echo "[fast-wosac-g3zr84tp] started at $(date '+%F %T')"
echo "[fast-wosac-g3zr84tp] repo=$(pwd)"
echo "[fast-wosac-g3zr84tp] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[fast-wosac-g3zr84tp] cache=${{CACHE_ROOT}}"
echo "[fast-wosac-g3zr84tp] ckpt=${{CKPT_PATH}}"
echo "[fast-wosac-g3zr84tp] scorer_scene_num=${{SCORER_SCENE_NUM}} val_batch_size=${{VAL_BATCH_SIZE}} limit_val_batches=${{LIMIT_VAL_BATCHES}}"
echo "[fast-wosac-g3zr84tp] n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} nproc=${{NPROC_PER_NODE}} precision=${{PRECISION}}"
echo

prepare_checkpoint() {{
  if [[ -f "$CKPT_PATH" ]]; then
    echo "[fast-wosac-g3zr84tp] using checkpoint: $CKPT_PATH"
    return 0
  fi

  python - <<'PY'
from pathlib import Path
import os
import shutil

target = Path(os.environ["CKPT_PATH"])
candidates = [
    target,
    Path("/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt"),
    Path("/mnt/nuplan/projects/catk/logs/flow_semi_continuous_pretrain_h100x4x2_bs26/runs/2026-05-03_19-01-34/checkpoints/epoch_last.ckpt"),
    Path("/mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_h100x4x2_bs26/run_g3zr84tp_v64/epoch_last.ckpt"),
    Path("/mnt/nuplan/projects/catk/checkpoints/wandb/flow_semi_continuous_pretrain_h100x4x2_bs26/epoch-last-g3zr84tp_v64/epoch_last.ckpt"),
]
for candidate in candidates:
    if candidate.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if candidate.resolve() != target.resolve():
            shutil.copy2(candidate, target)
        print(f"[fast-wosac-g3zr84tp] checkpoint ready from local path: {{target}}")
        raise SystemExit(0)
raise SystemExit(1)
PY
  local status=$?
  if (( status == 0 )); then
    return 0
  fi

  mkdir -p "$(dirname "$CKPT_PATH")" "$WANDB_DOWNLOAD_DIR"
  echo "[fast-wosac-g3zr84tp] local checkpoint not found; downloading W&B artifact: $WANDB_ARTIFACT"
  python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_ARTIFACT"]
download_dir = Path(os.environ["WANDB_DOWNLOAD_DIR"])
target_ckpt = Path(os.environ["CKPT_PATH"])
download_dir.mkdir(parents=True, exist_ok=True)
target_ckpt.parent.mkdir(parents=True, exist_ok=True)

try:
    import wandb
except Exception as exc:
    print(f"ERROR: failed to import wandb: {{exc}}", file=sys.stderr)
    sys.exit(2)

api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact_dir = Path(artifact.download(root=download_dir)).resolve()

candidates = []
preferred = artifact_dir / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred.as_posix())
candidates.extend(glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))
if not candidates:
    print(f"ERROR: no checkpoint file found in artifact dir: {{artifact_dir}}", file=sys.stderr)
    sys.exit(3)

source = Path(candidates[0]).resolve()
if source != target_ckpt.resolve():
    shutil.copy2(source, target_ckpt)
print(f"[fast-wosac-g3zr84tp] checkpoint ready from W&B artifact: {{target_ckpt}}")
PY
}}

prepare_checkpoint
status=$?
if (( status != 0 )); then
  echo "[fast-wosac-g3zr84tp] checkpoint preparation failed with status $status" >&2
  echo "[fast-wosac-g3zr84tp] leaving shell open for inspection"
  exec bash
fi

if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[fast-wosac-g3zr84tp] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  echo "[fast-wosac-g3zr84tp] leaving shell open for inspection"
  exec bash
fi

RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
OUTPUT_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
mkdir -p "$OUTPUT_DIR"

ARGS=(
  -m src.run
  experiment=local_val_flow
  action=validate
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$CATK_LOG_DIR"
  ckpt_path="$CKPT_PATH"
  task_name="$TASK_NAME"
  hydra.run.dir="$OUTPUT_DIR"
  trainer=ddp
  trainer.devices="$NPROC_PER_NODE"
  trainer.num_nodes=1
  trainer.precision="$PRECISION"
  trainer.limit_val_batches="$LIMIT_VAL_BATCHES"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.num_workers="$DATA_NUM_WORKERS"
  data.prefetch_factor="$PREFETCH_FACTOR"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val="$N_ROLLOUT_CLOSED_VAL"
  model.model_config.scorer_scene_num="$SCORER_SCENE_NUM"
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=false
  model.model_config.decoder.use_kinematic_control_flow=false
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=false
  model.model_config.decoder.use_stop_motion=true
  model.model_config.sim_agents_submission.is_active=false
  logger.wandb.name="$TASK_NAME"
  logger.wandb.group=fast_wosac_validation
  "logger.wandb.tags=[fast_wosac,g3zr84tp,wo-pvc-800,pose_space]"
)

if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($CATK_EXTRA_OVERRIDES)
  ARGS+=("${{EXTRA_ARGS[@]}}")
fi

echo "[fast-wosac-g3zr84tp] output_dir=$OUTPUT_DIR"
echo "[fast-wosac-g3zr84tp] command: python -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE ${{ARGS[*]}}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${{ARGS[@]}}"
status=$?

echo
echo "[fast-wosac-g3zr84tp] exited with status $status at $(date '+%F %T')"
echo "[fast-wosac-g3zr84tp] output_dir=$OUTPUT_DIR"
echo "[fast-wosac-g3zr84tp] leaving shell open for inspection"
exec bash
"""


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_start_command(args: argparse.Namespace) -> str:
    root = run_root(args)
    env_file = f"{root}/{args.pod}.env"
    worker_file = f"{root}/{args.pod}_worker.sh"
    monitor_file = f"{root}/{args.pod}_monitor.sh"
    tmux_log = f"{root}/{args.pod}.tmux.log"

    pull_block = ""
    if args.pull:
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

    if args.replace:
        session_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
"""
    else:
        session_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  echo "[launcher] attach with: tmux attach -t {args.session}" >&2
  exit 3
fi
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{render_monitor_script(args.monitor_interval, args.task_name).rstrip()}
CATK_MONITOR
chmod +x {shq(monitor_file)}
tmux split-window -v -l 12 -t {shq(args.session)} {shq(monitor_file)}
tmux select-pane -t {shq(args.session)}
"""

    return f"""set -Eeuo pipefail
if [ ! -d {shq(args.project_root)}/.git ]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
{session_block}
mkdir -p {shq(root)}
cat > {shq(env_file)} <<'CATK_ENV'
{render_env(args).rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(args.project_root, env_file).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.session} on {args.pod}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_stop_command(session: str) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
fi
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
    if args.dry_run:
        print("kubectl " + " ".join(shq(part) for part in command))
        return
    run_kubectl(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start Fast WOSAC validation for the g3zr84tp pose-space checkpoint "
            "inside the existing wo-pvc-800 pod."
        )
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--wandb-artifact", default=DEFAULT_WANDB_ARTIFACT)
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--limit-val-batches", default="auto")
    parser.add_argument("--scorer-scene-num", type=int, default=1680)
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.scorer_scene_num < 1:
        parser.error("--scorer-scene-num must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.limit_val_batches != "auto":
        try:
            parsed_limit = float(args.limit_val_batches)
        except ValueError:
            parser.error("--limit-val-batches must be 'auto' or a numeric value")
        if parsed_limit <= 0:
            parser.error("--limit-val-batches must be positive")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session))
        return

    limit_val_batches = (
        default_limit_val_batches(args)
        if args.limit_val_batches == "auto"
        else args.limit_val_batches
    )
    estimated_scenes = (
        int(limit_val_batches) * args.val_batch_size * args.nproc_per_node
        if isinstance(limit_val_batches, int) or str(limit_val_batches).isdigit()
        else "unknown"
    )

    print(f"[launcher] pod:       {args.pod}")
    print(f"[launcher] branch:    {args.branch}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] ckpt:      {args.ckpt_path}")
    print(f"[launcher] artifact:  {args.wandb_artifact}")
    print(f"[launcher] scorer_scene_num: {args.scorer_scene_num}")
    print(f"[launcher] limit_val_batches: {limit_val_batches} (estimated scenes={estimated_scenes})")
    print(f"[launcher] n_rollout_closed_val: {args.n_rollout_closed_val}")

    exec_in_pod(args, render_start_command(args))

    print("\nAttach command:")
    print(
        "kubectl exec -it "
        f"-n {shq(args.namespace)} {shq(args.pod)} -c {shq(args.container)} "
        f"-- tmux attach -t {shq(args.session)}"
    )


if __name__ == "__main__":
    main()
