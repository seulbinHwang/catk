#!/usr/bin/env python3
"""Launch quick Fast WOSAC/RMM validation for the A100x4x2 epoch-last checkpoint.

The validation runs inside the already-running ``testa`` pod on its four A100
GPUs. It uses the latest ``epoch_last.ckpt`` from the completed
``flow_control_space_pretrain_a100x4x2_prefix_roundtrip2_lr6e-4_bs26`` run,
verifies that the checkpoint metadata says epoch 63 by default, and starts a
tmux session in the pod. The launcher does not create, delete, or restart pods.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
import time


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "testa"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_SOURCE_TASK = "flow_control_space_pretrain_a100x4x2_prefix_roundtrip2_lr6e-4_bs26"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_a100x4x2_prefix_roundtrip2_lr6e-4_bs26_"
    "epoch063_fast_rmm_val64"
)
DEFAULT_SESSION = "catk-fast-rmm-a100x4-prefix-e63"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(command: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
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


def run_kubectl(args: argparse.Namespace, pod_script: str, *, capture: bool = False) -> str:
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
            pod_script,
        ],
        capture=capture,
        dry_run=args.dry_run,
    )


def default_limit_val_batches(args: argparse.Namespace) -> int:
    per_rank_scenes = math.ceil(args.scorer_scene_num / args.nproc_per_node)
    return max(1, math.ceil(per_rank_scenes / args.val_batch_size))


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_fast_rmm_val/{safe_task}"


def status_file(args: argparse.Namespace) -> str:
    return f"{run_root(args)}/{args.pod}.status"


def tmux_log(args: argparse.Namespace) -> str:
    return f"{run_root(args)}/{args.pod}.tmux.log"


def resolve_ckpt_path(args: argparse.Namespace) -> str:
    if args.ckpt_path != "auto":
        return args.ckpt_path
    if args.dry_run:
        return (
            f"{args.log_dir.rstrip('/')}/{args.source_task}/runs/<latest>/"
            "checkpoints/epoch_last.ckpt"
        )

    find_script = f"""
set -Eeuo pipefail
find {shq(args.log_dir.rstrip('/') + '/' + args.source_task)}/runs \\
  -path '*/checkpoints/epoch_last.ckpt' -type f \\
  -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
"""
    ckpt_path = run_kubectl(args, find_script, capture=True)
    if not ckpt_path:
        raise RuntimeError(f"failed to find epoch_last.ckpt for task {args.source_task}")
    return ckpt_path


def checkpoint_epoch(args: argparse.Namespace, ckpt_path: str) -> int:
    if args.dry_run:
        return args.expected_epoch

    inspect_script = f"""
set -Eeuo pipefail
ckpt={shq(ckpt_path)}
if [[ -x /mnt/nuplan/miniforge/envs/catk/bin/python ]]; then
  PYTHON_BIN=/mnt/nuplan/miniforge/envs/catk/bin/python
else
  PYTHON_BIN="$(command -v python)"
fi
"$PYTHON_BIN" - "$ckpt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(checkpoint.get("epoch", ""))
PY
"""
    output = run_kubectl(args, inspect_script, capture=True)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"failed to inspect checkpoint epoch: {ckpt_path}")
    return int(lines[-1])


def verify_checkpoint_epoch(args: argparse.Namespace, ckpt_path: str) -> int:
    epoch = checkpoint_epoch(args, ckpt_path)
    if args.expected_epoch >= 0 and epoch != args.expected_epoch and not args.allow_epoch_mismatch:
        raise RuntimeError(
            f"checkpoint epoch mismatch: expected {args.expected_epoch}, got {epoch}. "
            "Pass --allow-epoch-mismatch only if this is intentional."
        )
    return epoch


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def render_env(args: argparse.Namespace, ckpt_path: str) -> str:
    limit_val_batches = (
        default_limit_val_batches(args)
        if args.limit_val_batches == "auto"
        else args.limit_val_batches
    )
    lines = [
        export_line("TASK_NAME", args.task_name),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("CKPT_PATH", ckpt_path),
        export_line("STATUS_FILE", status_file(args)),
        export_line("OUTPUT_DIR_FILE", f"{run_root(args)}/{args.pod}.output_dir"),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("LIMIT_VAL_BATCHES", limit_val_batches),
        export_line("SCORER_SCENE_NUM", args.scorer_scene_num),
        export_line("N_ROLLOUT_CLOSED_VAL", args.n_rollout_closed_val),
        export_line("PRECISION", args.precision),
        export_line("DATA_NUM_WORKERS", args.num_workers),
        export_line("PREFETCH_FACTOR", args.prefetch_factor),
        export_line("WANDB_MODE", args.wandb_mode),
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

if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

if [[ -x /mnt/nuplan/miniforge/envs/catk/bin/python ]]; then
  PYTHON_BIN=/mnt/nuplan/miniforge/envs/catk/bin/python
else
  PYTHON_BIN="$(command -v python)"
fi

echo "[fast-rmm-a100x4] pod=$(hostname) task=${{TASK_NAME}}"
echo "[fast-rmm-a100x4] started at $(date '+%F %T')"
echo "[fast-rmm-a100x4] repo=$(pwd)"
echo "[fast-rmm-a100x4] branch=$(git branch --show-current 2>/dev/null)"
echo "[fast-rmm-a100x4] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[fast-rmm-a100x4] python=${{PYTHON_BIN}}"
echo "[fast-rmm-a100x4] cache=${{CACHE_ROOT}}"
echo "[fast-rmm-a100x4] ckpt=${{CKPT_PATH}}"
echo "[fast-rmm-a100x4] scorer_scene_num=${{SCORER_SCENE_NUM}} val_batch_size=${{VAL_BATCH_SIZE}} limit_val_batches=${{LIMIT_VAL_BATCHES}}"
echo "[fast-rmm-a100x4] n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} nproc=${{NPROC_PER_NODE}} precision=${{PRECISION}}"
echo

rm -f "$STATUS_FILE"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[fast-rmm-a100x4] checkpoint does not exist: $CKPT_PATH" >&2
  echo 2 > "$STATUS_FILE"
  exec bash
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[fast-rmm-a100x4] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  echo 2 > "$STATUS_FILE"
  exec bash
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[fast-rmm-a100x4] python is not available after conda activation" >&2
  echo 2 > "$STATUS_FILE"
  exec bash
fi

RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
OUTPUT_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
mkdir -p "$OUTPUT_DIR"
echo "$OUTPUT_DIR" > "$OUTPUT_DIR_FILE"

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
  +trainer.enable_progress_bar=false
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.num_workers="$DATA_NUM_WORKERS"
  data.prefetch_factor="$PREFETCH_FACTOR"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val="$N_ROLLOUT_CLOSED_VAL"
  model.model_config.scorer_scene_num="$SCORER_SCENE_NUM"
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=true
  model.model_config.decoder.use_kinematic_control_flow=true
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=true
  model.model_config.token_processor.control_round_trip_max_position_error_m=2.0
  model.model_config.token_processor.control_pos_scale_m=1.0
  model.model_config.token_processor.control_vehicle_yaw_scale_rad=0.025
  model.model_config.token_processor.control_pedestrian_yaw_scale_rad=0.20
  model.model_config.token_processor.control_cyclist_yaw_scale_rad=0.06
  model.model_config.decoder.use_stop_motion=true
  model.model_config.sim_agents_submission.is_active=false
  logger.wandb.name="$TASK_NAME"
  logger.wandb.group=fast_rmm_validation
  "logger.wandb.tags=[fast_rmm,fast_wosac,a100x4,testa,epoch063,control_space,semi_control]"
)

if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($CATK_EXTRA_OVERRIDES)
  ARGS+=("${{EXTRA_ARGS[@]}}")
fi

echo "[fast-rmm-a100x4] output_dir=$OUTPUT_DIR"
echo "[fast-rmm-a100x4] command: $PYTHON_BIN -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE ${{ARGS[*]}}"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${{ARGS[@]}}"
status=$?
echo "$status" > "$STATUS_FILE"

echo
echo "[fast-rmm-a100x4] exited with status $status at $(date '+%F %T')"
echo "[fast-rmm-a100x4] output_dir=$OUTPUT_DIR"
echo "[fast-rmm-a100x4] leaving shell open for inspection"
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


def render_start_command(args: argparse.Namespace, ckpt_path: str) -> str:
    root = run_root(args)
    env_file = f"{root}/{args.pod}.env"
    worker_file = f"{root}/{args.pod}_worker.sh"
    monitor_file = f"{root}/{args.pod}_monitor.sh"
    log_file = tmux_log(args)

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
if [[ ! -d {shq(args.project_root)}/.git ]]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
{session_block}
mkdir -p {shq(root)}
rm -f {shq(status_file(args))} {shq(root + '/' + args.pod + '.output_dir')}
cat > {shq(env_file)} <<'CATK_ENV'
{render_env(args, ckpt_path).rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(args.project_root, env_file).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(log_file)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq('cat >> ' + shq(log_file))}
{monitor_block}
echo "[launcher] started {args.session} on {args.pod}"
echo "[launcher] tmux log: {log_file}"
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


def exec_in_pod(args: argparse.Namespace, script: str) -> str:
    return run_kubectl(args, script, capture=args.dry_run)


def tail_tmux_log(args: argparse.Namespace, *, lines: int = 80, chars: int = 60000) -> str:
    script = f"""
set +e
if [[ -f {shq(tmux_log(args))} ]]; then
  tail -c {int(chars)} {shq(tmux_log(args))} | tr '\\r' '\\n' | tail -n {int(lines)}
fi
"""
    return run_kubectl(args, script, capture=True)


def wait_for_completion(args: argparse.Namespace) -> None:
    if args.dry_run:
        return

    print(f"[launcher] waiting for completion via {status_file(args)}")
    deadline = time.monotonic() + args.timeout_sec if args.timeout_sec > 0 else None
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError(f"timed out waiting for validation after {args.timeout_sec} seconds")

        poll_script = f"""
set +e
if [[ -f {shq(status_file(args))} ]]; then
  printf 'STATUS='
  cat {shq(status_file(args))}
  exit 0
fi
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo RUNNING
else
  echo MISSING
fi
"""
        state = run_kubectl(args, poll_script, capture=True).splitlines()[-1].strip()
        if state.startswith("STATUS="):
            status = int(state.split("=", 1)[1])
            print(tail_tmux_log(args, lines=80))
            if status != 0:
                raise RuntimeError(f"validation exited with status {status}")
            return
        if state == "MISSING":
            print(tail_tmux_log(args, lines=80))
            raise RuntimeError("tmux session ended before writing a status file")

        print(f"[launcher] still running; next poll in {args.wait_interval_sec}s")
        time.sleep(args.wait_interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start quick Fast WOSAC/RMM validation for the A100x4x2 epoch_last "
            "checkpoint inside the existing testa pod."
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
    parser.add_argument("--source-task", default=DEFAULT_SOURCE_TASK)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--ckpt-path", default="auto")
    parser.add_argument("--expected-epoch", type=int, default=63)
    parser.add_argument("--allow-epoch-mismatch", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--limit-val-batches", default="auto")
    parser.add_argument("--scorer-scene-num", type=int, default=64)
    parser.add_argument("--n-rollout-closed-val", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-interval-sec", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=0)
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
    if args.wait_interval_sec < 1:
        parser.error("--wait-interval-sec must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session))
        return

    ckpt_path = resolve_ckpt_path(args)
    epoch = verify_checkpoint_epoch(args, ckpt_path)
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
    print(f"[launcher] ckpt:      {ckpt_path}")
    print(f"[launcher] ckpt_epoch:{epoch}")
    print(f"[launcher] scorer_scene_num: {args.scorer_scene_num}")
    print(f"[launcher] limit_val_batches: {limit_val_batches} (estimated scenes={estimated_scenes})")
    print(f"[launcher] n_rollout_closed_val: {args.n_rollout_closed_val}")

    exec_in_pod(args, render_start_command(args, ckpt_path))

    print("\nAttach command:")
    print(
        "kubectl exec -it "
        f"-n {shq(args.namespace)} {shq(args.pod)} -c {shq(args.container)} "
        f"-- tmux attach -t {shq(args.session)}"
    )
    if args.wait:
        wait_for_completion(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[launcher] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
