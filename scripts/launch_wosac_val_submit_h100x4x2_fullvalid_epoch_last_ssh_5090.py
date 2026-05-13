#!/usr/bin/env python3
"""Launch full validation-set WOSAC submission for H100x4x2 full-valid run.

The launcher copies ``epoch_last.ckpt`` from the ``hsb-npc-training`` pod to
the RTX 5090 SSH host, then starts ``sim_agents_sub_flow`` validation export
with Waymo auto submission enabled inside the existing ``hsb-rl-train`` tmux
session.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile


DEFAULT_SSH_HOST = "user@10.60.188.78"
DEFAULT_REMOTE_PROJECT_ROOT = "/media/user/E/projects/catk"
DEFAULT_REMOTE_CACHE_ROOT = "/media/user/E/dataset/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/media/user/D/catk_wosac_val_submit_logs"
DEFAULT_SOURCE_NAMESPACE = "p-pnc"
DEFAULT_SOURCE_POD = "hsb-npc-training"
DEFAULT_SOURCE_CONTAINER = "main"
DEFAULT_SOURCE_TASK = "flow_control_space_pretrain_h100x4x2_fullvalid_roundtrip2_lr6e-4_bs26"
DEFAULT_REMOTE_CKPT_PATH = (
    "/media/user/E/projects/catk/checkpoints/from_pods/"
    "flow_control_space_pretrain_h100x4x2_fullvalid_roundtrip2_lr6e-4_bs26/"
    "epoch_last.ckpt"
)
DEFAULT_TMUX_SESSION = "hsb-rl-train"
DEFAULT_WINDOW_NAME = "wosac-submit-h100x4x2"
DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_h100x4x2_fullvalid_roundtrip2_lr6e-4_bs26_"
    "epoch_last_wosac_val_submit"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "semi_control"
    branch = result.stdout.strip()
    return branch if branch else "semi_control"


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


def run_shell(script: str, *, capture: bool = False, dry_run: bool = False) -> str:
    return run(["bash", "-lc", script], capture=capture, dry_run=dry_run)


def run_ssh(args: argparse.Namespace, script: str, *, capture: bool = False) -> str:
    return run(
        ["ssh", args.ssh_host, "bash -lc " + shq(script)],
        capture=capture,
        dry_run=args.dry_run,
    )


def run_kubectl(args: argparse.Namespace, script: str, *, capture: bool = False) -> str:
    return run(
        [
            "kubectl",
            "exec",
            "-n",
            args.source_namespace,
            args.source_pod,
            "-c",
            args.source_container,
            "--",
            "bash",
            "-lc",
            script,
        ],
        capture=capture,
        dry_run=args.dry_run,
    )


def resolve_source_ckpt(args: argparse.Namespace) -> str:
    if args.source_ckpt != "auto":
        return args.source_ckpt
    if args.dry_run:
        return (
            "/mnt/nuplan/projects/catk/logs/"
            f"{args.source_task}/runs/<latest>/checkpoints/epoch_last.ckpt"
        )

    script = f"""
set -Eeuo pipefail
task={shq(args.source_task)}
find /mnt/nuplan/projects/catk/logs/"$task"/runs \\
  -path '*/checkpoints/epoch_last.ckpt' -type f \\
  -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
"""
    source = run_kubectl(args, script, capture=True)
    if not source:
        raise RuntimeError(f"failed to find epoch_last.ckpt for task {args.source_task}")
    return source


def source_file_size(args: argparse.Namespace, source_ckpt: str) -> int:
    return int(run_kubectl(args, f"stat -c %s {shq(source_ckpt)}", capture=True))


def remote_file_size(args: argparse.Namespace, remote_path: str) -> int:
    return int(run_ssh(args, f"stat -c %s {shq(remote_path)}", capture=True))


def copy_checkpoint_to_remote(args: argparse.Namespace, source_ckpt: str) -> None:
    remote_dir = "/".join(args.remote_ckpt_path.rstrip("/").split("/")[:-1])
    run_ssh(args, f"mkdir -p {shq(remote_dir)}")
    if args.dry_run:
        print(f"[launcher] copying checkpoint from {args.source_pod}:{source_ckpt}")
        print(f"[launcher] remote checkpoint: {args.remote_ckpt_path}")
        print(
            "+ kubectl exec ... cat "
            + shq(source_ckpt)
            + " | ssh "
            + shq(args.ssh_host)
            + " 'cat > "
            + shq(args.remote_ckpt_path)
            + "'"
        )
        return

    source_size = source_file_size(args, source_ckpt)
    if args.skip_ckpt_copy:
        try:
            if remote_file_size(args, args.remote_ckpt_path) == source_size:
                print(f"[launcher] remote checkpoint already matches source size: {source_size}")
                return
        except Exception:
            pass

    print(f"[launcher] copying checkpoint from {args.source_pod}:{source_ckpt}")
    print(f"[launcher] remote checkpoint: {args.remote_ckpt_path} ({source_size} bytes)")
    tmp_path = args.remote_ckpt_path + ".tmp"
    pipe = (
        "set -o pipefail; "
        f"kubectl exec -n {shq(args.source_namespace)} {shq(args.source_pod)} "
        f"-c {shq(args.source_container)} -- cat {shq(source_ckpt)} | "
        f"ssh {shq(args.ssh_host)} "
        f"{shq('cat > ' + shq(tmp_path) + ' && mv ' + shq(tmp_path) + ' ' + shq(args.remote_ckpt_path))}"
    )
    run_shell(pipe)

    remote_size = remote_file_size(args, args.remote_ckpt_path)
    if remote_size == source_size:
        return

    print(
        "[launcher] stream copy size mismatch; retrying via kubectl cp + scp "
        f"(source={source_size}, remote={remote_size})"
    )
    with tempfile.TemporaryDirectory(prefix="catk-h100x4x2-submit-") as tmpdir:
        local_ckpt = os.path.join(tmpdir, "epoch_last.ckpt")
        run(
            [
                "kubectl",
                "cp",
                "-n",
                args.source_namespace,
                "-c",
                args.source_container,
                f"{args.source_pod}:{source_ckpt}",
                local_ckpt,
            ]
        )
        local_size = os.path.getsize(local_ckpt)
        if local_size != source_size:
            raise RuntimeError(f"kubectl cp size mismatch: source={source_size}, local={local_size}")
        remote_tmp_path = args.remote_ckpt_path + ".scp_tmp"
        run(["scp", local_ckpt, f"{args.ssh_host}:{remote_tmp_path}"])
        run_ssh(args, f"mv {shq(remote_tmp_path)} {shq(args.remote_ckpt_path)}")
        remote_size = remote_file_size(args, args.remote_ckpt_path)
        if remote_size != source_size:
            raise RuntimeError(
                f"checkpoint copy size mismatch after retry: source={source_size}, remote={remote_size}"
            )


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} host=$(hostname)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_worker_script(args: argparse.Namespace) -> str:
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
export CUDA_VISIBLE_DEVICES={shq(args.cuda_visible_devices)}

cd {shq(args.remote_project_root)}
if [[ -f scripts/_activate_conda.sh ]]; then
  # shellcheck source=/dev/null
  . scripts/_activate_conda.sh
elif [[ -f /media/user/E/miniforge/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  . /media/user/E/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi

TASK_NAME={shq(args.task_name)}
CACHE_ROOT={shq(args.remote_cache_root)}
CATK_LOG_DIR={shq(args.remote_log_dir)}
CKPT_PATH={shq(args.remote_ckpt_path)}
VAL_BATCH_SIZE={shq(args.val_batch_size)}
MIN_VAL_BATCH_SIZE={shq(args.min_val_batch_size)}
VAL_BATCH_SIZE_STEP={shq(args.val_batch_size_step)}
NPROC_PER_NODE={shq(args.nproc_per_node)}
PRECISION={shq(args.precision)}
DATA_NUM_WORKERS={shq(args.num_workers)}
PREFETCH_FACTOR={shq(args.prefetch_factor)}
N_ROLLOUT_CLOSED_VAL={shq(args.n_rollout_closed_val)}
WAYMO_SUBMISSION_ENABLED={shq(str(args.waymo_submission_enabled).lower())}
WAYMO_STORAGE_STATE_PATH={shq(args.waymo_storage_state_path)}
OOM_REGEX={shq(args.oom_regex)}

echo "[wosac-submit-h100x4x2] host=$(hostname) task=${{TASK_NAME}}"
echo "[wosac-submit-h100x4x2] started at $(date '+%F %T')"
echo "[wosac-submit-h100x4x2] repo=$(pwd)"
echo "[wosac-submit-h100x4x2] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[wosac-submit-h100x4x2] cache=${{CACHE_ROOT}}"
echo "[wosac-submit-h100x4x2] ckpt=${{CKPT_PATH}}"
echo "[wosac-submit-h100x4x2] val_batch_size=${{VAL_BATCH_SIZE}} nproc=${{NPROC_PER_NODE}} precision=${{PRECISION}}"
echo "[wosac-submit-h100x4x2] waymo_submission.enabled=${{WAYMO_SUBMISSION_ENABLED}}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[wosac-submit-h100x4x2] checkpoint does not exist: $CKPT_PATH" >&2
  exec bash
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[wosac-submit-h100x4x2] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  exec bash
fi

COMMON_ARGS=(
  -m src.run
  experiment=sim_agents_sub_flow
  action=validate
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$CATK_LOG_DIR"
  ckpt_path="$CKPT_PATH"
  task_name="$TASK_NAME"
  trainer.limit_val_batches=1.0
  trainer.precision="$PRECISION"
  model.model_config.val_open_loop=false
  model.model_config.val_closed_loop=true
  model.model_config.n_rollout_closed_val="$N_ROLLOUT_CLOSED_VAL"
  model.model_config.decoder.flow_window_steps=20
  model.model_config.token_processor.flow_window_steps=20
  model.model_config.token_processor.use_kinematic_control_flow=true
  model.model_config.decoder.use_kinematic_control_flow=true
  model.model_config.token_processor.use_prefix_valid_future_loss_mask=false
  model.model_config.token_processor.control_round_trip_max_position_error_m=2.0
  model.model_config.token_processor.control_pos_scale_m=1.0
  model.model_config.token_processor.control_vehicle_yaw_scale_rad=0.025
  model.model_config.token_processor.control_pedestrian_yaw_scale_rad=0.20
  model.model_config.token_processor.control_cyclist_yaw_scale_rad=0.06
  model.model_config.decoder.use_stop_motion=true
  model.model_config.sim_agents_submission.method_name="SMART-control-h100x4x2-fullvalid-rt2"
  "model.model_config.sim_agents_submission.authors=[Seulbin Hwang,Kiyoung Om]"
  model.model_config.sim_agents_submission.affiliation=NaverLabs
  model.model_config.sim_agents_submission.description="Control-space Flow Matching H100x4x2 full-valid round-trip 2.0 validation submission."
  model.model_config.sim_agents_submission.method_link="not available yet"
  model.model_config.sim_agents_submission.account_name="h.sb@naverlabs.com"
  waymo_submission.enabled="$WAYMO_SUBMISSION_ENABLED"
  waymo_submission.submit_validate=true
  waymo_submission.submit_test=false
  waymo_submission.evaluation_set=validation
  waymo_submission.poll_submission_status=false
  logger.wandb.name="$TASK_NAME"
  logger.wandb.group=wosac_validation_submission
  "logger.wandb.tags=[wosac_submission,h100x4x2_fullvalid,epoch_last,rtx5090,control_space]"
)

if [[ -n "$WAYMO_STORAGE_STATE_PATH" ]]; then
  COMMON_ARGS+=(waymo_submission.storage_state_path="$WAYMO_STORAGE_STATE_PATH")
fi
if [[ -n "$DATA_NUM_WORKERS" ]]; then
  COMMON_ARGS+=(data.num_workers="$DATA_NUM_WORKERS")
fi
if [[ -n "$PREFETCH_FACTOR" ]]; then
  COMMON_ARGS+=(data.prefetch_factor="$PREFETCH_FACTOR")
fi

is_positive_int() {{
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}}

if ! is_positive_int "$VAL_BATCH_SIZE" || ! is_positive_int "$MIN_VAL_BATCH_SIZE" || ! is_positive_int "$VAL_BATCH_SIZE_STEP"; then
  echo "[wosac-submit-h100x4x2] invalid val batch settings" >&2
  exec bash
fi

ATTEMPT_LOG_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/retry_attempts"
mkdir -p "$ATTEMPT_LOG_DIR"

attempt=1
current_val_bs="$VAL_BATCH_SIZE"
while true; do
  RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')-try$(printf '%02d' "$attempt")-valbs${{current_val_bs}}"
  OUTPUT_DIR="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
  ATTEMPT_LOG="${{ATTEMPT_LOG_DIR}}/${{RUN_ID}}.log"
  mkdir -p "$OUTPUT_DIR" "$ATTEMPT_LOG_DIR"

  echo
  echo "[wosac-submit-h100x4x2] attempt $attempt val_batch_size=$current_val_bs"
  echo "[wosac-submit-h100x4x2] output_dir=$OUTPUT_DIR"
  echo "[wosac-submit-h100x4x2] log=$ATTEMPT_LOG"

  ATTEMPT_ARGS=(
    "${{COMMON_ARGS[@]}}"
    data.val_batch_size="$current_val_bs"
    hydra.run.dir="$OUTPUT_DIR"
  )

  set +e
  if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    python "${{ATTEMPT_ARGS[@]}}" trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto 2>&1 | tee "$ATTEMPT_LOG"
    status="${{PIPESTATUS[0]}}"
  else
    python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${{ATTEMPT_ARGS[@]}}" trainer=ddp trainer.devices="$NPROC_PER_NODE" 2>&1 | tee "$ATTEMPT_LOG"
    status="${{PIPESTATUS[0]}}"
  fi
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "[wosac-submit-h100x4x2] SUCCESS val_batch_size=$current_val_bs"
    echo "[wosac-submit-h100x4x2] final_output_dir=$OUTPUT_DIR"
    break
  fi

  if ! grep -Eqi "$OOM_REGEX" "$ATTEMPT_LOG"; then
    echo "[wosac-submit-h100x4x2] failed with status $status and no OOM marker; not retrying." >&2
    break
  fi

  next_val_bs=$(( current_val_bs - VAL_BATCH_SIZE_STEP ))
  if [[ "$next_val_bs" -lt "$MIN_VAL_BATCH_SIZE" ]]; then
    echo "[wosac-submit-h100x4x2] OOM but next val_batch_size $next_val_bs is below minimum $MIN_VAL_BATCH_SIZE." >&2
    break
  fi
  echo "[wosac-submit-h100x4x2] OOM detected; retrying val_batch_size $current_val_bs -> $next_val_bs."
  current_val_bs="$next_val_bs"
  attempt=$(( attempt + 1 ))
done

echo
echo "[wosac-submit-h100x4x2] finished at $(date '+%F %T')"
echo "[wosac-submit-h100x4x2] leaving shell open for inspection"
exec bash
"""


def render_remote_start(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    run_root = f"{args.remote_log_dir.rstrip('/')}/tmux_wosac_val_submit/{safe_task}"
    worker_file = f"{run_root}/worker.sh"
    monitor_file = f"{run_root}/monitor.sh"
    tmux_log = f"{run_root}/tmux.log"

    sync_block = ""
    if args.pull:
        sync_block = f"""
git config --global --add safe.directory {shq(args.remote_project_root)} || true
git fetch origin {shq(args.branch)}
TRACKED_DIRTY="$(git status --porcelain --untracked-files=no)"
CURRENT_BRANCH="$(git branch --show-current || true)"
if [[ -n "$TRACKED_DIRTY" && "$CURRENT_BRANCH" != {shq(args.branch)} ]]; then
  echo "[launcher] tracked working tree is dirty on $CURRENT_BRANCH; refusing to checkout {args.branch}" >&2
  exit 4
fi
if [[ -z "$TRACKED_DIRTY" ]]; then
  git checkout {shq(args.branch)}
  git pull --ff-only origin {shq(args.branch)}
else
  echo "[launcher] tracked working tree is dirty; staying on $CURRENT_BRANCH and skipping pull."
fi
"""

    replace_block = ""
    if args.replace:
        replace_block = f"""
while IFS=: read -r window_id window_name; do
  if [[ "$window_name" == {shq(args.window_name)} ]]; then
    tmux kill-window -t {shq(args.tmux_session)}:"$window_id" || true
  fi
done < <(tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_index}}:#{{window_name}}' 2>/dev/null || true)
"""
    else:
        replace_block = f"""
if tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_name}}' 2>/dev/null | grep -Fx {shq(args.window_name)} >/dev/null; then
  echo "[launcher] tmux window already exists: {args.tmux_session}:{args.window_name}" >&2
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
tmux split-window -v -l 10 -t {shq(args.tmux_session)}:{shq(args.window_name)} {shq(monitor_file)}
tmux select-pane -t {shq(args.tmux_session)}:{shq(args.window_name)}.0
"""

    return f"""set -Eeuo pipefail
if [[ ! -d {shq(args.remote_project_root)}/.git ]]; then
  echo "[launcher] remote project root is not a git checkout: {args.remote_project_root}" >&2
  exit 2
fi
cd {shq(args.remote_project_root)}
{sync_block}
if ! tmux has-session -t {shq(args.tmux_session)} 2>/dev/null; then
  tmux new-session -d -s {shq(args.tmux_session)} -c {shq(args.remote_project_root)}
fi
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(args).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-window -t {shq(args.tmux_session)} -n {shq(args.window_name)} -c {shq(args.remote_project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.tmux_session)}:{shq(args.window_name)} -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.tmux_session}:{args.window_name}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_remote_stop(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
if ! tmux has-session -t {shq(args.tmux_session)} 2>/dev/null; then
  echo "[launcher] tmux session not found: {args.tmux_session}"
  exit 0
fi
found=0
while IFS=: read -r window_id window_name; do
  if [[ "$window_name" == {shq(args.window_name)} ]]; then
    tmux kill-window -t {shq(args.tmux_session)}:"$window_id" || true
    found=1
  fi
done < <(tmux list-windows -t {shq(args.tmux_session)} -F '#{{window_index}}:#{{window_name}}')
if (( found )); then
  echo "[launcher] stopped tmux window {args.tmux_session}:{args.window_name}"
else
  echo "[launcher] tmux window not found: {args.tmux_session}:{args.window_name}"
fi
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy H100x4x2 full-valid epoch_last.ckpt from hsb-npc-training "
            "and launch full validation-set Waymo/WOSAC auto submission on "
            "the RTX 5090 host."
        )
    )
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT)
    parser.add_argument("--remote-cache-root", default=DEFAULT_REMOTE_CACHE_ROOT)
    parser.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    parser.add_argument("--remote-ckpt-path", default=DEFAULT_REMOTE_CKPT_PATH)
    parser.add_argument("--source-namespace", default=DEFAULT_SOURCE_NAMESPACE)
    parser.add_argument("--source-pod", default=DEFAULT_SOURCE_POD)
    parser.add_argument("--source-container", default=DEFAULT_SOURCE_CONTAINER)
    parser.add_argument("--source-task", default=DEFAULT_SOURCE_TASK)
    parser.add_argument("--source-ckpt", default="auto")
    parser.add_argument("--branch", default=current_branch())
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--min-val-batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size-step", type=int, default=2)
    parser.add_argument("--n-rollout-closed-val", type=int, default=32)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--waymo-submission-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--waymo-storage-state-path",
        default="/media/user/E/projects/catk/secrets/waymo/waymo_storage_state.json",
    )
    parser.add_argument(
        "--oom-regex",
        default=(
            "OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|"
            "torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY"
        ),
    )
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--skip-ckpt-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.min_val_batch_size < 1:
        parser.error("--min-val-batch-size must be >= 1")
    if args.val_batch_size_step < 1:
        parser.error("--val-batch-size-step must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        run_ssh(args, render_remote_stop(args))
        return

    source_ckpt = resolve_source_ckpt(args)
    print(f"[launcher] ssh_host:       {args.ssh_host}")
    print(f"[launcher] remote repo:    {args.remote_project_root}")
    print(f"[launcher] branch:         {args.branch}")
    print(f"[launcher] tmux target:    {args.tmux_session}:{args.window_name}")
    print(f"[launcher] source ckpt:    {args.source_pod}:{source_ckpt}")
    print(f"[launcher] remote ckpt:    {args.remote_ckpt_path}")
    print(f"[launcher] task_name:      {args.task_name}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")

    copy_checkpoint_to_remote(args, source_ckpt)
    run_ssh(args, render_remote_start(args))

    print("\nAttach command:")
    print(f"ssh -t {shq(args.ssh_host)} {shq('tmux attach -t ' + args.tmux_session)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[launcher] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
