#!/usr/bin/env python3
"""Launch semi_mdg MDG-style pretrain on the single testas A100x7 pod.

The script runs from a workstation with kubectl access. It never creates,
deletes, or restarts the pod. It only syncs the requested git branch inside the
already-running pod and starts a tmux session that runs torchrun with 7 local
A100 GPUs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "testas"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_mdg"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "mdg_pretrain_h100x3x2"
DEFAULT_SESSION = "catk-semi-mdg-testas-a100x7"


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


def current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return DEFAULT_BRANCH
    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else DEFAULT_BRANCH


def default_task_name() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"semi_mdg_pretrain_testas_a100x7_{stamp}"


def render_remote_run_script(args: argparse.Namespace, run_root: str) -> str:
    extra_overrides = args.extra_hydra_overrides.strip()
    if args.train_sidecar_dir:
        extra_overrides = " ".join(
            part
            for part in [
                extra_overrides,
                f"data.train_sidecar_dir={args.train_sidecar_dir}",
            ]
            if part
        )
    if args.wandb_mode == "offline":
        extra_overrides = " ".join(
            part
            for part in [
                extra_overrides,
                "logger.wandb.offline=true",
                "callbacks.epoch_last_checkpoint.upload_to_wandb=false",
                "logger.wandb.log_model=false",
            ]
            if part
        )
    else:
        extra_overrides = " ".join(
            part
            for part in [
                extra_overrides,
                "logger.wandb.offline=false",
                "logger.wandb.group=semi_mdg_pretrain_testas_a100x7",
                "logger.wandb.job_type=pretrain",
            ]
            if part
        )

    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1
export CATK_REMOTE_PYTHON="${{CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}}"
export WANDB_MODE={shq(args.wandb_mode)}

log() {{
  printf '[%s] %s\\n' "$(date '+%F %T %Z')" "$*"
}}

activate_conda_if_available() {{
  if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
    conda activate "${{CATK_CONDA_ENV:-catk}}" 2>/dev/null \\
      || conda activate base 2>/dev/null \\
      || true
  fi
  log "conda env: ${{CONDA_DEFAULT_ENV:-unknown}}"
}}

task_process_pids() {{
  pgrep -f "task_name={args.task_name}" 2>/dev/null | while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
      echo "$pid"
    fi
  done
}}

terminate_task_processes() {{
  local reason="${{1:-cleanup}}"
  local pids=()
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} == 0 )); then
    return 0
  fi
  log "terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep "${{TASK_PROCESS_KILL_GRACE_SEC:-20}}"
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} > 0 )); then
    log "force killing task processes for $reason: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
}}

find_latest_epoch_last_ckpt() {{
  local runs_dir={shq(args.remote_log_dir.rstrip("/") + "/" + args.task_name + "/runs")}
  {{ ls -t "$runs_dir"/*/checkpoints/epoch_last.ckpt 2>/dev/null; \\
     ls -t "$runs_dir"/*/checkpoints/last.ckpt 2>/dev/null; }} | head -1
}}

on_interrupt() {{
  log "interrupt received; stopping torchrun cleanly"
  terminate_task_processes interrupt
  exit 130
}}
trap on_interrupt INT TERM

cd {shq(args.project_root)}
activate_conda_if_available
mkdir -p {shq(run_root)}
test -d {shq(args.cache_root)} || {{ log "ERROR: cache root missing: {args.cache_root}"; exit 2; }}

export LOGLEVEL="${{LOGLEVEL:-INFO}}"
export HYDRA_FULL_ERROR="${{HYDRA_FULL_ERROR:-1}}"
export TF_CPP_MIN_LOG_LEVEL="${{TF_CPP_MIN_LOG_LEVEL:-2}}"
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
export TORCH_NCCL_BLOCKING_WAIT="${{TORCH_NCCL_BLOCKING_WAIT:-0}}"
export CATK_ATTENTION_GRAPH_FP32="${{CATK_ATTENTION_GRAPH_FP32:-1}}"

export CACHE_ROOT={shq(args.cache_root)}
export NNODES=1
export NPROC_PER_NODE={args.nproc_per_node}
export TRAINER_DEVICES={args.nproc_per_node}
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT={shq(args.master_port)}
export CATK_EXPERIMENT={shq(args.experiment)}
export CATK_ACTION=fit
export TASK_NAME={shq(args.task_name)}
export LOG_DIR={shq(args.remote_log_dir)}
export VAL_BATCH_SIZE={shq(args.val_batch_size)}
export MAX_EPOCHS={shq(args.max_epochs)}
export CATK_HYDRA_OVERRIDES={shq(extra_overrides)}
export CATK_AUTO_SQRT_LR={"1" if args.auto_sqrt_lr else "0"}
export CATK_BASE_LR={shq(args.base_lr)}
export CATK_BASE_GLOBAL_BATCH_SIZE={shq(args.base_global_batch_size)}
if [[ "$CATK_AUTO_SQRT_LR" != "1" && -n {shq(args.learning_rate)} ]]; then
  export CATK_LR={shq(args.learning_rate)}
fi
if [[ -n {shq(args.limit_train_batches)} ]]; then
  export LIMIT_TRAIN_BATCHES={shq(args.limit_train_batches)}
fi
if [[ -n {shq(args.limit_val_batches)} ]]; then
  export LIMIT_VAL_BATCHES={shq(args.limit_val_batches)}
fi

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'
bs={args.initial_bs}
attempt=0
status_file={shq(run_root + "/torchrun_status")}
rm -f "$status_file"

log "semi_mdg testas A100x7 pretrain launcher"
log "task_name={args.task_name}"
log "experiment={args.experiment}"
log "cache_root={args.cache_root}"
log "train_sidecar_dir={args.train_sidecar_dir or '<disabled>'}"
log "nproc_per_node={args.nproc_per_node}"
log "initial_bs={args.initial_bs} oom_step={args.oom_step} min_bs={args.min_bs}"
log "val_batch_size={args.val_batch_size} max_epochs={args.max_epochs}"
if [[ "$CATK_AUTO_SQRT_LR" == "1" ]]; then
  log "auto sqrt lr enabled: base_lr={args.base_lr}, base_global_batch_size={args.base_global_batch_size}"
fi

while (( bs >= {args.min_bs} )); do
  attempt=$(( attempt + 1 ))
  export TRAIN_BATCH_SIZE="$bs"
  if [[ "$CATK_AUTO_SQRT_LR" == "1" ]]; then
    export CATK_LR="$(awk -v base="$CATK_BASE_LR" -v bs="$bs" -v nproc="$NPROC_PER_NODE" -v ref="$CATK_BASE_GLOBAL_BATCH_SIZE" 'BEGIN {{ printf "%.8g", base * sqrt((bs * nproc) / ref) }}')"
    log "attempt #$attempt: auto sqrt lr=$CATK_LR, global_batch=$(( bs * NPROC_PER_NODE ))"
  fi
  latest_ckpt="$(find_latest_epoch_last_ckpt)"
  unset CATK_CKPT_PATH
  if [[ -n "$latest_ckpt" ]]; then
    export CATK_CKPT_PATH="$latest_ckpt"
    log "attempt #$attempt: bs=$bs, resume ckpt=$latest_ckpt"
  else
    log "attempt #$attempt: bs=$bs, fresh fit"
  fi
  attempt_log={shq(run_root)}/attempt_$(printf '%03d' "$attempt")_bs${{bs}}.log
  log "attempt log: $attempt_log"
  terminate_task_processes pre_attempt_cleanup
  bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee "$attempt_log"
  status="${{PIPESTATUS[0]}}"
  echo "$status" > "$status_file"
  log "attempt #$attempt exited with status=$status"
  if [[ "$status" == "0" ]]; then
    log "training completed successfully at bs=$bs"
    exec bash
  fi
  if grep -Eq "$OOM_REGEX" "$attempt_log"; then
    new_bs=$(( bs - {args.oom_step} ))
    log "OOM detected at bs=$bs; retrying with bs=$new_bs"
    if (( new_bs < {args.min_bs} )); then
      log "next bs is below min_bs; aborting"
      exec bash
    fi
    bs="$new_bs"
    continue
  fi
  log "non-OOM failure; leaving tmux shell open"
  exec bash
done

log "no valid batch size left"
exec bash
"""


def render_remote_start_command(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    run_root = f"{args.remote_log_dir.rstrip('/')}/tmux_testas_a100x7_semi_mdg/{safe_task}"
    run_file = f"{run_root}/run.sh"
    tmux_log = f"{run_root}/tmux.log"
    monitor_file = f"{run_root}/monitor.sh"
    run_script = render_remote_run_script(args, run_root)
    monitor_script = f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T %Z') task={args.task_name} pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(args.monitor_interval)}
done
"""
    replace_block = ""
    if args.replace:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux send-keys -t {shq(args.session)} C-c || true
  sleep 10
  tmux kill-session -t {shq(args.session)} 2>/dev/null || true
fi
"""
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  exit 3
fi
"""
    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{monitor_script.rstrip()}
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
git config --global --add safe.directory {shq(args.project_root)} || true
if [[ -n "$(git status --porcelain)" ]]; then
  git stash push -u -m {shq("auto-stash before semi_mdg testas launch " + args.task_name)} || true
fi
git fetch origin {shq("+refs/heads/" + args.branch + ":refs/remotes/origin/" + args.branch)}
git checkout -B {shq(args.branch)} {shq("origin/" + args.branch)}
git reset --hard {shq("origin/" + args.branch)}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(run_file)} <<'CATK_RUN'
{run_script.rstrip()}
CATK_RUN
chmod +x {shq(run_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq("cat >> " + tmux_log)}
{monitor_block}
echo "[launcher] started tmux session {args.session} on {args.pod}"
echo "[launcher] task_name: {args.task_name}"
echo "[launcher] tmux log: {tmux_log}"
echo "[launcher] attach: tmux attach -t {args.session}"
"""


def render_remote_stop_command(args: argparse.Namespace) -> str:
    return f"""set +e
TASK_NAME_TO_STOP={shq(args.task_name)}
SESSION_TO_STOP={shq(args.session)}
if tmux has-session -t "$SESSION_TO_STOP" 2>/dev/null; then
  tmux send-keys -t "$SESSION_TO_STOP" C-c || true
  sleep 20
fi
mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 20
  mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
tmux kill-session -t "$SESSION_TO_STOP" 2>/dev/null || true
echo "[launcher] stop completed for task=$TASK_NAME_TO_STOP session=$SESSION_TO_STOP"
"""


def exec_in_pod(args: argparse.Namespace, script: str) -> None:
    cmd = [
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
        print("kubectl " + " ".join(shq(part) for part in cmd))
        return
    run_kubectl(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch semi_mdg MDG-style pretrain on testas A100x7.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH") or current_branch())
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--train-sidecar-dir",
        default="",
        help="Optional Semi-MDG training sidecar directory. Missing sidecars fail fast.",
    )
    parser.add_argument("--remote-log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=default_task_name())
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--nproc-per-node", type=int, default=7)
    parser.add_argument("--initial-bs", type=int, default=20)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", default="12")
    parser.add_argument("--max-epochs", default="64")
    parser.add_argument("--learning-rate", default="")
    parser.add_argument(
        "--auto-sqrt-lr",
        action="store_true",
        help=(
            "Recompute CATK_LR for each OOM retry attempt as "
            "base_lr * sqrt((train_batch_size * nproc_per_node) / base_global_batch_size)."
        ),
    )
    parser.add_argument("--base-lr", default="0.0006")
    parser.add_argument("--base-global-batch-size", default="108")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--master-port", default="29541")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.initial_bs < 1 or args.min_bs < 1:
        parser.error("--initial-bs and --min-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.auto_sqrt_lr and args.learning_rate:
        parser.error("--auto-sqrt-lr and --learning-rate are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_remote_stop_command(args))
        return
    print(f"[launcher] pod:       {args.pod}")
    print(f"[launcher] branch:    {args.branch}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] cache:     {args.cache_root}")
    print(f"[launcher] bs:        {args.initial_bs} -> min {args.min_bs} step {args.oom_step}")
    exec_in_pod(args, render_remote_start_command(args))
    print(
        "\nAttach:\n"
        f"  kubectl exec -it -n {args.namespace} {args.pod} -c {args.container} -- "
        f"tmux attach -t {args.session}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
