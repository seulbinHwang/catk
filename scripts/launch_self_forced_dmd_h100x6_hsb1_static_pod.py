#!/usr/bin/env python3
"""Launch DMD-style self-forced Flow fine-tuning on hsb-npc-training-1 H100x6.

The launcher never creates, deletes, or restarts pods. It only starts/stops a
tmux session inside the existing pod, prepares the requested pretrained
Generator checkpoint, and runs the H100x6 OOM-retry training script.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "hsb-npc-training-1"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_h100_6"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-mqfq3u39:v121"
)
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_self_forced_dmd_h100x6_hsb1_pretrain_epoch116_mqfq3u39/"
    "v121/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_self_forced_dmd_h100x6_hsb1_pretrain_epoch116_mqfq3u39/"
    "v121/artifact"
)
DEFAULT_TASK_NAME = (
    "flow_self_forced_dmd_h100x6_hsb1_epoch116_activecontrol_"
    "sample16_backprop8_lr1e-6_bs21_oomretry"
)
DEFAULT_SESSION = "catk-self-forced-dmd-h100x6-hsb1"


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


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_self_forced_dmd_h100x6_hsb1/{safe_task}"


def render_env(args: argparse.Namespace) -> str:
    lines = [
        export_line("PRETRAIN_CKPT", args.pretrain_ckpt),
        export_line("WANDB_PRETRAIN_ARTIFACT", args.wandb_pretrain_artifact),
        export_line("WANDB_PRETRAIN_DOWNLOAD_DIR", args.pretrain_download_dir),
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("INITIAL_BS", args.initial_bs),
        export_line("OOM_STEP", args.oom_step),
        export_line("MIN_BS", args.min_bs),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("CATK_LR", args.learning_rate),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("MAX_EPOCHS", args.max_epochs),
        export_line("CHECK_VAL_EVERY_N_EPOCH", args.check_val_every_n_epoch),
        export_line("LIMIT_VAL_BATCHES", args.limit_val_batches),
        export_line("TRAIN_EPOCH_SAMPLE_FRACTION", args.train_epoch_sample_fraction),
        export_line("RANDOM_TERMINAL_SCOPE", args.random_terminal_scope),
        export_line("RANDOM_TERMINAL_POLICY", args.random_terminal_policy),
        export_line("BACKPROP_LAST_K", args.backprop_last_k),
    ]
    optional = {
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "EMA_WEIGHT": args.ema_weight,
        "EMA_START_STEP": args.ema_start_step,
        "CLEAN_DMD_NORMALIZER_EPS": args.clean_dmd_normalizer_eps,
        "CLEAN_DMD_TAU_LOW": args.clean_dmd_tau_low,
        "CLEAN_DMD_TAU_HIGH": args.clean_dmd_tau_high,
        "CATK_EXTRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
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

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

echo "[self-forced-dmd-h100x6-hsb1] pod=$(hostname) task=${{TASK_NAME}}"
echo "[self-forced-dmd-h100x6-hsb1] started at $(date '+%F %T')"
echo "[self-forced-dmd-h100x6-hsb1] experiment=${{EXPERIMENT}} initial_bs=${{INITIAL_BS}} oom_step=${{OOM_STEP}}"
echo "[self-forced-dmd-h100x6-hsb1] lr=${{CATK_LR}} sample_steps=16 backprop_last_k=${{BACKPROP_LAST_K}} policy=${{RANDOM_TERMINAL_POLICY}}"
echo "[self-forced-dmd-h100x6-hsb1] pretrain_artifact=${{WANDB_PRETRAIN_ARTIFACT}}"
echo "[self-forced-dmd-h100x6-hsb1] pretrain_ckpt=${{PRETRAIN_CKPT}}"
echo "[self-forced-dmd-h100x6-hsb1] attach survives after exit; press Ctrl-b d to detach"
echo

ensure_pretrain_checkpoint() {{
  if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[self-forced-dmd-h100x6-hsb1] using cached pretrain checkpoint: $PRETRAIN_CKPT"
    return 0
  fi

  mkdir -p "$(dirname "$PRETRAIN_CKPT")" "$WANDB_PRETRAIN_DOWNLOAD_DIR"
  lock_dir="${{PRETRAIN_CKPT}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[self-forced-dmd-h100x6-hsb1] downloading W&B artifact: $WANDB_PRETRAIN_ARTIFACT"
    python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_PRETRAIN_ARTIFACT"]
download_dir = os.environ["WANDB_PRETRAIN_DOWNLOAD_DIR"]
target_ckpt = os.environ["PRETRAIN_CKPT"]

try:
    import wandb
except Exception as exc:
    print(f"ERROR: failed to import wandb: {{exc}}", file=sys.stderr)
    sys.exit(2)

Path(download_dir).mkdir(parents=True, exist_ok=True)
Path(target_ckpt).parent.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact_dir = artifact.download(root=download_dir)

candidates = []
preferred = Path(artifact_dir) / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred.as_posix())
candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))

if not candidates:
    print(f"ERROR: no checkpoint file found in artifact dir: {{artifact_dir}}", file=sys.stderr)
    sys.exit(3)

source = candidates[0]
if os.path.abspath(source) != os.path.abspath(target_ckpt):
    shutil.copy2(source, target_ckpt)
print(f"Downloaded pretrain checkpoint: {{target_ckpt}}")
PY
    status=$?
    rm -rf "$lock_dir"
    if (( status != 0 )); then
      echo "[self-forced-dmd-h100x6-hsb1] W&B artifact download failed with status $status" >&2
      return "$status"
    fi
  else
    echo "[self-forced-dmd-h100x6-hsb1] waiting for checkpoint download lock: $lock_dir"
    for _ in $(seq 1 180); do
      if [[ -f "$PRETRAIN_CKPT" ]]; then
        echo "[self-forced-dmd-h100x6-hsb1] checkpoint appeared: $PRETRAIN_CKPT"
        return 0
      fi
      sleep 10
    done
    echo "[self-forced-dmd-h100x6-hsb1] timed out waiting for $PRETRAIN_CKPT" >&2
    return 4
  fi

  test -f "$PRETRAIN_CKPT"
}}

ensure_pretrain_checkpoint
status=$?
if (( status != 0 )); then
  echo "[self-forced-dmd-h100x6-hsb1] checkpoint preparation failed with status $status"
  echo "[self-forced-dmd-h100x6-hsb1] leaving shell open for inspection"
  exec bash
fi

bash scripts/self_forced_h100_6_with_oom_retry.sh
status=$?

echo
echo "[self-forced-dmd-h100x6-hsb1] exited with status $status at $(date '+%F %T')"
echo "[self-forced-dmd-h100x6-hsb1] leaving shell open for inspection"
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
git fetch origin {shq(args.branch)}:refs/remotes/origin/{shq(args.branch)}
if git show-ref --verify --quiet refs/heads/{shq(args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} origin/{shq(args.branch)}
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
tmux select-pane -t {shq(args.session)}:0.0
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
    run_kubectl(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch DMD self-forced H100x6 training on hsb-npc-training-1.",
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
    parser.add_argument("--wandb-pretrain-artifact", default=DEFAULT_WANDB_PRETRAIN_ARTIFACT)
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--pretrain-download-dir", default=DEFAULT_PRETRAIN_DOWNLOAD_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5")
    parser.add_argument("--nproc-per-node", type=int, default=6)
    parser.add_argument("--initial-bs", type=int, default=21)
    parser.add_argument("--oom-step", type=int, default=1)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", default="16")
    parser.add_argument("--test-batch-size", default="16")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--max-epochs", default="12")
    parser.add_argument("--check-val-every-n-epoch", default="2")
    parser.add_argument("--learning-rate", default="1.0e-6")
    parser.add_argument("--train-epoch-sample-fraction", default="0.5")
    parser.add_argument("--random-terminal-scope", default="global_batch")
    parser.add_argument("--random-terminal-policy", default="all")
    parser.add_argument("--backprop-last-k", default="8")
    parser.add_argument("--ema-weight", default="")
    parser.add_argument("--ema-start-step", default="")
    parser.add_argument("--clean-dmd-normalizer-eps", default="")
    parser.add_argument("--clean-dmd-tau-low", default="")
    parser.add_argument("--clean-dmd-tau-high", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if args.nproc_per_node != 6:
        parser.error("--nproc-per-node must be 6 for the H100x6 preset")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.random_terminal_scope not in {"", "global_batch"}:
        parser.error("--random-terminal-scope must be empty or 'global_batch'")
    if args.random_terminal_policy not in {"", "all", "paper_uniform"}:
        parser.error("--random-terminal-policy must be empty, 'all', or 'paper_uniform'")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if not args.pretrain_ckpt:
        parser.error("--pretrain-ckpt must not be empty unless --stop is set")
    if not args.wandb_pretrain_artifact:
        parser.error("--wandb-pretrain-artifact must not be empty unless --stop is set")
    if not args.pretrain_download_dir:
        parser.error("--pretrain-download-dir must not be empty unless --stop is set")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session))
        return

    print(f"[launcher] pod:       {args.pod}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] experiment:{args.experiment}")
    print(f"[launcher] artifact:  {args.wandb_pretrain_artifact}")
    print(f"[launcher] ckpt path: {args.pretrain_ckpt}")
    print(f"[launcher] bs fallback: {args.initial_bs}->{args.min_bs} step {args.oom_step}")

    exec_in_pod(args, render_start_command(args))

    print("\nAttach command:")
    print(
        f"  kubectl exec -it -n {args.namespace} {args.pod} "
        f"-c {args.container} -- tmux attach -t {args.session}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
