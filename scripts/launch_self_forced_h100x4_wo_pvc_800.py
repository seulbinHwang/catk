#!/usr/bin/env python3
"""Launch self-forced H100x4 training in tmux on the existing wo-pvc-800 pod.

This launcher never creates, deletes, or restarts pods. It only uses
``kubectl exec`` to start or stop a tmux session inside the already-running
pod. Use ``--dry-run`` to render the command without touching the pod.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "wo-pvc-800"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing_w_track_loss"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_h100x4_wo_pvc_800"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64"
)
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "v64/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "v64/artifact"
)
DEFAULT_TASK_NAME = (
    "flow_self_forced_h100x4_wo_pvc_800_"
    "use_stop_motion_false_estimator_warmup_4_lr1e-6_bs28"
)
DEFAULT_SESSION = "catk-sf-h100x4-wo-pvc-800"


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
    return f"{args.log_dir.rstrip('/')}/tmux_self_forced_h100x4_wo_pvc_800/{safe_task}"


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
        export_line("ESTIMATOR_WARMUP_EPOCHS", args.estimator_warmup_epochs),
        export_line("SELF_FORCED_USE_STOP_MOTION", args.self_forced_use_stop_motion),
    ]
    optional = {
        "VAL_BATCH_SIZE": args.val_batch_size,
        "TEST_BATCH_SIZE": args.test_batch_size,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "CHECK_VAL_EVERY_N_EPOCH": args.check_val_every_n_epoch,
        "UNFROZEN_RANGE": args.unfrozen_range,
        "DECODER_USE_STOP_MOTION": args.decoder_use_stop_motion,
        "RANDOM_TERMINAL_SCOPE": args.random_terminal_scope,
        "RANDOM_TERMINAL_POLICY": args.random_terminal_policy,
        "BACKPROP_LAST_K": args.backprop_last_k,
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

echo "[self-forced-h100x4-wo-pvc-800] pod=$(hostname) task=${{TASK_NAME}}"
echo "[self-forced-h100x4-wo-pvc-800] started at $(date '+%F %T')"
echo "[self-forced-h100x4-wo-pvc-800] experiment=${{EXPERIMENT}} initial_bs=${{INITIAL_BS}}"
echo "[self-forced-h100x4-wo-pvc-800] lr=${{CATK_LR}} estimator_warmup=${{ESTIMATOR_WARMUP_EPOCHS}} self_forced_use_stop_motion=${{SELF_FORCED_USE_STOP_MOTION}}"
echo "[self-forced-h100x4-wo-pvc-800] pretrain_artifact=${{WANDB_PRETRAIN_ARTIFACT}}"
echo "[self-forced-h100x4-wo-pvc-800] pretrain_ckpt=${{PRETRAIN_CKPT}}"
echo "[self-forced-h100x4-wo-pvc-800] attach survives after exit; press Ctrl-b d to detach"
echo

ensure_pretrain_checkpoint() {{
  if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[self-forced-h100x4-wo-pvc-800] using cached pretrain checkpoint: $PRETRAIN_CKPT"
    return 0
  fi

  mkdir -p "$(dirname "$PRETRAIN_CKPT")" "$WANDB_PRETRAIN_DOWNLOAD_DIR"
  lock_dir="${{PRETRAIN_CKPT}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[self-forced-h100x4-wo-pvc-800] downloading W&B artifact: $WANDB_PRETRAIN_ARTIFACT"
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
    print("ERROR: failed to import wandb: {{}}".format(exc), file=sys.stderr)
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
    print("ERROR: no checkpoint file found in artifact dir: {{}}".format(artifact_dir), file=sys.stderr)
    sys.exit(3)

source = candidates[0]
if os.path.abspath(source) != os.path.abspath(target_ckpt):
    shutil.copy2(source, target_ckpt)
print("Downloaded pretrain checkpoint: {{}}".format(target_ckpt))
PY
    status=$?
    rm -rf "$lock_dir"
    if (( status != 0 )); then
      echo "[self-forced-h100x4-wo-pvc-800] W&B artifact download failed with status $status" >&2
      return "$status"
    fi
  else
    echo "[self-forced-h100x4-wo-pvc-800] waiting for checkpoint download lock: $lock_dir"
    for _ in $(seq 1 180); do
      if [[ -f "$PRETRAIN_CKPT" ]]; then
        echo "[self-forced-h100x4-wo-pvc-800] checkpoint appeared: $PRETRAIN_CKPT"
        return 0
      fi
      sleep 10
    done
    echo "[self-forced-h100x4-wo-pvc-800] timed out waiting for $PRETRAIN_CKPT" >&2
    return 4
  fi

  test -f "$PRETRAIN_CKPT"
}}

ensure_pretrain_checkpoint
status=$?
if (( status != 0 )); then
  echo "[self-forced-h100x4-wo-pvc-800] checkpoint preparation failed with status $status"
  echo "[self-forced-h100x4-wo-pvc-800] leaving shell open for inspection"
  exec bash
fi

bash scripts/self_forced_h100_4_with_oom_retry.sh
status=$?

echo
echo "[self-forced-h100x4-wo-pvc-800] exited with status $status at $(date '+%F %T')"
echo "[self-forced-h100x4-wo-pvc-800] leaving shell open for inspection"
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
        branch_ref = f"refs/heads/{args.branch}"
        origin_ref = f"origin/{args.branch}"
        fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq(branch_ref)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq(origin_ref)}
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
        description="Launch H100x4 self-forced training on the existing wo-pvc-800 pod.",
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
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--initial-bs", type=int, default=28)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", default="")
    parser.add_argument("--test-batch-size", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="10")
    parser.add_argument("--check-val-every-n-epoch", default="2")
    parser.add_argument("--learning-rate", default="1.0e-6")
    parser.add_argument("--estimator-warmup-epochs", default="4")
    parser.add_argument("--self-forced-use-stop-motion", default="false")
    parser.add_argument("--decoder-use-stop-motion", default="")
    parser.add_argument("--unfrozen-range", default="")
    parser.add_argument("--random-terminal-scope", default="")
    parser.add_argument("--random-terminal-policy", default="")
    parser.add_argument("--backprop-last-k", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if args.nproc_per_node != 4:
        parser.error("--nproc-per-node must be 4 for the H100x4 preset")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.self_forced_use_stop_motion not in {"true", "false"}:
        parser.error("--self-forced-use-stop-motion must be 'true' or 'false'")
    if args.decoder_use_stop_motion not in {"", "true", "false"}:
        parser.error("--decoder-use-stop-motion must be empty, 'true', or 'false'")
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
