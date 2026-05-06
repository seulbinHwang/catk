#!/usr/bin/env python3
"""Launch DRaFT fine-tuning on existing V100x3x5 static pods.

This launcher never creates, deletes, or restarts pods. It only uses
``kubectl exec`` to start or stop a tmux session inside already-running pods.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["fv", "fvv", "fvvv", "fvvvv", "fvvvvv"]
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing_w_track_loss"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "finetune_draft_flow_v100x3x5"
DEFAULT_TASK_NAME = "flow_finetune_draft_v100x3x5_bs24_soft1_topk20_commit1_noslip"
DEFAULT_SESSION = "catk-draft-v100x3x5-bs24"


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


def pod_ip(namespace: str, pod: str) -> str:
    return run_kubectl(
        ["get", "pod", pod, "-n", namespace, "-o", "jsonpath={.status.podIP}"],
        capture=True,
    )


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_static_multinode/{safe_task}"


def render_env(args: argparse.Namespace, *, rank: int, master_addr: str) -> str:
    lines = [
        export_line("PROJECT_ROOT", args.project_root),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CKPT_PATH", args.ckpt_path),
        export_line("WANDB_ARTIFACT", args.wandb_artifact),
        export_line("ARTIFACT_DOWNLOAD_DIR", args.artifact_download_dir),
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("NNODES", len(args.pods)),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("NODE_RANK", rank),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("LOG_DIR", args.log_dir),
        export_line("RUN_ROOT", run_root(args)),
        export_line("TRAIN_BATCH_SIZE", args.train_batch_size),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("PRECISION", args.precision),
        export_line("CATK_LR", args.learning_rate),
        export_line("CATK_EXTRA_OVERRIDES", args.extra_hydra_overrides),
    ]
    return "\n".join(lines) + "\n"


def render_worker_script(env_file: str) -> str:
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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${{TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}}"
export TORCH_NCCL_BLOCKING_WAIT="${{TORCH_NCCL_BLOCKING_WAIT:-0}}"

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

set -a
source {shq(env_file)}
set +a

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"

echo "[draft-v100x3x5] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[draft-v100x3x5] started at $(date '+%F %T')"
echo "[draft-v100x3x5] experiment=${{EXPERIMENT}} bs=${{TRAIN_BATCH_SIZE}} precision=${{PRECISION}}"
echo "[draft-v100x3x5] ckpt_path=${{CKPT_PATH}}"
echo "[draft-v100x3x5] attach survives after exit; press Ctrl-b d to detach"
echo

ensure_checkpoint() {{
  if [[ -f "$CKPT_PATH" ]]; then
    echo "[draft-v100x3x5] using checkpoint: $CKPT_PATH"
    return 0
  fi
  if [[ -z "$WANDB_ARTIFACT" ]]; then
    echo "[draft-v100x3x5] ERROR: checkpoint not found and WANDB_ARTIFACT is empty: $CKPT_PATH" >&2
    return 2
  fi

  local download_dir="${{ARTIFACT_DOWNLOAD_DIR:-$(dirname "$CKPT_PATH")/artifact}}"
  mkdir -p "$(dirname "$CKPT_PATH")" "$download_dir"
  local lock_dir="${{CKPT_PATH}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[draft-v100x3x5] downloading W&B artifact: $WANDB_ARTIFACT"
    python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_ARTIFACT"]
download_dir = os.environ["ARTIFACT_DOWNLOAD_DIR"] or str(Path(os.environ["CKPT_PATH"]).parent / "artifact")
target_ckpt = os.environ["CKPT_PATH"]

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
print(f"Downloaded checkpoint: {{target_ckpt}}")
PY
    status=$?
    rm -rf "$lock_dir"
    return "$status"
  fi

  echo "[draft-v100x3x5] waiting for checkpoint download lock: $lock_dir"
  for _ in $(seq 1 180); do
    if [[ -f "$CKPT_PATH" ]]; then
      echo "[draft-v100x3x5] checkpoint appeared: $CKPT_PATH"
      return 0
    fi
    sleep 10
  done
  echo "[draft-v100x3x5] timed out waiting for $CKPT_PATH" >&2
  return 4
}}

ensure_checkpoint || exit $?

extra_overrides=()
if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
  read -r -a extra_overrides <<< "$CATK_EXTRA_OVERRIDES"
fi

torchrun_args=(
  --nnodes "$NNODES"
  --nproc_per_node "$NPROC_PER_NODE"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  -m src.run
  experiment="$EXPERIMENT"
  action=finetune
  trainer=ddp
  trainer.devices="$NPROC_PER_NODE"
  trainer.num_nodes="$NNODES"
  trainer.precision="$PRECISION"
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$LOG_DIR"
  task_name="$TASK_NAME"
  ckpt_path="$CKPT_PATH"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.test_batch_size="$TEST_BATCH_SIZE"
  model.model_config.lr="$CATK_LR"
  model.model_config.draft.enabled=true
  model.model_config.draft.loss_enabled=true
  model.model_config.draft.physics.soft_limit_ratio=1.0
  model.model_config.draft.physics.topk_violation_k=20
  model.model_config.draft.physics.commit_loss_weight=1.0
  model.model_config.draft.physics.use_slip_penalty=false
)
torchrun_args+=("${{extra_overrides[@]}}")

printf '[draft-v100x3x5] torchrun'
printf ' %q' "${{torchrun_args[@]}}"
printf '\\n'

exec torchrun "${{torchrun_args[@]}}"
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


def render_start_command(
    args: argparse.Namespace,
    *,
    pod: str,
    rank: int,
    master_addr: str,
) -> str:
    root = run_root(args)
    env_file = f"{root}/{pod}.env"
    worker_file = f"{root}/{pod}_worker.sh"
    monitor_file = f"{root}/{pod}_monitor.sh"
    tmux_log = f"{root}/{pod}.tmux.log"

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

    replace_block = ""
    if args.replace:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
"""
    else:
        replace_block = f"""
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
{replace_block}
mkdir -p {shq(root)}
cat > {shq(env_file)} <<'CATK_ENV'
{render_env(args, rank=rank, master_addr=master_addr).rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(env_file).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.session} on {pod}"
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


def exec_in_pod(args: argparse.Namespace, pod: str, script: str) -> None:
    command = [
        "exec",
        "-n",
        args.namespace,
        pod,
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
        description="Launch DRaFT fine-tuning on existing V100x3x5 static pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument(
        "--wandb-artifact",
        default="",
        help="Optional W&B artifact full name to download if --ckpt-path is missing.",
    )
    parser.add_argument(
        "--artifact-download-dir",
        default="",
        help="Optional artifact download directory. Defaults to dirname(--ckpt-path)/artifact.",
    )
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29573")
    parser.add_argument("--nproc-per-node", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--test-batch-size", type=int, default=2)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--learning-rate", default="2.0e-4")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if len(args.pods) != 5:
        parser.error("--pods must contain exactly five pods for the V100x3x5 preset")
    if args.nproc_per_node != 3:
        parser.error("--nproc-per-node must be 3 for the V100x3x5 preset")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.test_batch_size < 1:
        parser.error("--test-batch-size must be >= 1")
    if not args.ckpt_path:
        parser.error("--ckpt-path is required unless --stop is set")
    return args


def main() -> None:
    args = parse_args()

    if args.stop:
        for pod in args.pods:
            exec_in_pod(args, pod, render_stop_command(args.session))
        return

    master_addr = args.master_addr or (
        "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    )
    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] ckpt path: {args.ckpt_path}")
    print(f"[launcher] batch:     train={args.train_batch_size} val={args.val_batch_size}")

    for rank, pod in enumerate(args.pods):
        script = render_start_command(args, pod=pod, rank=rank, master_addr=master_addr)
        exec_in_pod(args, pod, script)

    print("\nAttach commands:")
    for pod in args.pods:
        print(
            f"  kubectl exec -it -n {args.namespace} {pod} "
            f"-c {args.container} -- tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
