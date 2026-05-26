#!/usr/bin/env python3
"""Launch Waymo validation submission from the best Fast-RMM sweep checkpoint.

This launcher targets the existing ``hsb-npc-training-1`` H100x6 pod. It never
creates, deletes, or restarts pods. By default it reads the summary produced by
``launch_fast_rmm_epoch_sweep_h100x6_hsb1_static_pod.py``, resolves the
``BEST_BY_RMM`` epoch to the downloaded checkpoint path in that sweep manifest,
and starts one tmux session that runs full validation submission export plus
Waymo auto-upload.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "hsb-npc-training-1"
DEFAULT_BRANCH = "semi_control_rolling"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_REMOTE_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_SWEEP_NAME = "fast_rmm_epoch_sweep_h100x6_hsb1"
DEFAULT_SESSION = "catk-flow-waymo-val-submission-h100x6-hsb1"
DEFAULT_TASK_NAME = "flow_control_waymo_val_best_rmm_h100x6_hsb1"
DEFAULT_EXPERIMENT = "sim_agents_sub_flow"
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29890"
DEFAULT_NPROC_PER_NODE = 6
DEFAULT_VAL_BATCH_SIZE = 48
DEFAULT_N_ROLLOUT_CLOSED_VAL = 32


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


def remote_git_prepare_script(args: argparse.Namespace) -> str:
    fetch_refspec = f"+{args.branch}:refs/remotes/origin/{args.branch}"
    remote_ref = f"refs/remotes/origin/{args.branch}"
    clean_remote_ref = f"""
git update-ref -d {shq(remote_ref)} 2>/dev/null || rm -f .git/{shq(remote_ref)}
"""
    if args.git_ref:
        return f"""
git config --global --add safe.directory {shq(args.project_root)} || true
{clean_remote_ref}
git fetch origin --prune {shq(fetch_refspec)}
git checkout -f {shq(args.git_ref)}
"""
    if args.no_pull:
        return f"git config --global --add safe.directory {shq(args.project_root)} || true"
    return f"""
git config --global --add safe.directory {shq(args.project_root)} || true
{clean_remote_ref}
git fetch origin --prune {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq('refs/heads/' + args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq('origin/' + args.branch)}
fi
git pull --ff-only origin {shq(args.branch)}
"""


def render_worker_script(args: argparse.Namespace) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.task_name}"
    status_file = f"{run_root}/{args.pod}.status"
    tags = (
        f"[waymo_submission,h100x6,{args.branch},{args.task_name},"
        "best_fast_rmm_epoch,validation]"
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
export CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL="${{CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL:-1}}"

PROJECT_ROOT={shq(args.project_root)}
CACHE_ROOT={shq(args.cache_root)}
RUN_ROOT={shq(run_root)}
STATUS_FILE={shq(status_file)}
CKPT_PATH={shq(args.ckpt_path)}
TASK_NAME={shq(args.task_name)}
EXPERIMENT={shq(args.experiment)}
MASTER_ADDR={shq(args.master_addr)}
MASTER_PORT={shq(args.master_port)}
NPROC_PER_NODE={int(args.nproc_per_node)}
VAL_BATCH_SIZE={int(args.val_batch_size)}
N_ROLLOUT_CLOSED_VAL={int(args.n_rollout_closed_val)}
WAYMO_STORAGE_STATE_PATH={shq(args.waymo_storage_state_path)}
WAYMO_SUBMISSION_EVALUATION_SET={shq(args.evaluation_set)}

mkdir -p "$RUN_ROOT" "$(dirname "$STATUS_FILE")"
touch "$STATUS_FILE"
cd "$PROJECT_ROOT"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint does not exist: $CKPT_PATH" | tee -a "$STATUS_FILE" >&2
  exit 2
fi
if [[ ! -f "$WAYMO_STORAGE_STATE_PATH" ]]; then
  echo "ERROR: Waymo storage state does not exist: $WAYMO_STORAGE_STATE_PATH" | tee -a "$STATUS_FILE" >&2
  exit 2
fi

echo "[$(date '+%F %T')] Waymo validation submission start" | tee -a "$STATUS_FILE"
echo "task_name=$TASK_NAME" | tee -a "$STATUS_FILE"
echo "ckpt_path=$CKPT_PATH" | tee -a "$STATUS_FILE"
echo "branch={args.branch} commit=$(git rev-parse --short HEAD)" | tee -a "$STATUS_FILE"

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
export CATK_CKPT_PATH="$CKPT_PATH"
export TASK_NAME
export LOG_DIR="$RUN_ROOT"
export VAL_BATCH_SIZE
  export CATK_HYDRA_OVERRIDES="trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy +trainer.strategy.cluster_environment._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousTorchElasticEnvironment model.model_config.val_closed_loop=true model.model_config.val_open_loop=false model.model_config.n_rollout_closed_val=${{N_ROLLOUT_CLOSED_VAL}} model.model_config.sim_agents_submission.is_active=true waymo_submission.enabled=true waymo_submission.submit_validate=true waymo_submission.submit_test=false waymo_submission.evaluation_set=${{WAYMO_SUBMISSION_EVALUATION_SET}} waymo_submission.storage_state_path=${{WAYMO_STORAGE_STATE_PATH}} logger.wandb.group={args.wandb_group} logger.wandb.job_type=waymo_submission logger.wandb.tags={tags} logger.wandb.log_model=false"

set +e
bash scripts/h100x4_multinode_pretrain.sh 2>&1 | tee "$RUN_ROOT/{args.pod}.log"
status="${{PIPESTATUS[0]}}"
set -e
if [[ "$status" != "0" ]]; then
  echo "[$(date '+%F %T')] Waymo validation submission failed status=$status" | tee -a "$STATUS_FILE"
  exit "$status"
fi
echo "[$(date '+%F %T')] Waymo validation submission complete" | tee -a "$STATUS_FILE"
"""


def render_remote_resolve_ckpt_script(args: argparse.Namespace) -> str:
    sweep_root = f"{args.remote_log_dir.rstrip('/')}/{args.sweep_name}"
    summary_file = f"{sweep_root}/epoch_sweep_summary.txt"
    manifest_file = f"{sweep_root}/epoch_sweep_manifest.tsv"
    return f"""python - <<'PY'
from __future__ import annotations

import csv
import re
from pathlib import Path

summary = Path({summary_file!r})
manifest = Path({manifest_file!r})
if not summary.is_file():
    raise SystemExit("missing sweep summary: " + str(summary))
if not manifest.is_file():
    raise SystemExit("missing sweep manifest: " + str(manifest))
text = summary.read_text(errors="ignore")
match = re.search(r"^BEST_BY_RMM\\t.*?epoch=(\\d+)\\b", text, flags=re.MULTILINE)
if match is None:
    raise SystemExit("BEST_BY_RMM line with epoch=... was not found in " + str(summary))
best_epoch = int(match.group(1))
with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if int(row["epoch"]) == best_epoch:
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.is_file():
                raise SystemExit("best checkpoint path does not exist: " + str(checkpoint))
            print(checkpoint.as_posix())
            raise SystemExit(0)
raise SystemExit("best epoch %d was not found in manifest %s" % (best_epoch, manifest))
PY
"""


def resolve_ckpt_path(args: argparse.Namespace) -> str:
    if args.ckpt_path:
        return args.ckpt_path
    script = render_remote_resolve_ckpt_script(args)
    output = run_kubectl(
        [
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
        ],
        capture=True,
        dry_run=args.dry_run,
    )
    return output.splitlines()[-1].strip() if output else "<resolved-from-sweep-summary>"


def render_start_command(args: argparse.Namespace) -> str:
    run_root = f"{args.remote_log_dir.rstrip('/')}/{args.task_name}"
    script_path = f"{run_root}/{args.pod}_run_waymo_submission.sh"
    worker = render_worker_script(args)
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
    training_guard = "" if args.allow_while_training else f"""
if pgrep -af 'python .*src\\.run|torchrun .*src\\.run' 2>/dev/null | grep -E 'action=fit|pretrain|control_space_pretrain' >/dev/null; then
  echo "[launcher] active pretrain process is still running on {args.pod}; refusing to start submission." >&2
  echo "[launcher] pass --allow-while-training only if you intentionally want to share GPUs." >&2
  exit 4
fi
"""
    return f"""set -Eeuo pipefail
if [[ ! -d {shq(args.project_root)}/.git ]]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{remote_git_prepare_script(args)}
{training_guard}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(script_path)} <<'CATK_WAYMO_SUBMISSION'
{worker.rstrip()}
CATK_WAYMO_SUBMISSION
chmod +x {shq(script_path)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(script_path)}
echo "[launcher] started {args.session} on {args.pod}"
echo "[launcher] script: {script_path}"
"""


def render_stop_command(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
tmux kill-session -t {shq(args.session)} 2>/dev/null || true
mapfile -t pids < <(
  pgrep -f {shq(args.task_name)} 2>/dev/null | while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
      echo "$pid"
    fi
  done
)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating submission processes: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(
    pgrep -f {shq(args.task_name)} 2>/dev/null | while read -r pid; do
      if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
        echo "$pid"
      fi
    done
  )
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing submission processes: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
echo "[launcher] stopped session/processes for {args.task_name}"
"""


def exec_in_pod(args: argparse.Namespace, script: str) -> None:
    run_kubectl(
        [
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
        ],
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Waymo validation submission on hsb-npc-training-1 H100x6 from "
            "the best Fast-RMM epoch sweep checkpoint."
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
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--ckpt-path", default=os.environ.get("CKPT_PATH", ""))
    parser.add_argument("--master-addr", default=DEFAULT_MASTER_ADDR)
    parser.add_argument("--master-port", default=DEFAULT_MASTER_PORT)
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--n-rollout-closed-val", type=int, default=DEFAULT_N_ROLLOUT_CLOSED_VAL)
    parser.add_argument("--wandb-group", default="waymo_submission_best_rmm_h100x6_hsb1")
    parser.add_argument(
        "--waymo-storage-state-path",
        default=os.environ.get(
            "WAYMO_STORAGE_STATE_PATH",
            "",
        ),
    )
    parser.add_argument("--evaluation-set", default="validation", choices=("validation", "test"))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-while-training", action="store_true")
    args = parser.parse_args()
    if args.nproc_per_node != 6:
        parser.error("this hsb1 launcher expects --nproc-per-node 6")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.n_rollout_closed_val < 1:
        parser.error("--n-rollout-closed-val must be >= 1")
    if args.evaluation_set != "validation":
        parser.error("automatic Waymo upload is limited to validation in this launcher")
    if not args.waymo_storage_state_path:
        args.waymo_storage_state_path = (
            f"{args.project_root.rstrip('/')}/secrets/waymo/waymo_storage_state.json"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args))
        return
    args.ckpt_path = resolve_ckpt_path(args)
    print(f"[launcher] pod: {args.pod}")
    print(f"[launcher] checkpoint: {args.ckpt_path}")
    print(f"[launcher] task: {args.task_name}")
    print(f"[launcher] val_batch_size: {args.val_batch_size}")
    print(f"[launcher] n_rollout_closed_val: {args.n_rollout_closed_val}")
    exec_in_pod(args, render_start_command(args))
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
