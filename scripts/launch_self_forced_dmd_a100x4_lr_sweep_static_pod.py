#!/usr/bin/env python3
"""Launch a sequential self-forced DMD LR sweep on one existing A100x4 pod.

The launcher never creates, deletes, or restarts pods. It starts a tmux session
inside an already-running pod and lets the existing A100 OOM-retry wrapper run
one task per learning rate.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_a100x4_testa"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57"
)

COMMON_EXTRA_OVERRIDES = [
    "model.model_config.val_open_loop=false",
    "model.model_config.decoder.detach_train_metric_clean=true",
    "model.model_config.self_forced.distribution_matching_objective=dmd",
    "model.model_config.self_forced.clean_dmd_normalizer_eps=0.05",
    "model.model_config.self_forced.clean_dmd_tau_low=0.02",
    "model.model_config.self_forced.clean_dmd_tau_high=0.98",
    "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch",
    "model.model_config.self_forced.sampling.random_terminal_step.policy=all",
    "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=16",
    "model.model_config.self_forced.sampling.backprop_last_k=8",
    "model.model_config.self_forced.dmd_use_stable_scale_filter=true",
    "model.model_config.self_forced.dmd_stable_scale_scope=agent",
    "model.model_config.self_forced.dmd_use_teacher_alignment_filter=false",
    "model.model_config.self_forced.dmd_use_trust_region_filter=false",
    "model.model_config.self_forced.dmd_use_injection_ramp=false",
    "model.model_config.self_forced.project_dmd_to_pose_space=false",
]


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, dry_run: bool = False) -> None:
    command = ["kubectl", *args]
    if dry_run:
        print("+ " + " ".join(shq(part) for part in command))
        return
    subprocess.run(command, check=True)


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def lr_tag(lr: str) -> str:
    return lr.replace(".", "p").replace("-", "m").replace("+", "p")


def default_pretrain_ckpt(pod: str) -> str:
    return (
        f"/workspace/flow_self_forced_dmd_a100x4_{pod}_pretrain_epoch061_x5f9g0ce/"
        "v57/epoch_061.ckpt"
    )


def default_pretrain_download_dir(pod: str) -> str:
    return (
        f"/workspace/flow_self_forced_dmd_a100x4_{pod}_pretrain_epoch061_x5f9g0ce/"
        "v57/artifact"
    )


def run_root(args: argparse.Namespace) -> str:
    return f"{args.log_dir.rstrip('/')}/lr_sweeps/{args.sweep_name}_{args.pod}"


def render_worker_script(args: argparse.Namespace) -> str:
    lrs = " ".join(shq(lr) for lr in args.lrs)
    common_overrides = " ".join(shq(item) for item in COMMON_EXTRA_OVERRIDES)
    condition_overrides = " ".join(shq(item) for item in args.extra_hydra_overrides)
    root = run_root(args)
    status_tsv = f"{root}/status.tsv"
    summary_tsv = f"{root}/summary.tsv"
    task_prefix = args.task_prefix.rstrip("_")
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

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

cd {shq(args.project_root)}
mkdir -p {shq(root)}

echo -e "timestamp\\tpod\\tlr\\ttask\\tstatus\\telapsed_sec" > {shq(status_tsv)}
echo -e "lr\\ttask\\tstatus\\telapsed_sec\\tbest_summary_hint" > {shq(summary_tsv)}

ensure_pretrain_checkpoint() {{
  if [[ -f {shq(args.pretrain_ckpt)} ]]; then
    echo "[sweep] using cached pretrain checkpoint: {args.pretrain_ckpt}"
    return 0
  fi
  mkdir -p {shq(args.pretrain_download_dir)} "$(dirname {shq(args.pretrain_ckpt)})"
  echo "[sweep] downloading W&B artifact: {args.wandb_pretrain_artifact}"
  PRETRAIN_CKPT={shq(args.pretrain_ckpt)} \
  WANDB_PRETRAIN_DOWNLOAD_DIR={shq(args.pretrain_download_dir)} \
  WANDB_PRETRAIN_ARTIFACT={shq(args.wandb_pretrain_artifact)} \
  python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

try:
    import wandb
except Exception as exc:
    print(f"ERROR: failed to import wandb: {{exc}}", file=sys.stderr)
    sys.exit(2)

artifact_name = os.environ["WANDB_PRETRAIN_ARTIFACT"]
download_dir = os.environ["WANDB_PRETRAIN_DOWNLOAD_DIR"]
target_ckpt = os.environ["PRETRAIN_CKPT"]
Path(download_dir).mkdir(parents=True, exist_ok=True)
Path(target_ckpt).parent.mkdir(parents=True, exist_ok=True)
artifact = wandb.Api().artifact(artifact_name)
artifact_dir = artifact.download(root=download_dir)
candidates = []
for name in ("epoch_061.ckpt", "epoch_last.ckpt"):
    candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / name), recursive=True))
candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))
if not candidates:
    print(f"ERROR: no checkpoint found in {{artifact_dir}}", file=sys.stderr)
    sys.exit(3)
shutil.copy2(candidates[0], target_ckpt)
print(f"Downloaded pretrain checkpoint: {{target_ckpt}}")
PY
}}

ensure_pretrain_checkpoint

COMMON_OVERRIDES={shq(common_overrides)}
CONDITION_OVERRIDES={shq(condition_overrides)}
LR_LIST=({lrs})

echo "[sweep] pod=$(hostname) sweep={args.sweep_name}"
echo "[sweep] branch={args.branch} commit=$(git rev-parse --short HEAD)"
echo "[sweep] lrs=${{LR_LIST[*]}}"
echo "[sweep] task_prefix={task_prefix}"
echo "[sweep] started at $(date '+%F %T')"

for LR in "${{LR_LIST[@]}}"; do
  TAG="${{LR//./p}}"
  TAG="${{TAG//-/m}}"
  TAG="${{TAG//+/p}}"
  export PRETRAIN_CKPT={shq(args.pretrain_ckpt)}
  export EXPERIMENT={shq(args.experiment)}
  export TASK_NAME="{task_prefix}_lr${{TAG}}"
  export CACHE_ROOT={shq(args.cache_root)}
  export CATK_LOG_DIR={shq(args.log_dir)}
  export INITIAL_BS={shq(args.initial_bs)}
  export OOM_STEP={shq(args.oom_step)}
  export MIN_BS={shq(args.min_bs)}
  export CUDA_VISIBLE_DEVICES={shq(args.cuda_visible_devices)}
  export NPROC_PER_NODE={shq(args.nproc_per_node)}
  export VAL_BATCH_SIZE={shq(args.val_batch_size)}
  export TEST_BATCH_SIZE={shq(args.test_batch_size)}
  export MAX_EPOCHS={shq(args.max_epochs)}
  export CHECK_VAL_EVERY_N_EPOCH={shq(args.check_val_every_n_epoch)}
  export LIMIT_VAL_BATCHES={shq(args.limit_val_batches)}
  export TRAIN_EPOCH_SAMPLE_FRACTION={shq(args.train_epoch_sample_fraction)}
  export TRAIN_MEMORY_BALANCED_BATCHES={shq(args.train_memory_balanced_batches)}
  export RANDOM_TERMINAL_SCOPE=global_batch
  export RANDOM_TERMINAL_POLICY=all
  export BACKPROP_LAST_K=8
  export ESTIMATOR_WARMUP_EPOCHS={shq(args.estimator_warmup_epochs)}
  export SELF_FORCED_USE_STOP_MOTION=false
  export DECODER_USE_STOP_MOTION=false
  export UNFROZEN_RANGE={shq(args.unfrozen_range)}
  export CATK_LR="$LR"
  export CATK_EXTRA_OVERRIDES="$COMMON_OVERRIDES $CONDITION_OVERRIDES"

  START_TS=$(date +%s)
  echo
  echo "[sweep] ============================================================"
  echo "[sweep] starting lr=$LR task=$TASK_NAME at $(date '+%F %T')"
  echo "[sweep] initial_bs=$INITIAL_BS min_bs=$MIN_BS oom_step=$OOM_STEP"
  echo "[sweep] overrides=$CATK_EXTRA_OVERRIDES"
  echo

  set +e
  bash scripts/self_forced_a100_4_with_oom_retry.sh
  STATUS=$?
  set -e

  END_TS=$(date +%s)
  ELAPSED=$(( END_TS - START_TS ))
  echo -e "$(date '+%F %T')\\t{args.pod}\\t$LR\\t$TASK_NAME\\t$STATUS\\t$ELAPSED" >> {shq(status_tsv)}

  HINT=""
  LATEST_RUN=$(ls -td "{args.log_dir.rstrip('/')}/${{TASK_NAME}}"/runs/* 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_RUN" ]]; then
    HINT="$LATEST_RUN"
  fi
  echo -e "$LR\\t$TASK_NAME\\t$STATUS\\t$ELAPSED\\t$HINT" >> {shq(summary_tsv)}

  if (( STATUS != 0 )); then
    echo "[sweep] lr=$LR failed with status $STATUS. Stopping sweep for inspection."
    exit "$STATUS"
  fi
  echo "[sweep] lr=$LR completed successfully in $ELAPSED seconds."
done

echo
echo "[sweep] all LR runs completed at $(date '+%F %T')"
cat {shq(status_tsv)}
exec bash
"""


def render_start_command(args: argparse.Namespace) -> str:
    root = run_root(args)
    worker_file = f"{root}/{args.pod}_worker.sh"
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
  exit 3
fi
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
cat > {shq(worker_file)} <<'CATK_SWEEP_WORKER'
{render_worker_script(args).rstrip()}
CATK_SWEEP_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq('cat >> ' + shq(tmux_log))}
echo "[launcher] started {args.session} on {args.pod}"
echo "[launcher] tmux log: {tmux_log}"
echo "[launcher] status: {root}/status.tsv"
echo "[launcher] summary: {root}/summary.tsv"
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
        description="Launch sequential self-forced DMD LR sweep on one A100x4 pod.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--wandb-pretrain-artifact", default=DEFAULT_WANDB_PRETRAIN_ARTIFACT)
    parser.add_argument("--pretrain-ckpt", default="")
    parser.add_argument("--pretrain-download-dir", default="")
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--lrs", nargs="+", required=True)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--initial-bs", type=int, default=96)
    parser.add_argument("--oom-step", type=int, default=16)
    parser.add_argument("--min-bs", type=int, default=64)
    parser.add_argument("--val-batch-size", default="8")
    parser.add_argument("--test-batch-size", default="8")
    parser.add_argument("--max-epochs", default="5")
    parser.add_argument("--check-val-every-n-epoch", default="1")
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--train-epoch-sample-fraction", default="0.25")
    parser.add_argument("--train-memory-balanced-batches", default="true")
    parser.add_argument("--estimator-warmup-epochs", default="2")
    parser.add_argument("--unfrozen-range", default="middle")
    parser.add_argument("--extra-hydra-overrides", nargs="*", default=[])
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.pretrain_ckpt:
        args.pretrain_ckpt = default_pretrain_ckpt(args.pod)
    if not args.pretrain_download_dir:
        args.pretrain_download_dir = default_pretrain_download_dir(args.pod)
    if args.nproc_per_node != 4:
        parser.error("--nproc-per-node must be 4 for one A100x4 pod")
    if args.initial_bs < args.min_bs:
        parser.error("--initial-bs must be >= --min-bs")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session))
        return
    print(f"[launcher] pod:      {args.pod}")
    print(f"[launcher] session:  {args.session}")
    print(f"[launcher] sweep:    {args.sweep_name}")
    print(f"[launcher] lrs:      {' '.join(args.lrs)}")
    print(f"[launcher] bs range: {args.initial_bs}->{args.min_bs} step {args.oom_step}")
    print(f"[launcher] started:  {datetime.now().isoformat(timespec='seconds')}")
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
