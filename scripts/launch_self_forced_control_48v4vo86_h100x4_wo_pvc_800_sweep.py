#!/usr/bin/env python3
"""Launch a 4xH100 control-space self-forcing sweep on wo-pvc-800.

The sweep fine-tunes from the W&B artifact
``jksg01019-naver-labs/SMART-FLOW/epoch-last-48v4vo86:v61`` and runs six
fresh experiments:

* estimator_warmup_epochs in {0, 1}
* lr in {1e-6, 5e-6, 1e-5}

Each experiment uses ``use_anchor_flow_matching_loss=false``,
``backprop_last_k=8``, ``max_epochs=4``, and the shared H100 OOM retry wrapper.
This launcher only starts/stops tmux inside the existing pod; it never creates,
deletes, or restarts pods.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_POD = "wo-pvc-800"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_SESSION = "catk-sf-control48v4vo86-h100x4-wo800-sweep"
DEFAULT_EXPERIMENT = "self_forced_npfm_h100x4_wo_pvc_800"
DEFAULT_WANDB_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-48v4vo86:v61"
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/"
    "v61/epoch_last.ckpt"
)
DEFAULT_DOWNLOAD_DIR = (
    "/workspace/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/"
    "v61/artifact"
)
DEFAULT_TASK_PREFIX = "flow_sf_control48v4vo86_h100x4wo800_dmd_k8"


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
    safe_prefix = args.task_prefix.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_self_forced_control_48v4vo86_h100x4_wo_pvc_800/{safe_prefix}"


def render_env(args: argparse.Namespace) -> str:
    lines = [
        export_line("SWEEP_TASK_PREFIX", args.task_prefix),
        export_line("EXPERIMENT", args.experiment),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("PRETRAIN_CKPT", args.pretrain_ckpt),
        export_line("WANDB_PRETRAIN_ARTIFACT", args.wandb_artifact),
        export_line("WANDB_PRETRAIN_DOWNLOAD_DIR", args.download_dir),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("INITIAL_BS", args.initial_bs),
        export_line("OOM_STEP", args.oom_step),
        export_line("MIN_BS", args.min_bs),
        export_line("MAX_EPOCHS", args.max_epochs),
        export_line("CHECK_VAL_EVERY_N_EPOCH", args.check_val_every_n_epoch),
        export_line("BACKPROP_LAST_K", args.backprop_last_k),
        export_line("SELF_FORCED_USE_STOP_MOTION", args.self_forced_use_stop_motion),
        export_line("DECODER_USE_STOP_MOTION", args.decoder_use_stop_motion),
        export_line("RANDOM_TERMINAL_POLICY", "all"),
        export_line("RANDOM_TERMINAL_SCOPE", "global_batch"),
        export_line("WARMUP_EPOCHS_LIST", " ".join(str(v) for v in args.warmup_epochs)),
        export_line("LR_LIST", " ".join(args.learning_rates)),
        export_line("SCORER_SCENE_NUM", args.scorer_scene_num),
        export_line("LIMIT_VAL_BATCHES", args.limit_val_batches),
    ]
    if args.val_batch_size:
        lines.append(export_line("VAL_BATCH_SIZE", args.val_batch_size))
    if args.test_batch_size:
        lines.append(export_line("TEST_BATCH_SIZE", args.test_batch_size))
    if args.limit_train_batches:
        lines.append(export_line("LIMIT_TRAIN_BATCHES", args.limit_train_batches))
    if args.extra_hydra_overrides:
        lines.append(export_line("USER_EXTRA_OVERRIDES", args.extra_hydra_overrides))
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

echo "[sf-control48v4vo86-sweep] pod=$(hostname)"
echo "[sf-control48v4vo86-sweep] started at $(date '+%F %T')"
echo "[sf-control48v4vo86-sweep] repo=$(pwd)"
echo "[sf-control48v4vo86-sweep] commit=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --pretty=%s 2>/dev/null)"
echo "[sf-control48v4vo86-sweep] artifact=$WANDB_PRETRAIN_ARTIFACT"
echo "[sf-control48v4vo86-sweep] checkpoint=$PRETRAIN_CKPT"
echo "[sf-control48v4vo86-sweep] initial_bs=$INITIAL_BS oom_step=$OOM_STEP min_bs=$MIN_BS backprop_last_k=$BACKPROP_LAST_K"
echo "[sf-control48v4vo86-sweep] warmups=$WARMUP_EPOCHS_LIST"
echo "[sf-control48v4vo86-sweep] lrs=$LR_LIST"
echo

prepare_checkpoint() {{
  if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[sf-control48v4vo86-sweep] using cached checkpoint: $PRETRAIN_CKPT"
    return 0
  fi

  python - <<'PY'
from pathlib import Path
import os
import shutil

target = Path(os.environ["PRETRAIN_CKPT"])
candidates = [
    target,
    Path("/workspace/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/v61/epoch_last.ckpt"),
    Path("/mnt/nuplan/projects/catk/checkpoints/wandb/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/epoch-last-48v4vo86_v61/epoch_last.ckpt"),
    Path("/mnt/nuplan/projects/catk/checkpoints/flow_control_space_pretrain_v100x47_prefix_roundtrip2_bs8/run_48v4vo86_v61/epoch_last.ckpt"),
]
for candidate in candidates:
    if candidate.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if candidate.resolve() != target.resolve():
            shutil.copy2(candidate, target)
        print(f"[sf-control48v4vo86-sweep] checkpoint ready from local path: {{target}}")
        raise SystemExit(0)
raise SystemExit(1)
PY
  local status=$?
  if (( status == 0 )); then
    return 0
  fi

  mkdir -p "$(dirname "$PRETRAIN_CKPT")" "$WANDB_PRETRAIN_DOWNLOAD_DIR"
  local lock_dir="${{PRETRAIN_CKPT}}.download.lock"
  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[sf-control48v4vo86-sweep] downloading W&B artifact: $WANDB_PRETRAIN_ARTIFACT"
    python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_PRETRAIN_ARTIFACT"]
download_dir = Path(os.environ["WANDB_PRETRAIN_DOWNLOAD_DIR"])
target_ckpt = Path(os.environ["PRETRAIN_CKPT"])
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
print(f"[sf-control48v4vo86-sweep] checkpoint ready from W&B artifact: {{target_ckpt}}")
PY
    status=$?
    rm -rf "$lock_dir"
    return "$status"
  fi

  echo "[sf-control48v4vo86-sweep] waiting for checkpoint download lock: $lock_dir"
  for _ in $(seq 1 180); do
    if [[ -f "$PRETRAIN_CKPT" ]]; then
      echo "[sf-control48v4vo86-sweep] checkpoint appeared: $PRETRAIN_CKPT"
      return 0
    fi
    sleep 10
  done
  echo "[sf-control48v4vo86-sweep] timed out waiting for checkpoint download" >&2
  return 4
}}

prepare_checkpoint
status=$?
if (( status != 0 )); then
  echo "[sf-control48v4vo86-sweep] checkpoint preparation failed with status $status" >&2
  echo "[sf-control48v4vo86-sweep] leaving shell open for inspection"
  exec bash
fi

if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "[sf-control48v4vo86-sweep] CACHE_ROOT does not exist: $CACHE_ROOT" >&2
  echo "[sf-control48v4vo86-sweep] leaving shell open for inspection"
  exec bash
fi

SWEEP_ID="${{SWEEP_ID:-$(date '+%Y%m%d_%H%M%S')}}"
STATUS_FILE="${{CATK_LOG_DIR%/}}/tmux_self_forced_control_48v4vo86_h100x4_wo_pvc_800/${{SWEEP_TASK_PREFIX}}/sweep_${{SWEEP_ID}}_status.tsv"
mkdir -p "$(dirname "$STATUS_FILE")"
echo -e "finished_at\\twarmup\\tlr\\tstatus\\ttask_name" > "$STATUS_FILE"

COMMON_OVERRIDES=(
  "model.model_config.token_processor.use_kinematic_control_flow=true"
  "model.model_config.decoder.use_kinematic_control_flow=true"
  "model.model_config.token_processor.use_prefix_valid_future_loss_mask=true"
  "model.model_config.token_processor.control_round_trip_max_position_error_m=2.0"
  "model.model_config.token_processor.control_pos_scale_m=1.0"
  "model.model_config.token_processor.control_vehicle_yaw_scale_rad=0.025"
  "model.model_config.token_processor.control_pedestrian_yaw_scale_rad=0.20"
  "model.model_config.token_processor.control_cyclist_yaw_scale_rad=0.06"
  "model.model_config.decoder.flow_window_steps=20"
  "model.model_config.token_processor.flow_window_steps=20"
  "model.model_config.self_forced.distribution_matching_objective=dmd"
  "model.model_config.self_forced.use_anchor_flow_matching_loss=false"
  "model.model_config.self_forced.anchor_weight=0.0"
  "model.model_config.self_forced.detach_block_transition=false"
  "model.model_config.self_forced.sampling.random_terminal_step.policy=all"
  "model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch"
  "model.model_config.self_forced.sampling.backprop_last_k=${{BACKPROP_LAST_K}}"
  "model.model_config.scorer_scene_num=${{SCORER_SCENE_NUM}}"
  "logger.wandb.group=sf_control48v4vo86_h100x4_sweep"
)
if [[ -n "${{LIMIT_VAL_BATCHES:-}}" ]]; then
  COMMON_OVERRIDES+=("trainer.limit_val_batches=${{LIMIT_VAL_BATCHES}}")
fi
if [[ -n "${{USER_EXTRA_OVERRIDES:-}}" ]]; then
  # shellcheck disable=SC2206
  USER_OVERRIDES=($USER_EXTRA_OVERRIDES)
  COMMON_OVERRIDES+=("${{USER_OVERRIDES[@]}}")
fi

for warmup in $WARMUP_EPOCHS_LIST; do
  for lr in $LR_LIST; do
    lr_tag="${{lr//./p}}"
    lr_tag="${{lr_tag//-/m}}"
    task="${{SWEEP_TASK_PREFIX}}_s${{SWEEP_ID}}_warmup${{warmup}}_lr${{lr_tag}}_bs${{INITIAL_BS}}"
    echo
    echo "[sf-control48v4vo86-sweep] >>> start task=$task warmup=$warmup lr=$lr"
    echo "[sf-control48v4vo86-sweep] status_file=$STATUS_FILE"

    export EXPERIMENT
    export TASK_NAME="$task"
    export CACHE_ROOT
    export CATK_LOG_DIR
    export INITIAL_BS
    export OOM_STEP
    export MIN_BS
    export CUDA_VISIBLE_DEVICES
    export NPROC_PER_NODE
    export PRETRAIN_CKPT
    export CATK_LR="$lr"
    export ESTIMATOR_WARMUP_EPOCHS="$warmup"
    export SELF_FORCED_USE_STOP_MOTION
    export DECODER_USE_STOP_MOTION
    export MAX_EPOCHS
    export CHECK_VAL_EVERY_N_EPOCH
    export RANDOM_TERMINAL_POLICY
    export RANDOM_TERMINAL_SCOPE
    export BACKPROP_LAST_K
    export CATK_EXTRA_OVERRIDES="${{COMMON_OVERRIDES[*]}}"

    bash scripts/self_forced_h100_4_with_oom_retry.sh
    status=$?
    echo -e "$(date '+%F %T')\\t${{warmup}}\\t${{lr}}\\t${{status}}\\t${{task}}" >> "$STATUS_FILE"
    if (( status != 0 )); then
      echo "[sf-control48v4vo86-sweep] failed task=$task status=$status"
      echo "[sf-control48v4vo86-sweep] stopped early. Status file: $STATUS_FILE"
      echo "[sf-control48v4vo86-sweep] leaving shell open for inspection"
      exec bash
    fi
    echo "[sf-control48v4vo86-sweep] <<< finished task=$task"
  done
done

echo
echo "[sf-control48v4vo86-sweep] all sweep jobs completed"
echo "[sf-control48v4vo86-sweep] status_file=$STATUS_FILE"
echo "[sf-control48v4vo86-sweep] leaving shell open for inspection"
exec bash
"""


def render_monitor_script(interval: int, task_prefix: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') sweep={task_prefix} pod=$(hostname)"
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
{render_monitor_script(args.monitor_interval, args.task_prefix).rstrip()}
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
            "Launch the 48v4vo86 control-space self-forcing warmup/lr sweep "
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
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--wandb-artifact", default=DEFAULT_WANDB_ARTIFACT)
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--initial-bs", type=int, default=26)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=2)
    parser.add_argument("--backprop-last-k", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--learning-rates", nargs="+", default=["1e-6", "5e-6", "1e-5"])
    parser.add_argument("--self-forced-use-stop-motion", default="false")
    parser.add_argument("--decoder-use-stop-motion", default="true")
    parser.add_argument("--val-batch-size", default="")
    parser.add_argument("--test-batch-size", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--scorer-scene-num", type=int, default=1680)
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
        parser.error("--nproc-per-node must be 4 for wo-pvc-800 H100x4")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.max_epochs < 1:
        parser.error("--max-epochs must be >= 1")
    if args.check_val_every_n_epoch < 1:
        parser.error("--check-val-every-n-epoch must be >= 1")
    if args.backprop_last_k < 1:
        parser.error("--backprop-last-k must be >= 1")
    if args.scorer_scene_num < 1:
        parser.error("--scorer-scene-num must be >= 1")
    if args.self_forced_use_stop_motion not in {"true", "false"}:
        parser.error("--self-forced-use-stop-motion must be 'true' or 'false'")
    if args.decoder_use_stop_motion not in {"true", "false"}:
        parser.error("--decoder-use-stop-motion must be 'true' or 'false'")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if not args.warmup_epochs:
        parser.error("--warmup-epochs must not be empty")
    if not args.learning_rates:
        parser.error("--learning-rates must not be empty")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args.session))
        return

    print(f"[launcher] pod:       {args.pod}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] branch:    {args.branch}")
    print(f"[launcher] artifact:  {args.wandb_artifact}")
    print(f"[launcher] ckpt:      {args.pretrain_ckpt}")
    print(f"[launcher] warmups:   {args.warmup_epochs}")
    print(f"[launcher] lrs:       {args.learning_rates}")
    print(f"[launcher] initial_bs:{args.initial_bs} -> min_bs {args.min_bs}")
    exec_in_pod(args, render_start_command(args))


if __name__ == "__main__":
    main()
