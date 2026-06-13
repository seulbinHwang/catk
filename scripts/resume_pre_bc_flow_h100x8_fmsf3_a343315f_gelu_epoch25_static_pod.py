#!/usr/bin/env python3
"""Resume the fm-sf-3 a343315f GELU pretrain from the epoch-25 checkpoint.

This launcher targets the already-running ``fm-sf-3`` pod only. It does not
create, delete, or restart pods. It starts a separate tmux session so the resume
verification logs do not overwrite the original training launcher logs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-3"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_2x4_h100"
DEFAULT_SOURCE_TASK_NAME = (
    "flow_open_loop_pretrain_a343315f_gelu_head_dim16_freq32_"
    "h100x8_fmsf3_bs18_lr6p5e-4_warm5_val8_membal"
)
DEFAULT_TASK_NAME = DEFAULT_SOURCE_TASK_NAME + "_resume_epoch26_from_epoch25_verify"
DEFAULT_SESSION = "catk-pretrain-a343315f-gelu-h100x8-fmsf3-resume-e25"
DEFAULT_RESUME_CKPT = (
    f"{DEFAULT_LOG_DIR}/{DEFAULT_SOURCE_TASK_NAME}/runs/2026-06-13_00-51-08/"
    "checkpoints/epoch_last.ckpt"
)


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


def run_root(args: argparse.Namespace) -> str:
    return f"{args.log_dir.rstrip('/')}/tmux_pre_bc_flow_h100x8_fmsf3/{args.task_name.replace('/', '_')}"


def render_env(args: argparse.Namespace) -> str:
    lines = [
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("RESUME_CKPT", args.resume_ckpt),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("INITIAL_BS", args.initial_bs),
        export_line("OOM_STEP", args.oom_step),
        export_line("MIN_BS", args.min_bs),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("MAX_EPOCHS", args.max_epochs),
        export_line("CHECK_VAL_EVERY_N_EPOCH", args.check_val_every_n_epoch),
        export_line("LIMIT_VAL_BATCHES", args.limit_val_batches),
        export_line("USE_DISTRIBUTED_SAMPLER", args.use_distributed_sampler),
        export_line("TRAIN_MEMORY_BALANCED_BATCHES", args.train_memory_balanced_batches),
        export_line("LEARNING_RATE", args.learning_rate),
        export_line("LR_WARMUP_STEPS", args.lr_warmup_steps),
        export_line("MAX_NON_OOM_RETRIES", args.max_non_oom_retries),
        export_line("WANDB_GROUP", args.wandb_group),
        export_line("WANDB_JOB_TYPE", args.wandb_job_type),
        export_line("WANDB_LOG_MODEL", args.wandb_log_model),
    ]
    if args.limit_train_batches:
        lines.append(export_line("LIMIT_TRAIN_BATCHES", args.limit_train_batches))
    if args.extra_hydra_overrides:
        lines.append(export_line("CATK_EXTRA_OVERRIDES", args.extra_hydra_overrides))
    return "\n".join(lines) + "\n"


def render_worker_script(project_root: str, env_file: str, branch: str, pull: bool) -> str:
    pull_block = ""
    if pull:
        pull_block = f"""
git config --global --add safe.directory {shq(project_root)} || true
git fetch origin +{shq(branch)}:refs/remotes/origin/{shq(branch)}
git checkout -B {shq(branch)} origin/{shq(branch)}
"""

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
{pull_block}

set -a
source {shq(env_file)}
set +a

RUN_ROOT="${{CATK_LOG_DIR%/}}/tmux_pre_bc_flow_h100x8_fmsf3/${{TASK_NAME//\\//_}}"
mkdir -p "$RUN_ROOT"

echo "[resume-pretrain-fmsf3] pod=$(hostname) task=${{TASK_NAME}}"
echo "[resume-pretrain-fmsf3] started at $(date '+%F %T')"
echo "[resume-pretrain-fmsf3] git=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "[resume-pretrain-fmsf3] resume_ckpt=${{RESUME_CKPT}}"
echo "[resume-pretrain-fmsf3] experiment=${{EXPERIMENT}} initial_bs=${{INITIAL_BS}} oom_step=${{OOM_STEP}} min_bs=${{MIN_BS}}"
echo "[resume-pretrain-fmsf3] max_epochs=${{MAX_EPOCHS}} check_val_every=${{CHECK_VAL_EVERY_N_EPOCH}} limit_val=${{LIMIT_VAL_BATCHES}}"
echo "[resume-pretrain-fmsf3] run_root=${{RUN_ROOT}}"

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "[resume-pretrain-fmsf3] missing resume checkpoint: $RESUME_CKPT" >&2
  exec bash
fi

python - "$RESUME_CKPT" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
print(
    "[resume-pretrain-fmsf3] checkpoint state: "
    f"epoch={{ckpt.get('epoch')}} global_step={{ckpt.get('global_step')}} "
    f"optimizer_states={{len(ckpt.get('optimizer_states', []))}} "
    f"lr_schedulers={{len(ckpt.get('lr_schedulers', []))}}"
)
PY

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

task_process_pids() {{
  ps -eo pid=,cmd= | awk -v task="$TASK_NAME" '
    index($0, task) && ($0 ~ /torchrun/ || $0 ~ / -m src\\.run/) && $0 !~ /awk -v task/ {{ print $1 }}
  '
}}

terminate_task_processes() {{
  local reason="${{1:-cleanup}}"
  local pids=()
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} == 0 )); then
    return 0
  fi
  echo "[resume-pretrain-fmsf3] terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 15
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[resume-pretrain-fmsf3] force killing task processes for $reason: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
}}

is_retryable_non_oom_exit() {{
  local status="$1"
  [[ "$status" == "134" || "$status" == "143" ]]
}}

bs="$INITIAL_BS"
attempt=0
non_oom_retry_count=0
final_status=1

while (( bs >= MIN_BS )); do
  attempt=$(( attempt + 1 ))
  attempt_log="$RUN_ROOT/$(hostname).attempt_$(printf '%03d' "$attempt")_bs${{bs}}.log"

  echo
  echo "[resume-pretrain-fmsf3] attempt #${{attempt}} bs=${{bs}} resume=${{RESUME_CKPT}}"
  echo "[resume-pretrain-fmsf3] attempt log: $attempt_log"
  terminate_task_processes "pre-attempt cleanup"

  HYDRA_OVERRIDES=(
    "experiment=${{EXPERIMENT}}"
    "action=fit"
    "task_name=${{TASK_NAME}}"
    "ckpt_path=${{RESUME_CKPT}}"
    "paths.cache_root=${{CACHE_ROOT}}"
    "trainer.devices=${{NPROC_PER_NODE}}"
    "trainer.num_nodes=1"
    "trainer.max_epochs=${{MAX_EPOCHS}}"
    "trainer.check_val_every_n_epoch=${{CHECK_VAL_EVERY_N_EPOCH}}"
    "trainer.limit_val_batches=${{LIMIT_VAL_BATCHES}}"
    "+trainer.use_distributed_sampler=${{USE_DISTRIBUTED_SAMPLER}}"
    "data.train_batch_size=${{bs}}"
    "data.val_batch_size=${{VAL_BATCH_SIZE}}"
    "data.test_batch_size=${{TEST_BATCH_SIZE}}"
    "data.train_memory_balanced_batches=${{TRAIN_MEMORY_BALANCED_BATCHES}}"
    "model.model_config.lr=${{LEARNING_RATE}}"
    "model.model_config.lr_warmup_steps=${{LR_WARMUP_STEPS}}"
    "logger.wandb.group=${{WANDB_GROUP}}"
    "logger.wandb.job_type=${{WANDB_JOB_TYPE}}"
    "logger.wandb.log_model=${{WANDB_LOG_MODEL}}"
  )
  if [[ -n "${{LIMIT_TRAIN_BATCHES:-}}" ]]; then
    HYDRA_OVERRIDES+=("trainer.limit_train_batches=${{LIMIT_TRAIN_BATCHES}}")
  fi
  if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES=( $CATK_EXTRA_OVERRIDES )
    HYDRA_OVERRIDES+=("${{EXTRA_OVERRIDES[@]}}")
  fi

  printf '[resume-pretrain-fmsf3] torchrun command:'
  printf ' %q' torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}"
  printf '\\n'

  torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}" 2>&1 | tee "$attempt_log"
  exit_code="${{PIPESTATUS[0]}}"
  echo "[resume-pretrain-fmsf3] attempt #${{attempt}} exited with status $exit_code"

  if (( exit_code == 0 )); then
    final_status=0
    break
  fi

  terminate_task_processes "post-attempt status=$exit_code"

  if grep -Eq "$OOM_REGEX" "$attempt_log"; then
    non_oom_retry_count=0
    new_bs=$(( bs - OOM_STEP ))
    echo "[resume-pretrain-fmsf3] OOM detected at bs=${{bs}}; lowering to bs=${{new_bs}}"
    if (( new_bs < MIN_BS )); then
      echo "[resume-pretrain-fmsf3] next bs=${{new_bs}} is below MIN_BS=${{MIN_BS}}; aborting"
      final_status=1
      break
    fi
    bs="$new_bs"
    continue
  fi

  if is_retryable_non_oom_exit "$exit_code" && (( non_oom_retry_count < MAX_NON_OOM_RETRIES )); then
    non_oom_retry_count=$(( non_oom_retry_count + 1 ))
    echo "[resume-pretrain-fmsf3] retryable non-OOM exit=${{exit_code}}; retrying bs=${{bs}} (${{non_oom_retry_count}}/${{MAX_NON_OOM_RETRIES}})"
    continue
  fi

  echo "[resume-pretrain-fmsf3] non-OOM failure; see $attempt_log"
  final_status="$exit_code"
  break
done

echo
echo "[resume-pretrain-fmsf3] finished with status $final_status at $(date '+%F %T')"
echo "[resume-pretrain-fmsf3] leaving shell open for inspection"
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
    pipe_command = f"cat >> {shq(tmux_log)}"
    env_text = render_env(args)
    worker_text = render_worker_script(
        args.project_root,
        env_file,
        args.branch,
        pull=not args.no_pull,
    )
    monitor_text = render_monitor_script(args.monitor_interval, args.task_name)

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
{monitor_text.rstrip()}
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
mkdir -p {shq(root)}
{replace_block}
cat > {shq(env_file)} <<'CATK_ENV'
{env_text.rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{worker_text.rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq(pipe_command)}
{monitor_block}
echo "[launcher] started tmux session {args.session} on pod {args.pod}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_stop_command(session: str, task_name: str) -> str:
    return f"""set +e
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
fi
TASK_NAME_TO_STOP={shq(task_name)}
task_process_pids() {{
  ps -eo pid=,cmd= | awk -v task="$TASK_NAME_TO_STOP" '
    index($0, task) && ($0 ~ /torchrun/ || $0 ~ / -m src\\.run/) && $0 !~ /awk -v task/ {{ print $1 }}
  '
}}
mapfile -t pids < <(task_process_pids || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 20
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume fm-sf-3 a343315f GELU pretrain from the epoch-25 checkpoint."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--resume-ckpt", default=DEFAULT_RESUME_CKPT)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--initial-bs", type=int, default=18)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=12)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=64)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=8)
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--learning-rate", default="6.5e-4")
    parser.add_argument("--lr-warmup-steps", default="5")
    parser.add_argument("--use-distributed-sampler", default="false")
    parser.add_argument("--train-memory-balanced-batches", default="true")
    parser.add_argument("--wandb-group", default="resume_verification")
    parser.add_argument("--wandb-job-type", default="resume_epoch26_from_epoch25_verify")
    parser.add_argument("--wandb-log-model", default="false")
    parser.add_argument("--max-non-oom-retries", type=int, default=3)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=60)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.nproc_per_node < 1:
        raise SystemExit("--nproc-per-node must be >= 1")
    if args.initial_bs < 1:
        raise SystemExit("--initial-bs must be >= 1")
    if args.oom_step < 1:
        raise SystemExit("--oom-step must be >= 1")
    if args.min_bs < 1:
        raise SystemExit("--min-bs must be >= 1")
    if args.initial_bs < args.min_bs:
        raise SystemExit("--initial-bs must be >= --min-bs")
    if not args.resume_ckpt and not args.stop:
        raise SystemExit("--resume-ckpt is required unless --stop is set")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    remote_cmd = (
        render_stop_command(args.session, args.task_name)
        if args.stop
        else render_start_command(args)
    )
    if args.dry_run:
        print(remote_cmd)
        return

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
            remote_cmd,
        ]
    )

    if not args.stop:
        root = run_root(args)
        print(f"[launcher] pod:         {args.pod}")
        print(f"[launcher] session:     {args.session}")
        print(f"[launcher] task_name:   {args.task_name}")
        print(f"[launcher] resume_ckpt: {args.resume_ckpt}")
        print(f"[launcher] tmux log:    {root}/{args.pod}.tmux.log")
        print(
            "[launcher] attach:      "
            f"kubectl exec -it -n {args.namespace} {args.pod} "
            f"-c {args.container} -- tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    main()
