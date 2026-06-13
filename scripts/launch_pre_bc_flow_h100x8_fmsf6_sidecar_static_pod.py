#!/usr/bin/env python3
"""Launch an H100x8 open-loop Flow pretrain on the existing fm-sf-6 pod.

The launcher never creates, deletes, or restarts pods. It only starts/stops a
tmux session inside the already-running pod. The remote worker retries CUDA OOM
failures by resuming from the latest Lightning checkpoint with a lower
``data.train_batch_size``. It can prebuild deterministic Flow target sidecars
and disables train-only open-loop metrics by default.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-6"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_decoder_last"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_2x4_h100"
DEFAULT_TASK_NAME = (
    "flow_open_loop_pretrain_global2s_refiner_sidecar_metricoff_"
    "h100x8_fmsf6_bs18_lr6p5e-4_warm5_val8_membal"
)
DEFAULT_SESSION = "catk-pretrain-sidecar-metricoff-h100x8-fmsf6"


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
    return f"{args.log_dir.rstrip('/')}/tmux_pre_bc_flow_h100x8_fmsf6/{safe_task}"


def render_env(args: argparse.Namespace) -> str:
    lines = [
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("INITIAL_BS", args.initial_bs),
        export_line("OOM_STEP", args.oom_step),
        export_line("MIN_BS", args.min_bs),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("MAX_EPOCHS", args.max_epochs),
        export_line("CHECK_VAL_EVERY_N_EPOCH", args.check_val_every_n_epoch),
        export_line("LIMIT_VAL_BATCHES", args.limit_val_batches),
        export_line("USE_DISTRIBUTED_SAMPLER", args.use_distributed_sampler),
        export_line("TRAIN_MEMORY_BALANCED_BATCHES", args.train_memory_balanced_batches),
        export_line("MAX_NON_OOM_RETRIES", args.max_non_oom_retries),
        export_line("FLOW_TARGET_SIDECAR_DIR", args.flow_target_sidecar_dir),
        export_line("FLOW_TARGET_SIDECAR_READ", args.flow_target_sidecar_read),
        export_line("FLOW_TARGET_SIDECAR_REQUIRED", args.flow_target_sidecar_required),
        export_line("TRAIN_OPEN_LOOP_METRICS", args.train_open_loop_metrics),
        export_line(
            "SKIP_EMPTY_OPEN_LOOP_OPTIMIZER_GUARD",
            args.skip_empty_open_loop_optimizer_guard,
        ),
        export_line("SIDECAR_PREBUILD", args.sidecar_prebuild),
        export_line("SIDECAR_PREBUILD_MAX_SAMPLES", args.sidecar_prebuild_max_samples),
    ]
    optional = {
        "LEARNING_RATE": args.learning_rate,
        "LR_WARMUP_STEPS": args.lr_warmup_steps,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "CATK_EXTRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
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

RUN_ROOT="${{CATK_LOG_DIR%/}}/tmux_pre_bc_flow_h100x8_fmsf6/${{TASK_NAME//\\//_}}"
mkdir -p "$RUN_ROOT"

echo "[pretrain-h100x8-fmsf6] pod=$(hostname) task=${{TASK_NAME}}"
echo "[pretrain-h100x8-fmsf6] started at $(date '+%F %T')"
echo "[pretrain-h100x8-fmsf6] git=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "[pretrain-h100x8-fmsf6] experiment=${{EXPERIMENT}} initial_bs=${{INITIAL_BS}} oom_step=${{OOM_STEP}} min_bs=${{MIN_BS}}"
echo "[pretrain-h100x8-fmsf6] nproc=${{NPROC_PER_NODE}} cache=${{CACHE_ROOT}}"
echo "[pretrain-h100x8-fmsf6] run_root=${{RUN_ROOT}}"
echo "[pretrain-h100x8-fmsf6] attach survives after exit; press Ctrl-b d to detach"
echo

if [[ "${{SIDECAR_PREBUILD}}" == "true" ]]; then
  echo "[pretrain-h100x8-fmsf6] prebuilding flow target sidecars dir=${{FLOW_TARGET_SIDECAR_DIR}} max_samples=${{SIDECAR_PREBUILD_MAX_SAMPLES}} nproc=${{NPROC_PER_NODE}}"
  sidecar_log="$RUN_ROOT/$(hostname).sidecar_prebuild.log"
  torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" scripts/build_flow_target_sidecars.py \
    --cache-root "$CACHE_ROOT" \
    --sidecar-dir "$FLOW_TARGET_SIDECAR_DIR" \
    --experiment "$EXPERIMENT" \
    --device auto \
    --max-samples "$SIDECAR_PREBUILD_MAX_SAMPLES" \
    --status-every 500 \
    "model.model_config.train_open_loop_metrics=${{TRAIN_OPEN_LOOP_METRICS}}" \
    2>&1 | tee "$sidecar_log"
  sidecar_status="${{PIPESTATUS[0]}}"
  echo "[pretrain-h100x8-fmsf6] sidecar prebuild exited with status $sidecar_status log=$sidecar_log"
  if (( sidecar_status != 0 )); then
    echo "[pretrain-h100x8-fmsf6] sidecar prebuild failed; aborting before training"
    exec bash
  fi
fi

FLOW_TARGET_SIDECAR_ROOT=""
if [[ "${{FLOW_TARGET_SIDECAR_READ}}" == "true" && -n "${{FLOW_TARGET_SIDECAR_DIR}}" ]]; then
  FLOW_TARGET_SIDECAR_ROOT="$(python - <<'PY'
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.smart.tokens.flow_token_processor import FlowTokenProcessor

config_dir = (Path.cwd() / "configs").as_posix()
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name="run",
        overrides=[
            f"experiment={{os.environ['EXPERIMENT']}}",
            f"paths.cache_root={{os.environ['CACHE_ROOT']}}",
            f"model.model_config.token_processor.flow_target_sidecar_dir={{os.environ['FLOW_TARGET_SIDECAR_DIR']}}",
            "model.model_config.token_processor.flow_target_sidecar_read=true",
            "model.model_config.token_processor.flow_target_sidecar_write=false",
        ],
    )
processor = FlowTokenProcessor(
    **OmegaConf.to_container(cfg.model.model_config.token_processor, resolve=True)
)
print(processor._flow_target_sidecar_root())
PY
)"
  echo "[pretrain-h100x8-fmsf6] dataloader sidecar root=${{FLOW_TARGET_SIDECAR_ROOT}}"
fi

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|cuda runtime error.*out of memory|torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'

latest_checkpoint() {{
  local runs_dir="${{CATK_LOG_DIR%/}}/${{TASK_NAME}}/runs"
  {{ ls -t "${{runs_dir}}"/*/checkpoints/epoch_last.ckpt 2>/dev/null; \
     ls -t "${{runs_dir}}"/*/checkpoints/last.ckpt 2>/dev/null; }} | head -1
}}

task_process_pids() {{
  pgrep -f "task_name=${{TASK_NAME}}" 2>/dev/null | while read -r pid; do
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
  echo "[pretrain-h100x8-fmsf6] terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 15
  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[pretrain-h100x8-fmsf6] force killing task processes for $reason: ${{pids[*]}}"
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
  latest_ckpt="$(latest_checkpoint || true)"

  echo
  if [[ -n "$latest_ckpt" ]]; then
    echo "[pretrain-h100x8-fmsf6] attempt #${{attempt}} bs=${{bs}} resume=${{latest_ckpt}}"
    ckpt_override="ckpt_path=${{latest_ckpt}}"
  else
    echo "[pretrain-h100x8-fmsf6] attempt #${{attempt}} bs=${{bs}} fresh fit"
    ckpt_override="ckpt_path=null"
  fi
  echo "[pretrain-h100x8-fmsf6] attempt log: $attempt_log"

  terminate_task_processes "pre-attempt cleanup"

  HYDRA_OVERRIDES=(
    "experiment=${{EXPERIMENT}}"
    "action=fit"
    "task_name=${{TASK_NAME}}"
    "$ckpt_override"
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
    "data.train_flow_target_sidecar_root=${{FLOW_TARGET_SIDECAR_ROOT}}"
    "data.train_flow_target_sidecar_required=${{FLOW_TARGET_SIDECAR_REQUIRED}}"
    "model.model_config.train_open_loop_metrics=${{TRAIN_OPEN_LOOP_METRICS}}"
    "model.model_config.token_processor.flow_target_sidecar_dir=${{FLOW_TARGET_SIDECAR_DIR}}"
    "model.model_config.token_processor.flow_target_sidecar_read=${{FLOW_TARGET_SIDECAR_READ}}"
    "model.model_config.token_processor.flow_target_sidecar_write=false"
    "model.model_config.token_processor.flow_target_sidecar_required=${{FLOW_TARGET_SIDECAR_REQUIRED}}"
    "model.model_config.skip_empty_open_loop_optimizer_guard=${{SKIP_EMPTY_OPEN_LOOP_OPTIMIZER_GUARD}}"
  )
  if [[ -n "${{LEARNING_RATE:-}}" ]]; then
    HYDRA_OVERRIDES+=("model.model_config.lr=${{LEARNING_RATE}}")
  fi
  if [[ -n "${{LR_WARMUP_STEPS:-}}" ]]; then
    HYDRA_OVERRIDES+=("model.model_config.lr_warmup_steps=${{LR_WARMUP_STEPS}}")
  fi
  if [[ -n "${{LIMIT_TRAIN_BATCHES:-}}" ]]; then
    HYDRA_OVERRIDES+=("trainer.limit_train_batches=${{LIMIT_TRAIN_BATCHES}}")
  fi
  if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES=( $CATK_EXTRA_OVERRIDES )
    HYDRA_OVERRIDES+=("${{EXTRA_OVERRIDES[@]}}")
  fi

  printf '[pretrain-h100x8-fmsf6] torchrun command:'
  printf ' %q' torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}"
  printf '\\n'

  torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}" 2>&1 | tee "$attempt_log"
  exit_code="${{PIPESTATUS[0]}}"
  echo "[pretrain-h100x8-fmsf6] attempt #${{attempt}} exited with status $exit_code"

  if (( exit_code == 0 )); then
    final_status=0
    break
  fi

  terminate_task_processes "post-attempt status=$exit_code"

  if grep -Eq "$OOM_REGEX" "$attempt_log"; then
    non_oom_retry_count=0
    new_bs=$(( bs - OOM_STEP ))
    echo "[pretrain-h100x8-fmsf6] OOM detected at bs=${{bs}}; lowering to bs=${{new_bs}}"
    if (( new_bs < MIN_BS )); then
      echo "[pretrain-h100x8-fmsf6] next bs=${{new_bs}} is below MIN_BS=${{MIN_BS}}; aborting"
      final_status=1
      break
    fi
    bs="$new_bs"
    continue
  fi

  if is_retryable_non_oom_exit "$exit_code" && (( non_oom_retry_count < MAX_NON_OOM_RETRIES )); then
    non_oom_retry_count=$(( non_oom_retry_count + 1 ))
    echo "[pretrain-h100x8-fmsf6] retryable non-OOM exit=${{exit_code}}; retrying bs=${{bs}} (${{non_oom_retry_count}}/${{MAX_NON_OOM_RETRIES}})"
    continue
  fi

  echo "[pretrain-h100x8-fmsf6] non-OOM failure; see $attempt_log"
  final_status="$exit_code"
  break
done

echo
echo "[pretrain-h100x8-fmsf6] finished with status $final_status at $(date '+%F %T')"
echo "[pretrain-h100x8-fmsf6] leaving shell open for inspection"
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
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch fm-sf-6 H100x8 open-loop pretrain for the "
            "semi_control_decoder_last global 2s refiner sidecar fast path."
        )
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
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
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
    parser.add_argument("--max-non-oom-retries", type=int, default=3)
    parser.add_argument(
        "--flow-target-sidecar-dir",
        default=f"{DEFAULT_CACHE_ROOT}/flow_target_sidecars",
    )
    parser.add_argument("--flow-target-sidecar-read", default="true")
    parser.add_argument("--flow-target-sidecar-required", default="true")
    parser.add_argument("--train-open-loop-metrics", default="false")
    parser.add_argument("--skip-empty-open-loop-optimizer-guard", default="true")
    parser.add_argument("--sidecar-prebuild", default="true")
    parser.add_argument("--sidecar-prebuild-max-samples", type=int, default=0)
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
        print(f"[launcher] pod:       {args.pod}")
        print(f"[launcher] session:   {args.session}")
        print(f"[launcher] task_name: {args.task_name}")
        print(f"[launcher] tmux log:  {root}/{args.pod}.tmux.log")
        print(
            "[launcher] attach:    "
            f"kubectl exec -it -n {args.namespace} {args.pod} "
            f"-c {args.container} -- tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    main()
