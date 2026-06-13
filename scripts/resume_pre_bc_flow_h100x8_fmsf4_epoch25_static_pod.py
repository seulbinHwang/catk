#!/usr/bin/env python3
"""Resume the fm-sf-4 H100x8 Flow pretrain from an epoch-24 checkpoint.

This launcher never creates, deletes, or restarts pods. It writes a temporary
worker script inside the already-running pod, starts it in tmux, and resumes
Lightning from the explicit checkpoint path supplied by ``--ckpt-path``.

The default arguments match the fm-sf-4 run:

  flow_open_loop_pretrain_freq64_flow80_h100x8_fmsf4_bs18_lr6p5e-4_warm5_val8_membal

Use the default task name for a real continuation. For a short smoke probe,
override ``--task-name`` so the W&B/log namespace does not collide with the
production run, while the optimizer/model/scheduler state still comes from the
same checkpoint.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-4"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_stable_2"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "pre_bc_flow_2x4_h100"
DEFAULT_TASK_NAME = "flow_open_loop_pretrain_freq64_flow80_h100x8_fmsf4_bs18_lr6p5e-4_warm5_val8_membal"
DEFAULT_CKPT_PATH = (
    "/mnt/nuplan/projects/catk/logs/"
    "flow_open_loop_pretrain_freq64_flow80_h100x8_fmsf4_bs18_lr6p5e-4_warm5_val8_membal/"
    "runs/2026-06-13_01-33-55/checkpoints/epoch_last.ckpt"
)
DEFAULT_SESSION = "catk-pretrain-freq64-flow80-h100x8-fmsf4-resume-e25"


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
    return f"{args.log_dir.rstrip('/')}/tmux_pre_bc_flow_h100x8_fmsf4_resume/{safe_task}"


def render_env(args: argparse.Namespace) -> str:
    lines = [
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("CKPT_PATH", args.ckpt_path),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CATK_LOG_DIR", args.log_dir),
        export_line("TRAIN_BATCH_SIZE", args.train_batch_size),
        export_line("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices),
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
        export_line("EXPECTED_CKPT_EPOCH", args.expected_ckpt_epoch),
    ]
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
set -euo pipefail
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

RUN_ROOT="${{CATK_LOG_DIR%/}}/tmux_pre_bc_flow_h100x8_fmsf4_resume/${{TASK_NAME//\\//_}}"
mkdir -p "$RUN_ROOT"
ATTEMPT_LOG="$RUN_ROOT/$(hostname).resume_epoch25.log"

echo "[resume-h100x8-fmsf4] pod=$(hostname) task=${{TASK_NAME}}"
echo "[resume-h100x8-fmsf4] started at $(date '+%F %T')"
echo "[resume-h100x8-fmsf4] git=$(git rev-parse HEAD 2>/dev/null || true)"
echo "[resume-h100x8-fmsf4] experiment=${{EXPERIMENT}} ckpt_path=${{CKPT_PATH}}"
echo "[resume-h100x8-fmsf4] bs=${{TRAIN_BATCH_SIZE}} lr=${{LEARNING_RATE}} warmup=${{LR_WARMUP_STEPS}} max_epochs=${{MAX_EPOCHS}}"
echo "[resume-h100x8-fmsf4] run_root=${{RUN_ROOT}}"
echo "[resume-h100x8-fmsf4] attempt_log=${{ATTEMPT_LOG}}"

python - <<'PY'
import os
import sys
import torch

path = os.environ["CKPT_PATH"]
expected_epoch = int(os.environ["EXPECTED_CKPT_EPOCH"])
ckpt = torch.load(path, map_location="cpu")
epoch = ckpt.get("epoch")
global_step = ckpt.get("global_step")
fit = ckpt.get("loops", {{}}).get("fit_loop", {{}})
epoch_progress = fit.get("epoch_progress")
print(f"[resume-h100x8-fmsf4] checkpoint_epoch={{epoch}} global_step={{global_step}} epoch_progress={{epoch_progress}}")
if epoch != expected_epoch:
    print(
        f"[resume-h100x8-fmsf4] expected checkpoint epoch {{expected_epoch}}, got {{epoch}}",
        file=sys.stderr,
    )
    sys.exit(2)
PY

HYDRA_OVERRIDES=(
  "experiment=${{EXPERIMENT}}"
  "action=fit"
  "task_name=${{TASK_NAME}}"
  "ckpt_path=${{CKPT_PATH}}"
  "paths.cache_root=${{CACHE_ROOT}}"
  "trainer.devices=${{NPROC_PER_NODE}}"
  "trainer.num_nodes=1"
  "trainer.max_epochs=${{MAX_EPOCHS}}"
  "trainer.check_val_every_n_epoch=${{CHECK_VAL_EVERY_N_EPOCH}}"
  "trainer.limit_val_batches=${{LIMIT_VAL_BATCHES}}"
  "+trainer.use_distributed_sampler=${{USE_DISTRIBUTED_SAMPLER}}"
  "data.train_batch_size=${{TRAIN_BATCH_SIZE}}"
  "data.val_batch_size=${{VAL_BATCH_SIZE}}"
  "data.test_batch_size=${{TEST_BATCH_SIZE}}"
  "data.train_memory_balanced_batches=${{TRAIN_MEMORY_BALANCED_BATCHES}}"
  "model.model_config.lr=${{LEARNING_RATE}}"
  "model.model_config.lr_warmup_steps=${{LR_WARMUP_STEPS}}"
)
if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_OVERRIDES=( $CATK_EXTRA_OVERRIDES )
  HYDRA_OVERRIDES+=("${{EXTRA_OVERRIDES[@]}}")
fi

printf '[resume-h100x8-fmsf4] torchrun command:'
printf ' %q' torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}"
printf '\\n'

torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" -m src.run "${{HYDRA_OVERRIDES[@]}}" 2>&1 | tee "$ATTEMPT_LOG"
status="${{PIPESTATUS[0]}}"
echo "[resume-h100x8-fmsf4] torchrun exited with status $status"
exit "$status"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--train-batch-size", type=int, default=18)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=64)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=8)
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--learning-rate", default="6.5e-4")
    parser.add_argument("--lr-warmup-steps", type=int, default=5)
    parser.add_argument("--expected-ckpt-epoch", type=int, default=24)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--use-distributed-sampler", default="false")
    parser.add_argument("--train-memory-balanced-batches", default="true")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--replace", action="store_true", help="Kill an existing tmux session with the same name before launching.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = run_root(args)
    env_file = f"{root}/resume_epoch25.env"
    worker_file = f"{root}/resume_epoch25_worker.sh"
    tmux_log = f"{root}/{args.pod}.tmux.log"

    if args.replace:
        run_kubectl(
            [
                "-n",
                args.namespace,
                "exec",
                args.pod,
                "-c",
                args.container,
                "--",
                "bash",
                "-lc",
                f"tmux has-session -t {shq(args.session)} 2>/dev/null && tmux kill-session -t {shq(args.session)} || true",
            ],
            dry_run=args.dry_run,
        )

    remote_script = (
        f"mkdir -p {shq(root)}\n"
        f"cat > {shq(env_file)} <<'CATK_ENV'\n{render_env(args)}CATK_ENV\n"
        f"cat > {shq(worker_file)} <<'CATK_WORKER'\n{render_worker_script(args.project_root, env_file, args.branch, args.pull)}CATK_WORKER\n"
        f"chmod +x {shq(worker_file)}\n"
        f": > {shq(tmux_log)}\n"
        f"tmux new-session -d -s {shq(args.session)} \"bash {shq(worker_file)} 2>&1 | tee -a {shq(tmux_log)}\"\n"
        f"echo {shq(tmux_log)}\n"
    )

    output = run_kubectl(
        [
            "-n",
            args.namespace,
            "exec",
            args.pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            remote_script,
        ],
        capture=True,
        dry_run=args.dry_run,
    )
    if output:
        print(output)


if __name__ == "__main__":
    main()
