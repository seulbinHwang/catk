#!/usr/bin/env python3
"""Launch closed-loop self-forcing DMD training on the static fm-sf-3 H100x8 pod."""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-3"
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/tmp/catk_self_forcing_closed_loop_fmsf3"
DEFAULT_BRANCH = "self_forcing_closed_loop"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/tmp/catk_self_forcing_closed_loop_logs"
DEFAULT_SESSION = "catk-closed-loop-sf-h100x8-fmsf3"
DEFAULT_TASK = "flow_closed_loop_self_forced_h100x8_fmsf3"
DEFAULT_EXPERIMENT = "self_forced_npfm_h100_6"
DEFAULT_PRETRAIN_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-mqfq3u39:v121"
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_closed_loop_self_forced_h100x8_fmsf3_pretrain_epoch116_mqfq3u39"
    "/v121/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_closed_loop_self_forced_h100x8_fmsf3_pretrain_epoch116_mqfq3u39"
    "/v121/artifact"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(cmd: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shq(part) for part in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=True, text=True)


def kubectl_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "kubectl",
        "-n",
        args.namespace,
        "exec",
        args.pod,
        "-c",
        args.container,
        "--",
    ]


def kubectl_bash(args: argparse.Namespace, script: str) -> list[str]:
    return [*kubectl_prefix(args), "bash", "-lc", script]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def env_line(key: str, value: object) -> str:
    return f"export {key}={shq(value)}"


def render_env(args: argparse.Namespace) -> str:
    generated_estimator_lr = args.generated_estimator_learning_rate or args.learning_rate
    env = {
        "PROJECT_ROOT": args.project_root,
        "EXPERIMENT": args.experiment,
        "ACTION": args.action,
        "TASK_NAME": args.task_name,
        "CACHE_ROOT": args.cache_root,
        "LOG_DIR": args.log_dir,
        "PRETRAIN_CKPT": args.pretrain_ckpt,
        "WANDB_PRETRAIN_ARTIFACT": args.wandb_pretrain_artifact,
        "WANDB_PRETRAIN_DOWNLOAD_DIR": args.pretrain_download_dir,
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "NPROC_PER_NODE": args.nproc_per_node,
        "MASTER_PORT": args.master_port,
        "INITIAL_BS": args.initial_bs,
        "OOM_STEP": args.oom_step,
        "MIN_BS": args.min_bs,
        "CATK_LR": args.learning_rate,
        "GENERATED_ESTIMATOR_LR": generated_estimator_lr,
        "ESTIMATOR_WARMUP_EPOCHS": args.estimator_warmup_epochs,
        "MAX_EPOCHS": args.max_epochs,
        "CHECK_VAL_EVERY_N_EPOCH": args.check_val_every_n_epoch,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "TEST_BATCH_SIZE": args.test_batch_size,
        "NUM_WORKERS": args.num_workers,
        "PREFETCH_FACTOR": args.prefetch_factor,
        "TRAIN_EPOCH_SAMPLE_FRACTION": args.train_epoch_sample_fraction,
        "TRAIN_MEMORY_BALANCED_BATCHES": bool_text(args.train_memory_balanced_batches),
        "TRAIN_EPOCH_SAMPLE_FRACTION_SHUFFLE_FLAG": bool_text(
            args.train_epoch_sample_fraction_shuffle_flag
        ),
        "ESTIMATOR_UPDATES_PER_STEP": args.estimator_updates_per_step,
        "UNFROZEN_RANGE": args.unfrozen_range,
        "PROJECT_DMD_TO_POSE_SPACE": bool_text(args.project_dmd_to_pose_space),
        "DMD_USE_STABLE_SCALE_FILTER": bool_text(args.dmd_use_stable_scale_filter),
        "DMD_STABLE_SCALE_SCOPE": args.dmd_stable_scale_scope,
        "DMD_USE_TEACHER_ALIGNMENT_FILTER": bool_text(args.dmd_use_teacher_alignment_filter),
        "DMD_USE_TRUST_REGION_FILTER": bool_text(args.dmd_use_trust_region_filter),
        "DMD_USE_INJECTION_RAMP": bool_text(args.dmd_use_injection_ramp),
        "DETACH_BLOCK_TRANSITION": bool_text(args.detach_block_transition),
        "USE_ANCHOR_FLOW_MATCHING_LOSS": bool_text(args.use_anchor_flow_matching_loss),
        "ANCHOR_WEIGHT": args.anchor_weight,
        "SAMPLE_STEPS": args.sample_steps,
        "SAMPLE_METHOD": args.sample_method,
        "NOISE_SCALE": args.noise_scale,
        "BACKPROP_LAST_K": args.backprop_last_k,
        "RANDOM_TERMINAL_SCOPE": args.random_terminal_scope,
        "RANDOM_TERMINAL_POLICY": args.random_terminal_policy,
        "MIN_EXECUTED_STEPS": args.min_executed_steps,
        "CLOSED_LOOP_SF_GLOBAL_MAX_STEP": args.closed_loop_sf_global_max_step,
        "CLOSED_LOOP_SF_LOCAL_MAX_STEP": args.closed_loop_sf_local_max_step,
        "UPDATE_OPEN_LOOP_TEACHER_WHEN_ROLL": bool_text(args.update_open_loop_teacher_when_roll),
        "GENERATED_ESTIMATOR_INIT_PATH": args.generated_estimator_init_path,
        "GENERATED_ESTIMATOR_SKIP_WARMUP_ON_LOAD": bool_text(
            args.generated_estimator_skip_warmup_on_load
        ),
        "WANDB_OFFLINE": bool_text(args.wandb_offline),
        "CATK_EXTRA_OVERRIDES": args.extra_hydra_overrides,
        "PYTHONUNBUFFERED": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    return "\n".join(env_line(key, value) for key, value in env.items()) + "\n"


def render_worker_script() -> str:
    return r"""#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_PROJECT_ROOT}/.closed_loop_h100x8_env"

mkdir -p "${LOG_DIR}/${TASK_NAME}"
RUN_ROOT="${LOG_DIR}/${TASK_NAME}"
cd "${PROJECT_ROOT}"

echo "[launcher] pod=$(hostname)"
echo "[launcher] project_root=${PROJECT_ROOT}"
echo "[launcher] git_head=$(git rev-parse HEAD 2>/dev/null || echo no-git)"
echo "[launcher] experiment=${EXPERIMENT} action=${ACTION} task=${TASK_NAME}"
echo "[launcher] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc_per_node=${NPROC_PER_NODE}"
echo "[launcher] closed_loop global=${CLOSED_LOOP_SF_GLOBAL_MAX_STEP} local=${CLOSED_LOOP_SF_LOCAL_MAX_STEP} update_teacher=${UPDATE_OPEN_LOOP_TEACHER_WHEN_ROLL}"
echo "[launcher] batch fallback: ${INITIAL_BS} down to ${MIN_BS} by ${OOM_STEP}"

source ~/.bashrc || true
if command -v conda >/dev/null 2>&1; then
  conda activate catk || true
fi

ensure_pretrain_checkpoint() {
  if [[ "${ACTION}" == "fit" ]]; then
    return 0
  fi
  if [[ -n "${PRETRAIN_CKPT}" && -f "${PRETRAIN_CKPT}" ]]; then
    echo "[launcher] found pretrain checkpoint: ${PRETRAIN_CKPT}"
    return 0
  fi
  if [[ -z "${WANDB_PRETRAIN_ARTIFACT}" || -z "${WANDB_PRETRAIN_DOWNLOAD_DIR}" ]]; then
    echo "[launcher] ERROR: pretrain checkpoint missing and no W&B artifact was provided." >&2
    return 1
  fi
  echo "[launcher] downloading pretrain artifact ${WANDB_PRETRAIN_ARTIFACT}"
  mkdir -p "${WANDB_PRETRAIN_DOWNLOAD_DIR}"
  python - <<PY
import os
import wandb
artifact_name = os.environ["WANDB_PRETRAIN_ARTIFACT"]
download_dir = os.environ["WANDB_PRETRAIN_DOWNLOAD_DIR"]
api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact.download(root=download_dir)
PY
  if [[ -n "${PRETRAIN_CKPT}" && ! -f "${PRETRAIN_CKPT}" ]]; then
    candidate="$(find "${WANDB_PRETRAIN_DOWNLOAD_DIR}" -name '*.ckpt' | sort | tail -n 1)"
    if [[ -n "${candidate}" ]]; then
      mkdir -p "$(dirname "${PRETRAIN_CKPT}")"
      ln -sf "${candidate}" "${PRETRAIN_CKPT}"
    fi
  fi
  [[ -n "${PRETRAIN_CKPT}" && -f "${PRETRAIN_CKPT}" ]]
}

append_extra_overrides() {
  if [[ -n "${CATK_EXTRA_OVERRIDES}" ]]; then
    # Extra overrides are expected to be whitespace-separated Hydra tokens.
    read -r -a extra_overrides <<< "${CATK_EXTRA_OVERRIDES}"
    HYDRA_OVERRIDES+=("${extra_overrides[@]}")
  fi
}

find_latest_resume_checkpoint() {
  local search_root="${PROJECT_ROOT}/logs/${TASK_NAME}/runs"
  if [[ ! -d "${search_root}" ]]; then
    return 0
  fi
  find "${search_root}" -type f \
    \( -name 'epoch_last.ckpt' -o -name 'last.ckpt' -o -name 'epoch_*.ckpt' \) \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

build_overrides() {
  local current_bs="$1"
  local resume_ckpt="${2:-}"
  HYDRA_OVERRIDES=(
    "experiment=${EXPERIMENT}"
    "action=${ACTION}"
    "task_name=${TASK_NAME}"
    "paths.cache_root=${CACHE_ROOT}"
    "trainer.devices=${NPROC_PER_NODE}"
    "trainer.max_epochs=${MAX_EPOCHS}"
    "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
    "trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}"
    "trainer.limit_val_batches=${LIMIT_VAL_BATCHES}"
    "data.train_batch_size=${current_bs}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.test_batch_size=${TEST_BATCH_SIZE}"
    "data.num_workers=${NUM_WORKERS}"
    "data.train_epoch_sample_fraction=${TRAIN_EPOCH_SAMPLE_FRACTION}"
    "data.train_memory_balanced_batches=${TRAIN_MEMORY_BALANCED_BATCHES}"
    "data.train_epoch_sample_fraction_shuffle_flag=${TRAIN_EPOCH_SAMPLE_FRACTION_SHUFFLE_FLAG}"
    "model.model_config.lr=${CATK_LR}"
    "model.model_config.self_forced.generated_estimator_lr=${GENERATED_ESTIMATOR_LR}"
    "model.model_config.self_forced.estimator_warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}"
    "model.model_config.self_forced.estimator_updates_per_step=${ESTIMATOR_UPDATES_PER_STEP}"
    "model.model_config.self_forced.unfrozen_range=${UNFROZEN_RANGE}"
    "model.model_config.self_forced.project_dmd_to_pose_space=${PROJECT_DMD_TO_POSE_SPACE}"
    "model.model_config.self_forced.dmd_use_stable_scale_filter=${DMD_USE_STABLE_SCALE_FILTER}"
    "model.model_config.self_forced.dmd_stable_scale_scope=${DMD_STABLE_SCALE_SCOPE}"
    "model.model_config.self_forced.dmd_use_teacher_alignment_filter=${DMD_USE_TEACHER_ALIGNMENT_FILTER}"
    "model.model_config.self_forced.dmd_use_trust_region_filter=${DMD_USE_TRUST_REGION_FILTER}"
    "model.model_config.self_forced.dmd_use_injection_ramp=${DMD_USE_INJECTION_RAMP}"
    "model.model_config.self_forced.detach_block_transition=${DETACH_BLOCK_TRANSITION}"
    "model.model_config.self_forced.use_anchor_flow_matching_loss=${USE_ANCHOR_FLOW_MATCHING_LOSS}"
    "model.model_config.self_forced.anchor_weight=${ANCHOR_WEIGHT}"
    "model.model_config.self_forced.sampling.sample_steps=${SAMPLE_STEPS}"
    "model.model_config.self_forced.sampling.sample_method=${SAMPLE_METHOD}"
    "model.model_config.self_forced.sampling.noise_scale=${NOISE_SCALE}"
    "model.model_config.self_forced.sampling.backprop_last_k=${BACKPROP_LAST_K}"
    "model.model_config.self_forced.sampling.random_terminal_step.scope=${RANDOM_TERMINAL_SCOPE}"
    "model.model_config.self_forced.sampling.random_terminal_step.policy=${RANDOM_TERMINAL_POLICY}"
    "model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps=${MIN_EXECUTED_STEPS}"
    "model.model_config.self_forced.closed_loop_sf_global_max_step=${CLOSED_LOOP_SF_GLOBAL_MAX_STEP}"
    "model.model_config.self_forced.closed_loop_sf_local_max_step=${CLOSED_LOOP_SF_LOCAL_MAX_STEP}"
    "model.model_config.self_forced.update_open_loop_teacher_when_roll=${UPDATE_OPEN_LOOP_TEACHER_WHEN_ROLL}"
    "model.model_config.self_forced.generated_estimator_skip_warmup_on_load=${GENERATED_ESTIMATOR_SKIP_WARMUP_ON_LOAD}"
    "logger.wandb.offline=${WANDB_OFFLINE}"
  )
  if [[ "${NUM_WORKERS}" == "0" ]]; then
    HYDRA_OVERRIDES+=("data.persistent_workers=false" "data.prefetch_factor=null")
  else
    HYDRA_OVERRIDES+=("data.prefetch_factor=${PREFETCH_FACTOR}")
  fi
  if [[ -n "${resume_ckpt}" ]]; then
    HYDRA_OVERRIDES+=("ckpt_path=${resume_ckpt}")
  elif [[ "${ACTION}" != "fit" && -n "${PRETRAIN_CKPT}" ]]; then
    HYDRA_OVERRIDES+=("ckpt_path=${PRETRAIN_CKPT}")
  fi
  if [[ -n "${GENERATED_ESTIMATOR_INIT_PATH}" ]]; then
    HYDRA_OVERRIDES+=("model.model_config.self_forced.generated_estimator_init_path=${GENERATED_ESTIMATOR_INIT_PATH}")
  fi
  append_extra_overrides
}

run_attempt() {
  local current_bs="$1"
  local resume_ckpt="${2:-}"
  ATTEMPT_LOG="${RUN_ROOT}/${TASK_NAME}.bs${current_bs}.torchrun.log"
  build_overrides "${current_bs}" "${resume_ckpt}"
  echo "[launcher] starting attempt batch_size=${current_bs}; log=${ATTEMPT_LOG}"
  if [[ -n "${resume_ckpt}" ]]; then
    echo "[launcher] resuming from checkpoint: ${resume_ckpt}"
  fi
  printf '[launcher] hydra overrides:'
  printf ' %q' "${HYDRA_OVERRIDES[@]}"
  printf '\n'

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
  NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}" \
  torchrun --standalone --master-port "${MASTER_PORT}" --nproc-per-node="${NPROC_PER_NODE}" \
    -m src.run "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "${ATTEMPT_LOG}"
  return "${PIPESTATUS[0]}"
}

ensure_pretrain_checkpoint || exit 2

bs="${INITIAL_BS}"
resume_ckpt=""
while true; do
  run_attempt "${bs}" "${resume_ckpt}"
  status="$?"
  if [[ "${status}" == "0" ]]; then
    echo "[launcher] training finished successfully at batch_size=${bs}"
    break
  fi
  if grep -Eqi 'OutOfMemory|CUDA out of memory' "${ATTEMPT_LOG}" && (( bs - OOM_STEP >= MIN_BS )); then
    next_bs=$((bs - OOM_STEP))
    resume_ckpt="$(find_latest_resume_checkpoint || true)"
    if [[ -n "${resume_ckpt}" ]]; then
      echo "[launcher] OOM detected at batch_size=${bs}; retrying with batch_size=${next_bs} from ${resume_ckpt}"
    else
      echo "[launcher] OOM detected at batch_size=${bs}; retrying with batch_size=${next_bs} from original checkpoint"
    fi
    bs="${next_bs}"
    continue
  fi
  echo "[launcher] training failed with status=${status}; no safe retry condition matched"
  exit "${status}"
done

echo "[launcher] worker complete; leaving tmux pane open for inspection"
exec bash
"""


def render_monitor_script() -> str:
    return r"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_PROJECT_ROOT}/.closed_loop_h100x8_env"
mkdir -p "${LOG_DIR}/${TASK_NAME}"
while true; do
  echo "===== $(date -Iseconds) ====="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo
  sleep 30
done
"""


def write_remote_file(args: argparse.Namespace, path: str, content: str, mode: str = "0644") -> None:
    quoted_path = shq(path)
    cmd = [
        "kubectl",
        "-n",
        args.namespace,
        "exec",
        "-i",
        args.pod,
        "-c",
        args.container,
        "--",
        "bash",
        "-lc",
        f"cat > {quoted_path} && chmod {shq(mode)} {quoted_path}",
    ]
    print("+", " ".join(shq(part) for part in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, input=content, check=True, text=True)


def ensure_remote_project(args: argparse.Namespace) -> None:
    if args.pull:
        script = f"""
set -euo pipefail
if [[ ! -d {shq(args.project_root)}/.git ]]; then
  echo "ERROR: {args.project_root} is not a git checkout. Create it first or pass --no-pull." >&2
  exit 1
fi
cd {shq(args.project_root)}
git fetch origin {shq(args.branch)}
git checkout {shq(args.branch)}
git pull --ff-only origin {shq(args.branch)}
"""
    else:
        script = f"""
set -euo pipefail
if [[ ! -d {shq(args.project_root)} ]]; then
  echo "ERROR: {args.project_root} does not exist." >&2
  exit 1
fi
if [[ ! -d {shq(args.project_root)}/src ]]; then
  echo "ERROR: {args.project_root} does not look like the catk project root." >&2
  exit 1
fi
"""
    run(kubectl_bash(args, script), dry_run=args.dry_run)


def stop_session(args: argparse.Namespace) -> None:
    script = f"""
set +e
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "Sending C-c to tmux session {args.session}"
  tmux send-keys -t {shq(args.session)} C-c
  sleep 10
  if tmux has-session -t {shq(args.session)} 2>/dev/null; then
    tmux send-keys -t {shq(args.session)} C-c
    sleep 5
  fi
  if tmux has-session -t {shq(args.session)} 2>/dev/null; then
    tmux kill-session -t {shq(args.session)}
  fi
fi
"""
    run(kubectl_bash(args, script), dry_run=args.dry_run)


def start_session(args: argparse.Namespace) -> None:
    validate_args(args)
    ensure_remote_project(args)
    env_path = f"{args.project_root}/.closed_loop_h100x8_env"
    worker_path = f"{args.project_root}/scripts/.closed_loop_h100x8_worker.sh"
    monitor_path = f"{args.project_root}/scripts/.closed_loop_h100x8_monitor.sh"
    write_remote_file(args, env_path, render_env(args), "0644")
    write_remote_file(args, worker_path, render_worker_script(), "0755")
    write_remote_file(args, monitor_path, render_monitor_script(), "0755")

    if args.replace:
        stop_session(args)

    monitor_part = ""
    if not args.no_monitor_pane:
        monitor_part = (
            f"tmux split-window -h -t {shq(args.session)} "
            f"{shq(f'bash {monitor_path} 2>&1 | tee -a {args.log_dir}/{args.task_name}/monitor.log')};"
            f"tmux select-layout -t {shq(args.session)} tiled;"
        )

    script = f"""
set -euo pipefail
mkdir -p {shq(args.log_dir)}/{shq(args.task_name)}
tmux new-session -d -s {shq(args.session)} {shq(f'bash {worker_path} 2>&1 | tee -a {args.log_dir}/{args.task_name}/{args.pod}.tmux.log')}
{monitor_part}
tmux ls
echo "log: {args.log_dir}/{args.task_name}/{args.pod}.tmux.log"
"""
    run(kubectl_bash(args, script), dry_run=args.dry_run)


def validate_args(args: argparse.Namespace) -> None:
    devices = [device.strip() for device in args.cuda_visible_devices.split(",") if device.strip()]
    if args.nproc_per_node != 8:
        raise SystemExit("This fm-sf-3 launcher is intentionally fixed to --nproc-per-node 8.")
    if len(devices) != 8:
        raise SystemExit("--cuda-visible-devices must list exactly 8 GPU ids.")
    if args.initial_bs < args.min_bs:
        raise SystemExit("--initial-bs must be >= --min-bs.")
    if args.oom_step <= 0:
        raise SystemExit("--oom-step must be positive.")
    if args.action != "fit" and not args.pretrain_ckpt and not args.wandb_pretrain_artifact:
        raise SystemExit("finetune/validate/test require --pretrain-ckpt or --wandb-pretrain-artifact.")
    if args.closed_loop_sf_global_max_step < 0:
        raise SystemExit("--closed-loop-sf-global-max-step must be >= 0.")
    if args.closed_loop_sf_local_max_step < 1:
        raise SystemExit("--closed-loop-sf-local-max-step must be >= 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch closed-loop self-forcing training on fm-sf-3 with H100 x8."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--pull", dest="pull", action="store_true", default=True)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--action", default="finetune", choices=["fit", "finetune", "validate", "test"])
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--wandb-pretrain-artifact", default=DEFAULT_PRETRAIN_ARTIFACT)
    parser.add_argument("--pretrain-download-dir", default=DEFAULT_PRETRAIN_DOWNLOAD_DIR)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29680)
    parser.add_argument("--initial-bs", type=int, default=72)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--learning-rate", default="5e-5")
    parser.add_argument("--generated-estimator-learning-rate", default="5e-5")
    parser.add_argument("--estimator-warmup-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--limit-train-batches", default="1.0")
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--train-epoch-sample-fraction", default="0.25")
    parser.add_argument("--train-memory-balanced-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-epoch-sample-fraction-shuffle-flag",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--estimator-updates-per-step", type=int, default=5)
    parser.add_argument("--unfrozen-range", default="middle")
    parser.add_argument("--project-dmd-to-pose-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dmd-use-stable-scale-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dmd-stable-scale-scope", default="agent")
    parser.add_argument("--dmd-use-teacher-alignment-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dmd-use-trust-region-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dmd-use-injection-ramp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detach-block-transition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-anchor-flow-matching-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-weight", default="0.1")
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--sample-method", default="euler")
    parser.add_argument("--noise-scale", default="1.0")
    parser.add_argument("--backprop-last-k", type=int, default=8)
    parser.add_argument("--random-terminal-scope", default="global_batch")
    parser.add_argument("--random-terminal-policy", default="all")
    parser.add_argument("--min-executed-steps", type=int, default=16)
    parser.add_argument("--closed-loop-sf-global-max-step", type=int, default=3)
    parser.add_argument("--closed-loop-sf-local-max-step", type=int, default=4)
    parser.add_argument(
        "--update-open-loop-teacher-when-roll",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--generated-estimator-init-path", default="")
    parser.add_argument(
        "--generated-estimator-skip-warmup-on-load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop:
        stop_session(args)
        return
    start_session(args)


if __name__ == "__main__":
    main()
