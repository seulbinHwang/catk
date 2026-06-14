#!/usr/bin/env bash
# Run SMART CAT-K/CLSFT fine-tuning on the testas A100x7 pod.
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

activate_conda_if_available() {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && command -v python >/dev/null 2>&1 && command -v torchrun >/dev/null 2>&1; then
    log "conda env already active: ${CONDA_DEFAULT_ENV}"
    return 0
  fi

  local conda_root="${CONDA_ROOT:-/mnt/nuplan/miniforge}"
  if [[ -f "$conda_root/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$conda_root/etc/profile.d/conda.sh"
    conda activate "${CATK_CONDA_ENV:-catk}" 2>/dev/null \
      || conda activate base 2>/dev/null \
      || true
    log "conda env: ${CONDA_DEFAULT_ENV:-unknown}"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CATK_CONDA_ENV:-catk}" 2>/dev/null \
      || conda activate base 2>/dev/null \
      || true
    log "conda env: ${CONDA_DEFAULT_ENV:-unknown}"
    return 0
  fi

  log "conda not found; using current Python."
}

download_wandb_ckpt() {
  local artifact="$1"
  local output_dir="$2"
  mkdir -p "$output_dir"
  CKPT_ARTIFACT="$artifact" CKPT_DOWNLOAD_DIR="$output_dir" python - <<'PY'
import os
from pathlib import Path

import wandb

artifact_name = os.environ["CKPT_ARTIFACT"]
download_dir = Path(os.environ["CKPT_DOWNLOAD_DIR"])
api = wandb.Api()
artifact = api.artifact(artifact_name, type="model")
path = Path(artifact.download(root=download_dir.as_posix()))
ckpt = path / "epoch_last.ckpt"
if not ckpt.exists():
    candidates = sorted(path.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt file found under {path}")
    ckpt = candidates[0]
print(ckpt.as_posix())
PY
}

main() {
  export LOGLEVEL="${LOGLEVEL:-INFO}"
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
  export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
  export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"

  activate_conda_if_available

  local cache_root="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
  local nproc_per_node="${NPROC_PER_NODE:-7}"
  local trainer_devices="${TRAINER_DEVICES:-7}"
  local master_addr="${MASTER_ADDR:-127.0.0.1}"
  local master_port="${MASTER_PORT:-29573}"
  local experiment="${CATK_EXPERIMENT:-clsft}"
  local action="${CATK_ACTION:-finetune}"
  local task_name="${TASK_NAME:-smart_clsft_testas_a100x7}"
  local run_id="${CATK_RUN_ID:-}"
  local ckpt_path="${CATK_CKPT_PATH:-${CKPT_PATH:-}}"
  local ckpt_artifact="${CATK_CKPT_ARTIFACT:-${CKPT_ARTIFACT:-}}"
  local ckpt_download_dir="${CATK_CKPT_DOWNLOAD_DIR:-${CKPT_DOWNLOAD_DIR:-/workspace/checkpoints/smart_clsft_testas_a100x7}}"

  case "$action" in
    finetune|validate|test) ;;
    *)
      log "ERROR: CATK_ACTION must be finetune, validate, or test; got: $action"
      exit 2
      ;;
  esac
  if [[ -z "$ckpt_path" ]]; then
    if [[ -z "$ckpt_artifact" ]]; then
      log "ERROR: CATK_CKPT_PATH/CKPT_PATH or CATK_CKPT_ARTIFACT/CKPT_ARTIFACT is required."
      exit 2
    fi
    log "downloading checkpoint artifact: $ckpt_artifact"
    ckpt_path="$(download_wandb_ckpt "$ckpt_artifact" "$ckpt_download_dir" | tail -1)"
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    log "ERROR: checkpoint does not exist: $ckpt_path"
    exit 2
  fi
  if [[ ! -d "$cache_root" ]]; then
    log "ERROR: CACHE_ROOT does not exist: $cache_root"
    exit 2
  fi

  local extra_overrides=()
  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides <<< "$CATK_HYDRA_OVERRIDES"
  fi

  local app_args=(
    -m src.run
    experiment="$experiment"
    action="$action"
    trainer=ddp
    trainer.devices="$trainer_devices"
    trainer.num_nodes=1
    ++trainer.enable_progress_bar=true
    trainer.precision=bf16-mixed
    trainer.strategy.find_unused_parameters=true
    trainer.sync_batchnorm=false
    +trainer.use_distributed_sampler=false
    paths.cache_root="$cache_root"
    task_name="$task_name"
    ckpt_path="$ckpt_path"
    data.train_batch_size="${TRAIN_BATCH_SIZE:-10}"
    data.val_batch_size="${VAL_BATCH_SIZE:-10}"
    data.test_batch_size="${TEST_BATCH_SIZE:-10}"
    data.train_use_eval_agent_selection="${TRAIN_USE_EVAL_AGENT_SELECTION:-false}"
    model.model_config.decoder.num_freq_bands="${NUM_FREQ_BANDS:-88}"
    model.model_config.scorer_scene_num="${SCORER_SCENE_NUM:-1680}"
    logger.wandb.group="${WANDB_GROUP:-smart_clsft_testas_a100x7}"
    logger.wandb.job_type=catk_finetune
  )
  if [[ -n "$run_id" ]]; then
    app_args+=("hydra.run.dir=${LOG_DIR:-${PWD}/logs}/${task_name}/runs/${run_id}")
  fi
  if [[ -n "${LOG_DIR:-}" ]]; then
    app_args+=(paths.log_dir="$LOG_DIR")
  fi
  if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES")
  fi
  if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_val_batches="$LIMIT_VAL_BATCHES")
  fi
  if [[ -n "${LIMIT_TEST_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_test_batches="$LIMIT_TEST_BATCHES")
  fi
  if [[ -n "${MAX_EPOCHS:-}" ]]; then
    app_args+=(trainer.max_epochs="$MAX_EPOCHS")
  fi
  if [[ -n "${CATK_LR:-}" ]]; then
    app_args+=(model.model_config.lr="$CATK_LR")
  fi
  if (( ${#extra_overrides[@]} > 0 )); then
    app_args+=("${extra_overrides[@]}")
  fi
  app_args+=("$@")

  local torchrun_args=(
    --nnodes 1
    --nproc_per_node "$nproc_per_node"
    --master_addr "$master_addr"
    --master_port "$master_port"
    "${app_args[@]}"
  )

  log "starting SMART CAT-K fine-tuning testas A100x7 run"
  log "  experiment:       $experiment"
  log "  action:           $action"
  log "  task_name:        $task_name"
  log "  run_id:           ${run_id:-auto}"
  log "  nproc_per_node:   $nproc_per_node"
  log "  trainer.devices:  $trainer_devices"
  log "  master_addr:      $master_addr"
  log "  master_port:      $master_port"
  log "  cache_root:       $cache_root"
  log "  ckpt_path:        $ckpt_path"
  log "  ckpt_artifact:    ${ckpt_artifact:-none}"
  log "  train_batch_size: ${TRAIN_BATCH_SIZE:-10}"
  log "torchrun command:"
  printf '  %q' torchrun "${torchrun_args[@]}"
  printf '\n'
  exec torchrun "${torchrun_args[@]}"
}

main "$@"
