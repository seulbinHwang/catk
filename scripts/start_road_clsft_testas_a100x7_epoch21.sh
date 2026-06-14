#!/usr/bin/env bash
# Run RoaD-style fine-tuning on the static testas A100x7 pod.
#
# This script is meant to run inside the pod from a main-branch checkout. It
# downloads the requested W&B checkpoint artifact if needed, then launches a
# single-node DDP RoaD fine-tune job over all seven visible A100 GPUs.
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
  local task_name="${TASK_NAME:-road_clsft_a100x7_testas_main_1iapr5ed_epoch21_bs12_$(date '+%Y%m%d_%H%M%S')}"
  local log_dir="${LOG_DIR:-${PWD}/logs}"
  local nproc_per_node="${NPROC_PER_NODE:-7}"
  local trainer_devices="${TRAINER_DEVICES:-7}"
  local train_batch_size="${TRAIN_BATCH_SIZE:-12}"
  local val_batch_size="${VAL_BATCH_SIZE:-12}"
  local test_batch_size="${TEST_BATCH_SIZE:-12}"
  local road_cache_root="${ROAD_CACHE_ROOT:-/workspace/road_cache/${task_name}}"
  local road_data_use_ratio="${ROAD_DATA_USE_RATIO:-1.0}"
  local road_num_rollouts="${ROAD_NUM_ROLLOUTS_PER_SCENARIO:-1}"
  local road_generation_batch_size="${ROAD_GENERATION_BATCH_SIZE:-160}"
  local road_num_workers="${ROAD_NUM_WORKERS:-10}"
  local ckpt_artifact="${CKPT_ARTIFACT:-jksg01019-naver-labs/SMART-FLOW/epoch-last-1iapr5ed:v21}"
  local ckpt_download_dir="${CKPT_DOWNLOAD_DIR:-/workspace/checkpoints/road_clsft_epoch21_1iapr5ed_v21}"
  local ckpt_path="${CKPT_PATH:-}"

  if [[ ! -d "$cache_root" ]]; then
    log "ERROR: CACHE_ROOT does not exist: $cache_root"
    exit 2
  fi

  if [[ -z "$ckpt_path" ]]; then
    log "downloading checkpoint artifact: $ckpt_artifact"
    ckpt_path="$(download_wandb_ckpt "$ckpt_artifact" "$ckpt_download_dir" | tail -1)"
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    log "ERROR: checkpoint does not exist: $ckpt_path"
    exit 2
  fi

  local app_args=(
    -m src.run
    experiment=road_clsft
    action=road_finetune
    trainer=ddp
    trainer.devices="$trainer_devices"
    trainer.num_nodes=1
    ++trainer.enable_progress_bar=true
    paths.cache_root="$cache_root"
    paths.log_dir="$log_dir"
    task_name="$task_name"
    ckpt_path="$ckpt_path"
    data.train_batch_size="$train_batch_size"
    data.val_batch_size="$val_batch_size"
    data.test_batch_size="$test_batch_size"
    data.train_use_eval_agent_selection=true
    road.cache_root_dir="$road_cache_root"
    road.road_data_use_ratio="$road_data_use_ratio"
    road.num_rollouts_per_scenario="$road_num_rollouts"
    road.generation_batch_size="$road_generation_batch_size"
    road.num_workers="$road_num_workers"
    road.pin_memory=true
    model.model_config.scorer_scene_num="${SCORER_SCENE_NUM:-1680}"
    model.model_config.decoder.num_freq_bands="${NUM_FREQ_BANDS:-88}"
  )

  if [[ -n "${MAX_EPOCHS:-}" ]]; then
    app_args+=(trainer.max_epochs="$MAX_EPOCHS")
  fi
  if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES")
  fi
  if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_val_batches="$LIMIT_VAL_BATCHES")
  fi
  if [[ -n "${ROAD_DELETE_AFTER_USE:-}" ]]; then
    app_args+=(road.delete_after_use="$ROAD_DELETE_AFTER_USE")
  fi
  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_overrides=( $CATK_HYDRA_OVERRIDES )
    app_args+=("${extra_overrides[@]}")
  fi
  app_args+=("$@")

  log "starting RoaD fine-tuning"
  log "  task_name:       $task_name"
  log "  checkpoint:      $ckpt_path"
  log "  ckpt artifact:   $ckpt_artifact"
  log "  cache_root:      $cache_root"
  log "  road cache root: $road_cache_root"
  log "  train bs/rank:   $train_batch_size"
  log "  nproc_per_node:  $nproc_per_node"
  log "  trainer.devices: $trainer_devices"
  log "  road ratio:      $road_data_use_ratio"
  log "  road rollouts:   $road_num_rollouts"
  log "  road gen bs/rank:$road_generation_batch_size"
  log "torchrun command:"
  printf '  %q' torchrun --nnodes 1 --nproc_per_node "$nproc_per_node" "${app_args[@]}"
  printf '\n'

  exec torchrun --nnodes 1 --nproc_per_node "$nproc_per_node" "${app_args[@]}"
}

main "$@"
