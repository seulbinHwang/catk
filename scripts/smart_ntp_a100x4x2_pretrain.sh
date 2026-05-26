#!/usr/bin/env bash
# Run SMART NTP pretrain across two existing static A100x4 pods/nodes.
#
# This script is executed inside every node. For the testa/testaa pair,
# prefer launching it through scripts/launch_smart_ntp_a100x4x2_testa.py.
# Manual use is also supported by setting NODE_RANK, MASTER_ADDR, MASTER_PORT,
# NNODES, and NPROC_PER_NODE. It never creates, deletes, or restarts pods.
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

default_cache_root() {
  local pod_name
  pod_name="$(hostname)"
  case "$pod_name" in
    testa*|testaa*)
      printf '%s\n' "/workspace/womd_v1_3/SMART_cache"
      ;;
    *)
      printf '%s\n' "/workspace/womd_v1_3/SMART_cache"
      ;;
  esac
}

activate_conda_if_available() {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    if command -v python >/dev/null 2>&1 && command -v torchrun >/dev/null 2>&1; then
      log "conda env already active: ${CONDA_DEFAULT_ENV}"
      return 0
    fi
    log "conda env marker is set (${CONDA_DEFAULT_ENV}), but PATH is incomplete; reactivating."
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

resolve_trainer_devices() {
  local requested="$1"
  case "$requested" in
    gpu|auto)
      python - <<'PY'
import torch

count = torch.cuda.device_count()
if count < 1:
    raise SystemExit("no CUDA devices are visible")
print(count)
PY
      ;;
    *)
      printf '%s\n' "$requested"
      ;;
  esac
}

validate_strict_a100_pretrain_overrides() {
  local experiment="$1"
  local action="$2"
  shift 2

  if [[ "$experiment" != "pre_bc_a100x4x2" || "$action" != "fit" ]]; then
    return 0
  fi

  local max_train_batch_size=24
  if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
    if ! [[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ ]] || (( TRAIN_BATCH_SIZE < 1 || TRAIN_BATCH_SIZE > max_train_batch_size )); then
      log "ERROR: pre_bc_a100x4x2 must use train_batch_size in [1, ${max_train_batch_size}]; got TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}."
      exit 2
    fi
  fi
  if [[ -n "${ACCUMULATE_GRAD_BATCHES:-}" && "${ACCUMULATE_GRAD_BATCHES}" != "1" ]]; then
    log "ERROR: pre_bc_a100x4x2 must use accumulate_grad_batches=1; got ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}."
    exit 2
  fi

  local override key value
  for override in "$@"; do
    key="${override%%=*}"
    value="${override#*=}"
    case "$key" in
      data.train_batch_size)
        if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > max_train_batch_size )); then
          log "ERROR: pre_bc_a100x4x2 must use data.train_batch_size in [1, ${max_train_batch_size}]; got override ${override}."
          exit 2
        fi
        ;;
      trainer.accumulate_grad_batches)
        if [[ "$value" != "1" ]]; then
          log "ERROR: pre_bc_a100x4x2 must use trainer.accumulate_grad_batches=1; got override ${override}."
          exit 2
        fi
        ;;
    esac
  done
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

  local cache_root="${CACHE_ROOT:-$(default_cache_root)}"
  local nnodes="${NNODES:-2}"
  local nproc_per_node="${NPROC_PER_NODE:-4}"
  local trainer_devices="${TRAINER_DEVICES:-}"
  local node_rank="${NODE_RANK:-}"
  local master_addr="${MASTER_ADDR:-}"
  local master_port="${MASTER_PORT:-29521}"
  local experiment="${CATK_EXPERIMENT:-pre_bc_a100x4x2}"
  local action="${CATK_ACTION:-fit}"
  local task_name="${TASK_NAME:-smart_ntp_pretrain_a100x4x2}"
  local ckpt_path="${CATK_CKPT_PATH:-${CKPT_PATH:-}}"
  local auto_resume="${CATK_AUTO_RESUME:-false}"
  local resume_task_name="${CATK_RESUME_TASK_NAME:-}"
  local resume_checkpoint_name="${CATK_RESUME_CHECKPOINT_NAME:-epoch_last.ckpt}"
  local resume_require_checkpoint="${CATK_RESUME_REQUIRE_CHECKPOINT:-true}"

  if [[ "$action" != "fit" && "$action" != "validate" && "$action" != "test" ]]; then
    log "ERROR: CATK_ACTION must be fit, validate, or test; got: $action"
    exit 2
  fi
  if [[ "$action" != "fit" && -z "$ckpt_path" ]]; then
    log "ERROR: CATK_CKPT_PATH or CKPT_PATH is required when CATK_ACTION=$action."
    exit 2
  fi
  if [[ -n "$ckpt_path" && ! -f "$ckpt_path" ]]; then
    log "ERROR: checkpoint does not exist in this node: $ckpt_path"
    exit 2
  fi
  if [[ "$nnodes" -gt 1 && -z "$node_rank" ]]; then
    log "ERROR: multi-node launch requires NODE_RANK."
    exit 2
  fi
  if [[ "$nnodes" -gt 1 && -z "$master_addr" ]]; then
    log "ERROR: multi-node launch requires MASTER_ADDR."
    exit 2
  fi
  if [[ ! -d "$cache_root" ]]; then
    log "ERROR: CACHE_ROOT does not exist in this node: $cache_root"
    exit 2
  fi
  if [[ -z "$trainer_devices" ]]; then
    trainer_devices="$(resolve_trainer_devices "$nproc_per_node")"
  fi
  if ! [[ "$trainer_devices" =~ ^[0-9]+$ ]] || (( trainer_devices < 1 )); then
    log "ERROR: resolved trainer.devices must be a positive integer; got: $trainer_devices"
    exit 2
  fi
  local extra_overrides=()
  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides <<< "$CATK_HYDRA_OVERRIDES"
  fi
  if (( ${#extra_overrides[@]} > 0 )); then
    validate_strict_a100_pretrain_overrides "$experiment" "$action" "${extra_overrides[@]}" "$@"
  else
    validate_strict_a100_pretrain_overrides "$experiment" "$action" "$@"
  fi

  log "starting SMART NTP A100x4x2 pretrain"
  log "  experiment:       $experiment"
  log "  action:           $action"
  log "  task_name:        $task_name"
  log "  nnodes:           $nnodes"
  log "  nproc_per_node:   $nproc_per_node"
  log "  trainer.devices:  $trainer_devices"
  log "  node_rank:        ${node_rank:-0}"
  log "  master_addr:      $master_addr"
  log "  master_port:      $master_port"
  log "  cache_root:       $cache_root"
  log "  graph_attn_fp32:  $CATK_ATTENTION_GRAPH_FP32"
  if [[ -n "$ckpt_path" ]]; then
    log "  ckpt_path:        $ckpt_path"
  elif [[ "$auto_resume" == "true" || "$auto_resume" == "1" ]]; then
    log "  resume.auto:      true"
    log "  resume task:      ${resume_task_name:-$task_name}"
    log "  resume checkpoint: $resume_checkpoint_name"
  fi

  local app_args=(
    -m src.run
    experiment="$experiment"
    action="$action"
    trainer=ddp
    trainer.devices="$trainer_devices"
    trainer.num_nodes="$nnodes"
    ++trainer.enable_progress_bar=true
    paths.cache_root="$cache_root"
    task_name="$task_name"
  )

  if [[ -n "$ckpt_path" ]]; then
    app_args+=(ckpt_path="$ckpt_path")
  elif [[ "$auto_resume" == "true" || "$auto_resume" == "1" ]]; then
    app_args+=(
      resume.auto=true
      resume.checkpoint_name="$resume_checkpoint_name"
      resume.require_checkpoint="$resume_require_checkpoint"
    )
    if [[ -n "$resume_task_name" ]]; then
      app_args+=(resume.task_name="$resume_task_name")
    fi
  fi
  if [[ -n "${LOG_DIR:-}" ]]; then
    app_args+=(paths.log_dir="$LOG_DIR")
  fi
  if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
    app_args+=(data.train_batch_size="$TRAIN_BATCH_SIZE")
  fi
  if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
    app_args+=(data.val_batch_size="$VAL_BATCH_SIZE")
  fi
  if [[ -n "${TEST_BATCH_SIZE:-}" ]]; then
    app_args+=(data.test_batch_size="$TEST_BATCH_SIZE")
  fi
  if [[ -n "${ACCUMULATE_GRAD_BATCHES:-}" ]]; then
    app_args+=(trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES")
  fi
  if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES")
  fi
  if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
    app_args+=(trainer.limit_val_batches="$LIMIT_VAL_BATCHES")
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
    --nnodes "$nnodes"
    --nproc_per_node "$nproc_per_node"
    --node_rank "${node_rank:-0}"
    --master_addr "$master_addr"
    --master_port "$master_port"
    "${app_args[@]}"
  )

  log "torchrun command:"
  printf '  %q' torchrun "${torchrun_args[@]}"
  printf '\n'
  exec torchrun "${torchrun_args[@]}"
}

main "$@"
