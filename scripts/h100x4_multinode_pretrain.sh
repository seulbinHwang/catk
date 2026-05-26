#!/usr/bin/env bash
# Run CAT-K Flow Matching pretrain across two existing static H100x4 pods/nodes.
#
# This script is executed inside every node. For the hsb-npc-training pair,
# prefer launching it through scripts/launch_h100x4_multinode_pretrain_tmux.py;
# manual use is also supported by setting NODE_RANK, MASTER_ADDR, MASTER_PORT,
# NNODES, and NPROC_PER_NODE. It never creates, deletes, or restarts pods.
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    log "ERROR: required environment variable $name is not set."
    exit 2
  fi
}

default_cache_root() {
  local pod_name
  pod_name="$(hostname)"
  case "$pod_name" in
    hsb-npc-training-2*)
      printf '%s\n' "/workspace/womd_v1_3/SMART_cache"
      ;;
    hsb-npc-training*)
      printf '%s\n' "/mnt/nuplan/womd_v1_3/SMART_cache"
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
  local nnodes="${PET_NNODES:-${NNODES:-2}}"
  local nproc_per_node="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-4}}"
  local trainer_devices="${TRAINER_DEVICES:-}"
  local node_rank="${NODE_RANK:-}"
  local master_addr="${MASTER_ADDR:-}"
  local master_port="${MASTER_PORT:-29511}"
  local rdzv_id="${PET_RDZV_ID:-${RDZV_ID:-catk-h100x4-pretrain}}"
  local rdzv_backend="${PET_RDZV_BACKEND:-${RDZV_BACKEND:-c10d}}"
  local rdzv_endpoint="${PET_RDZV_ENDPOINT:-${RDZV_ENDPOINT:-}}"
  local experiment="${CATK_EXPERIMENT:-pre_bc_flow_2x4_h100}"
  local action="${CATK_ACTION:-fit}"
  local task_name="${TASK_NAME:-flow_semi_continuous_pretrain_h1004x2}"
  local ckpt_path="${CATK_CKPT_PATH:-${CKPT_PATH:-}}"
  local manual_rank_offset="${MANUAL_RANK_OFFSET:-}"
  local manual_world_size="${MANUAL_WORLD_SIZE:-}"

  if [[ "$action" != "fit" && "$action" != "validate" && "$action" != "test" ]]; then
    log "ERROR: CATK_ACTION must be fit, validate, or test for this pretrain wrapper; got: $action"
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
  if [[ "$nnodes" -gt 1 && -z "$node_rank" && -z "$rdzv_endpoint" ]]; then
    log "ERROR: multi-node launch needs either static NODE_RANK/MASTER_ADDR or RDZV_ENDPOINT."
    exit 2
  fi
  if [[ -n "$node_rank" && -z "$master_addr" ]]; then
    log "ERROR: static multi-node mode requires MASTER_ADDR when NODE_RANK is set."
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

  log "starting CAT-K H100x4 multi-node pretrain"
  log "  experiment:       $experiment"
  log "  action:           $action"
  log "  task_name:        $task_name"
  log "  nnodes:           $nnodes"
  log "  nproc_per_node:   $nproc_per_node"
  log "  trainer.devices:  $trainer_devices"
  log "  cache_root:       $cache_root"
  if [[ -n "$node_rank" ]]; then
    log "  launch_mode:      static"
    log "  node_rank:        $node_rank"
    log "  master_addr:      $master_addr"
    log "  master_port:      $master_port"
  else
    log "  launch_mode:      elastic"
    log "  rdzv_backend:     $rdzv_backend"
    log "  rdzv_endpoint:    $rdzv_endpoint"
  fi
  if [[ -n "$ckpt_path" ]]; then
    log "  ckpt_path:        $ckpt_path"
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
  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides <<< "$CATK_HYDRA_OVERRIDES"
    app_args+=("${extra_overrides[@]}")
  fi
  app_args+=("$@")

  if [[ -n "$manual_rank_offset" || -n "$manual_world_size" ]]; then
    if ! [[ "$manual_rank_offset" =~ ^[0-9]+$ && "$manual_world_size" =~ ^[0-9]+$ ]]; then
      log "ERROR: MANUAL_RANK_OFFSET and MANUAL_WORLD_SIZE must be non-negative integers."
      exit 2
    fi
    if ! [[ "$nproc_per_node" =~ ^[0-9]+$ ]] || (( nproc_per_node < 1 )); then
      log "ERROR: manual launch requires integer NPROC_PER_NODE; got: $nproc_per_node"
      exit 2
    fi
    if [[ -z "$master_addr" ]]; then
      log "ERROR: manual launch requires MASTER_ADDR."
      exit 2
    fi

    log "manual heterogeneous launch:"
    log "  rank_offset:      $manual_rank_offset"
    log "  world_size:       $manual_world_size"
    local -a pids=()
    local local_rank global_rank pid remaining status
    for (( local_rank = 0; local_rank < nproc_per_node; local_rank++ )); do
      global_rank=$(( manual_rank_offset + local_rank ))
      (
        export MASTER_ADDR="$master_addr"
        export MASTER_PORT="$master_port"
        export WORLD_SIZE="$manual_world_size"
        export LOCAL_WORLD_SIZE="$nproc_per_node"
        export RANK="$global_rank"
        export LOCAL_RANK="$local_rank"
        export GROUP_RANK="${node_rank:-0}"
        export GROUP_WORLD_SIZE="$nnodes"
        export ROLE_RANK="$global_rank"
        export ROLE_WORLD_SIZE="$manual_world_size"
        export TORCHELASTIC_RUN_ID="${TORCHELASTIC_RUN_ID:-$task_name}"
        export TORCHELASTIC_RESTART_COUNT="${TORCHELASTIC_RESTART_COUNT:-0}"
        export TORCHELASTIC_MAX_RESTARTS="${TORCHELASTIC_MAX_RESTARTS:-0}"
        log "manual worker local_rank=$LOCAL_RANK rank=$RANK/$WORLD_SIZE"
        exec python "${app_args[@]}"
      ) &
      pid=$!
      pids+=("$pid")
    done

    status=0
    remaining="${#pids[@]}"
    while (( remaining > 0 )); do
      wait -n
      status=$?
      if (( status != 0 )); then
        log "manual worker failed with status $status; terminating peer workers"
        kill -TERM "${pids[@]}" 2>/dev/null || true
        sleep "${LOCAL_KILL_GRACE_SEC:-10}"
        kill -KILL "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
        exit "$status"
      fi
      remaining=$(( remaining - 1 ))
    done
    exit 0
  fi

  local torchrun_args=(
    --nnodes "$nnodes"
    --nproc_per_node "$nproc_per_node"
  )

  if [[ -n "$node_rank" ]]; then
    torchrun_args+=(
      --node_rank "$node_rank"
      --master_addr "$master_addr"
      --master_port "$master_port"
    )
  else
    torchrun_args+=(
      --rdzv_id "$rdzv_id"
      --rdzv_backend "$rdzv_backend"
      --rdzv_endpoint "$rdzv_endpoint"
    )
  fi
  if [[ -n "${LOCAL_ADDR:-}" ]]; then
    torchrun_args+=(--local_addr "$LOCAL_ADDR")
  fi
  torchrun_args+=("${app_args[@]}")

  log "torchrun command:"
  printf '  %q' torchrun "${torchrun_args[@]}"
  printf '\n'
  exec torchrun "${torchrun_args[@]}"
}

main "$@"
