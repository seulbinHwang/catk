#!/usr/bin/env bash
# Run UniMM Anchor-Based-4s across hsb-npc-training-3-{1,2}
# as an equal-size 2 node x 3 GPU H100 pod fleet.
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

activate_conda_if_available() {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
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
  fi
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
  export CATK_ATTENTION_GRAPH_FP32="${CATK_ATTENTION_GRAPH_FP32:-1}"

  activate_conda_if_available

  local cache_root="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
  local anchor_file="${UNIMM_ANCHOR_FILE:-${PWD}/src/unimm/anchors/unimm_anchors_8s_k2048.pkl}"
  local nnodes="${NNODES:-2}"
  local nproc_per_node="${NPROC_PER_NODE:-3}"
  local trainer_devices="${TRAINER_DEVICES:-$nproc_per_node}"
  local trainer_precision="${TRAINER_PRECISION:-bf16-mixed}"
  local node_rank="${NODE_RANK:-}"
  local master_addr="${MASTER_ADDR:-}"
  local master_port="${MASTER_PORT:-29541}"
  local manual_rank_offset="${MANUAL_RANK_OFFSET:-}"
  local manual_world_size="${MANUAL_WORLD_SIZE:-}"
  local action="${CATK_ACTION:-fit}"
  local task_name="${TASK_NAME:-unimm_anchor_based_4s_h100x3x2}"

  if [[ "$action" != "fit" && "$action" != "validate" && "$action" != "test" ]]; then
    log "ERROR: CATK_ACTION must be fit, validate, or test; got: $action"
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
  if [[ ! -f "$anchor_file" ]]; then
    log "ERROR: UniMM anchor file does not exist in this node: $anchor_file"
    log "Build it first with scripts/build_unimm_anchors.sh or use the committed anchor file in src/unimm/anchors."
    exit 2
  fi

  local app_args=(
    -m src.run
    experiment=unimm_anchor_based_4s
    action="$action"
    trainer=ddp
    trainer.devices="$trainer_devices"
    trainer.num_nodes="$nnodes"
    trainer.precision="$trainer_precision"
    +trainer.enable_progress_bar=true
    paths.cache_root="$cache_root"
    task_name="$task_name"
    model.model_config.anchor_file="$anchor_file"
  )

  if [[ -n "${CKPT_PATH:-}" ]]; then
    app_args+=(ckpt_path="$CKPT_PATH")
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
  if [[ -n "${LEARNING_RATE:-}" ]]; then
    app_args+=(model.model_config.lr="$LEARNING_RATE")
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
  if [[ -n "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE
  fi
  if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_overrides=( $CATK_HYDRA_OVERRIDES )
    app_args+=("${extra_overrides[@]}")
  fi
  app_args+=("$@")

  log "starting UniMM Anchor-Based-4s"
  log "  action:          $action"
  log "  task_name:       $task_name"
  log "  nnodes:          $nnodes"
  log "  nproc_per_node:  $nproc_per_node"
  log "  node_rank:       ${node_rank:-0}"
  log "  precision:       $trainer_precision"
  log "  learning_rate:   ${LEARNING_RATE:-config default}"
  log "  master_addr:     $master_addr"
  log "  master_port:     $master_port"
  log "  cache_root:      $cache_root"
  log "  anchor_file:     $anchor_file"

  if [[ -n "$manual_rank_offset" || -n "$manual_world_size" ]]; then
    if ! [[ "$manual_rank_offset" =~ ^[0-9]+$ && "$manual_world_size" =~ ^[0-9]+$ ]]; then
      log "ERROR: MANUAL_RANK_OFFSET and MANUAL_WORLD_SIZE must be non-negative integers."
      exit 2
    fi
    if ! [[ "$nproc_per_node" =~ ^[0-9]+$ ]] || (( nproc_per_node < 1 )); then
      log "ERROR: manual launch requires integer NPROC_PER_NODE; got: $nproc_per_node"
      exit 2
    fi

    log "manual DDP launch:"
    log "  rank_offset:     $manual_rank_offset"
    log "  world_size:      $manual_world_size"
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

  exec torchrun \
    --nnodes "$nnodes" \
    --nproc_per_node "$nproc_per_node" \
    --node_rank "${node_rank:-0}" \
    --master_addr "$master_addr" \
    --master_port "$master_port" \
    "${app_args[@]}"
}

main "$@"
