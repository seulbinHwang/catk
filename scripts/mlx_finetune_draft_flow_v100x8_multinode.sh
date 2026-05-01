#!/usr/bin/env bash
# Run CAT-K DRaFT fine-tuning inside an MLX Kubeflow PyTorchJob worker.
#
# This script is intentionally launched by every Worker pod. Kubeflow injects
# PET_NNODES, PET_NPROC_PER_NODE, PET_RDZV_*; torchrun then starts one process
# per GPU on each pod and connects all pods into a single Lightning DDP run.
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

activate_conda_if_available() {
    local conda_root="${CONDA_ROOT:-/mnt/nuplan/miniforge}"
    if [[ -f "$conda_root/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$conda_root/etc/profile.d/conda.sh"
        conda activate "${CATK_CONDA_ENV:-catk}" 2>/dev/null \
            || conda activate base 2>/dev/null \
            || true
        log "conda env: ${CONDA_DEFAULT_ENV:-unknown}"
    else
        log "conda root not found at $conda_root; using image Python."
    fi
}

maybe_install_requirements() {
    local mode="${CATK_INSTALL_REQUIREMENTS:-auto}"
    if [[ "$mode" == "0" || "$mode" == "false" || "$mode" == "False" ]]; then
        return 0
    fi

    if python - <<'PY' >/dev/null 2>&1
import lightning
import torch
PY
    then
        log "required Python packages already import cleanly."
        return 0
    fi

    if [[ "$mode" == "auto" || "$mode" == "1" || "$mode" == "true" || "$mode" == "True" ]]; then
        log "installing Python requirements because torch/lightning import failed."
        python -m pip install --no-cache-dir -r install/requirements.txt
        return 0
    fi

    log "ERROR: torch/lightning import failed and CATK_INSTALL_REQUIREMENTS=$mode."
    exit 2
}

compute_auto_lr() {
    local nnodes="$1"
    python - "$nnodes" <<'PY'
import sys
nnodes = int(sys.argv[1])
print(2e-4 * nnodes)
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

    activate_conda_if_available

    require_env CACHE_ROOT

    local nnodes="${PET_NNODES:-${NNODES:-1}}"
    local nproc_per_node="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
    local rdzv_id="${PET_RDZV_ID:-${RDZV_ID:-catk-draft-flow}}"
    local rdzv_backend="${PET_RDZV_BACKEND:-${RDZV_BACKEND:-c10d}}"
    local rdzv_endpoint="${PET_RDZV_ENDPOINT:-${RDZV_ENDPOINT:-}}"
    local node_rank="${NODE_RANK:-}"
    local master_addr="${MASTER_ADDR:-}"
    local master_port="${MASTER_PORT:-29500}"
    local task_name="${TASK_NAME:-flow_semi_continuous_finetune_v100x8x${nnodes}}"
    local experiment="${CATK_EXPERIMENT:-finetune_draft_flow_v100x8}"
    local action="${CATK_ACTION:-finetune}"
    local ckpt_path="${CATK_CKPT_PATH:-${PRETRAIN_CKPT:-}}"
    local lr="${CATK_LR:-auto}"

    if [[ "$nnodes" -gt 1 && -z "$node_rank" && -z "$rdzv_endpoint" ]]; then
        log "ERROR: multi-node elastic mode requires PET_RDZV_ENDPOINT/RDZV_ENDPOINT."
        log "       For existing fixed pods, set NODE_RANK, MASTER_ADDR, and MASTER_PORT instead."
        exit 2
    fi
    if [[ -n "$node_rank" && -z "$master_addr" ]]; then
        log "ERROR: static multi-node mode requires MASTER_ADDR when NODE_RANK is set."
        exit 2
    fi
    if [[ ! -d "$CACHE_ROOT" ]]; then
        log "ERROR: CACHE_ROOT does not exist in this pod: $CACHE_ROOT"
        exit 2
    fi
    if [[ "$action" != "finetune" && "$action" != "fit" ]]; then
        log "ERROR: CATK_ACTION must be finetune or fit, got: $action"
        exit 2
    fi
    if [[ -z "$ckpt_path" ]]; then
        log "ERROR: PRETRAIN_CKPT or CATK_CKPT_PATH must be set."
        exit 2
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        log "ERROR: checkpoint does not exist in this pod: $ckpt_path"
        exit 2
    fi

    maybe_install_requirements

    if [[ -n "${WANDB_API_KEY:-}" ]] && command -v wandb >/dev/null 2>&1; then
        wandb login --relogin "$WANDB_API_KEY" || log "wandb login failed; continuing."
    fi

    if [[ "$lr" == "auto" ]]; then
        lr="$(compute_auto_lr "$nnodes")"
    fi

    log "starting CAT-K multi-node fine-tune"
    log "  experiment:       $experiment"
    log "  action:           $action"
    log "  task_name:        $task_name"
    log "  nnodes:           $nnodes"
    log "  nproc_per_node:   $nproc_per_node"
    if [[ -n "$node_rank" ]]; then
        log "  launch_mode:      static"
        log "  node_rank:        $node_rank"
        log "  master_addr:      $master_addr"
        log "  master_port:      $master_port"
    else
        log "  launch_mode:      elastic"
        log "  rdzv_backend:     $rdzv_backend"
        log "  rdzv_endpoint:    ${rdzv_endpoint:-<empty>}"
    fi
    log "  cache_root:       $CACHE_ROOT"
    log "  ckpt_path:        $ckpt_path"
    log "  lr:               $lr"

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
        )
        torchrun_args+=(--rdzv_endpoint "$rdzv_endpoint")
    fi
    if [[ -n "${LOCAL_ADDR:-}" ]]; then
        torchrun_args+=(--local_addr "$LOCAL_ADDR")
    fi

    torchrun_args+=(
        -m src.run
        experiment="$experiment"
        action="$action"
        trainer=ddp
        trainer.devices="$nproc_per_node"
        trainer.num_nodes="$nnodes"
        paths.cache_root="$CACHE_ROOT"
        ckpt_path="$ckpt_path"
        task_name="$task_name"
        model.model_config.lr="$lr"
    )

    if [[ -n "${LOG_DIR:-}" ]]; then
        torchrun_args+=(paths.log_dir="$LOG_DIR")
    fi
    if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
        torchrun_args+=(trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES")
    fi
    if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
        torchrun_args+=(trainer.limit_val_batches="$LIMIT_VAL_BATCHES")
    fi
    if [[ -n "${SOFT_LIMIT_RATIO:-}" ]]; then
        torchrun_args+=(model.model_config.draft.physics.soft_limit_ratio="$SOFT_LIMIT_RATIO")
    fi
    if [[ -n "${TOPK_VIOLATION_K:-}" ]]; then
        torchrun_args+=(model.model_config.draft.physics.topk_violation_k="$TOPK_VIOLATION_K")
    fi
    if [[ -n "${BACKPROP_LAST_K:-}" ]]; then
        torchrun_args+=(model.model_config.draft.sampling.backprop_last_k="$BACKPROP_LAST_K")
    fi
    if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
        torchrun_args+=(data.train_batch_size="$TRAIN_BATCH_SIZE")
    fi
    if [[ -n "${ACCUMULATE_GRAD_BATCHES:-}" ]]; then
        torchrun_args+=(trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES")
    fi
    if [[ -n "${CATK_HYDRA_OVERRIDES:-}" ]]; then
        # Simple whitespace splitting is enough for scalar overrides such as
        # trainer.limit_train_batches=40. Use script arguments for quoted values.
        read -r -a extra_overrides <<< "$CATK_HYDRA_OVERRIDES"
        torchrun_args+=("${extra_overrides[@]}")
    fi
    torchrun_args+=("$@")

    log "torchrun command:"
    printf '  %q' torchrun "${torchrun_args[@]}"
    printf '\n'
    exec torchrun "${torchrun_args[@]}"
}

main "$@"
