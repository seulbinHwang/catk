#!/bin/bash
set -euo pipefail

# Usage: bash scripts/probe_batch_size.sh <train_batch_size> <log_path>
BS="${1:?usage: probe_batch_size.sh <train_batch_size> <log_path>}"
LOG_PATH="${2:?usage: probe_batch_size.sh <train_batch_size> <log_path>}"

export LOGLEVEL=WARNING
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export WANDB_DISABLED=true

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
# shellcheck disable=SC1091
. "$(dirname "$0")/_activate_conda.sh"

CACHE_ROOT="${CACHE_ROOT:-/mnt/nuplan/womd_v1_3/SMART_cache}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}" \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
    experiment=pre_bc_flow \
    trainer=ddp \
    trainer.devices=6 \
    paths.cache_root="${CACHE_ROOT}" \
    task_name="batchsize_probe_bs${BS}" \
    data.train_batch_size="${BS}" \
    trainer.limit_train_batches=${PROBE_STEPS:-15} \
    trainer.max_epochs=1 \
    trainer.check_val_every_n_epoch=999 \
    trainer.num_sanity_val_steps=0 \
    ~callbacks.model_checkpoint \
    ~callbacks.epoch_last_checkpoint \
    ~callbacks.wandb_runtime_metrics \
    ~callbacks.learning_rate_monitor \
    ~callbacks.model_summary \
    +callbacks.probe_timing._target_=src.utils.probe_timing_callback.ProbeTimingCallback \
    +callbacks.probe_timing.skip_warmup_steps=3 \
    ~logger \
  2>&1 | tee "${LOG_PATH}"
