#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────────────
# GOD-FM  Step 2: Fine-tuning
#
# Fine-tunes the pretrained model with 50% GOD-FM recovery pairs
# and 50% standard GT flow matching.
#
# Runs in a single process launch:
#   - normal GOD-FM training
#   - periodic online recollect from current policy
#
# If GODFM_PAIR_DIR is empty, training starts from GT branch and
# online recollect fills the GOD-FM buffer after warmup.
#
# Edit the three variables below, then run:
#   bash scripts/train_godfm.sh
# ──────────────────────────────────────────────────────────────────────────────
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# ── User settings (edit these) ────────────────────────────────────────────────
# Pretrained SMARTFlow checkpoint to start fine-tuning from
PRETRAINED_CKPT="/home2/pnc2/repos_python/project_2/logs/pretrained/epoch_last.ckpt"

# Optional seed directory written by collect_godfm.sh (pairs_*.pt files).
# Leave empty ("") for pure online start.
GODFM_PAIR_DIR="data/godfm_pairs"

MY_EXPERIMENT="godfm_finetune_flow"
MY_TASK_NAME="${MY_EXPERIMENT}-sim_agents_2025"

# Training / validation knobs
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=8
NUM_WORKERS=4
PREFETCH_FACTOR=1
MAX_EPOCHS=10
CHECK_VAL_EVERY_N_EPOCH=1
VAL_CHECK_INTERVAL_STEPS=100
LIMIT_TRAIN_BATCHES=0.02
LIMIT_VAL_BATCHES=1
N_ROLLOUT_CLOSED_VAL=32
N_BATCH_SIM_AGENTS_METRIC=1
N_VIS_BATCH=0
N_VIS_SCENARIO=0
N_VIS_ROLLOUT=0
DELETE_LOCAL_VIDEOS_AFTER_WANDB_UPLOAD=false

# Online recollect knobs (effective only when godfm.enabled=true)
ONLINE_ENABLED=true
ONLINE_COLLECT_EVERY_N_STEPS=100
ONLINE_WARMUP_STEPS=500
ONLINE_MAX_BUFFER_PAIRS=200000
ONLINE_MAX_PAIRS_PER_COLLECT=1024

# ── Conda env ─────────────────────────────────────────────────────────────────
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found. Please install/init conda first."
  exit 1
fi
conda activate "$CATK_CONDA_ENV"

# ── Run ───────────────────────────────────────────────────────────────────────
python -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer.devices=1 \
  trainer.max_epochs=$MAX_EPOCHS \
  trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH \
  +trainer.val_check_interval=$VAL_CHECK_INTERVAL_STEPS \
  trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
  trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
  task_name=$MY_TASK_NAME \
  ckpt_path="$PRETRAINED_CKPT" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.num_workers=$NUM_WORKERS \
  data.prefetch_factor=$PREFETCH_FACTOR \
  model.model_config.n_rollout_closed_val=$N_ROLLOUT_CLOSED_VAL \
  model.model_config.n_batch_sim_agents_metric=$N_BATCH_SIM_AGENTS_METRIC \
  model.model_config.n_vis_batch=$N_VIS_BATCH \
  model.model_config.n_vis_scenario=$N_VIS_SCENARIO \
  model.model_config.n_vis_rollout=$N_VIS_ROLLOUT \
  model.model_config.delete_local_videos_after_wandb_upload=$DELETE_LOCAL_VIDEOS_AFTER_WANDB_UPLOAD \
  model.model_config.godfm.pair_dir="$GODFM_PAIR_DIR" \
  model.model_config.godfm.online_enabled=$ONLINE_ENABLED \
  model.model_config.godfm.online_collect_every_n_steps=$ONLINE_COLLECT_EVERY_N_STEPS \
  model.model_config.godfm.online_warmup_steps=$ONLINE_WARMUP_STEPS \
  model.model_config.godfm.online_max_buffer_pairs=$ONLINE_MAX_BUFFER_PAIRS \
  model.model_config.godfm.online_max_pairs_per_collect=$ONLINE_MAX_PAIRS_PER_COLLECT

echo "train_godfm.sh done!"
