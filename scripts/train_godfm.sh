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
NUM_WORKERS=64
PREFETCH_FACTOR=1
LR=2e-4
LR_WARMUP_STEPS=1000
LR_TOTAL_STEPS=-1
LR_MIN_RATIO=0.1
LR_SCHEDULER_UNIT="epoch"
MAX_EPOCHS=100
CHECK_VAL_EVERY_N_EPOCH=1
VAL_CHECK_INTERVAL_STEPS=1000
LIMIT_TRAIN_BATCHES=1.0
LIMIT_VAL_BATCHES=1
N_ROLLOUT_CLOSED_VAL=32
N_BATCH_SIM_AGENTS_METRIC=1
VAL_OPEN_LOOP=true
VAL_CLOSED_LOOP=true
N_VIS_BATCH=0
N_VIS_SCENARIO=0
N_VIS_ROLLOUT=0
DELETE_LOCAL_VIDEOS_AFTER_WANDB_UPLOAD=false
CKPT_MONITOR_METRIC="val_closed/sim_agents_2025/realism_meta_metric"
CKPT_MONITOR_MODE="max"
# "official" = TF + multiprocessing (slow, full metrics); "torch" = pure-GPU (fast, metametric only)
SIM_AGENTS_METRIC_BACKEND="torch"

# GOD-FM collecting knobs
# - GODFM_P_AUG is probability of sampling GOD-FM branch.
#   Set 0.0 for GT-only, 1.0 for GOD-FM-only.
# - N_ROLLOUT_COLLECT controls rollout drift horizon during collect.
# - The rollout is 2Hz: 1 step = 0.5s drift, 4 steps = 2.0s drift.
GODFM_P_AUG=1.0
INPAINT_STEPS=10
N_ROLLOUT_COLLECT=2

# Online recollect knobs (effective only when godfm.online_enabled=true)
ONLINE_ENABLED=true
ONLINE_COLLECT_EVERY_N_STEPS=100
ONLINE_WARMUP_STEPS=0
ONLINE_MAX_BUFFER_PAIRS=200000
ONLINE_MAX_PAIRS_PER_COLLECT=2000

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
  callbacks.model_checkpoint.monitor="$CKPT_MONITOR_METRIC" \
  callbacks.model_checkpoint.mode=$CKPT_MONITOR_MODE \
  task_name=$MY_TASK_NAME \
  ckpt_path="$PRETRAINED_CKPT" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.num_workers=$NUM_WORKERS \
  data.prefetch_factor=$PREFETCH_FACTOR \
  model.model_config.lr=$LR \
  model.model_config.lr_warmup_steps=$LR_WARMUP_STEPS \
  model.model_config.lr_total_steps=$LR_TOTAL_STEPS \
  model.model_config.lr_min_ratio=$LR_MIN_RATIO \
  model.model_config.lr_scheduler_unit=$LR_SCHEDULER_UNIT \
  model.model_config.n_rollout_closed_val=$N_ROLLOUT_CLOSED_VAL \
  model.model_config.n_batch_sim_agents_metric=$N_BATCH_SIM_AGENTS_METRIC \
  model.model_config.val_open_loop=$VAL_OPEN_LOOP \
  model.model_config.val_closed_loop=$VAL_CLOSED_LOOP \
  model.model_config.n_vis_batch=$N_VIS_BATCH \
  model.model_config.n_vis_scenario=$N_VIS_SCENARIO \
  model.model_config.n_vis_rollout=$N_VIS_ROLLOUT \
  model.model_config.delete_local_videos_after_wandb_upload=$DELETE_LOCAL_VIDEOS_AFTER_WANDB_UPLOAD \
  model.model_config.sim_agents_metric_backend=$SIM_AGENTS_METRIC_BACKEND \
  model.model_config.godfm.pair_dir="$GODFM_PAIR_DIR" \
  model.model_config.godfm.p_aug=$GODFM_P_AUG \
  model.model_config.godfm.inpaint_steps=$INPAINT_STEPS \
  model.model_config.godfm.n_rollout_collect=$N_ROLLOUT_COLLECT \
  model.model_config.godfm.online_enabled=$ONLINE_ENABLED \
  model.model_config.godfm.online_collect_every_n_steps=$ONLINE_COLLECT_EVERY_N_STEPS \
  model.model_config.godfm.online_warmup_steps=$ONLINE_WARMUP_STEPS \
  model.model_config.godfm.online_max_buffer_pairs=$ONLINE_MAX_BUFFER_PAIRS \
  model.model_config.godfm.online_max_pairs_per_collect=$ONLINE_MAX_PAIRS_PER_COLLECT

echo "train_godfm.sh done!"
