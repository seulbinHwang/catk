#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# GOD-FM  Step 2: Fine-tuning
#
# Fine-tunes the pretrained model with 50% GOD-FM recovery pairs
# and 50% standard GT flow matching.
#
# Requires collect_godfm.sh to have been run first so that GODFM_PAIR_DIR
# contains at least one pairs_*.pt file.
#
# Edit the three variables below, then run:
#   bash scripts/train_godfm.sh
# ──────────────────────────────────────────────────────────────────────────────
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3

# ── User settings (edit these) ────────────────────────────────────────────────
# Pretrained SMARTFlow checkpoint to start fine-tuning from
PRETRAINED_CKPT="/home2/pnc2/repos_python/project_2/logs/pretrained/epoch_last.ckpt"

# Directory written by collect_godfm.sh (contains pairs_*.pt files)
GODFM_PAIR_DIR="data/godfm_pairs"

MY_EXPERIMENT="godfm_finetune_flow"
MY_TASK_NAME="${MY_EXPERIMENT}-sim_agents_2025"

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
torchrun \
  --nproc_per_node=2 \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer.devices=2 \
  task_name=$MY_TASK_NAME \
  ckpt_path="$PRETRAINED_CKPT" \
  model.model_config.godfm.pair_dir="$GODFM_PAIR_DIR"

echo "train_godfm.sh done!"
