#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# GOD-FM  Step 1: Offline dataset collection
#
# Runs closed-loop rollout with the pretrained model (single GPU, no DDP)
# and saves (anchor_hidden_c_shift, tau_target) pairs to GODFM_PAIR_DIR.
#
# Edit the three variables below, then run:
#   bash scripts/collect_godfm.sh
# ──────────────────────────────────────────────────────────────────────────────
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── User settings (edit these) ────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0

# Pretrained SMARTFlow checkpoint produced by train_flow.sh
PRETRAINED_CKPT="/home2/pnc2/repos_python/project_2/logs/pretrained/epoch_last.ckpt"

# Output directory for the pair .pt chunk files
GODFM_PAIR_DIR="data/godfm_pairs"

# ── Rollout / inpainting knobs ────────────────────────────────────────────────
N_ROLLOUT_COLLECT=4   # 2Hz steps of drift to simulate   (4 = 2 s)
INPAINT_STEPS=10      # Teacher ODE integration steps
GOAL_WEIGHT=5.0       # Endpoint guidance strength
CHUNK_SIZE=5000       # Pairs per .pt file

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

# ── Run (single GPU, no torchrun) ─────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -m scripts.generate_godfm_dataset \
  checkpoint="$PRETRAINED_CKPT" \
  output_dir="$GODFM_PAIR_DIR" \
  n_rollout_collect=$N_ROLLOUT_COLLECT \
  inpaint_steps=$INPAINT_STEPS \
  goal_weight=$GOAL_WEIGHT \
  chunk_size=$CHUNK_SIZE

echo "collect_godfm.sh done  →  pairs written to $GODFM_PAIR_DIR"
