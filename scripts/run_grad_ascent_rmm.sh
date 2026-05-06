#!/bin/bash
# Gradient Ascent RMM Test
# Usage: bash scripts/run_grad_ascent_rmm.sh [n_scenarios] [n_steps] [lr]
#
# Examples:
#   bash scripts/run_grad_ascent_rmm.sh          # defaults: 6 scenarios, 100 steps
#   bash scripts/run_grad_ascent_rmm.sh 20 150   # 20 scenarios, 150 steps

cd "$(dirname "$0")/.."

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
. "$(dirname "$0")/_activate_conda.sh" 2>/dev/null || conda activate "${CATK_CONDA_ENV}" 2>/dev/null

export CUDA_VISIBLE_DEVICES=2,3
export TF_CPP_MIN_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1

# Override constants via env if args given
if [ -n "$1" ]; then
  sed -i "s/^N_SCENARIOS    = .*/N_SCENARIOS    = $1/" scripts/test_grad_ascent_rmm.py
fi
if [ -n "$2" ]; then
  sed -i "s/^N_ASCENT_STEPS = .*/N_ASCENT_STEPS = $2/" scripts/test_grad_ascent_rmm.py
fi
if [ -n "$3" ]; then
  sed -i "s/^ASCENT_LR      = .*/ASCENT_LR      = $3/" scripts/test_grad_ascent_rmm.py
fi

echo "=== Grad Ascent RMM Test ==="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
grep -E "^N_SCENARIOS|^N_ROLLOUTS|^N_ASCENT_STEPS|^ASCENT_LR" scripts/test_grad_ascent_rmm.py
echo ""

python scripts/test_grad_ascent_rmm.py 2>&1 | tee /tmp/grad_ascent_rmm_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Done. Logs at /tmp/grad_ascent_rmm_*.log"
echo "Videos at /tmp/grad_ascent_rmm_test/videos/"
