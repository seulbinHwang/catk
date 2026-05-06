#!/bin/bash
# 실험 모니터링: rmm_delta, rmm_soft, grad_norm 추출
LOG2="${1:-/home2/pnc2/repos_python/project/logs/exp_logs/gpu2_noreg.log}"
LOG3="${2:-/home2/pnc2/repos_python/project/logs/exp_logs/gpu3_b2_steps8.log}"

extract_delta() {
  local logfile="$1"
  local label="$2"
  echo "=== $label ==="
  # rmm info lines from log.info
  grep -oE "\[rmm\] step=[0-9]+ rmm_soft=[-0-9.]+" "$logfile" 2>/dev/null | tail -20
  # rmm_delta from Lightning metric output
  grep -oE "rmm_delta=[-0-9.]+" "$logfile" 2>/dev/null | tail -20
  # grad norm
  grep -oE "grad_norm_velocity_head=[-0-9.]+" "$logfile" 2>/dev/null | tail -5
  # last few lines for errors
  echo "--- last lines ---"
  tail -5 "$logfile" 2>/dev/null
  echo ""
}

echo "=== $(date) ==="
echo ""
extract_delta "$LOG2" "GPU2: no-reg (FLOW_REG_LAMBDA=0, LR=5e-6, B=4)"
extract_delta "$LOG3" "GPU3: b2-steps8 (FLOW_REG_LAMBDA=1.0, LR=5e-6, B=2, steps=8)"
