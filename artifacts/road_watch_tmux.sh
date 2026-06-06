#!/usr/bin/env bash
set -euo pipefail

cd /home2/pnc2/repos_python/kinematic_flow

LOG="artifacts/road_ft_main_prealign_lr1e5_b32_v16_val200_lval002_nbatch27_nofixednoise_2gpu_ddpunused.log"

while true; do
  clear
  date -Is
  printf '\n[progress]\n'
  rg -n 'Epoch 0:|Validation DataLoader|Validation:|val_closed/sim_agents|Traceback|RuntimeError|CUDA out|out of memory|limit_val_batches를' "$LOG" | tail -n 20 || true
  printf '\n[gpu]\n'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  if rg -q 'Validation DataLoader|Validation:|val_closed/sim_agents' "$LOG"; then
    printf '\n[watch] validation reached\n'
    sleep 300
  fi
  if rg -q 'Traceback|RuntimeError|CUDA out|out of memory' "$LOG"; then
    printf '\n[watch] failure detected\n'
    sleep 300
  fi
  sleep 60
done
