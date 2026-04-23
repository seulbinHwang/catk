#!/usr/bin/env bash
# Sweep val_batch_size for `experiment=sim_agents_sub_flow action=validate`
# on 8x V100-SXM2 32GB.
#
# Goal: find the max val_batch_size that does not OOM so the full 44,097
# validation-split submission export completes as fast as possible.
#
# Env:
#   CACHE_ROOT  (required) Waymo SMART cache root.
#   CKPT        (required) path to epoch_last.ckpt
#   LIMIT       (optional) trainer.limit_val_batches (default 8 — ~5 warmup+timed)
#   CANDIDATES  (optional) space-separated list of val_batch_size values.
#   SUFFIX      (optional) extra string appended to task_name / log file.
#
# Results appended to scripts/bench/v100x8_sim_agents_sub_results.log.
set -u

: "${CACHE_ROOT:?CACHE_ROOT must be exported}"
: "${CKPT:?CKPT must be exported (path to epoch_last.ckpt)}"
LIMIT="${LIMIT:-8}"
SUFFIX="${SUFFIX:-}"
CANDIDATES_DEFAULT="1 2 4 6"
CANDIDATES="${CANDIDATES:-$CANDIDATES_DEFAULT}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RESULT_FILE="$ROOT/scripts/bench/v100x8_sim_agents_sub_results.log"
mkdir -p "$(dirname "$RESULT_FILE")"
: > "$RESULT_FILE"

run_one() {
  local bs="$1"
  local tag="sim_agents_sub_val_bs${bs}${SUFFIX}"
  local log="$ROOT/scripts/bench/_run_${tag}.log"

  echo "=== START $tag  val_bs=$bs ===" | tee -a "$RESULT_FILE"
  local t0
  t0=$(date +%s)

  # V100 needs precision=16-mixed (no bf16 hardware). Disable wandb logger
  # for the benchmark. Submission is active (is_active=true is already set in
  # the experiment file); for a benchmark run the partial submission files
  # written to disk are inconsequential and the write path still exercises
  # the same per-batch memory pattern.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  WANDB_DISABLED=true \
  WANDB_MODE=offline \
  timeout 900 \
  torchrun \
    --standalone \
    --nproc_per_node=8 \
    -m src.run \
    experiment=sim_agents_sub_flow \
    action=validate \
    trainer=ddp \
    trainer.devices=8 \
    trainer.precision=16-mixed \
    trainer.limit_val_batches="$LIMIT" \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=1 \
    data.val_batch_size="$bs" \
    model.model_config.val_open_loop=false \
    model.model_config.val_closed_loop=true \
    paths.cache_root="$CACHE_ROOT" \
    ckpt_path="$CKPT" \
    task_name="bench_${tag}" \
    callbacks=val_benchmark \
    '~logger' \
    >"$log" 2>&1
  local rc=$?

  local dt=$(( $(date +%s) - t0 ))

  local bench_line
  bench_line=$(grep -E '^\[VALBENCH\]' "$log" | tail -n1)

  local oom=""
  if grep -q "CUDA out of memory\|OutOfMemoryError" "$log"; then
    oom=" OOM=1"
  fi

  if [[ -n "$bench_line" ]]; then
    echo "$bench_line elapsed=${dt}s rc=${rc}${oom}" | tee -a "$RESULT_FILE"
  else
    echo "[FAIL] tag=$tag rc=${rc} elapsed=${dt}s${oom} (see $log)" | tee -a "$RESULT_FILE"
  fi
  echo | tee -a "$RESULT_FILE"
}

for bs in $CANDIDATES; do
  run_one "$bs"
done

echo
echo "=== SUMMARY (sorted by total_samples_s desc) ==="
grep -E '^\[VALBENCH\]' "$RESULT_FILE" \
  | awk '{
      bs=""; batch_ms=""; total_s=""; vram="";
      for (i=1;i<=NF;i++) {
        split($i, kv, "=");
        if      (kv[1]=="val_bs") bs=kv[2];
        else if (kv[1]=="batch_ms") batch_ms=kv[2];
        else if (kv[1]=="total_samples_s") total_s=kv[2];
        else if (kv[1]=="peak_vram_mib") vram=kv[2];
      }
      printf "%s %s %s %s\n", total_s, bs, batch_ms, vram
    }' \
  | sort -k1 -n -r \
  | awk 'BEGIN { printf "%-9s  %-6s  %-9s  %-9s\n", "tot_s/s", "val_bs", "batch_ms", "peak_MiB" }
         { printf "%-9s  %-6s  %-9s  %-9s\n", $1, $2, $3, $4 }'
