#!/usr/bin/env bash
# Resume the interrupted 20260605 H100x4+H100x2 TrajTok run on testas A100x7.
#
# The bootstrap checkpoint is downloaded from the latest W&B epoch_last artifact,
# copied into the testas pod, patched to the A100x7 sqrt-scaled LR, and then
# passed to the standard OOM-retry launcher. Subsequent retries resume from the
# latest task-local epoch_last.ckpt.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_TASK_NAME="${SOURCE_TASK_NAME:-smart_ntp_pretrain_h100x4_h100x2_globalbs108_lr581e4_oom_retry_trajtok_hidden128_renewedvocab_fulltraj_topk12_trainselectfalse_20260605}"
NAMESPACE="${NAMESPACE:-p-pnc}"
CONTAINER="${CONTAINER:-main}"
POD="${POD:-testas}"
INITIAL_BS="${INITIAL_BS:-18}"
TOTAL_GPU_COUNT="${TOTAL_GPU_COUNT:-7}"
BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-108}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-0.0005809475}"
CKPT_LOCAL_ROOT="${CKPT_LOCAL_ROOT:-${REPO_ROOT}/logs/_wandb_epoch_last/${SOURCE_TASK_NAME}}"
REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/mnt/nuplan/projects/catk/checkpoints/trajtok_resume_h100x4x2_20260605}"
REMOTE_SOURCE_CKPT_PATH="${REMOTE_SOURCE_CKPT_PATH:-${REMOTE_CKPT_DIR}/epoch_last_from_wandb.ckpt}"
REMOTE_CKPT_PATH="${REMOTE_CKPT_PATH:-${REMOTE_CKPT_DIR}/epoch_last_from_wandb_lrpatched_bs${INITIAL_BS}.ckpt}"

resolve_learning_rate() {
  python - "$BASE_LEARNING_RATE" "$BASE_TOTAL_BATCH_SIZE" "$TOTAL_GPU_COUNT" "$INITIAL_BS" <<'PY'
import math
import sys

base_lr = float(sys.argv[1])
base_total_batch = int(sys.argv[2])
total_gpu_count = int(sys.argv[3])
per_rank_batch = int(sys.argv[4])
print(f"{base_lr * math.sqrt((total_gpu_count * per_rank_batch) / base_total_batch):.10g}")
PY
}

if [[ -n "${LEARNING_RATE:-}" && "$LEARNING_RATE" != "auto" ]]; then
  PATCHED_BASE_LR="$LEARNING_RATE"
else
  PATCHED_BASE_LR="$(resolve_learning_rate)"
fi
LR_TAG="$(
  python - <<PY
lr = float("${PATCHED_BASE_LR}")
print(f"lr{int(round(lr * 1_000_000))}e-6")
PY
)"

mkdir -p "$CKPT_LOCAL_ROOT"
download_output="$(
  python scripts/download_wandb_epoch_last_artifact.py \
    --run-name "$SOURCE_TASK_NAME" \
    --output-dir "$CKPT_LOCAL_ROOT"
)"
printf '%s\n' "$download_output"
SOURCE_CKPT_PATH="$(awk -F= '/^CKPT_PATH=/{print $2}' <<< "$download_output" | tail -1)"
if [[ -z "$SOURCE_CKPT_PATH" || ! -f "$SOURCE_CKPT_PATH" ]]; then
  echo "ERROR: failed to resolve downloaded checkpoint path." >&2
  exit 1
fi

kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- mkdir -p "$REMOTE_CKPT_DIR"
kubectl cp -n "$NAMESPACE" -c "$CONTAINER" "$SOURCE_CKPT_PATH" "${POD}:${REMOTE_SOURCE_CKPT_PATH}"
kubectl cp -n "$NAMESPACE" -c "$CONTAINER" "scripts/patch_checkpoint_lr.py" "${POD}:${REMOTE_CKPT_DIR}/patch_checkpoint_lr.py"
kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- bash -lc "
set -Eeuo pipefail
source /mnt/nuplan/miniforge/etc/profile.d/conda.sh 2>/dev/null || true
conda activate catk 2>/dev/null || true
python $(printf '%q' "${REMOTE_CKPT_DIR}/patch_checkpoint_lr.py") \
  --input $(printf '%q' "$REMOTE_SOURCE_CKPT_PATH") \
  --output $(printf '%q' "$REMOTE_CKPT_PATH") \
  --base-lr $(printf '%q' "$PATCHED_BASE_LR")
"
kubectl exec -n "$NAMESPACE" "$POD" -c "$CONTAINER" -- ls -lh "$REMOTE_CKPT_PATH"

export PODS="${PODS:-testas}"
export PROJECT_ROOT="${PROJECT_ROOT:-/tmp/catk_smart_ntp_testas_a100x7_trajtok_resume_h100x4x2_20260605}"
export BRANCH="${BRANCH:-trajtok}"
export TASK_NAME="${TASK_NAME:-smart_ntp_resume_testas_a100x7_from_h100x4x2_20260605_bs${INITIAL_BS}_${LR_TAG}}"
export SESSION="${SESSION:-catk-smart-ntp-testas-a100x7-trajtok-resume-h100x4x2}"
export CACHE_ROOT="${CACHE_ROOT:-/workspace/womd_v1_3/SMART_cache}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export INITIAL_BS
export OOM_STEP="${OOM_STEP:-2}"
export MIN_BS="${MIN_BS:-14}"
export TOTAL_GPU_COUNT
export BASE_TOTAL_BATCH_SIZE
export BASE_LEARNING_RATE
export LEARNING_RATE="${LEARNING_RATE:-auto}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-12}"
export BOOTSTRAP_CKPT_PATH="$REMOTE_CKPT_PATH"
export PATCH_RESUME_CKPT_LR="${PATCH_RESUME_CKPT_LR:-true}"
export PATCH_CHECKPOINT_LR_SCRIPT="${PATCH_CHECKPOINT_LR_SCRIPT:-${REMOTE_CKPT_DIR}/patch_checkpoint_lr.py}"
export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"

bash scripts/start_smart_ntp_testas_a100x7_trajtok_pretrain_oom_retry.sh "$@"
