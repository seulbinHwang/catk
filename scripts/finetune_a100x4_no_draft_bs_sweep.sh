#!/usr/bin/env bash
# =============================================================================
# A100x4 no-DRaFT fine-tune with adaptive train_batch_size on CUDA OOM.
# -----------------------------------------------------------------------------
# Behavior:
#   - First attempt:  train_batch_size=64, action=finetune (load pretrained
#                     weights, start at epoch 0).
#   - On CUDA OOM:    decrement train_batch_size by 4 and retry. Each retry
#                     uses action=fit + ckpt_path=<latest epoch_last.ckpt>,
#                     which is Lightning's full-resume path and restores
#                     epoch counter / optimizer / scheduler state. Net effect:
#                     training resumes from the last fully-completed epoch.
#   - Non-OOM error:  abort (no point reducing batch size for unrelated bugs).
#   - bs floor:       4 (matches the experiment's default)
#
# Resume mechanics:
#   Lightning's ModelCheckpoint with save_last=True writes epoch_last.ckpt at
#   end-of-epoch, so a mid-epoch OOM leaves a checkpoint at the last *fully
#   completed* epoch. We point `action=fit ckpt_path=<that ckpt>` at that
#   file; Lightning resumes everything (weights, optimizer, scheduler, epoch).
#
#   If attempt 1 OOMs before any epoch completes (no epoch_last.ckpt yet),
#   the next attempt falls back to action=finetune from the pretrain ckpt
#   with the smaller batch size (epoch 0 retry).
#
# Usage:
#   cd /mnt/nuplan/projects/catk
#   bash scripts/finetune_a100x4_no_draft_bs_sweep.sh
#
# Or as a long-running background job:
#   nohup bash scripts/finetune_a100x4_no_draft_bs_sweep.sh \
#       > /tmp/finetune_a100x4_no_draft_sweep.log 2>&1 &
#   disown
#   tail -f /tmp/finetune_a100x4_no_draft_sweep.log
# =============================================================================
set -uo pipefail

REPO_DIR=/mnt/nuplan/projects/catk
cd "$REPO_DIR"

# --- Fixed inputs --------------------------------------------------------
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
PRETRAIN_CKPT=$REPO_DIR/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/run_4pxhrpv8_v70/epoch_last.ckpt
TASK_NAME=flow_semi_continuous_finetune_a100x4_no_draft
TASK_LOG_ROOT=$REPO_DIR/logs/$TASK_NAME/runs

# --- Batch-size sweep ----------------------------------------------------
START_BS=64
STEP=4
MIN_BS=4

# --- Linear LR scaling ---------------------------------------------------
# Reference point matches the original A100x4 preset header (line 17-22 of
# configs/experiment/finetune_draft_flow_a100x4.yaml):
#   per_gpu_bs * gpus * accumulate = 36 * 4 * 2 = 288 -> lr = 2e-4
# We pin accumulate_grad_batches=1 here, so global batch = bs * 4. Linear
# scaling: lr = BASE_LR * (bs * 4) / BASE_GLOBAL_BATCH.
#
# Caveat: the LR override is consumed by the optimizer factory at model
# __init__ time. On `action=fit` retries Lightning restores the saved
# optimizer state, so the new LR only takes effect on `action=finetune`
# attempts (i.e. the very first attempt + any "no epoch completed yet"
# retries that fall back to pretrain). Once the first epoch is committed,
# the LR is effectively frozen at whatever it was when that epoch ran.
BASE_LR=2e-4
BASE_GLOBAL_BATCH=288
NUM_GPUS=4

compute_lr() {
    # Linear scaling. Output as float (Hydra/OmegaConf accepts e-notation).
    python3 -c "print($BASE_LR * ($1 * $NUM_GPUS) / $BASE_GLOBAL_BATCH)"
}

# --- OOM detection regex -------------------------------------------------
# Covers torch CUDA OOM, cgroup OOM-killer, and NCCL OOM propagation.
OOM_PATTERN='CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|CUDA error: out of memory|Killed|signal 9|SIGKILL'

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

find_latest_epoch_last_ckpt() {
    if [[ ! -d "$TASK_LOG_ROOT" ]]; then
        echo ""
        return
    fi
    find "$TASK_LOG_ROOT" -name epoch_last.ckpt -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}'
}

wait_for_gpu_release() {
    log "  cleaning stragglers and waiting for GPU memory release..."
    pkill -9 -f "torchrun.*$TASK_NAME" 2>/dev/null || true
    pkill -9 -f "python.*-m src.run.*$TASK_NAME" 2>/dev/null || true
    sleep 20
    for i in $(seq 1 12); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        if [[ "$used" -lt 2000 ]]; then
            log "  GPUs released (total used: ${used} MiB)"
            return 0
        fi
        log "  GPUs still holding ${used} MiB (attempt $i/12); sleep 15s"
        sleep 15
    done
    log "  WARNING: GPUs still holding memory after wait. Proceeding anyway."
}

# --- Pre-flight ---------------------------------------------------------
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
    log "ERROR: pretrained checkpoint not found: $PRETRAIN_CKPT"
    exit 1
fi
if [[ ! -d "$CACHE_ROOT/training" && ! -d "$CACHE_ROOT" ]]; then
    log "ERROR: cache root missing: $CACHE_ROOT"
    exit 1
fi

log "Starting A100x4 no-DRaFT fine-tune with adaptive batch size."
log "  pretrain ckpt: $PRETRAIN_CKPT"
log "  cache root:    $CACHE_ROOT"
log "  task_name:     $TASK_NAME"
log "  bs sweep:      $START_BS -> ... -> $MIN_BS (step $STEP)"

bs=$START_BS
attempt=0
while [[ "$bs" -ge "$MIN_BS" ]]; do
    attempt=$((attempt + 1))

    latest_ckpt=$(find_latest_epoch_last_ckpt)
    if [[ -z "$latest_ckpt" ]]; then
        # No completed epoch from any prior attempt. Start fresh from pretrain.
        action=finetune
        ckpt=$PRETRAIN_CKPT
        resume_note="fresh from pretrain (epoch 0)"
    else
        action=fit
        ckpt=$latest_ckpt
        resume_note="Lightning-resume from $latest_ckpt"
    fi

    lr=$(compute_lr "$bs")
    attempt_log=/tmp/${TASK_NAME}_attempt${attempt}_bs${bs}.log

    if [[ "$action" == "fit" ]]; then
        lr_note="$lr (override ignored on resume — optimizer state restored from ckpt)"
    else
        lr_note="$lr (linear-scaled from base ${BASE_LR} @ global=${BASE_GLOBAL_BATCH})"
    fi

    log ""
    log "============================================================"
    log "Attempt $attempt"
    log "  train_batch_size: $bs"
    log "  lr:               $lr_note"
    log "  action:           $action"
    log "  ckpt:             $ckpt"
    log "  resume:           $resume_note"
    log "  attempt log:      $attempt_log"
    log "============================================================"

    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
        torchrun \
            --standalone \
            --nproc_per_node=4 \
            -m src.run \
            experiment=finetune_draft_flow_a100x4 \
            action="$action" \
            trainer=ddp \
            trainer.devices=4 \
            paths.cache_root="$CACHE_ROOT" \
            ckpt_path="$ckpt" \
            model.model_config.draft.enabled=false \
            model.model_config.lr="$lr" \
            data.train_batch_size="$bs" \
            trainer.accumulate_grad_batches=1 \
            task_name="$TASK_NAME" \
            2>&1 | tee "$attempt_log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -eq 0 ]]; then
        log ""
        log "Attempt $attempt SUCCEEDED (train_batch_size=$bs)."
        log "Output dir(s) under: $TASK_LOG_ROOT"
        exit 0
    fi

    if grep -qE "$OOM_PATTERN" "$attempt_log"; then
        log ""
        log "OOM detected at train_batch_size=$bs (rc=$rc)."
        wait_for_gpu_release
        bs=$((bs - STEP))
        if [[ "$bs" -ge "$MIN_BS" ]]; then
            log "  retrying with train_batch_size=$bs..."
        fi
    else
        log ""
        log "Attempt $attempt failed with a NON-OOM error (rc=$rc)."
        log "Reducing batch size will not fix this; aborting."
        log ""
        log "--- last 40 lines of attempt log ---"
        tail -40 "$attempt_log"
        exit 1
    fi
done

log ""
log "ERROR: exhausted batch sizes (down to $MIN_BS) without success."
exit 1
