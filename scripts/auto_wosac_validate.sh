#!/usr/bin/env bash
# =============================================================================
# auto_wosac_validate.sh
# -----------------------------------------------------------------------------
# 왼쪽 터미널에서 돌고 있는 학습 (task_name=$TRAIN_TASK) 이 "코드실행이 완전히
# 종료"될 때까지 10분마다 polling 한 뒤, 성공 종료를 확인하면 다음을 실행합니다.
#
#   1) sim_agents_sub_flow action=validate 로 WOSAC 2025 validation
#      제출 파일 (sim_agents_2025_submission.tar.gz) 생성
#   2) waymo_submission.enabled=true 로 Waymo 공식 사이트에 자동 업로드
#      (이 서버엔 chromium 시스템 lib 이 없어 HTTP fallback 경로로 동작)
#
# 4x A100-80GB 전용 프리셋 `experiment=sim_agents_sub_flow_a100_4` 를 사용합니다
# (val_batch_size=16, 실측 peak ~30 GiB/GPU 로 80GB 중 63% 여유, throughput
#  포화 지점. 벤치 근거는 configs/experiment/sim_agents_sub_flow_a100_4.yaml
#  헤더 주석 참조.)
#
# 사용법 (터미널 닫혀도 계속 돌게 nohup 또는 tmux 권장):
#   cd /mnt/nuplan/projects/catk
#   nohup bash scripts/auto_wosac_validate.sh > /tmp/auto_wosac.log 2>&1 &
#   tail -f /tmp/auto_wosac.log
# =============================================================================
set -euo pipefail

# --- 환경 설정 ---
REPO_DIR=/mnt/nuplan/projects/catk
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
TRAIN_TASK=flow_semi_continuous_finetune_inv_best_a_100
TRAIN_RUN_DIR=$REPO_DIR/logs/$TRAIN_TASK/runs/2026-04-20_19-12-52
TRAIN_LOG=$TRAIN_RUN_DIR/$TRAIN_TASK.log

# Checkpoint 선택
#   last.ckpt       → best-monitored (val realism_meta_metric 최고점 심볼릭)  [기본]
#   epoch_last.ckpt → 최종 epoch 의 rolling 상태
CKPT_NAME="last.ckpt"
CKPT_PATH=$TRAIN_RUN_DIR/checkpoints/$CKPT_NAME

SUBMIT_TASK=flow_sim_agents_val_a100x4_auto
POLL_INTERVAL=600  # 10 min
STORAGE_STATE_PATH=$REPO_DIR/secrets/waymo/waymo_storage_state.json

cd "$REPO_DIR"

# --- 0) 사전 환경 체크 (빨리 실패) -----------------------------------------
# 학습이 끝난 뒤에 환경이 깨져서 submission 을 못 날리는 최악의 경우를 막기 위해
# 10분 대기 루프 들어가기 전에 필수 전제조건을 먼저 검증합니다.
echo "[$(date '+%F %T')] Pre-flight environment checks..."

if ! command -v torchrun > /dev/null; then
    echo "!!! ERROR: torchrun not found in PATH."
    echo "!!! Activate the catk conda env before running this script:"
    echo "!!!   conda activate catk"
    echo "!!! Current PATH: $PATH"
    exit 1
fi
echo "  torchrun:    $(command -v torchrun)"

if ! command -v nvidia-smi > /dev/null; then
    echo "!!! ERROR: nvidia-smi not found."
    exit 1
fi
echo "  nvidia-smi:  $(command -v nvidia-smi)"

if [ ! -d "$CACHE_ROOT/validation" ] || [ ! -d "$CACHE_ROOT/validation_tfrecords_splitted" ]; then
    echo "!!! ERROR: validation cache incomplete under $CACHE_ROOT"
    echo "!!! Expected both validation/ and validation_tfrecords_splitted/"
    exit 1
fi
echo "  cache_root:  $CACHE_ROOT (validation OK)"

if [ ! -s "$STORAGE_STATE_PATH" ]; then
    echo "!!! ERROR: waymo_storage_state.json missing or empty at:"
    echo "!!!   $STORAGE_STATE_PATH"
    echo "!!! Without it, waymo_submission.enabled=true cannot auto-upload."
    exit 1
fi
echo "  storage:     $STORAGE_STATE_PATH ($(stat -c %s "$STORAGE_STATE_PATH") bytes)"

if [ ! -d "$TRAIN_RUN_DIR" ]; then
    echo "!!! ERROR: training run dir does not exist: $TRAIN_RUN_DIR"
    exit 1
fi
echo "  train_run:   $TRAIN_RUN_DIR"

if [ ! -f "$TRAIN_LOG" ]; then
    echo "!!! ERROR: training log file not found: $TRAIN_LOG"
    exit 1
fi
echo "  train_log:   $TRAIN_LOG"

echo "[$(date '+%F %T')] Pre-flight OK."
echo ""

# --- 1) 학습 완료 대기 (10 min polling) ---
echo "========================================================"
echo "[$(date '+%F %T')] Watching training ($TRAIN_TASK)"
echo "  run dir:     $TRAIN_RUN_DIR"
echo "  poll every:  ${POLL_INTERVAL}s"
echo "========================================================"

while true; do
    if pgrep -f "torchrun.*$TRAIN_TASK" > /dev/null; then
        last_mt=$(stat -c '%y' "$TRAIN_RUN_DIR/checkpoints/epoch_last.ckpt" 2>/dev/null \
                  | cut -d'.' -f1 || echo "N/A")
        echo "[$(date '+%F %T')] still running (latest epoch_last.ckpt mtime: $last_mt)"
        sleep "$POLL_INTERVAL"
    else
        echo "[$(date '+%F %T')] torchrun processes gone -- checking completion markers..."
        break
    fi
done

# --- 2) 완료 건강도 검증 ---
if ! grep -q "run.py DONE" "$TRAIN_LOG"; then
    echo ""
    echo "!!! 'run.py DONE' marker NOT found in log."
    echo "!!! Training probably crashed or was killed. Aborting to avoid"
    echo "!!! wasting a WOSAC validation submission attempt on a broken ckpt."
    echo ""
    echo "--- last 20 log lines ---"
    tail -20 "$TRAIN_LOG"
    exit 1
fi

if [ ! -e "$CKPT_PATH" ]; then
    echo "!!! Expected checkpoint not found: $CKPT_PATH"
    echo "Available checkpoints:"
    ls -la "$TRAIN_RUN_DIR/checkpoints/" || true
    exit 1
fi

ACTUAL_CKPT=$(readlink -f "$CKPT_PATH")
echo ""
echo "[$(date '+%F %T')] Training verified complete."
echo "  symlink: $CKPT_PATH"
echo "  actual:  $ACTUAL_CKPT"
ls -la "$CKPT_PATH" "$ACTUAL_CKPT" 2>/dev/null || true

# --- 3) GPU 해제 대기 (최대 6 min) ---
echo ""
echo "[$(date '+%F %T')] Waiting for 4x A100 to fully release memory..."
for i in $(seq 1 12); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '{s+=$1} END {print s+0}')
    if [ "$used" -lt 2000 ]; then
        echo "  GPUs released (total used: ${used} MiB)"
        break
    fi
    echo "  GPUs still holding ${used} MiB (attempt $i/12); sleep 30s"
    sleep 30
done

# --- 4) Submission 생성 + 자동 업로드 (OOM 시 val_batch_size 단계적 감소로 retry) ---
# WOSAC 은 scene 당 rollout 32개가 고정 요구사항이라 n_rollout_closed_val=32 는
# 절대 안 건드립니다. 대신 val_batch_size 를 16→12→8→4→2 로 점진적으로 줄여서
# memory 스파이크를 흡수합니다.
#
# 실측 peak mem (4xA100-80GB, euler/32, n_rollout_closed_val=32):
#   bs=16 → 30 GiB/GPU (63% 여유)  ← 기본 (throughput 포화 지점)
#   bs=12 → ~24 GiB/GPU (70% 여유)
#   bs= 8 → 18 GiB/GPU (78% 여유)
#   bs= 4 → 13 GiB/GPU (84% 여유)  ← 원래 보수적 기본값
#   bs= 2 → 이론상 반드시 성공해야 하는 하한
# bs=16 이 OOM 날 가능성은 매우 낮지만 혹시 극단적 heavy scene 이 있을 경우
# 대비해서 fallback chain 은 남겨둡니다.
RETRY_VAL_BATCHES=(16 12 8 4 2)

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

# OOM 판정용 regex: torch CUDA OOM + cgroup OOM killer + NCCL OOM 전파
OOM_PATTERN='CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|CUDA error: out of memory|Killed|signal 9|SIGKILL'

SUCCESS=0
LAST_ATTEMPT_TMP_LOG=""

for attempt_idx in "${!RETRY_VAL_BATCHES[@]}"; do
    vbs=${RETRY_VAL_BATCHES[$attempt_idx]}
    attempt_num=$((attempt_idx + 1))
    attempt_task="${SUBMIT_TASK}_bs${vbs}"
    tmp_log=$(mktemp /tmp/auto_wosac_attempt_XXXX.log)
    LAST_ATTEMPT_TMP_LOG=$tmp_log

    echo ""
    echo "========================================================"
    echo "[$(date '+%F %T')] Attempt $attempt_num/${#RETRY_VAL_BATCHES[@]}"
    echo "  task_name:     $attempt_task"
    echo "  ckpt:          $ACTUAL_CKPT"
    echo "  val_batch:     $vbs   (rollouts/scene: 32 → $((vbs*32)) materialized/rank)"
    echo "  solver:        euler, 32 steps   (commit b4a59ae intent)"
    echo "  auto upload:   yes (HTTP fallback)"
    echo "  attempt log:   $tmp_log"
    echo "========================================================"
    echo ""

    # torchrun 실행. `if !` 로 감싸면 set -e 에 걸리지 않고 실패 코드를 잡을 수 있음.
    # pipefail 가 걸려있어 tee 로 복사해도 torchrun 의 실패가 파이프라인 실패로 전파됨.
    #
    # Solver override 에 대한 메모:
    #   b4a59ae 커밋은 `configs/model/smart_flow.yaml` 의 `decoder.flow_solver_*`
    #   (= fallback 값) 을 euler/32 로 바꿨지만, 실제 WOSAC rollout 은
    #   `model_config.validation_rollout_sampling.{sample_steps,sample_method}`
    #   을 "우선 참조" 하기 때문에 해당 키를 CLI override 로 같이 덮어써야
    #   euler/32 가 실제로 적용됩니다 (flow_agent_decoder.py:336-345 참조).
    if CUDA_VISIBLE_DEVICES=0,1,2,3 \
         torchrun \
           --standalone \
           --nproc_per_node=4 \
           -m src.run \
           experiment=sim_agents_sub_flow_a100_4 \
           action=validate \
           trainer=ddp \
           trainer.devices=4 \
           paths.cache_root="$CACHE_ROOT" \
           ckpt_path="$CKPT_PATH" \
           task_name="$attempt_task" \
           trainer.limit_val_batches=1.0 \
           data.val_batch_size="$vbs" \
           model.model_config.validation_rollout_sampling.sample_method=euler \
           model.model_config.validation_rollout_sampling.sample_steps=32 \
           waymo_submission.enabled=true \
           waymo_submission.poll_submission_status=false \
           2>&1 | tee "$tmp_log" ; then
        echo ""
        echo "[$(date '+%F %T')] Attempt $attempt_num SUCCEEDED (val_batch_size=$vbs)."
        SUCCESS=1
        break
    fi

    # 실패 → OOM 인지 판별
    if grep -qE "$OOM_PATTERN" "$tmp_log"; then
        echo ""
        echo "[$(date '+%F %T')] OOM detected at val_batch_size=$vbs."
        if [ "$attempt_num" -lt "${#RETRY_VAL_BATCHES[@]}" ]; then
            next_vbs=${RETRY_VAL_BATCHES[$((attempt_idx + 1))]}
            echo "  → retrying with val_batch_size=$next_vbs after 30s GPU cooldown..."

            # GPU 정리: 혹시 살아남은 worker 가 있을 수 있으니 강제 정리
            pkill -9 -f "torchrun.*$attempt_task" 2>/dev/null || true
            pkill -9 -f "src.run.*$attempt_task"  2>/dev/null || true
            sleep 30
            # GPU 해제 재확인
            for i in $(seq 1 10); do
                used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                       | awk '{s+=$1} END {print s+0}')
                if [ "$used" -lt 2000 ]; then
                    echo "  GPUs cooled down (total used: ${used} MiB)"
                    break
                fi
                echo "  GPUs still holding ${used} MiB; sleep 15s"
                sleep 15
            done
        else
            echo "  !!! already at smallest val_batch_size (=$vbs); cannot reduce further."
        fi
    else
        # OOM 이 아닌 다른 에러 (모델 버그, I/O, 권한 등). Infinite-retry 의미 없음.
        echo ""
        echo "!!! Attempt $attempt_num failed with a NON-OOM error."
        echo "!!! Stopping retries (reducing bs would not fix this)."
        echo ""
        echo "--- last 40 lines of attempt log ---"
        tail -40 "$tmp_log"
        exit 1
    fi
done

if [ "$SUCCESS" -ne 1 ]; then
    echo ""
    echo "!!! All ${#RETRY_VAL_BATCHES[@]} attempts OOMed (val_batch_size down to 1)."
    echo "!!! This is unexpected on 4x A100-80GB. Inspect the last attempt log:"
    echo "!!!   $LAST_ATTEMPT_TMP_LOG"
    echo "!!! Suggestions:"
    echo "!!!   - check if another process is holding GPU memory"
    echo "!!!     (nvidia-smi → fuser -v /dev/nvidia*)"
    echo "!!!   - consider lowering model.model_config.n_rollout_closed_val=16"
    echo "!!!     (WOSAC requires 32 → submission score may be affected)"
    exit 1
fi

# 성공한 attempt 의 임시 로그는 정리 (실패 로그는 디버깅용으로 /tmp 에 남겨둠)
rm -f "$LAST_ATTEMPT_TMP_LOG"

final_vbs=${RETRY_VAL_BATCHES[$attempt_idx]}
final_task="${SUBMIT_TASK}_bs${final_vbs}"

echo ""
echo "[$(date '+%F %T')] Pipeline complete."
echo ""
echo "최종 성공 configuration:"
echo "  val_batch_size = $final_vbs"
echo "  task_name      = $final_task"
echo ""
echo "산출물 위치:"
echo "  logs/$final_task/runs/<timestamp>/sim_agents_2025_submission.tar.gz"
echo "  logs/$final_task/runs/<timestamp>/sim_agents_2025_submission/"
echo ""
echo "업로드가 성공했다면 리더보드의 Validation 섹션에서 method_name"
echo "= 'Flow Agents 7M' 결과를 확인할 수 있습니다."
