#!/bin/sh
# =============================================================================
# Flow-BPTT (rmm_bptt_ft) 샘플 실행 스크립트
# =============================================================================
# configs/experiment/flow_bptt_ft.yaml 기준:
#   - validation split 경로로 train 로더 구성 (soft RMM / tfrecord 정합)
#   - finetune.mode=rmm_bptt_ft, bptt_n_rollouts 등
# 기본값은 짧은 스모크에 가깝게 잡혀 있음. 본 학습 시 LIMIT_TRAIN_BATCHES / MAX_EPOCHS 등을 환경변수로 늘리면 됨.
# 예: sh scripts/train_flow_bptt_ft.sh
#     MAX_EPOCHS=10 LIMIT_TRAIN_BATCHES=1.0 WANDB_MODE=online sh scripts/train_flow_bptt_ft.sh
#
# 빠른 validation (비디오 끄고, val 배치·RMM 배치만 소량):
#   N_VIS_BATCH=0 N_BATCH_SIM_AGENTS_METRIC=10 LIMIT_VAL_BATCHES=10 sh scripts/train_flow_bptt_ft.sh
#   - N_VIS_BATCH=0 이면 closed-loop W&B 비디오 생성 안 함 (batch_idx < n_vis_batch 일 때만 생성)
#   - n_batch_sim_agents_metric: official SimAgents RMM(CPU 풀)에 넣는 val 배치 수 상한
#   - LIMIT_VAL_BATCHES: 정수면 그만큼의 val 배치만 전체 검증 루프에서 사용 (open+closed 포함)
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2, 3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_bptt_ft}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-a100-bpttft}"
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-0.1}"
# 정수(예: 10) = val 배치 최대 개수. 0~1 실수 = 데이터셋 비율. 빠른 RMM 스모크는 10 권장.
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
# val_check_interval: 정수면 "N training step마다" 검증, 0~1 실수면 "에폭의 해당 비율마다" 검증.
# limit_train_batches가 작으면 정수 N은 N 이하로 맞출 것(그렇지 않으면 Lightning 설정 오류).
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"

TRAIN_B="${TRAIN_B:-4}"
VAL_B="${VAL_B:-4}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
# DataLoader 워커는 GPU 프로세스마다 따로 뜸: (NPROC_PER_NODE × NUM_WORKERS) + α.
# 예: 2GPU × 63워커 ≈ 126개 워커만으로도 RAM·파일 디스크립터·스케줄링 폭주 → 몇 step 후 OOM/Killed/멈춤이 잦음.
# 단일 GPU에서도 63은 과한 경우 많음. 필요 시 NUM_WORKERS=16 등으로 올려서 튜닝.
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

# rmm_bptt_ft: LR를 너무 키우면 soft RMM 역전파·옵티마 스텝에서 NaN/폭주가 나기 쉬움. 문제 시 1e-6 전후로 낮춰볼 것.
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

ROLLOUT_STEPS="${ROLLOUT_STEPS:-3}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"
# 0 이면 validation 비디오 생성 안 함 (Waymo rollout MP4 + W&B 업로드 스킵)
N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

# official closed-loop SimAgents RMM 갱신에 쓰는 val 배치 수 (CPU 멀티프로세스 구간)
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10}"

BPTT_N_ROLLOUTS="${BPTT_N_ROLLOUTS:-3}"
RMM_BPTT_USE_REF_MODEL="${RMM_BPTT_USE_REF_MODEL:-false}"
# OOM 발생 시 true로 설정: flow ODE model_fn 호출을 gradient checkpoint으로 감쌈
# (Neural ODE adjoint 이산 버전) — solver_steps×activation 메모리를 activation 수준으로 절감
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"

WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

echo "Experiment=${MY_EXPERIMENT}"
echo "CACHE_ROOT=${CACHE_ROOT} (flow_bptt_ft: train_* → validation split under cache)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} MAX_EPOCHS=${MAX_EPOCHS} WANDB_MODE=${WANDB_MODE}"
echo "LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS} val_check_interval=${VAL_CHECK_INTERVAL} check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
echo "BPTT_N_ROLLOUTS=${BPTT_N_ROLLOUTS} RMM_BPTT_USE_REF_MODEL=${RMM_BPTT_USE_REF_MODEL} BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT}"
echo "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} N_VIS_BATCH=${N_VIS_BATCH} N_BATCH_SIM_AGENTS_METRIC=${N_BATCH_SIM_AGENTS_METRIC}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} NUM_WORKERS=${NUM_WORKERS} (≈ ${NPROC_PER_NODE}×${NUM_WORKERS} dataloader worker 프로세스 + 메인)"

PORT="$(get_free_port)"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.train_max_num="${TRAIN_MAX_NUM}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRAD_CLIP_VAL}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.finetune.rollout_steps="${ROLLOUT_STEPS}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.bptt_n_rollouts="${BPTT_N_ROLLOUTS}" \
  model.model_config.finetune.rmm_bptt_use_ref_model="${RMM_BPTT_USE_REF_MODEL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  ${EXTRA_ARGS}

