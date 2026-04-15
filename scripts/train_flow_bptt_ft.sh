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
#   - N_ROLLOUT_CLOSED_VAL: closed-loop val 시 시나리오당 rollout 수 (yaml 16과 동일 기본, 낮출수록 빠름)
#
# 전체 flow_decoder 학습(velocity_head만이 아님): FLOW_VELOCITY_HEAD_ONLY=false
#
# coarse step 수 제한 (전체 16이 아니라 앞 N coarse만; 그 구간 전체 역전파·soft RMM):
#   BPTT_MAX_COARSE_STEPS=3 sh scripts/train_flow_bptt_ft.sh
#   (비우면 yaml 기본: null = coarse 전부)
#
# training split + splitted tfrecords (train/val 분리, rmm_bptt_ft 유지):
#   A) 기존 training/ 캐시를 그대로 쓰고 tfrecord 만 올릴 때:
#        python -m src.data_preprocess --input_dir <womd_scenario> --output_dir "${CACHE_ROOT}" \
#          --split training --write_tfrecords always
#   B) closed-loop 전용 폴더(validation 과 동일 레이아웃, plain training/ 과 분리):
#        python -m src.data_preprocess --input_dir <womd_scenario> --output_dir "${CACHE_ROOT}" \
#          --split training --output_split closed_loop_train --write_tfrecords always
#      → closed_loop_train/ + closed_loop_train_tfrecords_splitted/
#   실행 예 (B):
#        EXTRA_ARGS="data.train_raw_dir=${CACHE_ROOT}/closed_loop_train data.train_tfrecords_splitted=${CACHE_ROOT}/closed_loop_train_tfrecords_splitted" sh scripts/train_flow_bptt_ft.sh
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

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

# 학습 데이터 경로 (pickle + splitted tfrecords)
# 기본: train_with_tfrecords 폴더 (tfrecord가 함께 전처리된 트레이닝 스플릿)
TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
# 정수(예: 10) = val 배치 최대 개수. 0~1 실수 = 데이터셋 비율. 빠른 RMM 스모크는 10 권장.
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
# val_check_interval: 정수면 "N training step마다" 검증, 0~1 실수면 "에폭의 해당 비율마다" 검증.
# limit_train_batches가 작으면 정수 N은 N 이하로 맞출 것(그렇지 않으면 Lightning 설정 오류).
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-0}"
TRAIN_B="${TRAIN_B:-4}"
VAL_B="${VAL_B:-1}"
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
# DataLoader 워커는 GPU 프로세스마다 따로 뜸: (NPROC_PER_NODE × NUM_WORKERS) + α.
# 예: 2GPU × 63워커 ≈ 126개 워커만으로도 RAM·파일 디스크립터·스케줄링 폭주 → 몇 step 후 OOM/Killed/멈춤이 잦음.
# 단일 GPU에서도 63은 과한 경우 많음. 필요 시 NUM_WORKERS=16 등으로 올려서 튜닝.
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

# rmm_bptt_ft: LR를 너무 키우면 soft RMM 역전파·옵티마 스텝에서 NaN/폭주가 나기 쉬움.
# multi-scenario(TRAIN_B>1)에서 gradient 방향이 시나리오마다 충돌하므로 단일 시나리오보다
# LR을 낮춰야 함. 권장: single-scenario=1e-6, multi-scenario=1e-6~1e-5.
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# GT FM 정규화 (rmm_bptt_ft): flow_train_clean_norm velocity FM MSE 가중치. 0 이면 비활성.
FLOW_REG_LAMBDA="${FLOW_REG_LAMBDA:-10.0}"

ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"
# 0 이면 validation 비디오 생성 안 함 (Waymo rollout MP4 + W&B 업로드 스킵)
N_VIS_BATCH="${N_VIS_BATCH:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-1}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

# closed-loop validation: 시나리오당 rollout N (best-of-N minADE 등; configs/experiment/flow_bptt_ft.yaml n_rollout_closed_val)
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-4}"

# official closed-loop SimAgents RMM 갱신에 쓰는 val 배치 수 (CPU 멀티프로세스 구간)
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-2}"
# "real": 공식 TF RMM (subprocess, 느림), "hard": PyTorch 인-프로세스 RMM (빠름, 수치 동등)
# 속도 추가 팁: WOSAC_TORCH_COMPILE=1 설정 시 dno/ttc/d_road 커널을 torch.compile 로 최적화
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-1}"

BPTT_N_ROLLOUTS="${BPTT_N_ROLLOUTS:-1}"
RMM_BPTT_USE_REF_MODEL="${RMM_BPTT_USE_REF_MODEL:-false}"
# true: training step 마다 ref G rollout (no_grad) → train/rmm_ref + train/rmm_delta 로깅
RMM_BPTT_REF_TRAIN="${RMM_BPTT_REF_TRAIN:-true}"
# true: validation 시 pretrained ref model 도 rollout → val_ref/rmm + val_delta/rmm (val 시간 ≈ 2배)
RMM_BPTT_REF_VAL="${RMM_BPTT_REF_VAL:-true}"
# OOM 발생 시 true로 설정: flow ODE model_fn 호출을 gradient checkpoint으로 감쌈
# (Neural ODE adjoint 이산 버전) — solver_steps×activation 메모리를 activation 수준으로 절감
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"
# 비어 있으면 오버라이드 없음 → configs/experiment 의 bptt_max_coarse_steps (null = 전체)
BPTT_MAX_COARSE_STEPS="${BPTT_MAX_COARSE_STEPS:-4}"
# true (기본): G rollout 을 1개씩 순차 실행 후 각각 backward → 피크 메모리 ≈ G 배 절감
BPTT_SEQUENTIAL_ROLLOUTS="${BPTT_SEQUENTIAL_ROLLOUTS:-false}"
# 앞 N coarse step 을 no_grad/detach (sliding-window BPTT). 0 = 비활성.
# 예: BPTT_WARM_COARSE_STEPS=12 이면 마지막 4 step (BPTT_MAX_COARSE_STEPS=16 기준)만 gradient.
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"
# true: HierarchicalFlowDecoder.velocity_head만 학습 (인코더·flow 트렁크·residual 동결)
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"

# =============================================================================
# Gradient norm 관련 하이퍼파라미터
# =============================================================================
# [1] trainer.gradient_clip_val (GRAD_CLIP_VAL, 위에서 정의):
#     Lightning이 loss.backward() 이후 optimizer.step() 직전에 적용하는
#     모델 전체 파라미터 gradient의 전역 L2 norm clip. 기본 1.0.
#
# [2] bptt_grad_clip_traj:
#     pred_traj / pred_head_traj tensor에 등록되는 backward hook.
#     soft RMM → pred_traj 방향으로 역전파되는 gradient의 L2 norm 상한.
#     0 이하면 비활성. element-wise clamp가 아닌 norm clip이므로 방향 유지.
#     주의: multi-scenario(TRAIN_B>1)에서는 pred_traj가
#           [n_sc*n_agent, G, T, 2] 크기라 single보다 norm이 자연히 커짐.
#           시나리오 간 gradient 상쇄로 실효 gradient는 오히려 작아질 수 있음.
#           → 너무 작으면 학습 신호 소멸, 너무 크면 폭주. 1.0~5.0 사이 탐색 권장.
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-0}"
#
# [3] bptt_debug:
#     true 시 첫 번째 시나리오·rollout의 sim_feature 극값과
#     per-metric likelihood를 WARNING 레벨로 출력.
#     train/grad_norm_velocity_head, train/grad_norm_total 은
#     on_after_backward에서 항상 로깅됨 (W&B 확인).
BPTT_DEBUG="${BPTT_DEBUG:-true}"

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
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "TRAIN_RAW_DIR=${TRAIN_RAW_DIR}"
echo "TRAIN_TFRECORDS_SPLITTED=${TRAIN_TFRECORDS_SPLITTED}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} MAX_EPOCHS=${MAX_EPOCHS} WANDB_MODE=${WANDB_MODE}"
echo "LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS} val_check_interval=${VAL_CHECK_INTERVAL} check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
echo "BPTT_N_ROLLOUTS=${BPTT_N_ROLLOUTS} FLOW_VELOCITY_HEAD_ONLY=${FLOW_VELOCITY_HEAD_ONLY} RMM_BPTT_USE_REF_MODEL=${RMM_BPTT_USE_REF_MODEL} BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT} BPTT_MAX_COARSE_STEPS=${BPTT_MAX_COARSE_STEPS:-"(yaml)"}"
echo "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} N_VIS_BATCH=${N_VIS_BATCH} N_ROLLOUT_CLOSED_VAL=${N_ROLLOUT_CLOSED_VAL} N_BATCH_SIM_AGENTS_METRIC=${N_BATCH_SIM_AGENTS_METRIC} VALIDATION_METRIC=${VALIDATION_METRIC}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} NUM_WORKERS=${NUM_WORKERS} (≈ ${NPROC_PER_NODE}×${NUM_WORKERS} dataloader worker 프로세스 + 메인)"
echo "[grad] GRAD_CLIP_VAL=${GRAD_CLIP_VAL} BPTT_GRAD_CLIP_TRAJ=${BPTT_GRAD_CLIP_TRAJ} LR=${LR} FLOW_REG_LAMBDA=${FLOW_REG_LAMBDA} BPTT_DEBUG=${BPTT_DEBUG}"

PORT="$(get_free_port)"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${PORT}" --rdzv_endpoint="127.0.0.1:${PORT}" -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action=finetune \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  data.train_raw_dir="${TRAIN_RAW_DIR}" \
  data.train_tfrecords_splitted="${TRAIN_TFRECORDS_SPLITTED}" \
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
  model.model_config.finetune.flow_reg_lambda="${FLOW_REG_LAMBDA}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.bptt_n_rollouts="${BPTT_N_ROLLOUTS}" \
  model.model_config.finetune.rmm_bptt_use_ref_model="${RMM_BPTT_USE_REF_MODEL}" \
  model.model_config.finetune.rmm_bptt_ref_train="${RMM_BPTT_REF_TRAIN}" \
  model.model_config.finetune.rmm_bptt_ref_val="${RMM_BPTT_REF_VAL}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_sequential_rollouts="${BPTT_SEQUENTIAL_ROLLOUTS}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.bptt_debug="${BPTT_DEBUG}" \
  ${BPTT_MAX_COARSE_STEPS:+model.model_config.finetune.bptt_max_coarse_steps="${BPTT_MAX_COARSE_STEPS}"} \
  ${EXTRA_ARGS}

