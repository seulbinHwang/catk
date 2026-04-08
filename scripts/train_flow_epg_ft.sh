#!/bin/sh
# =============================================================================
# Flow-EPG Fine-tuning 실행 스크립트
# =============================================================================
# 목적:
#   - 동일 초기 상태에서 G개 closed-loop rollout → Waymo Sim Agents metametric(RMM) 점수
#   - 그룹 내 RMM으로 advantage 만든 뒤 EPG(Exact Policy Gradient + ELBO log p)로 flow decoder 학습
# 엔트리:
#   torchrun … -m src.run → Hydra configs/experiment/flow_epg_ft.yaml + 아래 CLI override
# 오버라이드:
#   거의 모든 값은 쉘 환경변수로 바꿀 수 있음 (예: TRAIN_B=32 sh scripts/train_flow_epg_ft.sh)
# =============================================================================

# -----------------------------------------------------------------------------
# 로그 / 경고 / 스레드 (학습 전에 export 되어야 하는 것들)
# -----------------------------------------------------------------------------
# Python/root 로거 레벨
export LOGLEVEL=INFO
# Hydra가 설정 오류 시 전체 트레이스백 출력 (디버깅에 유리)
export HYDRA_FULL_ERROR=1
# TensorFlow C++ 로그 억제 (RMM 서브프로세스에서 TF 쓸 때 터미널 정리)
export TF_CPP_MIN_LOG_LEVEL=2
# PyTorch CUDA 메모리 단편화 완화 옵션
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CPU BLAS/OpenMP 스레드 (데이터 전처리·numpy 등). GPU 학습과 CPU 과다 점유 줄이려 조정
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# W&B: online/offline/disabled 등. 오프라인이면 export WANDB_MODE=offline
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
# 이 프로세스에서 보이는 GPU 번호 (물리 GPU와 매핑). 단일 GPU면 "0" 등
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# -----------------------------------------------------------------------------
# 실험 식별 / Conda
# -----------------------------------------------------------------------------
# Hydra experiment 이름 → configs/experiment/${MY_EXPERIMENT}.yaml 로드
MY_EXPERIMENT="${MY_EXPERIMENT:-flow_epg_ft}"
# Hydra task_name / 출력 디렉터리·로그에 붙는 Run 이름
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-a100-epgft}"
# 활성화할 conda 환경 이름
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"

CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# -----------------------------------------------------------------------------
# 데이터 / 체크포인트 경로
# -----------------------------------------------------------------------------
# Waymo(또는 동일 레이아웃) 전처리 캐시 루트. 하위에 train/val/tfrecord 등이 있다고 가정
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1}"
# action=finetune 일 때 run.py 가 torch.load로 읽어 model.load_state_dict(strict=False) 하는 경로
# Lightning 전체 체크포인트(.ckpt)도 동작 (내부에서 weights_only fail 시 폴백)
CKPT_PATH="${CKPT_PATH:-/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt}"

# -----------------------------------------------------------------------------
# Trainer / DataLoader (Lightning Trainer & Hydra data.*)
# -----------------------------------------------------------------------------
# DDP: 프로세스당 1 GPU 권장 → 보통 GPU 개수와 동일
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# 한 epoch당 학습 step 상한 비율. 1.0 = 전체, 0.05 = 5%만 (스모크 테스트용)
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
# 검증: epoch 대비 비율(소수) 또는 step 수. 0.01 = val 데이터의 1% 배치만
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.01}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
# global step 기준 몇 step마다 validation 할지 (check_val_every_n_epoch=null 일 때 주로 사용)
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
# 매 N epoch마다만 val 하려면 숫자. 매 step val 느낌이면 null + VAL_CHECK_INTERVAL 사용
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
# Lightning precision: 32-true, bf16-mixed 등
PRECISION="${PRECISION:-32-true}"
# 그래디언트 클리핑 (0이면 비활성화인지는 Lightning 설정 따름; 여기선 1.0)
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-1.0}"

# 배치: 한 step당 묶는 시나리오 수(정의에 따라 다름). OOM 나면 줄일 것
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-8}"
# data.train_max_num → MultiDataModule 에서 Train transform 생성 시 사용.
# train_use_val_transform=true(EPG yaml 기본)면 사실상 val transform을 쓰므로
# 이 값이 안 쓰일 수 있으나, Hydra/data 설정 바꿀 때를 위해 그대로 전달
TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-8}"
# DataLoader worker 수. 0이면 메인 프로세스만 로드(디버그 편함, 느릴 수 있음)
NUM_WORKERS="${NUM_WORKERS:-4}"
# worker당 미리 꺼내 둘 배치 수 (num_workers>0 일 때 의미 있음)
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
# epoch 사이에 worker 프로세스 유지 (num_workers>0)
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
# CPU 텐서를 pinned memory로 올려 GPU H2D 조금 빠르게
PIN_MEMORY="${PIN_MEMORY:-true}"

# -----------------------------------------------------------------------------
# 옵티마이저 / 스케줄 (model.model_config.* → SMARTFlow)
# -----------------------------------------------------------------------------
LR="${LR:-5e-5}"
# Step scheduler: 초기 linear warmup step 수
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
# Cosine 등 전체 step (-1이면 trainer/모델이 다른 방식으로 결정)
# 참고: ${VAR:--1} 는 unset일 때 기본값 -1
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:--1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# finetune.rollout_steps: Adjoint/terminal-cost 계열에서 flow 시간 [eps,1] 구간을 몇 등분할지.
# EPG 본 loss와 직접 연결은 약하나, epg_bc_lambda>0 인 BC 등에선 영향 가능
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"
# finetune.rollout_noise_scale: 해당 rollout 경로 초기 잡음 스케일(모듈별 해석)
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"
# validation/test 시 video 생성 관련 기본값 (필요 시 env로 오버라이드)
N_VIS_BATCH="${N_VIS_BATCH:-1}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-2}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-4}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

# -----------------------------------------------------------------------------
# Flow-EPG 전용 (model.model_config.finetune.*)
# -----------------------------------------------------------------------------
# epg_n_rollouts (G): 시나리오당 같은 초기 상태에서 몇 개 rollout을 뽑을지.
#   많을수록 RMM ranking 신호는 좋아질 수 있으나 메모리·시간·RMM 서브프로세스 비용 증가
EPG_N_ROLLOUTS="${EPG_N_ROLLOUTS:-4}"
# epg_beta: KL(π_θ || π_ref) 가중치. 클수록 참조 모델에 가깝게
EPG_BETA="${EPG_BETA:-0.1}"
# epg_n_samples: ELBO/log p 근사용 MC 샘플 수 K (shared_t/shared_x0 공유)
EPG_N_SAMPLES="${EPG_N_SAMPLES:-4}"
# true면 사전학습 복제한 frozen ref decoder로 KL 항 계산
EPG_USE_REF_MODEL="${EPG_USE_REF_MODEL:-true}"
# >0 이면 GT flow-matching BC 보조 손실
EPG_BC_LAMBDA="${EPG_BC_LAMBDA:-0.0}"

# -----------------------------------------------------------------------------
# W&B
# -----------------------------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"

# -----------------------------------------------------------------------------
# Hydra 추가 인자 (공백 구분 그대로 tail에 붙음)
# -----------------------------------------------------------------------------
# 예: val closed-loop rollout 수 줄이기
#   EXTRA_ARGS='model.model_config.n_rollout_closed_val=8'
# 예: 데이터 transform/경로
#   EXTRA_ARGS='data.train_use_val_transform=false'
EXTRA_ARGS="${EXTRA_ARGS:-data.train_tfrecords_splitted=/dev/shm/validation_tfrecords_splitted}"

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
echo "Task=${MY_TASK_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TRAIN_B=${TRAIN_B} VAL_B=${VAL_B} NUM_WORKERS=${NUM_WORKERS}"
echo "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} MAX_EPOCHS=${MAX_EPOCHS}"
echo "EPG_N_ROLLOUTS=${EPG_N_ROLLOUTS} EPG_BETA=${EPG_BETA} EPG_N_SAMPLES=${EPG_N_SAMPLES}"
echo "EPG_USE_REF_MODEL=${EPG_USE_REF_MODEL} EPG_BC_LAMBDA=${EPG_BC_LAMBDA}"
echo "N_VIS_BATCH=${N_VIS_BATCH} N_VIS_SCENARIO=${N_VIS_SCENARIO} N_VIS_ROLLOUT=${N_VIS_ROLLOUT}"
echo "WANDB_MODE=${WANDB_MODE} WANDB_ENTITY=${WANDB_ENTITY}"

PORT="$(get_free_port)"
echo "==== Start training (flow_epg_ft): train_batch_size=${TRAIN_B}, val_batch_size=${VAL_B} ===="

# torchrun: 각 rank가 python -m src.run … 동일 인자 수신
# --master_port / rdzv_endpoint: 단일 노드 DDP용 rendezvous
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
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.finetune.epg_n_rollouts="${EPG_N_ROLLOUTS}" \
  model.model_config.finetune.epg_beta="${EPG_BETA}" \
  model.model_config.finetune.epg_n_samples="${EPG_N_SAMPLES}" \
  model.model_config.finetune.epg_use_ref_model="${EPG_USE_REF_MODEL}" \
  model.model_config.finetune.epg_bc_lambda="${EPG_BC_LAMBDA}" \
  ${EXTRA_ARGS}
