#!/bin/sh
# =============================================================================
# Ref-NLL fine-tuning 실행 스크립트 (closed-loop → open-loop likelihood)
# configs/experiment/flow_ref_nll.yaml 기준
# =============================================================================
# 알고리즘:
#   1. Closed-loop generation (BPTT): 0.5s 씩 commit 하며 pred_max_steps × 0.5s
#      (기본 2s) 만큼 굴려 world-frame 의 closed-loop 궤적을 모은다.
#   2. 모은 2s 궤적을 **rollout 시작 시점 pose** 의 local frame 으로 변환해 단일
#      x₁ (= [n_active, 20, 4]) 을 만든다.
#   3. Frozen pretrained ref (open-loop) 로 backward ODE + Hutchinson →
#      log p_ref(τ_2s | initial_anchor) 와 ∂ log p_ref / ∂ x₁ 계산.
#   4. Straight-through loss: L = -mean(∂ log p_ref / ∂ x₁ · x₁)
#      → gradient = -(∂ log p_ref / ∂ x₁) 가 closed-loop AR joint likelihood 를
#      open-loop p(τ|initial) 로 끌어올리는 covariate-shift 보정 fine-tuning.
#
# 사용법:
#   sh scripts/train_flow_ref_nll.sh
#   CKPT_PATH=/path/to/pretrained.ckpt sh scripts/train_flow_ref_nll.sh
#   MAX_EPOCHS=10 WANDB_MODE=online sh scripts/train_flow_ref_nll.sh
#
# ── 핵심 파라미터 ──────────────────────────────────────────────────────────
# REF_NLL_N_ROLLOUTS       : G (시나리오당 closed-loop rollout 수, 권장 2)
# REF_NLL_PRED_MAX_STEPS   : coarse step 수 (반드시 4 = open-loop horizon 2s)
# REF_NLL_N_HUTCH_SAMPLES  : Hutchinson 발산 추정 probe 수 (1 = 빠름)
# REF_NLL_USE_FULL_DIV_GRAD: true → 발산 항 2차 미분 (정확하지만 느림, 보통 false)
# REF_NLL_FM_REG_LAMBDA    : GT FM regularization 가중치 (0 = 비활성, 발산 시 0.05~0.2)
# REF_NLL_LOSS_SCALE       : 전체 loss 스케일 인수
#
# ── BPTT tricks ────────────────────────────────────────────────────────────
# BPTT_USE_ADJOINT         : ODE gradient checkpoint (기본 true, OOM 방지)
# BPTT_GRAD_CLIP_TRAJ      : pred_traj gradient L2 norm clip (기본 1.0; log-prob grad 폭주 방지)
# BPTT_WARM_COARSE_STEPS   : 앞 N coarse step no_grad (sliding-window BPTT)
# BPTT_LAST_N_SOLVER_STEPS : ODE solver 마지막 N step 에만 velocity gradient (기본 2)
# FLOW_VELOCITY_HEAD_ONLY  : true → velocity_head 만 학습 (보수적, 권장)
#
# ── 첫 실행 권장 프로필 (24 GB GPU 가정) ─────────────────────────────────────
#   TRAIN_B=4  REF_NLL_N_ROLLOUTS=2  BPTT_USE_ADJOINT=true
#   BPTT_LAST_N_SOLVER_STEPS=2  BPTT_GRAD_CLIP_TRAJ=1.0  FLOW_VELOCITY_HEAD_ONLY=true
#   LR=5e-6  REF_NLL_LOSS_SCALE=1.0  REF_NLL_FM_REG_LAMBDA=0.0
#   → train/ref_nll_log_p 가 꾸준히 상승 + val_closed/realism_meta_metric 모니터링.
#   ▸ OOM 시: TRAIN_B=2 또는 BPTT_WARM_COARSE_STEPS=2 추가
#   ▸ 발산 (loss → -inf, val drop) 시: REF_NLL_FM_REG_LAMBDA=0.1 또는 REF_NLL_LOSS_SCALE=0.5
# =============================================================================

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_ref_nll}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-main_exp}"
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

TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-${CACHE_ROOT}/train_with_tfrecords}"
TRAIN_TFRECORDS_SPLITTED="${TRAIN_TFRECORDS_SPLITTED:-${CACHE_ROOT}/train_with_tfrecords_tfrecords_splitted}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
PRECISION="${PRECISION:-32-true}"
GRAD_CLIP_VAL="${GRAD_CLIP_VAL:-0}"
# closed-loop BPTT(4 coarse × 8 ODE) + ref-side backward Hutchinson 그래프가 동시에
# 살아있어 rmm_bptt 보다 무거움. adjoint=true 와 함께 4 부터 시작 권장.
TRAIN_B="${TRAIN_B:-16}"
VAL_B="${VAL_B:-16}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-1}"
VALIDATION_METRIC="${VALIDATION_METRIC:-hard}"
WOSAC_TORCH_COMPILE="${WOSAC_TORCH_COMPILE:-0}"

TRAIN_MAX_NUM="${TRAIN_MAX_NUM:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-false}"

LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
if [ -z "${LR_TOTAL_STEPS}" ] || [ "${LR_TOTAL_STEPS}" = "-1" ]; then
  LR_TOTAL_STEPS=$(python3 - <<PY
import pathlib, math
p = pathlib.Path("${TRAIN_RAW_DIR}")
n = len(list(p.glob("*.pkl")))
if n > 0:
    steps_per_epoch = math.ceil(n / (${TRAIN_B} * ${NPROC_PER_NODE}))
    print(steps_per_epoch * ${MAX_EPOCHS})
else:
    print(1000)
PY
  )
  echo "[LR schedule] auto LR_TOTAL_STEPS=${LR_TOTAL_STEPS}"
fi

FLOW_SOLVER_METHOD="${FLOW_SOLVER_METHOD:-euler}"
FLOW_SOLVER_STEPS="${FLOW_SOLVER_STEPS:-16}"
ROLLOUT_NOISE_SCALE="${ROLLOUT_NOISE_SCALE:-1.0}"

# ── Ref-NLL 파라미터 ────────────────────────────────────────────────────────
# REF_NLL_N_ROLLOUTS (기본 2)
#   시나리오당 closed-loop rollout 횟수 G.
#   G개의 rollout 에서 수집한 log p 를 평균내어 분산을 줄임.
#   크게 할수록 gradient 추정이 안정적이지만 메모리·시간 비례 증가.
REF_NLL_N_ROLLOUTS="${REF_NLL_N_ROLLOUTS:-4}"

# REF_NLL_PRED_MAX_STEPS (1 step = 0.5s, **must be 4**)
#   closed-loop rollout 길이 (coarse step 수). 새 구현은 0.5s commit × 4 = 2s 궤적을
#   초기 frame 으로 합쳐 open-loop ref (horizon=20 fine step) 의 likelihood 를 잰다.
#   따라서 ``pred_max_steps × shift == 20`` 이어야 하며, 다르면 코드가 ValueError.
REF_NLL_PRED_MAX_STEPS="${REF_NLL_PRED_MAX_STEPS:-4}"

# REF_NLL_N_HUTCH_SAMPLES (기본 1)
#   Hutchinson trace estimator 의 probe 벡터 수.
#   div(v) = E_ε[εᵀ (∂v/∂x) ε] 의 Monte Carlo 추정 횟수.
#   1이면 unbiased 이지만 분산 큼. 2~4 이상이면 분산 감소, 속도는 비례 감소.
#   실용적으로 1로 충분 (noise 는 rollout 다양성으로 상쇄됨).
REF_NLL_N_HUTCH_SAMPLES="${REF_NLL_N_HUTCH_SAMPLES:-4}"

# REF_NLL_USE_FULL_DIV_GRAD (기본 false)
#   false (권장): ∂ log p / ∂ x₁ 계산 시 발산(divergence) 항의 gradient 를 끊음.
#                 prior log p₀(x₀) 항의 gradient 만 사용 → 1차 근사, 빠름.
#   true         : 발산 항까지 2차 미분 유지 (create_graph=True in Hutchinson).
#                 더 정확한 gradient 이지만 메모리·연산 비용이 크게 증가.
REF_NLL_USE_FULL_DIV_GRAD="${REF_NLL_USE_FULL_DIV_GRAD:-true}"

# REF_NLL_FM_REG_LAMBDA (기본 0.0)
#   GT flow-matching regularization 가중치.
#   0 이면 비활성. 양수이면 L_total = L_ref_nll + λ * L_fm 으로 정규화.
#   ref model 에서 너무 멀리 벗어나는 것을 방지하는 anchoring 역할.
REF_NLL_FM_REG_LAMBDA="${REF_NLL_FM_REG_LAMBDA:-0.0}"
REF_NLL_FM_REG_LAMBDA="${REF_NLL_FM_REG_LAMBDA#=}"

# REF_NLL_LOSS_SCALE (기본 1.0)
#   전체 ref-NLL loss 에 곱하는 스케일 인수.
#   gradient 크기를 LR 조정 없이 빠르게 조절할 때 사용.
REF_NLL_LOSS_SCALE="${REF_NLL_LOSS_SCALE:-1.0}"

# ── BPTT tricks ─────────────────────────────────────────────────────────────
# BPTT_USE_ADJOINT (기본 false)
#   true 로 설정하면 ODE solver 의 역전파를 adjoint method 로 수행.
#   일반 BPTT 대비 메모리 절약 (O(1) vs O(steps)), 단 속도는 약간 느림.
#   GPU OOM 이 발생할 때 먼저 시도.
BPTT_USE_ADJOINT="${BPTT_USE_ADJOINT:-true}"

# BPTT_WARM_COARSE_STEPS (기본 0)
#   앞 N 개의 coarse step 을 no_grad + detach 로 처리 (sliding-window BPTT).
#   0이면 전체 rollout 에 gradient 를 흘림.
#   N > 0이면 마지막 (pred_max_steps - N) step 만 BPTT → 메모리 절약.
BPTT_WARM_COARSE_STEPS="${BPTT_WARM_COARSE_STEPS:-0}"

# BPTT_LAST_N_SOLVER_STEPS (기본 2 권장)
#   각 coarse step 내 ODE solver (flow ODE) 의 마지막 N solver step 에만 gradient.
#   0이면 전체 solver step 에 gradient. N > 0이면 메모리 절약.
#   ref-NLL 은 ref-side 그래프가 추가로 살아있어 0 보다 2~4 권장.
BPTT_LAST_N_SOLVER_STEPS="${BPTT_LAST_N_SOLVER_STEPS:-0}"

# BPTT_GRAD_CLIP_TRAJ (기본 1.0)
#   per-step x₁ (y_hat_norm) tensor 의 gradient L2 norm clip 값.
#   0 이면 clip 비활성. 양수이면 각 coarse step x₁ 의 gradient 를 해당 norm 이하로 제한.
#   exploding gradient 방지용.
BPTT_GRAD_CLIP_TRAJ="${BPTT_GRAD_CLIP_TRAJ:-1.0}"

# FLOW_VELOCITY_HEAD_ONLY (기본 true)
#   true : velocity_head 파라미터만 학습 (나머지 flow decoder 는 frozen).
#   false: flow decoder 전체 학습.
#   처음에는 true 로 시작하고, 안정적이면 false 로 전환하는 것을 권장.
FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"

# FLOW_REG_LAMBDA (기본 0.0)
#   GT flow-matching loss 를 추가 정규화로 사용할 때의 가중치.
#   REF_NLL_FM_REG_LAMBDA 와 독립적으로 동작.
FLOW_REG_LAMBDA="${FLOW_REG_LAMBDA:-0.0}"

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
echo "CKPT_PATH=${CKPT_PATH}"
echo "REF_NLL_N_ROLLOUTS=${REF_NLL_N_ROLLOUTS}  REF_NLL_PRED_MAX_STEPS=${REF_NLL_PRED_MAX_STEPS}  REF_NLL_N_HUTCH_SAMPLES=${REF_NLL_N_HUTCH_SAMPLES}"
echo "REF_NLL_USE_FULL_DIV_GRAD=${REF_NLL_USE_FULL_DIV_GRAD}  REF_NLL_FM_REG_LAMBDA=${REF_NLL_FM_REG_LAMBDA}  REF_NLL_LOSS_SCALE=${REF_NLL_LOSS_SCALE}"
echo "BPTT_USE_ADJOINT=${BPTT_USE_ADJOINT}  BPTT_GRAD_CLIP_TRAJ=${BPTT_GRAD_CLIP_TRAJ}  BPTT_WARM_COARSE_STEPS=${BPTT_WARM_COARSE_STEPS}"
echo "FLOW_VELOCITY_HEAD_ONLY=${FLOW_VELOCITY_HEAD_ONLY}  LR=${LR}  FLOW_REG_LAMBDA=${FLOW_REG_LAMBDA}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}  TRAIN_B=${TRAIN_B}  MAX_EPOCHS=${MAX_EPOCHS}"

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
  logger.wandb.entity="${WANDB_ENTITY}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.validation_metric="${VALIDATION_METRIC}" \
  model.model_config.wosac_torch_compile="${WOSAC_TORCH_COMPILE}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.decoder.flow_solver_method="${FLOW_SOLVER_METHOD}" \
  model.model_config.decoder.flow_solver_steps="${FLOW_SOLVER_STEPS}" \
  model.model_config.finetune.rollout_noise_scale="${ROLLOUT_NOISE_SCALE}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.flow_reg_lambda="${FLOW_REG_LAMBDA}" \
  model.model_config.finetune.ref_nll_n_rollouts="${REF_NLL_N_ROLLOUTS}" \
  model.model_config.finetune.ref_nll_pred_max_steps="${REF_NLL_PRED_MAX_STEPS}" \
  model.model_config.finetune.ref_nll_n_hutch_samples="${REF_NLL_N_HUTCH_SAMPLES}" \
  model.model_config.finetune.ref_nll_use_full_div_grad="${REF_NLL_USE_FULL_DIV_GRAD}" \
  model.model_config.finetune.ref_nll_fm_reg_lambda="${REF_NLL_FM_REG_LAMBDA}" \
  model.model_config.finetune.ref_nll_loss_scale="${REF_NLL_LOSS_SCALE}" \
  model.model_config.finetune.bptt_use_adjoint="${BPTT_USE_ADJOINT}" \
  model.model_config.finetune.bptt_warm_coarse_steps="${BPTT_WARM_COARSE_STEPS}" \
  model.model_config.finetune.bptt_last_n_solver_steps="${BPTT_LAST_N_SOLVER_STEPS}" \
  model.model_config.finetune.bptt_grad_clip_traj="${BPTT_GRAD_CLIP_TRAJ}" \
  model.model_config.finetune.gradient_clip_val="${GRAD_CLIP_VAL}" \
  ${EXTRA_ARGS}
