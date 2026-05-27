#!/bin/sh
# ==========================================================================
# Self-Forced N-second Path-Flow Matching — Pareto(RMM × CPD) fine-tuning
# ==========================================================================
#
# 이 스크립트는 `configs/experiment/self_forced_npfm_pareto.yaml` 을 entry로 써서
# closed-loop covariate shift 해소 + RMM(현실성) × CPD(다양성) Pareto fine-tuning
# 을 1대 또는 여러 GPU에서 실행합니다.
#
# ── 알고리즘 한 줄 ────────────────────────────────────────────────────────
# Generator(=encoder)가 closed-loop autoregressive로 만든 path X 를 frozen
# pretrained teacher(R) 분포와 trainable critic(F) 분포 사이의 distribution
# matching loss(DMD or SiD)로 학습합니다. 이때 entropy knob `dmd_beta`로
# realism ↔ diversity 균형을 잡습니다 (β=1.0 vanilla, β<1 diversity↑, β>1
# sharpening).
#
# 핵심 metric:
#   - RMM = val_closed/sim_agents_2025/realism_meta_metric    (↑)
#   - CPD = val_closed/WOSAC-CPD/value                        (↑)
#
# ── 사용 예 ───────────────────────────────────────────────────────────────
#
# 1. β=1.0 vanilla DMD, 1 GPU smoke (cuda:3 단일):
#    CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 \
#      MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=2 LIMIT_VAL_BATCHES=2 \
#      bash scripts/train_self_forced_npfm_pareto.sh
#
# 2. diversity 방향 β=0.75 sweep, 2 GPU(2,3):
#    CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
#      DMD_BETA=0.75 MY_TASK_NAME=pareto_beta0.75 \
#      bash scripts/train_self_forced_npfm_pareto.sh
#
# 3. SiD-lite로 baseline 비교:
#    CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
#      DM_OBJECTIVE=sid SID_ALPHA=1.0 MY_TASK_NAME=pareto_sid \
#      bash scripts/train_self_forced_npfm_pareto.sh
#
# 4. anchor FM regularizer 같이 켜기 (covariate shift는 더 줄지만 다양성↓):
#    CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
#      USE_ANCHOR_FM=true ANCHOR_WEIGHT=0.1 MY_TASK_NAME=pareto_anchor \
#      bash scripts/train_self_forced_npfm_pareto.sh
#
# 5. local_val (=validate only) 한 번 더 돌리기:
#    CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 ACTION=validate \
#      CKPT_PATH=logs/<task>/runs/<ts>/checkpoints/last.ckpt \
#      bash scripts/train_self_forced_npfm_pareto.sh
#
# ── 룰 (CLAUDE.md §6) ─────────────────────────────────────────────────────
# GPU는 2번, 3번만 사용합니다.  default CUDA_VISIBLE_DEVICES=3 (1 GPU smoke).
#
# ==========================================================================

# ────────────────────────────────────────────────────────────────────────
# 0. Shell / 환경
# ────────────────────────────────────────────────────────────────────────
export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
# CLAUDE.md §6: 이 repo의 모든 학습/평가는 GPU 2, 3만 사용합니다.
# 1 GPU smoke 시 single GPU 3을 default로 두고, 2 GPU는 CUDA_VISIBLE_DEVICES=2,3 로 override.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# WandB 모드. offline / disabled 도 가능.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

# ────────────────────────────────────────────────────────────────────────
# 1. Conda env activate
# ────────────────────────────────────────────────────────────────────────
CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi

# ────────────────────────────────────────────────────────────────────────
# 2. Top-level Hydra entry & 경로 / 이름
# ────────────────────────────────────────────────────────────────────────
MY_EXPERIMENT="${MY_EXPERIMENT:-self_forced_npfm_pareto}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-debug}"

# Hydra action: fit / finetune / road_finetune / validate / test
#   - fit       : Lightning 전체 resume (optimizer/scheduler/EMA 포함, self-forced 재개용)
#   - finetune  : weight-only 로딩(strict=False) + 새 optimizer (★ pretrain → fine-tune)
#   - validate  : trainer.validate (RMM + CPD 로컬 계산만)
#   - test      : trainer.test
ACTION="${ACTION:-finetune}"

# Random seed (run.yaml default 817)
SEED="${SEED:-817}"

# WOMD cache root (configs/paths/default.yaml 의 cache_root override).
# 이 repo default 는 /scratch/cache/SMART; 실험 머신마다 override 권장.
CACHE_ROOT="${CACHE_ROOT:-/scratch/cache/SMART}"

# Pretrained backbone checkpoint — fine-tune entry point.
# 기본값은 이 repo 안의 logs/pretrained/pretrained.ckpt (CLAUDE.md §5 참조).
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

# ────────────────────────────────────────────────────────────────────────
# 3. Trainer (Lightning)
# ────────────────────────────────────────────────────────────────────────
# 분산 학습
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"          # 노드당 프로세스 수 (= GPU 수)
NUM_NODES="${NUM_NODES:-1}"
# epoch / step
MAX_EPOCHS="${MAX_EPOCHS:-16}"                 # self_forced_npfm 기본 16
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"   # 1.0=전체, smoke 시 2 등 정수도 가능
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.1}"      # self_forced_npfm 기본 0.1
# Validation 빈도:
#   - step 단위 평가 (잘 되는 세팅 빠르게 탐색용; default 200 step):
#       VAL_CHECK_INTERVAL=<int>  +  CHECK_VAL_EVERY_N_EPOCH=null
#   - epoch 단위 평가 (긴 학습용):
#       VAL_CHECK_INTERVAL=null   +  CHECK_VAL_EVERY_N_EPOCH=<int>
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
# precision: bf16-mixed (default), fp16-mixed, 32-true (재현 디버그용)
PRECISION="${PRECISION:-bf16-mixed}"
# self-forced 는 manual optimization 이라 Lightning 의 trainer.gradient_clip_val 을 쓰지 않음.
# 실제 grad clip 은 model.model_config.self_forced.gradient_clip_val (아래 SF_GRAD_CLIP) 이 담당.
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-null}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-false}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
TRAINER_DETERMINISTIC="${TRAINER_DETERMINISTIC:-false}"

# ────────────────────────────────────────────────────────────────────────
# 4. Data (configs/data/waymo.yaml)
# ────────────────────────────────────────────────────────────────────────
TRAIN_B="${TRAIN_B:-8}"                        # self_forced_npfm 기본 8
VAL_B="${VAL_B:-16}"
TEST_B="${TEST_B:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
# self-forced 학습 시 train 시나리오 중 일부만 epoch 당 사용 (rollout 비용 ↑).
# self_forced_npfm default 0.5; 1.0 으로 두면 모든 시나리오 사용.
TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-0.5}"
TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-true}"

# ────────────────────────────────────────────────────────────────────────
# 5. Generator 학습 (model.model_config)
# ────────────────────────────────────────────────────────────────────────
# self_forced_npfm 기본 lr 2e-4. distillation 단계라 pretrain LR (5e-4) 보다 낮게.
LR="${LR:-2e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-${MAX_EPOCHS}}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1e-2}"

# Validation rollout 비용 제어
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-32}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-10}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-1680}"
SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-0}"

# Visualization (epoch 마다 video upload)
N_VIS_BATCH="${N_VIS_BATCH:-0}"
N_VIS_SCENARIO="${N_VIS_SCENARIO:-0}"
N_VIS_ROLLOUT="${N_VIS_ROLLOUT:-0}"
DELETE_LOCAL_VIDEOS_AFTER_UPLOAD="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD:-true}"

# Open-loop / closed-loop validation toggle
VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-true}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"

# Closed-loop rollout 모드 (decoder.closed_loop_rollout_mode):
#   - raw_fm              : 네트워크 raw FM 출력을 그대로 외부 export (★ 기본)
#   - matched_token_chunk : 외부 export에 retokenize 한 token chunk 반영
CLOSED_LOOP_ROLLOUT_MODE="${CLOSED_LOOP_ROLLOUT_MODE:-raw_fm}"
DECODER_USE_LQR="${DECODER_USE_LQR:-false}"

# ────────────────────────────────────────────────────────────────────────
# 6. Self-Forced (DMD/SiD distribution matching) — 가장 중요 ★
# ────────────────────────────────────────────────────────────────────────
# Generator distribution matching objective:
#   - dmd : Distribution Matching Distillation (default; entropy knob DMD_BETA 사용)
#   - sid : Score Identity Distillation        (sid_alpha 사용; DMD_BETA 무시)
DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"

# ── DMD 전용 knob ─────────────────────────────────────────────────────────
# Entropy knob (build_clean_dmd_direction):
#   - 1.0 (default) : vanilla DMD, 기존 동작 그대로 (R - F)/normalizer
#   - <1.0          : fake(F) 항이 1/β 배 커져 entropy↑/diversity↑
#   - >1.0          : fake 항 작아져 sharpening (realism↑, mode collapse 위험)
DMD_BETA="${DMD_BETA:-1.0}"
# direction normalizer agent-wise 분모 최소값 (수치 안정)
CLEAN_DMD_NORMALIZER_EPS="${CLEAN_DMD_NORMALIZER_EPS:-1.0e-3}"
# teacher/estimator 가 보는 noisy path 의 tau 범위.
# self-forced 학습은 (0.02, 0.98) 안에서 uniform 샘플.
CLEAN_DMD_TAU_LOW="${CLEAN_DMD_TAU_LOW:-0.02}"
CLEAN_DMD_TAU_HIGH="${CLEAN_DMD_TAU_HIGH:-0.98}"

# ── SiD 전용 knob (DM_OBJECTIVE=sid 일 때만 효과) ─────────────────────────
SID_ALPHA="${SID_ALPHA:-1.0}"
SID_NORMALIZER_EPS="${SID_NORMALIZER_EPS:-1.0e-3}"

# ── 공통 self-forced knob ────────────────────────────────────────────────
SF_ENABLED="${SF_ENABLED:-true}"
SF_START_EPOCH="${SF_START_EPOCH:-0}"             # 어느 epoch 부터 self-forced 활성
SF_WEIGHT="${SF_WEIGHT:-1.0}"                     # distribution matching loss weight
SF_PATH_STEP_SIZE="${SF_PATH_STEP_SIZE:-0.05}"    # DMD target = X + step * direction; SiD에는 무관
# anchor (open-loop FM) regularizer — closed-loop matching 만으로 free-form 해지지 않게 잡아줌
USE_ANCHOR_FM="${USE_ANCHOR_FM:-false}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.1}"

# Block transition 단에서 backward graph 끊기 (메모리 절약, gradient flow 줄임)
SF_DETACH_BLOCK_TRANSITION="${SF_DETACH_BLOCK_TRANSITION:-false}"

# critic (generated estimator) 업데이트 cadence — generator 1 step 당 estimator N step.
# reference Self-Forcing 의 dfake_gen_update_ratio 대응.
ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-5}"
# critic 만 학습하는 초기 warmup 구간. step / epoch 두 단위가 OR 로 결합되며,
# 둘 다 0 이면 warmup 없이 바로 generator+critic 동시 학습으로 들어갑니다.
# default 는 self_forced_npfm_pareto.yaml 의 200 step / 0 epoch 입니다
# (잘 되는 세팅 빠른 탐색용; epoch 기반은 1 epoch 가 너무 길 때를 피하기 위해 끔).
ESTIMATOR_WARMUP_STEPS="${ESTIMATOR_WARMUP_STEPS:-200}"
ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"
# fit_start 시점에 teacher/estimator 를 main encoder 의 weight 로 재동기화할지 (fresh finetune ↔ resume).
SF_INIT_AUX_FROM_GEN="${SF_INIT_AUX_FROM_GEN:-true}"

# 학습 가능한 범위.
#   - except_map_encoder    : map encoder 만 freeze (default)
#   - all_unfrozen          : 모두 학습
#   - flow_decoder_only     : flow_decoder 만 학습
#   - full                  : encoder/decoder 모두 학습 (예전 호환)
SF_UNFROZEN_RANGE="${SF_UNFROZEN_RANGE:-except_map_encoder}"

# Generator EMA — Self-Forcing paper 기본 0.99
SF_EMA_WEIGHT="${SF_EMA_WEIGHT:-0.99}"
SF_EMA_START_STEP="${SF_EMA_START_STEP:-50}"

# Manual optimization 안에서 직접 거는 grad clip (Trainer.gradient_clip_val 대신)
SF_GRAD_CLIP="${SF_GRAD_CLIP:-1.0}"

# ── Self-forced rollout sampling (closed-loop denoising step 정책) ───────
# sample_steps  : closed-loop ODE 한 chunk 당 denoising step 수
# sample_method : euler | midpoint
# noise_scale   : init noise 분산 multiplier. >1.0 이면 entropy↑/diversity↑.
SAMPLING_SAMPLE_STEPS="${SAMPLING_SAMPLE_STEPS:-16}"
SAMPLING_SAMPLE_METHOD="${SAMPLING_SAMPLE_METHOD:-euler}"
SAMPLING_NOISE_SCALE="${SAMPLING_NOISE_SCALE:-1.0}"

# Random terminal step (gradient 흘릴 위치 무작위화)
SAMPLING_RTS_ENABLED="${SAMPLING_RTS_ENABLED:-true}"
SAMPLING_RTS_SCOPE="${SAMPLING_RTS_SCOPE:-global_batch}"    # global_batch | per_scenario
# Policy:
#   - paper_uniform : 실행 denoising step K 를 min_executed_steps..sample_steps 범위에서 균등.
#   - all           : 항상 sample_steps 전체 실행, gradient 는 마지막 backprop_last_k 개에만.
SAMPLING_RTS_POLICY="${SAMPLING_RTS_POLICY:-paper_uniform}"
SAMPLING_RTS_MIN_EXECUTED_STEPS="${SAMPLING_RTS_MIN_EXECUTED_STEPS:-16}"
SAMPLING_RTS_BACKPROP_LAST_K="${SAMPLING_RTS_BACKPROP_LAST_K:-8}"  # policy=all 일 때만

# ────────────────────────────────────────────────────────────────────────
# 6.5. Validation rollout sampling — main 학습 시 val closed-loop와 동기화 ★
# ────────────────────────────────────────────────────────────────────────
# smart_flow.yaml 의 model_config.validation_rollout_sampling 은 다음 두 곳에서 함께 쓰입니다:
#   (A) Main flow pretrain / fine-tune yaml (pre_bc_flow_*, finetune_flow_*) 의
#       validation 단계 closed-loop rollout
#   (B) Self-forced fine-tuning (이 launcher) 의 validation 단계 closed-loop rollout
#
# Training-time self-forced rollout sampling (SAMPLING_* 위 섹션) 과 분리되어 있어,
# DMD_BETA / SAMPLING_NOISE_SCALE 같은 sweep knob 을 학습 측에서만 바꾸면 validation 측
# rollout 분포는 default 16/euler/1.0 그대로 유지됩니다. RMM/CPD 측정 일관성을 위해
# 이 hook 의 default 를 학습 측 값과 자동 동기화해 두고, 필요 시 별도 override 가능.
#
# Note: validation rollout 은 random terminal step 이 없습니다 (deterministic 평가).
# 그래서 RTS 관련 키는 여기 없고 sample_steps / sample_method / noise_scale 만 노출합니다.
VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-${SAMPLING_SAMPLE_STEPS}}"
VAL_SAMPLE_METHOD="${VAL_SAMPLE_METHOD:-${SAMPLING_SAMPLE_METHOD}}"
VAL_NOISE_SCALE="${VAL_NOISE_SCALE:-${SAMPLING_NOISE_SCALE}}"

# ────────────────────────────────────────────────────────────────────────
# 7. Pareto(RMM × CPD) 추적
# ────────────────────────────────────────────────────────────────────────
# CPD 정규화 scale.  실험 간 비교 가능성을 위해 같은 값을 유지하는 게 좋습니다.
# 기본값은 smart_flow.yaml 의 training-cache offline 추정값:
#   [vehicle, pedestrian, cyclist] = [22.3461620418, 4.5793447978, 18.5374388830]
WOSAC_DIST_TYPE_SCALE="${WOSAC_DIST_TYPE_SCALE:-[22.3461620418, 4.5793447978, 18.5374388830]}"

# CPD 보존율 (DPR) tracking 기준. baseline (e.g. pretrain) 의 CPD 값을 넣으면
# val_closed/WOSAC-CPD/DPR = (현재 CPD) / (기준 CPD) 가 자동 로그됩니다.
# 비워두면 (null) DPR 로그 생략.
WOSAC_CPD_REFERENCE="${WOSAC_CPD_REFERENCE:-null}"

# ────────────────────────────────────────────────────────────────────────
# 8. Checkpoint / Callbacks
# ────────────────────────────────────────────────────────────────────────
# 어떤 metric 으로 best ckpt 를 고를지.  basis 로 RMM 사용 (default).
# CPD 도 같이 보고 싶으면 별도 sweep 또는 추후 second ModelCheckpoint 추가.
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"

# ────────────────────────────────────────────────────────────────────────
# 9. Logger (WandB)
# ────────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
# 개인 계정으로 보내려면 WANDB_ENTITY 를 override.
WANDB_TAGS="${WANDB_TAGS:-[]}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-all}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

# ────────────────────────────────────────────────────────────────────────
# 10. Hydra CLI extras
# ────────────────────────────────────────────────────────────────────────
# 추가 override 가 필요하면 EXTRA_ARGS 에 공백 분리로 넣으면 그대로 붙습니다.
# 예: EXTRA_ARGS="model.model_config.decoder.flow_solver_steps=32"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ────────────────────────────────────────────────────────────────────────
# 11. Sanity check
# ────────────────────────────────────────────────────────────────────────
if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  echo "        Set CKPT_PATH=<path-to-pretrained.ckpt> or use ACTION=fit (resume)."
  exit 1
fi

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}
PORT="$(get_free_port)"

# ────────────────────────────────────────────────────────────────────────
# 12. Banner
# ────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[self_forced_npfm_pareto] launching ..."
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NPROC=${NPROC_PER_NODE}  NUM_NODES=${NUM_NODES}  ACTION=${ACTION}"
echo "  EXPERIMENT=${MY_EXPERIMENT}  TASK=${MY_TASK_NAME}"
echo "  CKPT_PATH=${CKPT_PATH}"
echo "  CACHE_ROOT=${CACHE_ROOT}"
echo "------------------------------------------------------------"
echo "  Trainer: max_epochs=${MAX_EPOCHS} precision=${PRECISION}"
echo "    limit_train=${LIMIT_TRAIN_BATCHES} limit_val=${LIMIT_VAL_BATCHES}"
echo "    val_check_interval=${VAL_CHECK_INTERVAL} check_val_every_n=${CHECK_VAL_EVERY_N_EPOCH} seed=${SEED}"
echo "  Data: train_B=${TRAIN_B} val_B=${VAL_B} workers=${NUM_WORKERS}"
echo "    epoch_sample_frac=${TRAIN_EPOCH_SAMPLE_FRACTION}"
echo "  Generator: lr=${LR} warmup=${LR_WARMUP_STEPS} total=${LR_TOTAL_STEPS}"
echo "    closed_loop_rollout_mode=${CLOSED_LOOP_ROLLOUT_MODE} use_lqr=${DECODER_USE_LQR}"
echo "  Self-Forced ★:"
echo "    objective=${DM_OBJECTIVE} dmd_beta=${DMD_BETA} sid_alpha=${SID_ALPHA}"
echo "    weight=${SF_WEIGHT} path_step=${SF_PATH_STEP_SIZE}"
echo "    use_anchor_fm=${USE_ANCHOR_FM} anchor_weight=${ANCHOR_WEIGHT}"
echo "    estimator_per_step=${ESTIMATOR_UPDATES_PER_STEP}"
echo "    estimator_warmup_steps=${ESTIMATOR_WARMUP_STEPS} estimator_warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}"
echo "    unfrozen_range=${SF_UNFROZEN_RANGE}"
echo "    ema_weight=${SF_EMA_WEIGHT} ema_start=${SF_EMA_START_STEP}"
echo "    grad_clip=${SF_GRAD_CLIP}"
echo "  Sampling (train self-forced rollout):"
echo "    sample_steps=${SAMPLING_SAMPLE_STEPS} method=${SAMPLING_SAMPLE_METHOD}"
echo "    noise_scale=${SAMPLING_NOISE_SCALE}"
echo "    rts: policy=${SAMPLING_RTS_POLICY} min_executed=${SAMPLING_RTS_MIN_EXECUTED_STEPS}"
echo "         backprop_last_k=${SAMPLING_RTS_BACKPROP_LAST_K} scope=${SAMPLING_RTS_SCOPE}"
echo "  Sampling (val closed-loop; main pretrain val 과 동기화):"
echo "    sample_steps=${VAL_SAMPLE_STEPS} method=${VAL_SAMPLE_METHOD} noise_scale=${VAL_NOISE_SCALE}"
echo "  Pareto:"
echo "    wosac_distribution_type_scale=${WOSAC_DIST_TYPE_SCALE}"
echo "    wosac_cpd_reference=${WOSAC_CPD_REFERENCE}"
echo "  Checkpoint: monitor=${CHECKPOINT_MONITOR} mode=${CHECKPOINT_MODE}"
echo "  WandB: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} offline=${WANDB_OFFLINE}"
echo "  Threads: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS}"
echo "============================================================"

# ────────────────────────────────────────────────────────────────────────
# 13. Launch
# ────────────────────────────────────────────────────────────────────────
PREFETCH_ARG=""
if [ "${NUM_WORKERS}" -gt 0 ]; then
  PREFETCH_ARG="data.prefetch_factor=${PREFETCH_FACTOR}"
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NUM_NODES}" \
  --master_port="${PORT}" \
  --rdzv_endpoint="127.0.0.1:${PORT}" \
  -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${NPROC_PER_NODE}" \
  trainer.num_nodes="${NUM_NODES}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.precision="${PRECISION}" \
  trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}" \
  trainer.sync_batchnorm="${SYNC_BATCHNORM}" \
  trainer.num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.deterministic="${TRAINER_DETERMINISTIC}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.test_batch_size="${TEST_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_use_eval_agent_selection="${TRAIN_USE_EVAL_AGENT_SELECTION}" \
  data.train_epoch_sample_fraction="${TRAIN_EPOCH_SAMPLE_FRACTION}" \
  callbacks.model_checkpoint.monitor="${CHECKPOINT_MONITOR}" \
  callbacks.model_checkpoint.mode="${CHECKPOINT_MODE}" \
  callbacks.model_checkpoint.save_top_k="${CHECKPOINT_SAVE_TOP_K}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.tags="${WANDB_TAGS}" \
  logger.wandb.log_model="${WANDB_LOG_MODEL}" \
  logger.wandb.offline="${WANDB_OFFLINE}" \
  model.model_config.lr="${LR}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}" \
  model.model_config.sim_agents_metric_workers="${SIM_AGENTS_METRIC_WORKERS}" \
  model.model_config.val_open_loop="${VAL_OPEN_LOOP}" \
  model.model_config.val_closed_loop="${VAL_CLOSED_LOOP}" \
  model.model_config.n_vis_batch="${N_VIS_BATCH}" \
  model.model_config.n_vis_scenario="${N_VIS_SCENARIO}" \
  model.model_config.n_vis_rollout="${N_VIS_ROLLOUT}" \
  model.model_config.delete_local_videos_after_wandb_upload="${DELETE_LOCAL_VIDEOS_AFTER_UPLOAD}" \
  model.model_config.decoder.closed_loop_rollout_mode="${CLOSED_LOOP_ROLLOUT_MODE}" \
  model.model_config.decoder.use_lqr="${DECODER_USE_LQR}" \
  model.model_config.wosac_distribution_type_scale="${WOSAC_DIST_TYPE_SCALE}" \
  model.model_config.wosac_cpd_reference="${WOSAC_CPD_REFERENCE}" \
  model.model_config.self_forced.enabled="${SF_ENABLED}" \
  model.model_config.self_forced.start_epoch="${SF_START_EPOCH}" \
  model.model_config.self_forced.weight="${SF_WEIGHT}" \
  model.model_config.self_forced.path_step_size="${SF_PATH_STEP_SIZE}" \
  model.model_config.self_forced.anchor_weight="${ANCHOR_WEIGHT}" \
  model.model_config.self_forced.use_anchor_flow_matching_loss="${USE_ANCHOR_FM}" \
  model.model_config.self_forced.distribution_matching_objective="${DM_OBJECTIVE}" \
  model.model_config.self_forced.dmd_beta="${DMD_BETA}" \
  model.model_config.self_forced.clean_dmd_normalizer_eps="${CLEAN_DMD_NORMALIZER_EPS}" \
  model.model_config.self_forced.clean_dmd_tau_low="${CLEAN_DMD_TAU_LOW}" \
  model.model_config.self_forced.clean_dmd_tau_high="${CLEAN_DMD_TAU_HIGH}" \
  model.model_config.self_forced.sid_alpha="${SID_ALPHA}" \
  model.model_config.self_forced.sid_normalizer_eps="${SID_NORMALIZER_EPS}" \
  model.model_config.self_forced.detach_block_transition="${SF_DETACH_BLOCK_TRANSITION}" \
  model.model_config.self_forced.estimator_updates_per_step="${ESTIMATOR_UPDATES_PER_STEP}" \
  model.model_config.self_forced.estimator_warmup_steps="${ESTIMATOR_WARMUP_STEPS}" \
  model.model_config.self_forced.estimator_warmup_epochs="${ESTIMATOR_WARMUP_EPOCHS}" \
  model.model_config.self_forced.initialize_aux_from_generator_on_fit_start="${SF_INIT_AUX_FROM_GEN}" \
  model.model_config.self_forced.unfrozen_range="${SF_UNFROZEN_RANGE}" \
  model.model_config.self_forced.ema_weight="${SF_EMA_WEIGHT}" \
  model.model_config.self_forced.ema_start_step="${SF_EMA_START_STEP}" \
  model.model_config.self_forced.gradient_clip_val="${SF_GRAD_CLIP}" \
  model.model_config.self_forced.sampling.sample_steps="${SAMPLING_SAMPLE_STEPS}" \
  model.model_config.self_forced.sampling.sample_method="${SAMPLING_SAMPLE_METHOD}" \
  model.model_config.self_forced.sampling.noise_scale="${SAMPLING_NOISE_SCALE}" \
  model.model_config.self_forced.sampling.random_terminal_step.enabled="${SAMPLING_RTS_ENABLED}" \
  model.model_config.self_forced.sampling.random_terminal_step.scope="${SAMPLING_RTS_SCOPE}" \
  model.model_config.self_forced.sampling.random_terminal_step.policy="${SAMPLING_RTS_POLICY}" \
  model.model_config.self_forced.sampling.random_terminal_step.min_executed_steps="${SAMPLING_RTS_MIN_EXECUTED_STEPS}" \
  model.model_config.self_forced.sampling.backprop_last_k="${SAMPLING_RTS_BACKPROP_LAST_K}" \
  model.model_config.validation_rollout_sampling.sample_steps="${VAL_SAMPLE_STEPS}" \
  model.model_config.validation_rollout_sampling.sample_method="${VAL_SAMPLE_METHOD}" \
  model.model_config.validation_rollout_sampling.noise_scale="${VAL_NOISE_SCALE}" \
  ${PREFETCH_ARG} \
  ${EXTRA_ARGS}

echo "bash $(basename "$0") done!"
