#!/bin/sh
# OCSC GT-target L2 fine-tuning -- MAIN experiment on real train/val data.
#
# Common overrides:
#   GPU=2,3 LR=5e-8 TRAIN_B=8 VAL_B=16 NPROC_PER_NODE=2 bash scripts/_ocsc_main.sh
#   SCOPE=velocity_head_only MATCH_FRAME=local GT_TARGET=false OCSC_N_OL_ROLLOUTS=8 bash scripts/_ocsc_main.sh
set -e

is_true() {
  case "$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

count_cuda_devices() {
  if [ -z "$1" ]; then
    printf "1"
    return
  fi
  printf "%s" "$1" | awk -F',' '{print NF}'
}

get_free_port() {
  python - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

# Runtime environment.
export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  . "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CATK_CONDA_ENV}" || true
fi
cd "${REPO_ROOT}"

# Paths / Hydra entry.
MY_EXPERIMENT="${MY_EXPERIMENT:-ocsc_ft}"
ACTION="${ACTION:-finetune}"
SEED="${SEED:-817}"
CACHE_ROOT="${CACHE_ROOT:-${R:-/home2/pnc2/repos_python/datasets/catk_cache}}"
CKPT_PATH="${CKPT_PATH:-${CKPT:-logs/pretrained/pretrained.ckpt}}"
TRAIN_RAW_DIR="${TRAIN_RAW_DIR:-}"
VAL_RAW_DIR="${VAL_RAW_DIR:-}"
VAL_TFRECORDS_SPLITTED="${VAL_TFRECORDS_SPLITTED:-}"

# GPU / distributed setup.  GPU is kept as a short alias for CUDA_VISIBLE_DEVICES.
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-2, 3}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_cuda_devices "${CUDA_VISIBLE_DEVICES}")}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${NPROC_PER_NODE}}"
if [ "${NPROC_PER_NODE}" -gt 1 ] || [ "${NUM_NODES}" -gt 1 ]; then
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
else
  TRAINER_STRATEGY="${TRAINER_STRATEGY:-auto}"
fi

# Trainer knobs.
MAX_EPOCHS="${MAX_EPOCHS:-16}"
PRECISION="${PRECISION:-32-true}"
ACCELERATOR="${ACCELERATOR:-gpu}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-${LIMIT_TRAIN:-1.0}}"
# null(=full) 이면 val 루프가 전체 ~2757 batch 를 돌아 closed-loop validation 이
# 수십 시간 걸린다.  유한 int 를 주면 모델의 _ensure_validation_limit_reaches_scorer_batches
# 가 이를 scorer 가 요구하는 batch 수(=ceil(scorer_scene_num/global_val_B), 440→28)로
# 자동 상향해 딱 scorer scope 만큼만 평가한다.  더 많은 open-loop scene 을 원하면 키우면 됨.
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-${LIMIT_VAL:-1}}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-${VAL_CHECK:-200}}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-null}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-false}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
DETERMINISTIC="${DETERMINISTIC:-false}"

# Data knobs.  BATCH is the old alias; TRAIN_B/VAL_B/TEST_B are preferred.
BATCH="${BATCH:-16}"
TRAIN_B="${TRAIN_B:-${BATCH}}"
VAL_B="${VAL_B:-${BATCH}}"
TEST_B="${TEST_B:-${VAL_B}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-${NUM_WORKERS}}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
EVAL_PERSISTENT_WORKERS="${EVAL_PERSISTENT_WORKERS:-true}"
EVAL_PIN_MEMORY="${EVAL_PIN_MEMORY:-${PIN_MEMORY}}"
EVAL_MULTIPROCESSING_CONTEXT="${EVAL_MULTIPROCESSING_CONTEXT:-spawn}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"   # 정합성 고정: run 간 train 순서까지 동일하게 (필요시 DATA_SHUFFLE=true)
TRAIN_EPOCH_SAMPLE_FRACTION="${TRAIN_EPOCH_SAMPLE_FRACTION:-1.0}"
TRAIN_USE_EVAL_AGENT_SELECTION="${TRAIN_USE_EVAL_AGENT_SELECTION:-true}"
TRAIN_MEMORY_BALANCED_BATCHES="${TRAIN_MEMORY_BALANCED_BATCHES:-false}"
TRAIN_MEMORY_BALANCE_METADATA_CACHE="${TRAIN_MEMORY_BALANCE_METADATA_CACHE:-null}"
TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS="${TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS:-8}"
TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING="${TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING:-true}"
TRAIN_MEMORY_BALANCE_AGENT_WEIGHT="${TRAIN_MEMORY_BALANCE_AGENT_WEIGHT:-1.0}"
TRAIN_MEMORY_BALANCE_CURRENT_VALID_AGENT_WEIGHT="${TRAIN_MEMORY_BALANCE_CURRENT_VALID_AGENT_WEIGHT:-1.0}"
TRAIN_MEMORY_BALANCE_VALID_AGENT_STEP_WEIGHT="${TRAIN_MEMORY_BALANCE_VALID_AGENT_STEP_WEIGHT:-0.0}"
TRAIN_MEMORY_BALANCE_MAP_WEIGHT="${TRAIN_MEMORY_BALANCE_MAP_WEIGHT:-0.02}"
TRAIN_MEMORY_BALANCE_SEED="${TRAIN_MEMORY_BALANCE_SEED:-0}"

# Optimizer / scheduler.
LR="${LR:-1e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-${MAX_EPOCHS}}"
LR_MIN_RATIO="${LR_MIN_RATIO:-1.0}"

# Validation / sampling.
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-440}"
N_BATCH_SIM_AGENTS_METRIC="${N_BATCH_SIM_AGENTS_METRIC:-${N_BATCH_METRIC:-100000}}"
SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-8}"
VAL_OPEN_LOOP="${VAL_OPEN_LOOP:-true}"
VAL_CLOSED_LOOP="${VAL_CLOSED_LOOP:-true}"
VALIDATION_OPEN_SEED="${VALIDATION_OPEN_SEED:-0}"
VALIDATION_CLOSED_SEED="${VALIDATION_CLOSED_SEED:-0}"
VALIDATION_FIXED_FLOW_NOISE="${VALIDATION_FIXED_FLOW_NOISE:-true}"
VALIDATION_SAMPLE_STEPS="${VALIDATION_SAMPLE_STEPS:-16}"
VALIDATION_SAMPLE_METHOD="${VALIDATION_SAMPLE_METHOD:-euler}"
VALIDATION_NOISE_SCALE="${VALIDATION_NOISE_SCALE:-1.0}"
OCSC_TRAIN_SAMPLE_STEPS="${OCSC_TRAIN_SAMPLE_STEPS:-${TRAIN_SAMPLE_STEPS:-${VALIDATION_SAMPLE_STEPS}}}"
OCSC_TRAIN_SAMPLE_METHOD="${OCSC_TRAIN_SAMPLE_METHOD:-${TRAIN_SAMPLE_METHOD:-${VALIDATION_SAMPLE_METHOD}}}"
OCSC_TRAIN_NOISE_SCALE="${OCSC_TRAIN_NOISE_SCALE:-${TRAIN_NOISE_SCALE:-${VALIDATION_NOISE_SCALE}}}"

# Flow/control sync knobs.  Defaults mirror configs/model/smart_flow.yaml.
USE_KINEMATIC_CONTROL_FLOW="${USE_KINEMATIC_CONTROL_FLOW:-true}"
USE_HOLONOMIC_MODEL_ONLY="${USE_HOLONOMIC_MODEL_ONLY:-false}"
USE_ROLLING_SUPERVISION="${USE_ROLLING_SUPERVISION:-true}"
CONTROL_POS_SCALE_M="${CONTROL_POS_SCALE_M:-1.0}"
CONTROL_VEHICLE_NO_SLIP_POINT_RATIO="${CONTROL_VEHICLE_NO_SLIP_POINT_RATIO:-0.2289518863}"
CONTROL_CYCLIST_NO_SLIP_POINT_RATIO="${CONTROL_CYCLIST_NO_SLIP_POINT_RATIO:-0.0495847873}"
CONTROL_VEHICLE_YAW_SCALE_RAD="${CONTROL_VEHICLE_YAW_SCALE_RAD:-0.025}"
CONTROL_PEDESTRIAN_YAW_SCALE_RAD="${CONTROL_PEDESTRIAN_YAW_SCALE_RAD:-0.20}"
CONTROL_CYCLIST_YAW_SCALE_RAD="${CONTROL_CYCLIST_YAW_SCALE_RAD:-0.06}"
CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M="${CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M:-0.5}"

# OCSC core knobs.  Old aliases G/MATCH_FRAME/GT_TARGET/STRIDE/POS_W/HEAD_W remain valid.
OCSC_N_ROLLOUTS="${OCSC_N_ROLLOUTS:-${G:-4}}"
OCSC_N_OL_ROLLOUTS="${OCSC_N_OL_ROLLOUTS:--1}"
OCSC_OL_NEAREST_MATCH="${OCSC_OL_NEAREST_MATCH:-true}"
OCSC_LOSS_TYPE="${OCSC_LOSS_TYPE:-l2}"
OCSC_GT_TARGET="${OCSC_GT_TARGET:-${GT_TARGET:-true}}"
OCSC_USE_PRETRAINED_REF="${OCSC_USE_PRETRAINED_REF:-true}"
OCSC_ANCHOR_IDX="${OCSC_ANCHOR_IDX:-0}"
OCSC_MATCH_SPACE="${OCSC_MATCH_SPACE:-pose}"
OCSC_MATCH_FRAME="${OCSC_MATCH_FRAME:-${MATCH_FRAME:-global}}"
OCSC_LOSS_WINDOW_STEPS="${OCSC_LOSS_WINDOW_STEPS:--1}"
OCSC_LOSS_TEMPORAL_STRIDE="${OCSC_LOSS_TEMPORAL_STRIDE:-${STRIDE:-1}}"
OCSC_STRICT_ACTIVE_MASK="${OCSC_STRICT_ACTIVE_MASK:-true}"
OCSC_POSITION_WEIGHT="${OCSC_POSITION_WEIGHT:-${POS_W:-1.0}}"
OCSC_HEADING_WEIGHT="${OCSC_HEADING_WEIGHT:-${HEAD_W:-0.01}}"
OCSC_GRADIENT_CLIP_VAL="${OCSC_GRADIENT_CLIP_VAL:-0.0}"
OCSC_USE_MMD="${OCSC_USE_MMD:-false}"
OCSC_PWIL_COUPLING="${OCSC_PWIL_COUPLING:-hungarian}"
OCSC_PWIL_USE_EXP_REWARD="${OCSC_PWIL_USE_EXP_REWARD:-true}"
OCSC_PWIL_ALPHA="${OCSC_PWIL_ALPHA:-1.0}"
OCSC_PWIL_BETA="${OCSC_PWIL_BETA:-5.0}"
OCSC_TARGET_MAX_STEPS="${OCSC_TARGET_MAX_STEPS:-4}"
OCSC_PRED_MAX_STEPS="${OCSC_PRED_MAX_STEPS:-4}"
OCSC_REL_DISP_WEIGHT="${OCSC_REL_DISP_WEIGHT:-0.0}"
OCSC_EVAL_HARD_RMM="${OCSC_EVAL_HARD_RMM:-false}"
OCSC_EVAL_HARD_RMM_INTERVAL="${OCSC_EVAL_HARD_RMM_INTERVAL:-1}"
OCSC_FM_REG_LAMBDA="${OCSC_FM_REG_LAMBDA:-0.0}"
OCSC_GT_RESOLUTION="${OCSC_GT_RESOLUTION:-2hz}"
OCSC_OL_RESOLUTION="${OCSC_OL_RESOLUTION:-10hz}"
OCSC_NEAREST_INCLUDE_GT="${OCSC_NEAREST_INCLUDE_GT:-false}"
OCSC_REF_REFRESH_MODE="${OCSC_REF_REFRESH_MODE:-frozen}"
OCSC_REF_REFRESH_INTERVAL="${OCSC_REF_REFRESH_INTERVAL:-0}"
OCSC_REF_EMA_DECAY="${OCSC_REF_EMA_DECAY:-0.999}"

# Trainable range.  SCOPE preserves the old three-value interface and also sets flow_ft_target.
SCOPE="${SCOPE:-except_map_encoder}"
case "${SCOPE}" in
  velocity_head_only|velocity_head)
    OCSC_EXCEPT_MAP_ENCODER="${OCSC_EXCEPT_MAP_ENCODER:-false}"
    OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-true}"
    OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-false}"
    FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-true}"
    FLOW_FT_TARGET="${FLOW_FT_TARGET:-velocity_head}"
    ;;
  except_map_encoder)
    OCSC_EXCEPT_MAP_ENCODER="${OCSC_EXCEPT_MAP_ENCODER:-true}"
    OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-false}"
    OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-false}"
    FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"
    FLOW_FT_TARGET="${FLOW_FT_TARGET:-except_map_encoder}"
    ;;
  full_flow_decoder|full)
    OCSC_EXCEPT_MAP_ENCODER="${OCSC_EXCEPT_MAP_ENCODER:-false}"
    OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-false}"
    OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-true}"
    FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"
    FLOW_FT_TARGET="${FLOW_FT_TARGET:-full}"
    ;;
  step_refiner_and_velocity_head|chunk_mixers_and_velocity_head)
    OCSC_EXCEPT_MAP_ENCODER="${OCSC_EXCEPT_MAP_ENCODER:-false}"
    OCSC_VELOCITY_HEAD_ONLY="${OCSC_VELOCITY_HEAD_ONLY:-false}"
    OCSC_FULL_FLOW_DECODER="${OCSC_FULL_FLOW_DECODER:-false}"
    FLOW_VELOCITY_HEAD_ONLY="${FLOW_VELOCITY_HEAD_ONLY:-false}"
    FLOW_FT_TARGET="${FLOW_FT_TARGET:-${SCOPE}}"
    ;;
  *)
    echo "[ERROR] unknown SCOPE=${SCOPE}"
    exit 1
    ;;
esac

# Checkpoint / W&B / optional early floor.
ENABLE_MODEL_CHECKPOINT="${ENABLE_MODEL_CHECKPOINT:-false}"
ENABLE_EPOCH_LAST_CHECKPOINT="${ENABLE_EPOCH_LAST_CHECKPOINT:-false}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
CHECKPOINT_SAVE_LAST="${CHECKPOINT_SAVE_LAST:-true}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"
RMM_FLOOR_ENABLED="${RMM_FLOOR_ENABLED:-false}"
RMM_FLOOR="${RMM_FLOOR:-0.700}"
RMM_FLOOR_MONITOR="${RMM_FLOOR_MONITOR:-val_closed/sim_agents_2025/realism_meta_metric}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-ocsc_main}"
WANDB_TAGS="${WANDB_TAGS:-[ocsc_main]}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

TS="$(date +%m%d_%H%M%S)"
TASK_DEFAULT="ocsc_main_lr${LR}_G${OCSC_N_ROLLOUTS}_${OCSC_MATCH_FRAME}_gt${OCSC_GT_TARGET}_${SCOPE}_b${TRAIN_B}x${NPROC_PER_NODE}_st${OCSC_LOSS_TEMPORAL_STRIDE}_${TS}"
MY_TASK_NAME="${MY_TASK_NAME:-${TASK:-${TASK_DEFAULT}}}"
TASK="${MY_TASK_NAME}"
LOG="${LOG:-artifacts/${TASK}.log}"
LOG_TO_FILE="${LOG_TO_FILE:-true}"
DRY_RUN="${DRY_RUN:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$(dirname "${LOG}")"

if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

set -- \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${TASK}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.accelerator="${ACCELERATOR}" \
  trainer.devices="${TRAINER_DEVICES}" \
  trainer.num_nodes="${NUM_NODES}" \
  trainer.precision="${PRECISION}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}" \
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  trainer.sync_batchnorm="${SYNC_BATCHNORM}" \
  trainer.num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.deterministic="${DETERMINISTIC}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.test_batch_size="${TEST_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.prefetch_factor="${PREFETCH_FACTOR}" \
  data.persistent_workers="${PERSISTENT_WORKERS}" \
  data.pin_memory="${PIN_MEMORY}" \
  data.eval_num_workers="${EVAL_NUM_WORKERS}" \
  data.eval_prefetch_factor="${EVAL_PREFETCH_FACTOR}" \
  data.eval_persistent_workers="${EVAL_PERSISTENT_WORKERS}" \
  data.eval_pin_memory="${EVAL_PIN_MEMORY}" \
  data.eval_multiprocessing_context="${EVAL_MULTIPROCESSING_CONTEXT}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.train_epoch_sample_fraction="${TRAIN_EPOCH_SAMPLE_FRACTION}" \
  data.train_use_eval_agent_selection="${TRAIN_USE_EVAL_AGENT_SELECTION}" \
  data.train_memory_balanced_batches="${TRAIN_MEMORY_BALANCED_BATCHES}" \
  data.train_memory_balance_metadata_cache="${TRAIN_MEMORY_BALANCE_METADATA_CACHE}" \
  data.train_memory_balance_metadata_num_workers="${TRAIN_MEMORY_BALANCE_METADATA_NUM_WORKERS}" \
  data.train_memory_balance_build_on_missing="${TRAIN_MEMORY_BALANCE_BUILD_ON_MISSING}" \
  data.train_memory_balance_agent_weight="${TRAIN_MEMORY_BALANCE_AGENT_WEIGHT}" \
  data.train_memory_balance_current_valid_agent_weight="${TRAIN_MEMORY_BALANCE_CURRENT_VALID_AGENT_WEIGHT}" \
  data.train_memory_balance_valid_agent_step_weight="${TRAIN_MEMORY_BALANCE_VALID_AGENT_STEP_WEIGHT}" \
  data.train_memory_balance_map_weight="${TRAIN_MEMORY_BALANCE_MAP_WEIGHT}" \
  data.train_memory_balance_seed="${TRAIN_MEMORY_BALANCE_SEED}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.job_type="${WANDB_JOB_TYPE}" \
  logger.wandb.tags="${WANDB_TAGS}" \
  logger.wandb.log_model="${WANDB_LOG_MODEL}" \
  logger.wandb.offline="${WANDB_OFFLINE}" \
  model.model_config.lr="${LR}" \
  model.model_config.weight_decay="${WEIGHT_DECAY}" \
  model.model_config.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  model.model_config.lr_total_steps="${LR_TOTAL_STEPS}" \
  model.model_config.lr_min_ratio="${LR_MIN_RATIO}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}" \
  model.model_config.n_batch_sim_agents_metric="${N_BATCH_SIM_AGENTS_METRIC}" \
  model.model_config.sim_agents_metric_workers="${SIM_AGENTS_METRIC_WORKERS}" \
  model.model_config.val_open_loop="${VAL_OPEN_LOOP}" \
  model.model_config.val_closed_loop="${VAL_CLOSED_LOOP}" \
  model.model_config.validation_open_seed="${VALIDATION_OPEN_SEED}" \
  model.model_config.validation_closed_seed="${VALIDATION_CLOSED_SEED}" \
  model.model_config.validation_fixed_flow_noise="${VALIDATION_FIXED_FLOW_NOISE}" \
  model.model_config.validation_rollout_sampling.sample_steps="${VALIDATION_SAMPLE_STEPS}" \
  model.model_config.validation_rollout_sampling.sample_method="${VALIDATION_SAMPLE_METHOD}" \
  model.model_config.validation_rollout_sampling.noise_scale="${VALIDATION_NOISE_SCALE}" \
  ++model.model_config.finetune.ocsc_train_rollout_sampling.sample_steps="${OCSC_TRAIN_SAMPLE_STEPS}" \
  ++model.model_config.finetune.ocsc_train_rollout_sampling.sample_method="${OCSC_TRAIN_SAMPLE_METHOD}" \
  ++model.model_config.finetune.ocsc_train_rollout_sampling.noise_scale="${OCSC_TRAIN_NOISE_SCALE}" \
  model.model_config.token_processor.use_kinematic_control_flow="${USE_KINEMATIC_CONTROL_FLOW}" \
  model.model_config.token_processor.use_holonomic_model_only="${USE_HOLONOMIC_MODEL_ONLY}" \
  model.model_config.token_processor.use_rolling_supervision="${USE_ROLLING_SUPERVISION}" \
  model.model_config.token_processor.control_pos_scale_m="${CONTROL_POS_SCALE_M}" \
  model.model_config.token_processor.control_vehicle_no_slip_point_ratio="${CONTROL_VEHICLE_NO_SLIP_POINT_RATIO}" \
  model.model_config.token_processor.control_cyclist_no_slip_point_ratio="${CONTROL_CYCLIST_NO_SLIP_POINT_RATIO}" \
  model.model_config.token_processor.control_vehicle_yaw_scale_rad="${CONTROL_VEHICLE_YAW_SCALE_RAD}" \
  model.model_config.token_processor.control_pedestrian_yaw_scale_rad="${CONTROL_PEDESTRIAN_YAW_SCALE_RAD}" \
  model.model_config.token_processor.control_cyclist_yaw_scale_rad="${CONTROL_CYCLIST_YAW_SCALE_RAD}" \
  model.model_config.token_processor.control_round_trip_max_position_error_m="${CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M}" \
  model.model_config.finetune.enabled=true \
  model.model_config.finetune.mode=ocsc_ft \
  model.model_config.finetune.train_except_map_encoder="${OCSC_EXCEPT_MAP_ENCODER}" \
  model.model_config.finetune.velocity_head_only="${OCSC_VELOCITY_HEAD_ONLY}" \
  model.model_config.finetune.train_full_flow_decoder_only="${OCSC_FULL_FLOW_DECODER}" \
  model.model_config.finetune.flow_velocity_head_only="${FLOW_VELOCITY_HEAD_ONLY}" \
  ++model.model_config.finetune.flow_ft_target="${FLOW_FT_TARGET}" \
  ++model.model_config.finetune.gradient_clip_val="${OCSC_GRADIENT_CLIP_VAL}" \
  model.model_config.finetune.ocsc_n_rollouts="${OCSC_N_ROLLOUTS}" \
  model.model_config.finetune.ocsc_n_ol_rollouts="${OCSC_N_OL_ROLLOUTS}" \
  model.model_config.finetune.ocsc_ol_nearest_match="${OCSC_OL_NEAREST_MATCH}" \
  model.model_config.finetune.ocsc_loss_type="${OCSC_LOSS_TYPE}" \
  ++model.model_config.finetune.ocsc_use_mmd="${OCSC_USE_MMD}" \
  ++model.model_config.finetune.ocsc_pwil_coupling="${OCSC_PWIL_COUPLING}" \
  ++model.model_config.finetune.ocsc_pwil_use_exp_reward="${OCSC_PWIL_USE_EXP_REWARD}" \
  ++model.model_config.finetune.ocsc_pwil_alpha="${OCSC_PWIL_ALPHA}" \
  ++model.model_config.finetune.ocsc_pwil_beta="${OCSC_PWIL_BETA}" \
  model.model_config.finetune.ocsc_gt_target="${OCSC_GT_TARGET}" \
  model.model_config.finetune.ocsc_use_pretrained_ref="${OCSC_USE_PRETRAINED_REF}" \
  model.model_config.finetune.ocsc_anchor_idx="${OCSC_ANCHOR_IDX}" \
  model.model_config.finetune.ocsc_match_space="${OCSC_MATCH_SPACE}" \
  model.model_config.finetune.ocsc_match_frame="${OCSC_MATCH_FRAME}" \
  model.model_config.finetune.ocsc_loss_window_steps="${OCSC_LOSS_WINDOW_STEPS}" \
  model.model_config.finetune.ocsc_loss_temporal_stride="${OCSC_LOSS_TEMPORAL_STRIDE}" \
  model.model_config.finetune.ocsc_strict_active_mask="${OCSC_STRICT_ACTIVE_MASK}" \
  model.model_config.finetune.ocsc_position_weight="${OCSC_POSITION_WEIGHT}" \
  model.model_config.finetune.ocsc_heading_weight="${OCSC_HEADING_WEIGHT}" \
  ++model.model_config.finetune.ocsc_target_max_steps="${OCSC_TARGET_MAX_STEPS}" \
  ++model.model_config.finetune.ocsc_pred_max_steps="${OCSC_PRED_MAX_STEPS}" \
  ++model.model_config.finetune.ocsc_rel_disp_weight="${OCSC_REL_DISP_WEIGHT}" \
  ++model.model_config.finetune.ocsc_eval_hard_rmm="${OCSC_EVAL_HARD_RMM}" \
  ++model.model_config.finetune.ocsc_eval_hard_rmm_interval="${OCSC_EVAL_HARD_RMM_INTERVAL}" \
  ++model.model_config.finetune.ocsc_fm_reg_lambda="${OCSC_FM_REG_LAMBDA}" \
  ++model.model_config.finetune.ocsc_gt_resolution="${OCSC_GT_RESOLUTION}" \
  ++model.model_config.finetune.ocsc_ol_resolution="${OCSC_OL_RESOLUTION}" \
  ++model.model_config.finetune.ocsc_nearest_include_gt="${OCSC_NEAREST_INCLUDE_GT}" \
  ++model.model_config.finetune.ocsc_ref_refresh_mode="${OCSC_REF_REFRESH_MODE}" \
  ++model.model_config.finetune.ocsc_ref_refresh_interval="${OCSC_REF_REFRESH_INTERVAL}" \
  ++model.model_config.finetune.ocsc_ref_ema_decay="${OCSC_REF_EMA_DECAY}" \
  model.model_config.self_forced.enabled=false

# Strategy.  OCSC loss back-propagates ONLY through the closed-loop rollout path,
# so under except_map_encoder/full_flow scopes some trainable params receive no
# gradient each step.  Plain `ddp` (find_unused_parameters=false) then deadlocks
# on the gradient all-reduce (util spikes during rollout, then idles forever).
# Keep the structured ddp.yaml strategy (timeout=14400 for the CPU-side scorer)
# and only flip find_unused_parameters=true -- mirrors the DMD full-mode launcher.
case "${TRAINER_STRATEGY}" in
  ddp)
    set -- "$@" trainer.strategy.find_unused_parameters=true
    ;;
  *)
    set -- "$@" trainer.strategy="${TRAINER_STRATEGY}"
    ;;
esac

if [ -n "${TRAIN_RAW_DIR}" ]; then
  set -- "$@" data.train_raw_dir="${TRAIN_RAW_DIR}"
fi
if [ -n "${VAL_RAW_DIR}" ]; then
  set -- "$@" data.val_raw_dir="${VAL_RAW_DIR}"
fi
if [ -n "${VAL_TFRECORDS_SPLITTED}" ]; then
  set -- "$@" data.val_tfrecords_splitted="${VAL_TFRECORDS_SPLITTED}"
fi
if [ -n "${WANDB_GROUP}" ]; then
  set -- "$@" logger.wandb.group="${WANDB_GROUP}"
fi
if is_true "${ENABLE_MODEL_CHECKPOINT}"; then
  set -- "$@" \
    callbacks.model_checkpoint.monitor="${CHECKPOINT_MONITOR}" \
    callbacks.model_checkpoint.mode="${CHECKPOINT_MODE}" \
    callbacks.model_checkpoint.save_last="${CHECKPOINT_SAVE_LAST}" \
    callbacks.model_checkpoint.save_top_k="${CHECKPOINT_SAVE_TOP_K}"
else
  set -- "$@" "~callbacks.model_checkpoint"
fi
if ! is_true "${ENABLE_EPOCH_LAST_CHECKPOINT}"; then
  set -- "$@" "~callbacks.epoch_last_checkpoint"
fi
if is_true "${RMM_FLOOR_ENABLED}"; then
  set -- "$@" \
    "++callbacks.rmm_floor._target_=src.utils.rmm_floor_callback.RmmFloorStop" \
    "++callbacks.rmm_floor.monitor=${RMM_FLOOR_MONITOR}" \
    "++callbacks.rmm_floor.floor=${RMM_FLOOR}"
fi

echo "============================================================"
echo "[ocsc_main] launching"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} nodes=${NUM_NODES} node_rank=${NODE_RANK}"
echo "  task=${TASK}"
echo "  ckpt=${CKPT_PATH}"
echo "  cache=${CACHE_ROOT}"
echo "  trainer: max_epochs=${MAX_EPOCHS} precision=${PRECISION} strategy=${TRAINER_STRATEGY} grad_clip=${GRADIENT_CLIP_VAL}"
echo "  data: train_B=${TRAIN_B} val_B=${VAL_B} workers=${NUM_WORKERS} shuffle=${DATA_SHUFFLE} sample_fraction=${TRAIN_EPOCH_SAMPLE_FRACTION}"
echo "  opt: lr=${LR} wd=${WEIGHT_DECAY} warmup=${LR_WARMUP_STEPS} total=${LR_TOTAL_STEPS} min_ratio=${LR_MIN_RATIO}"
echo "  sample: train=${OCSC_TRAIN_SAMPLE_METHOD}/${OCSC_TRAIN_SAMPLE_STEPS} noise=${OCSC_TRAIN_NOISE_SCALE} | val=${VALIDATION_SAMPLE_METHOD}/${VALIDATION_SAMPLE_STEPS} noise=${VALIDATION_NOISE_SCALE}"
echo "  val: scorer_scene_num=${SCORER_SCENE_NUM} n_rollout_closed_val=${N_ROLLOUT_CLOSED_VAL}"
echo "  ocsc: G=${OCSC_N_ROLLOUTS} M=${OCSC_N_OL_ROLLOUTS} gt=${OCSC_GT_TARGET} nearest=${OCSC_OL_NEAREST_MATCH} space=${OCSC_MATCH_SPACE} frame=${OCSC_MATCH_FRAME} stride=${OCSC_LOSS_TEMPORAL_STRIDE}"
echo "  scope: ${SCOPE} flow_ft_target=${FLOW_FT_TARGET} except_map=${OCSC_EXCEPT_MAP_ENCODER} velocity_head=${OCSC_VELOCITY_HEAD_ONLY} full_flow=${OCSC_FULL_FLOW_DECODER}"
echo "  wandb=${WANDB_ENTITY}/${WANDB_PROJECT} log=${LOG}"
echo "============================================================"

if is_true "${DRY_RUN}"; then
  printf "[ocsc_main] dry-run command:\n"
  printf "torchrun --nproc_per_node=%s --nnodes=%s --node_rank=%s --master_addr=%s --master_port=%s -m src.run" \
    "${NPROC_PER_NODE}" "${NUM_NODES}" "${NODE_RANK}" "${MASTER_ADDR}" "${MASTER_PORT}"
  for arg in "$@"; do
    printf " %s" "${arg}"
  done
  if [ -n "${EXTRA_ARGS}" ]; then
    printf " %s" "${EXTRA_ARGS}"
  fi
  printf "\n"
  exit 0
fi

if is_true "${LOG_TO_FILE}"; then
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m src.run \
    "$@" \
    ${EXTRA_ARGS} \
    > "${LOG}" 2>&1
else
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m src.run \
    "$@" \
    ${EXTRA_ARGS}
fi

status=$?
echo "[ocsc_main] done status=${status} log=${LOG}"
exit "${status}"
