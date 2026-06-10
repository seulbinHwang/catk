#!/bin/sh
# Self-forcing(DMD) fine-tuning — cadence 기반 fake:gen 업데이트.
#
# 새 cadence 의미: fake(critic)는 매 batch 1회, generator 는 N batch 마다 1회(서로 다른
# batch 들에서). 같은 시나리오를 여러 번 돌리지 않는다.  (estimator_updates_per_step=1)
#
# 기본 런치: cadence 5:1, gen lr=fake lr=1e-7, EMA off, DMD, train B=8, val 1000 batch,
#            scorer 440 scene + val_b 16 (scene 수 자동 조정), GPU 2,3.
#   bash scripts/_self_forcing_update.sh
set -e

is_true() { case "$(printf %s "$1" | tr A-Z a-z)" in 1|true|yes|on) return 0;; *) return 1;; esac; }
get_free_port() { python - <<'PY'
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
}

export LOGLEVEL="${LOGLEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
[ -f "${CONDA_SH}" ] && . "${CONDA_SH}"
command -v conda >/dev/null 2>&1 && conda activate "${CATK_CONDA_ENV:-catk}" || true
cd "${REPO_ROOT}"

# --- core knobs ---
MY_EXPERIMENT="${MY_EXPERIMENT:-self_forced_npfm}"
ACTION="${ACTION:-finetune}"
SEED="${SEED:-817}"
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/catk_cache}"
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="${GPU}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(printf %s "${GPU}" | awk -F, '{print NF}')}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"

# self-forcing cadence / lr / ema / objective
CADENCE="${CADENCE:-5}"                         # fake:gen = N:1
ESTIMATOR_UPDATES_PER_STEP="${ESTIMATOR_UPDATES_PER_STEP:-1}"   # fake updates per batch (1 = distinct batches)
ESTIMATOR_INIT_CKPT="${ESTIMATOR_INIT_CKPT:-}"   # warmup된 fake critic ckpt(F_psi override). 빈값=generator 복사본
GEN_LR="${GEN_LR:-2e-6}"
FAKE_LR="${FAKE_LR:-4e-7}"
USE_EMA="${USE_EMA:-false}"
DM_OBJECTIVE="${DM_OBJECTIVE:-dmd}"
# normalize on(기본): direction 이 normalizer 로 O(1) 스케일 → step≈1.0 이 원본 DMD 정합.
# normalize off(raw) 로 쓸 땐 raw gap 이 작아 2.0 같은 큰 값 필요.
PATH_STEP_SIZE="${PATH_STEP_SIZE:-1.0}"   # DMD direction step (normalize on 이면 ≈1.0)
NORMALIZE_DIRECTION="${NORMALIZE_DIRECTION:-true}"  # false=거리-나눗셈 제거(raw, 수렴형)
# normalizer 모드. false(기본)=시간+채널 전체 평균(agent당 스칼라, 원본 DMD 정합, 분모
# 안정). 죽은 채널(non-holonomic delta_n)은 direction/normalizer 양쪽에서 masking 제외.
# true=시간축만 평균(채널별 분모) — 분모 불안정 → push 폭발하던 기존 방식.
PER_CHANNEL_NORMALIZER="${PER_CHANNEL_NORMALIZER:-false}"
# DMD direction normalizer 분모 하한(eps). normalizer=|committed-teacher| 평균이 이 값보다
# 작으면 floor. 기본 0.05 는 수렴 구간 push 증폭(분모 0 근처 폭발)을 강하게 억제. 1e-3 으로
# 낮추면 원본 정합. config 키: self_forced.clean_dmd_normalizer_eps.
NORMALIZER_EPS="${NORMALIZER_EPS:-0.05}"
# gradient 경로 정책.
#   all(기본)=random terminal 미생성 → 블록 간 detach 제거(전 horizon grad) + backprop_last_k
#     경로. BACKPROP_LAST_K=16=sample_steps 면 16 ODE step 전부 grad. (full-gradient)
#   paper_uniform=원본식 truncation(블록 detach + 마지막 1개 ODE step만 grad, τ≈1 고정).
TERMINAL_POLICY="${TERMINAL_POLICY:-all}"
BACKPROP_LAST_K="${BACKPROP_LAST_K:-16}"   # policy=all 일 때 grad 남길 마지막 ODE step 수
ESTIMATOR_WARMUP_EPOCHS="${ESTIMATOR_WARMUP_EPOCHS:-0}"
# 반복 warmup/joint zone 스케줄(step 기준). 둘 다 양수면 warmup zone(critic만)과
# joint zone(기존 cadence DMD)을 step 기준으로 번갈아 무한 반복. 0/0 이면 비활성.
WARMUP_ZONE_STEPS="${WARMUP_ZONE_STEPS:-0}"
JOINT_ZONE_STEPS="${JOINT_ZONE_STEPS:-0}"
# 학습할 파라미터 범위. except_map_encoder(기본) | middle(flow decoder + 마지막 agent
# 문맥 블록만) | full_flow_decoder(flow decoder만). 빈값이면 config 기본값 유지.
UNFROZEN_RANGE="${UNFROZEN_RANGE:-}"
# FM regularization: DMD loss 에 open-loop flow-matching loss 를 anchor_weight 로 더해
# generator 가 teacher 의 open-loop FM 에서 drift 하는 것을 억제한다. false=기존(off).
USE_ANCHOR_FM_LOSS="${USE_ANCHOR_FM_LOSS:-false}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.05}"

# data / trainer
TRAIN_B="${TRAIN_B:-8}"
VAL_B="${VAL_B:-16}"
SCORER_SCENE_NUM="${SCORER_SCENE_NUM:-440}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"      # scorer_scene_num 가 자동 상향
MAX_EPOCHS="${MAX_EPOCHS:-16}"
PRECISION="${PRECISION:-32-true}"   # fp32
NUM_WORKERS="${NUM_WORKERS:-8}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
SIM_AGENTS_METRIC_WORKERS="${SIM_AGENTS_METRIC_WORKERS:-8}"   # 0=직렬(느림). 병렬로 val scorer 단축.
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
# checkpoint 저장 여부. true(기본)=best(RMM 최대)+last 저장. false=콜백 제거(빠른 실험).
SAVE_CKPT="${SAVE_CKPT:-true}"

# wandb
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"

TS="$(date +%m%d_%H%M%S)"
TASK_DEFAULT="sfupdate_cad${CADENCE}_gen${GEN_LR}_fake${FAKE_LR}_${DM_OBJECTIVE}_b${TRAIN_B}x${NPROC_PER_NODE}_${TS}"
TASK="${MY_TASK_NAME:-${TASK_DEFAULT}}"
LOG="${LOG:-artifacts/${TASK}.log}"
mkdir -p artifacts

if [ ! -f "${CKPT_PATH}" ] && [ "${ACTION}" != "fit" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"; exit 1
fi

set -- \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${TASK}" \
  ckpt_path="${CKPT_PATH}" \
  seed="${SEED}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${NPROC_PER_NODE}" \
  "~trainer.strategy" \
  "+trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy" \
  "+trainer.strategy.find_unused_parameters=true" \
  "+trainer.strategy.gradient_as_bucket_view=true" \
  "+trainer.strategy.timeout._target_=datetime.timedelta" \
  "+trainer.strategy.timeout.seconds=14400" \
  trainer.precision="${PRECISION}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  ++trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch=1 \
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  data.shuffle="${DATA_SHUFFLE}" \
  model.model_config.lr="${GEN_LR}" \
  model.model_config.scorer_scene_num="${SCORER_SCENE_NUM}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.self_forced.distribution_matching_objective="${DM_OBJECTIVE}" \
  model.model_config.self_forced.path_step_size="${PATH_STEP_SIZE}" \
  model.model_config.self_forced.normalize_direction="${NORMALIZE_DIRECTION}" \
  model.model_config.self_forced.clean_dmd_per_channel_normalizer="${PER_CHANNEL_NORMALIZER}" \
  model.model_config.self_forced.clean_dmd_normalizer_eps="${NORMALIZER_EPS}" \
  model.model_config.self_forced.sampling.random_terminal_step.policy="${TERMINAL_POLICY}" \
  model.model_config.self_forced.sampling.backprop_last_k="${BACKPROP_LAST_K}" \
  model.model_config.self_forced.use_anchor_flow_matching_loss="${USE_ANCHOR_FM_LOSS}" \
  model.model_config.self_forced.anchor_weight="${ANCHOR_WEIGHT}" \
  model.model_config.sim_agents_metric_workers="${SIM_AGENTS_METRIC_WORKERS}" \
  model.model_config.self_forced.cadence="${CADENCE}" \
  model.model_config.self_forced.estimator_updates_per_step="${ESTIMATOR_UPDATES_PER_STEP}" \
  model.model_config.self_forced.estimator_lr="${FAKE_LR}" \
  model.model_config.self_forced.use_ema="${USE_EMA}" \
  model.model_config.self_forced.estimator_warmup_epochs="${ESTIMATOR_WARMUP_EPOCHS}" \
  model.model_config.self_forced.warmup_zone_steps="${WARMUP_ZONE_STEPS}" \
  model.model_config.self_forced.joint_zone_steps="${JOINT_ZONE_STEPS}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.project="${WANDB_PROJECT}"

# warmup된 fake critic override (빈값이면 생략 → config null 유지)
if [ -n "${ESTIMATOR_INIT_CKPT}" ]; then
  set -- "$@" model.model_config.self_forced.estimator_init_ckpt="${ESTIMATOR_INIT_CKPT}"
fi

# 학습 파라미터 범위 override (빈값이면 생략 → config 기본값 유지)
if [ -n "${UNFROZEN_RANGE}" ]; then
  set -- "$@" model.model_config.self_forced.unfrozen_range="${UNFROZEN_RANGE}"
fi

# checkpoint 저장. SAVE_CKPT=true(기본): model_checkpoint(best, monitor=RMM/mode=max,
# save_top_k=1 + save_last 링크) + epoch_last_checkpoint(epoch_last.ckpt) 유지 → best+last.
# false: 두 콜백 제거(빠른 실험용, 디스크 미사용).
if ! is_true "${SAVE_CKPT}"; then
  set -- "$@" "~callbacks.model_checkpoint" "~callbacks.epoch_last_checkpoint"
fi

echo "============================================================"
echo "[sf-update] task=${TASK}"
echo "  GPU=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}  ckpt=${CKPT_PATH}"
echo "  cadence(fake:gen)=${CADENCE}:1  est_updates/batch=${ESTIMATOR_UPDATES_PER_STEP}  gen_lr=${GEN_LR} fake_lr=${FAKE_LR}"
echo "  estimator_init_ckpt=${ESTIMATOR_INIT_CKPT:-<none>}"
echo "  objective=${DM_OBJECTIVE} use_ema=${USE_EMA} warmup_epochs=${ESTIMATOR_WARMUP_EPOCHS}"
echo "  normalize_dir=${NORMALIZE_DIRECTION} per_channel_norm=${PER_CHANNEL_NORMALIZER} path_step=${PATH_STEP_SIZE} normalizer_eps=${NORMALIZER_EPS}"
echo "  grad_policy=${TERMINAL_POLICY} backprop_last_k=${BACKPROP_LAST_K} (all+16=full-gradient: 블록 detach 없음 + 16 ODE step 전부)"
echo "  anchor_fm_loss=${USE_ANCHOR_FM_LOSS} anchor_weight=${ANCHOR_WEIGHT}"
echo "  zone_schedule(warmup:joint steps)=${WARMUP_ZONE_STEPS}:${JOINT_ZONE_STEPS} (0:0=off)"
echo "  unfrozen_range=${UNFROZEN_RANGE:-<config default>}"
echo "  train_B=${TRAIN_B} val_B=${VAL_B} scorer_scene=${SCORER_SCENE_NUM} val_check=${VAL_CHECK_INTERVAL} precision=${PRECISION}"
echo "  save_ckpt=${SAVE_CKPT} (true=best+last, false=none)"
echo "  log=${LOG}"
echo "============================================================"

if is_true "${DRY_RUN:-false}"; then
  printf "torchrun --standalone --nproc_per_node=%s -m src.run" "${NPROC_PER_NODE}"
  for a in "$@"; do printf " %s" "$a"; done; printf "\n"; exit 0
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  -m src.run "$@" > "${LOG}" 2>&1
echo "[sf-update] done status=$? log=${LOG}"
