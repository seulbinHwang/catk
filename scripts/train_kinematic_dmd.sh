#!/bin/sh
# ============================================================================
# Kinematic DMD fine-tuning launcher
#   - 알고리즘: OCSC self_forcing_dmd (GT-grounded per-anchor rollout, 🅐)
#   - 모델/inference: 현재 kinematic_flow backbone (수정 X)
#   - 기본값 = 방금 돌린 세팅 (wandb run t3or3qf1 / experiment=flow_dmd)
#
# 사용:
#   bash scripts/train_kinematic_dmd.sh                      # 기본(방금 세팅)으로 학습
#   FAKE_LR=1e-5 bash scripts/train_kinematic_dmd.sh         # critic lr만 변경
#   N_ANCHORS=2 ANCHOR_STRIDE=4 bash scripts/train_kinematic_dmd.sh
#   ACTION=fit CKPT_PATH=<dmd_run_ckpt> bash scripts/train_kinematic_dmd.sh   # resume
#   CUDA_VISIBLE_DEVICES=2 bash scripts/train_kinematic_dmd.sh                # 단일 GPU
#
# 주의(repo 정책): 학습/평가는 GPU 2,3 만 사용.
# ============================================================================
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
# repo 정책: GPU 2,3. 단일 GPU 면 CUDA_VISIBLE_DEVICES=2 (또는 3).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
# conda 활성화 (self-contained; _activate_conda.sh 의존 X).
CONDA_SH="${CONDA_SH:-/home2/pnc2/miniforge3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then . "${CONDA_SH}"; fi
if command -v conda >/dev/null 2>&1; then conda activate "${CATK_CONDA_ENV}" || true; fi

# ── 실행 메타 ───────────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-flow_dmd}"
ACTION="${ACTION:-finetune}"           # fresh=finetune, resume=fit
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"
CACHE_ROOT="${CACHE_ROOT:-/home2/pnc2/repos_python/datasets/catk_cache}"
WANDB_ENTITY="${WANDB_ENTITY:-se99an}"
WANDB_PROJECT="${WANDB_PROJECT:-clsft-catk}"

# ── 하이퍼파라미터 (기본 = 방금 돌린 값; flow_dmd.yaml 과 동일, override 가능) ──
GEN_LR="${GEN_LR:-5.0e-5}"             # generator lr (static, no decay) — sweep 검증 peak
FAKE_LR="${FAKE_LR:-1.0e-4}"           # critic(fake_score) lr
UNFROZEN_RANGE="${UNFROZEN_RANGE:-velocity_head_only}"   # flow head only
# cadence fake:gen = n_anchors(4)×updates = 16:1 — 단일 scene sweep 유일 안정-상승 지점.
ESTIMATOR_UPDATES="${ESTIMATOR_UPDATES:-4}"
FAKE_WARMUP="${FAKE_WARMUP:-200}"                        # fake-only warmup steps
N_ANCHORS="${N_ANCHORS:-4}"                             # GT-grounded time-anchor 수
ANCHOR_STRIDE="${ANCHOR_STRIDE:-4}"                     # 4 coarse step = 2초 간격
DMD_BETA="${DMD_BETA:-1.0}"                            # 1=vanilla, <1=diversity↑
SAMPLE_STEPS="${SAMPLE_STEPS:-16}"                      # 학습 closed-loop ODE step (기본 16)
# full backprop: random_terminal_step 비활성 → rollout 이 16 ODE step 전체에 gradient
# (true=full BPTT, false=마지막 ODE step 만 grad).  validation ODE 는 별도(=16) 유지.
FULL_BACKPROP="${FULL_BACKPROP:-true}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"                          # G (variance reduction)
if [ "${FULL_BACKPROP}" = "true" ]; then RT_ENABLED=false; else RT_ENABLED=true; fi

# ── trainer / data (val 세팅은 pareto 정합; flow_dmd.yaml 기본) ──────────────
TRAIN_B="${TRAIN_B:-32}"
VAL_B="${VAL_B:-16}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-400}"
MAX_EPOCHS="${MAX_EPOCHS:-16}"
PRECISION="${PRECISION:-bf16-mixed}"
N_ROLLOUT_CLOSED_VAL="${N_ROLLOUT_CLOSED_VAL:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ── DDP device 수 = CUDA_VISIBLE_DEVICES 개수 ────────────────────────────────
N_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
if [ "${N_DEVICES}" -gt 1 ]; then
  STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"  # warmup 중 gen idle 대비
else
  STRATEGY="${STRATEGY:-auto}"
fi

TS="$(date +%m%d_%H%M%S)"
TASK_NAME="${TASK_NAME:-kindmd_head_genlr${GEN_LR}_fakelr${FAKE_LR}_c${ESTIMATOR_UPDATES}w${FAKE_WARMUP}_a${N_ANCHORS}s${ANCHOR_STRIDE}_${TS}}"
mkdir -p artifacts
LOG="artifacts/${TASK_NAME}.log"
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}"
  exit 1
fi

echo "[kinematic-dmd] EXPERIMENT=${EXPERIMENT} ACTION=${ACTION}"
echo "  GPU=${CUDA_VISIBLE_DEVICES} (devices=${N_DEVICES}, strategy=${STRATEGY})"
echo "  GEN_LR=${GEN_LR} FAKE_LR=${FAKE_LR} scope=${UNFROZEN_RANGE} cadence(fake:gen)=${ESTIMATOR_UPDATES}:1 warmup=${FAKE_WARMUP}"
echo "  anchor: n=${N_ANCHORS} stride=${ANCHOR_STRIDE}(2s) | dmd_beta=${DMD_BETA} train_ODE=${SAMPLE_STEPS}(full_backprop=${FULL_BACKPROP}, val_ODE=16) G=${N_ROLLOUTS}"
echo "  TRAIN_B=${TRAIN_B} VAL_B=${VAL_B} val_check=${VAL_CHECK_INTERVAL} n_rollout_closed_val=${N_ROLLOUT_CLOSED_VAL}"
echo "  wandb=${WANDB_ENTITY}/${WANDB_PROJECT}  ckpt=${CKPT_PATH}"
echo "  STDOUT_LOG=$(pwd)/${LOG}"

torchrun --nproc_per_node="${N_DEVICES}" --master_port="${PORT}" -m src.run \
  experiment="${EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  paths.cache_root="${CACHE_ROOT}" \
  trainer.devices="${N_DEVICES}" \
  trainer.strategy="${STRATEGY}" \
  trainer.precision="${PRECISION}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  data.train_batch_size="${TRAIN_B}" \
  data.val_batch_size="${VAL_B}" \
  data.num_workers="${NUM_WORKERS}" \
  logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.project="${WANDB_PROJECT}" \
  model.model_config.lr="${GEN_LR}" \
  model.model_config.n_rollout_closed_val="${N_ROLLOUT_CLOSED_VAL}" \
  model.model_config.self_forced.estimator_lr="${FAKE_LR}" \
  model.model_config.self_forced.unfrozen_range="${UNFROZEN_RANGE}" \
  model.model_config.self_forced.estimator_updates_per_step="${ESTIMATOR_UPDATES}" \
  model.model_config.self_forced.estimator_warmup_steps="${FAKE_WARMUP}" \
  model.model_config.self_forced.n_anchors="${N_ANCHORS}" \
  model.model_config.self_forced.anchor_stride="${ANCHOR_STRIDE}" \
  model.model_config.self_forced.dmd_beta="${DMD_BETA}" \
  model.model_config.self_forced.n_rollouts="${N_ROLLOUTS}" \
  model.model_config.self_forced.sampling.sample_steps="${SAMPLE_STEPS}" \
  model.model_config.self_forced.sampling.random_terminal_step.enabled="${RT_ENABLED}" \
  ${EXTRA_ARGS} \
  2>&1 | tee "${LOG}"

echo "[kinematic-dmd] done. log=${LOG}"
