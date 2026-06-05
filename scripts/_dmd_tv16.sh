#!/bin/sh
# DMD "같은 16-scene train/val" overfit (셔플 없음). env: GPU, CLIP 필수.
# train_raw_dir=val_raw_dir=validation, batch16, limit_batches=1, shuffle off.
set -e
export LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
. /home2/pnc2/miniforge3/etc/profile.d/conda.sh
conda activate catk
cd "$(dirname "$0")/.."

R=/home2/pnc2/repos_python/datasets/catk_cache
CKPT=logs/pretrained/pretrained.ckpt
TS="$(date +%m%d_%H%M%S)"
# 베이스에서 단일 변경용 knob (안 주면 베이스값)
CLIP="${CLIP:-1.0}"
UPDATES="${UPDATES:-3}"                 # cadence = n_anchors(1)×UPDATES
SCOPE="${SCOPE:-except_map_encoder}"    # except_map_encoder | velocity_head_only | full_flow_decoder
GEN_LR="${GEN_LR:-1e-6}"               # generator(main encoder) lr
CRITIC_LR="${CRITIC_LR:-1e-5}"         # critic(fake_score) lr
DMD_NORMALIZE="${DMD_NORMALIZE:-true}" # DMD direction abs-mean normalizer on/off
N_ANCHORS="${N_ANCHORS:-1}"            # anchor 개수 (cadence = n_anchors×UPDATES : 1)
ANCHOR_STRIDE="${ANCHOR_STRIDE:-4}"    # anchor 간격(coarse 2Hz step)
MAX_EPOCHS="${MAX_EPOCHS:-375}"        # 1 epoch=1 step, wandb _step≈epoch×5.35 (375≈2000 step)
RMM_FLOOR="${RMM_FLOOR:-0.775}"        # val RMM 이 이 값 밑이면 run 자동 종료(다음 큐로)
DMD_BETA="${DMD_BETA:-1}"              # DMD β: 1=vanilla, <1=diversity↑, >1=sharpening(realism↑)
SAMPLE_STEPS="${SAMPLE_STEPS:-16}"     # ★ training self-forced ODE step (val=16 고정, 안 건드림)
PRECISION="${PRECISION:-32-true}"      # trainer precision: 32-true | 16-mixed | bf16-mixed
BACKPROP_K="${BACKPROP_K:-}"           # self-forced denoising step 중 마지막 K개만 backprop (빈값=전체 16)
ESTIMATOR_WARMUP="${ESTIMATOR_WARMUP:-0}"  # critic(fake) warmup step 수 (generator 시작 전 critic만 학습)
export CUDA_VISIBLE_DEVICES="${GPU}"
NORMTAG=""; [ "${DMD_NORMALIZE}" = "false" ] && NORMTAG="_nonorm"
ANCHORTAG=""; [ "${N_ANCHORS}" != "1" ] && ANCHORTAG="_a${N_ANCHORS}s${ANCHOR_STRIDE}"
BETATAG=""; [ "${DMD_BETA}" != "1" ] && BETATAG="_b${DMD_BETA}"
STEPTAG=""; [ "${SAMPLE_STEPS}" != "16" ] && STEPTAG="_s${SAMPLE_STEPS}"
PRECTAG=""; [ "${PRECISION}" != "32-true" ] && PRECTAG="_$(echo ${PRECISION} | sed 's/-.*//')p"
WTAG=""; [ "${ESTIMATOR_WARMUP}" != "0" ] && WTAG="_w${ESTIMATOR_WARMUP}"
BKTAG=""; BK_OVERRIDE=""
[ -n "${BACKPROP_K}" ] && BKTAG="_bk${BACKPROP_K}" && BK_OVERRIDE="+model.model_config.self_forced.sampling.backprop_last_k=${BACKPROP_K}"
TASK="dmd_tv16_g${GEN_LR}_f${CRITIC_LR}_clip${CLIP}_u${UPDATES}_${SCOPE}${ANCHORTAG}${BETATAG}${STEPTAG}${PRECTAG}${BKTAG}${WTAG}${NORMTAG}_${TS}"
LOG="artifacts/${TASK}.log"
mkdir -p artifacts
echo "[tv16] GPU=${GPU} CLIP=${CLIP} -> ${LOG}"

python -m src.run \
  experiment=flow_dmd action=finetune \
  ckpt_path="${CKPT}" task_name="${TASK}" \
  paths.cache_root="${R}" \
  data.train_raw_dir="${R}/validation" data.val_raw_dir="${R}/validation" \
  data.val_tfrecords_splitted="${R}/validation_tfrecords_splitted" \
  data.train_batch_size=16 data.val_batch_size=16 data.num_workers=2 \
  data.shuffle=false data.train_epoch_sample_fraction=1.0 \
  trainer.devices=1 trainer.strategy=auto trainer.precision="${PRECISION}" \
  trainer.limit_train_batches=1 trainer.limit_val_batches=1 \
  trainer.val_check_interval=null trainer.check_val_every_n_epoch=10 \
  trainer.max_epochs="${MAX_EPOCHS}" \
  model.model_config.n_rollout_closed_val=16 \
  model.model_config.scorer_scene_num=null \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.lr="${GEN_LR}" \
  model.model_config.self_forced.estimator_lr="${CRITIC_LR}" \
  model.model_config.self_forced.estimator_warmup_steps="${ESTIMATOR_WARMUP}" \
  model.model_config.self_forced.unfrozen_range="${SCOPE}" \
  model.model_config.self_forced.estimator_updates_per_step="${UPDATES}" \
  model.model_config.self_forced.n_anchors="${N_ANCHORS}" \
  model.model_config.self_forced.anchor_stride="${ANCHOR_STRIDE}" \
  model.model_config.self_forced.dmd_beta="${DMD_BETA}" \
  model.model_config.self_forced.dmd_normalize="${DMD_NORMALIZE}" \
  model.model_config.self_forced.n_rollouts=1 \
  model.model_config.self_forced.gradient_clip_val="${CLIP}" \
  model.model_config.self_forced.sampling.sample_steps="${SAMPLE_STEPS}" \
  ${BK_OVERRIDE} \
  model.model_config.self_forced.sampling.noise_scale=1.0 \
  model.model_config.self_forced.sampling.random_terminal_step.enabled=false \
  logger.wandb.entity=se99an logger.wandb.project=clsft-catk \
  logger.wandb.log_model=false \
  '~callbacks.model_checkpoint' '~callbacks.epoch_last_checkpoint' \
  '+callbacks.rmm_floor._target_=src.utils.rmm_floor_callback.RmmFloorStop' \
  '+callbacks.rmm_floor.monitor=val_closed/sim_agents_2025/realism_meta_metric' \
  "+callbacks.rmm_floor.floor=${RMM_FLOOR}" \
  > "${LOG}" 2>&1
echo "[tv16] done CLIP=${CLIP} log=${LOG}"
