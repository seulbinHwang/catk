#!/bin/sh
# OCSC GT-target L2 fine-tuning, tv16 "Train=Val 16-scene overfit" (DMD 검증과 동일 환경).
# student closed-loop rollout(2초) 을 GT future 에 매칭(L2). flow head only. env: GPU 필수.
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
# knob
LR="${LR:-1e-6}"                       # generator lr (flow head)
G="${G:-4}"                            # ocsc_n_rollouts (student CL rollout 수)
MATCH_FRAME="${MATCH_FRAME:-global}"   # global(raw world meter L2) | local(anchor-0 정규화)
GT_TARGET="${GT_TARGET:-true}"         # GT future 에 매칭(true) | OL-ref(false)
SCOPE="${SCOPE:-velocity_head_only}"   # velocity_head_only | except_map_encoder | full_flow_decoder
BATCH="${BATCH:-16}"                    # train=val batch size (=시나리오 개수, limit_batches=1)
STRIDE="${STRIDE:-}"                    # ocsc_loss_temporal_stride (10Hz step). 1=dense(전체0.1초), 5=0.5초 coarse, 빈값=기본(-1→shift=5=0.5초)
POS_W="${POS_W:-1.0}"
HEAD_W="${HEAD_W:-0.01}"
MAX_EPOCHS="${MAX_EPOCHS:-375}"
RMM_FLOOR="${RMM_FLOOR:-0.700}"        # 상승 관찰용: 낮게 둬서 일찍 안 끊기게
export CUDA_VISIBLE_DEVICES="${GPU}"
# OCSC scope → finetune flag 3종
case "${SCOPE}" in
  velocity_head_only) SCOPE_OVR="model.model_config.finetune.velocity_head_only=true model.model_config.finetune.train_except_map_encoder=false model.model_config.finetune.train_full_flow_decoder_only=false" ;;
  except_map_encoder) SCOPE_OVR="model.model_config.finetune.velocity_head_only=false model.model_config.finetune.train_except_map_encoder=true model.model_config.finetune.train_full_flow_decoder_only=false" ;;
  full_flow_decoder)  SCOPE_OVR="model.model_config.finetune.velocity_head_only=false model.model_config.finetune.train_except_map_encoder=false model.model_config.finetune.train_full_flow_decoder_only=true" ;;
  *) echo "unknown SCOPE=${SCOPE}"; exit 1 ;;
esac
BATCHTAG=""; [ "${BATCH}" != "16" ] && BATCHTAG="_b${BATCH}"
STRIDETAG=""; STRIDE_OVR=""
[ -n "${STRIDE}" ] && STRIDETAG="_st${STRIDE}" && STRIDE_OVR="model.model_config.finetune.ocsc_loss_temporal_stride=${STRIDE}"
TASK="ocsc_tv16_lr${LR}_G${G}_${MATCH_FRAME}_gt${GT_TARGET}_${SCOPE}${BATCHTAG}${STRIDETAG}_${TS}"
LOG="artifacts/${TASK}.log"
mkdir -p artifacts
echo "[ocsc_tv16] GPU=${GPU} frame=${MATCH_FRAME} G=${G} lr=${LR} -> ${LOG}"

python -m src.run \
  experiment=ocsc_ft action=finetune \
  ckpt_path="${CKPT}" task_name="${TASK}" \
  paths.cache_root="${R}" \
  data.train_raw_dir="${R}/validation" data.val_raw_dir="${R}/validation" \
  data.val_tfrecords_splitted="${R}/validation_tfrecords_splitted" \
  data.train_batch_size="${BATCH}" data.val_batch_size="${BATCH}" data.num_workers=2 \
  data.shuffle=false data.train_epoch_sample_fraction=1.0 \
  trainer.devices=1 trainer.strategy=auto trainer.precision=32-true \
  trainer.limit_train_batches=1 trainer.limit_val_batches=1 \
  trainer.val_check_interval=null trainer.check_val_every_n_epoch=10 \
  trainer.max_epochs="${MAX_EPOCHS}" \
  model.model_config.lr="${LR}" \
  model.model_config.n_rollout_closed_val=16 \
  model.model_config.scorer_scene_num=null \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.finetune.enabled=true \
  model.model_config.finetune.mode=ocsc_ft \
  ${SCOPE_OVR} \
  model.model_config.finetune.ocsc_n_rollouts="${G}" \
  model.model_config.finetune.ocsc_gt_target="${GT_TARGET}" \
  model.model_config.finetune.ocsc_match_space=pose \
  model.model_config.finetune.ocsc_match_frame="${MATCH_FRAME}" \
  model.model_config.finetune.ocsc_loss_window_steps=-1 \
  ${STRIDE_OVR} \
  model.model_config.finetune.ocsc_position_weight="${POS_W}" \
  model.model_config.finetune.ocsc_heading_weight="${HEAD_W}" \
  logger.wandb.entity=se99an logger.wandb.project=clsft-catk \
  logger.wandb.log_model=false \
  '~callbacks.model_checkpoint' '~callbacks.epoch_last_checkpoint' \
  '+callbacks.rmm_floor._target_=src.utils.rmm_floor_callback.RmmFloorStop' \
  '+callbacks.rmm_floor.monitor=val_closed/sim_agents_2025/realism_meta_metric' \
  "+callbacks.rmm_floor.floor=${RMM_FLOOR}" \
  > "${LOG}" 2>&1
echo "[ocsc_tv16] done log=${LOG}"
