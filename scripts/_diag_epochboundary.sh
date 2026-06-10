#!/usr/bin/env bash
# 진단용: epoch 0->1 경계에서의 NCCL 데드락 재현 + desync collective 식별.
# limit_train_batches=40 으로 epoch 을 분 단위로 끝내고, timeout 을 120s 로 줄여
# 데드락 시 watchdog 가 갈린 collective 를 빠르게 dump 하도록 한다.
set -u
cd /home2/pnc2/repos_python/kinematic_flow

source /home2/pnc2/miniforge3/etc/profile.d/conda.sh
conda activate catk

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
# --- 진단 핵심 env ---
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export NCCL_DEBUG=WARN
export SF_CTRACE=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

LOG=artifacts/DIAG_LOGKEY_limit24_gpu23.log

torchrun --standalone --nproc_per_node=2 --master_port=29555 \
  -m src.run \
  experiment=self_forced_npfm action=finetune task_name=DIAG_LOGKEY_limit24_gpu23 \
  ckpt_path=logs/pretrained/pretrained.ckpt seed=817 \
  paths.cache_root=/home2/pnc2/repos_python/datasets/catk_cache \
  trainer.devices=2 \
  ~trainer.strategy \
  +trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy \
  +trainer.strategy.find_unused_parameters=true \
  +trainer.strategy.gradient_as_bucket_view=true \
  +trainer.strategy.timeout._target_=datetime.timedelta \
  +trainer.strategy.timeout.seconds=14400 \
  ++trainer.strategy.timeout.seconds=120 \
  trainer.precision=32-true trainer.max_epochs=2 \
  ++trainer.val_check_interval=1.0 trainer.check_val_every_n_epoch=1 \
  trainer.limit_val_batches=1 \
  ++trainer.limit_train_batches=24 \
  data.train_batch_size=24 data.val_batch_size=32 data.num_workers=8 data.shuffle=false \
  model.model_config.lr=1e-7 model.model_config.scorer_scene_num=64 \
  model.model_config.n_rollout_closed_val=16 \
  model.model_config.self_forced.distribution_matching_objective=dmd \
  model.model_config.self_forced.path_step_size=1.0 \
  model.model_config.self_forced.normalize_direction=true \
  model.model_config.self_forced.clean_dmd_per_channel_normalizer=false \
  model.model_config.self_forced.clean_dmd_normalizer_eps=0.05 \
  model.model_config.self_forced.sampling.random_terminal_step.policy=all \
  model.model_config.self_forced.sampling.backprop_last_k=16 \
  model.model_config.self_forced.use_anchor_flow_matching_loss=false \
  model.model_config.self_forced.anchor_weight=0.05 \
  model.model_config.sim_agents_metric_workers=8 \
  model.model_config.self_forced.cadence=25 \
  model.model_config.self_forced.estimator_updates_per_step=1 \
  model.model_config.self_forced.estimator_lr=1e-7 \
  model.model_config.self_forced.use_ema=false \
  model.model_config.self_forced.estimator_warmup_epochs=0 \
  model.model_config.self_forced.warmup_zone_steps=0 \
  model.model_config.self_forced.joint_zone_steps=0 \
  logger.wandb.entity=se99an logger.wandb.project=clsft-catk \
  model.model_config.self_forced.unfrozen_range=middle \
  ~callbacks.model_checkpoint ~callbacks.epoch_last_checkpoint \
  > "${LOG}" 2>&1
echo "[diag] done status=$? log=${LOG}"
