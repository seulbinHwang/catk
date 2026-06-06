#!/usr/bin/env bash
set -euo pipefail

ROOT="/home2/pnc2/repos_python/kinematic_flow"
PY="/home2/pnc2/miniforge3/envs/catk/bin/python"
CACHE_ROOT="/home2/pnc2/repos_python/datasets/catk_cache"
VAL_PKL_DIR="${CACHE_ROOT}/validation"
VAL_TFR_DIR="${CACHE_ROOT}/validation_tfrecords_splitted"
GPU_ID="${GPU_ID:-3}"
CHAIN_TS="${CHAIN_TS:-$(date +%m%d_%H%M%S)}"
WANDB_GROUP="dmd_b_sweep_oldadam_g1e-4_c1e-5_${CHAIN_TS}"

cd "${ROOT}"
mkdir -p artifacts

prepare_subset() {
  local batch_size="$1"
  local subset_root="${CACHE_ROOT}/_b${batch_size}_scene_validation_head${batch_size}"
  local scene_dir="${subset_root}/scene"
  local tfr_dir="${subset_root}/tfrecords"
  rm -rf "${scene_dir}" "${tfr_dir}"
  mkdir -p "${scene_dir}" "${tfr_dir}"

  mapfile -t scene_paths < <(find "${VAL_PKL_DIR}" -maxdepth 1 -name '*.pkl' | sort | head -n "${batch_size}")
  if [ "${#scene_paths[@]}" -ne "${batch_size}" ]; then
    echo "[oldadam-g1e4][ERROR] requested B=${batch_size}, found ${#scene_paths[@]} validation pkl files"
    exit 1
  fi

  for pkl_path in "${scene_paths[@]}"; do
    local file
    local scene_id
    file="$(basename "${pkl_path}")"
    scene_id="${file%.pkl}"
    ln -f "${pkl_path}" "${scene_dir}/${file}"
    ln -f "${VAL_TFR_DIR}/${scene_id}.tfrecords" "${tfr_dir}/${scene_id}.tfrecords"
  done

  echo "${scene_dir}|${tfr_dir}"
}

launch_and_wait() {
  local batch_size="$1"
  local subset
  local scene_dir
  local tfr_dir
  subset="$(prepare_subset "${batch_size}")"
  scene_dir="${subset%%|*}"
  tfr_dir="${subset##*|}"

  local ts
  local task
  local log
  ts="$(date +%m%d_%H%M%S)"
  task="dmd_b${batch_size}_noshuf_velhead_oldadam_g1e-4_c1e-5_a4u3_gpu${GPU_ID}_${ts}"
  log="artifacts/${task}.log"

  echo "[oldadam-g1e4] launching B=${batch_size} on GPU ${GPU_ID}"
  echo "[oldadam-g1e4] scene_dir=${scene_dir}"
  echo "[oldadam-g1e4] log=${log}"

  setsid env LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PY}" -m src.run \
    experiment=flow_dmd \
    action=finetune \
    ckpt_path=logs/pretrained/pretrained.ckpt \
    paths.cache_root="${CACHE_ROOT}" \
    data.train_raw_dir="${scene_dir}" \
    data.val_raw_dir="${scene_dir}" \
    data.val_tfrecords_splitted="${tfr_dir}" \
    data.train_batch_size="${batch_size}" \
    data.val_batch_size="${batch_size}" \
    data.shuffle=false \
    data.num_workers=1 \
    data.train_epoch_sample_fraction=1.0 \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1 \
    trainer.val_check_interval=null \
    trainer.check_val_every_n_epoch=10 \
    trainer.log_every_n_steps=1 \
    trainer.max_epochs=180 \
    model.model_config.n_rollout_closed_val=16 \
    model.model_config.scorer_scene_num="${batch_size}" \
    model.model_config.n_batch_sim_agents_metric=1 \
    model.model_config.self_forced.estimator_warmup_steps=0 \
    model.model_config.lr=1e-4 \
    model.model_config.self_forced.estimator_lr=1e-5 \
    model.model_config.self_forced.unfrozen_range=velocity_head_only \
    model.model_config.self_forced.estimator_updates_per_step=3 \
    model.model_config.self_forced.n_anchors=4 \
    model.model_config.self_forced.anchor_stride=4 \
    model.model_config.self_forced.dmd_beta=1 \
    model.model_config.self_forced.n_rollouts=4 \
    model.model_config.self_forced.gradient_clip_val=10.0 \
    model.model_config.self_forced.sampling.sample_steps=16 \
    model.model_config.self_forced.sampling.noise_scale=1.0 \
    model.model_config.self_forced.sampling.random_terminal_step.enabled=false \
    model.model_config.self_forced.adam_beta1=0.9 \
    model.model_config.self_forced.adam_beta2=0.999 \
    model.model_config.self_forced.adam_eps=1.0e-8 \
    logger.wandb.entity=se99an \
    logger.wandb.project=clsft-catk \
    logger.wandb.group="${WANDB_GROUP}" \
    task_name="${task}" \
    > "${log}" 2>&1 < /dev/null &

  local pid=$!
  echo "[oldadam-g1e4] B=${batch_size} pid=${pid} task=${task}"
  wait "${pid}"
  echo "[oldadam-g1e4] B=${batch_size} complete"
}

launch_and_wait 4
launch_and_wait 8
launch_and_wait 16
echo "[oldadam-g1e4] done"
