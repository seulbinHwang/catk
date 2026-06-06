#!/usr/bin/env bash
set -euo pipefail

ROOT="/home2/pnc2/repos_python/kinematic_flow"
PY="/home2/pnc2/miniforge3/envs/catk/bin/python"
CACHE_ROOT="/home2/pnc2/repos_python/datasets/catk_cache"
VAL_PKL_DIR="${CACHE_ROOT}/validation"
VAL_TFR_DIR="${CACHE_ROOT}/validation_tfrecords_splitted"
CHAIN_TS="${CHAIN_TS:-$(date +%m%d_%H%M%S)}"
WANDB_GROUP="dmd_b_sweep_b4_b8_b16_${CHAIN_TS}"

cd "${ROOT}"
mkdir -p artifacts

wait_for_pids() {
  local label="$1"
  shift
  local pids=("$@")
  echo "[watcher] waiting for ${label}: ${pids[*]}"
  while true; do
    local alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive+=("${pid}")
      fi
    done
    if [ "${#alive[@]}" -eq 0 ]; then
      echo "[watcher] ${label} complete"
      return 0
    fi
    echo "[watcher] ${label} still running: ${alive[*]}"
    sleep 60
  done
}

prepare_subset() {
  local batch_size="$1"
  local subset_root="${CACHE_ROOT}/_b${batch_size}_scene_validation_head${batch_size}"
  local scene_dir="${subset_root}/scene"
  local tfr_dir="${subset_root}/tfrecords"
  rm -rf "${scene_dir}" "${tfr_dir}"
  mkdir -p "${scene_dir}" "${tfr_dir}"

  mapfile -t scene_paths < <(find "${VAL_PKL_DIR}" -maxdepth 1 -name '*.pkl' | sort | head -n "${batch_size}")
  if [ "${#scene_paths[@]}" -ne "${batch_size}" ]; then
    echo "[watcher][ERROR] requested B=${batch_size}, found ${#scene_paths[@]} validation pkl files"
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

launch_pair() {
  local batch_size="$1"
  local subset
  local scene_dir
  local tfr_dir
  subset="$(prepare_subset "${batch_size}")"
  scene_dir="${subset%%|*}"
  tfr_dir="${subset##*|}"

  local ts
  local task_old
  local task_new
  local log_old
  local log_new
  ts="$(date +%m%d_%H%M%S)"
  task_old="dmd_b${batch_size}_noshuf_velhead_adam09_g1e-5_c1e-5_a4u3_gpu2_${ts}"
  task_new="dmd_b${batch_size}_noshuf_velhead_adam00_g1e-5_c1e-5_a4u3_gpu3_${ts}"
  log_old="artifacts/${task_old}.log"
  log_new="artifacts/${task_new}.log"

  local common_args=(
    experiment=flow_dmd
    action=finetune
    ckpt_path=logs/pretrained/pretrained.ckpt
    paths.cache_root="${CACHE_ROOT}"
    data.train_raw_dir="${scene_dir}"
    data.val_raw_dir="${scene_dir}"
    data.val_tfrecords_splitted="${tfr_dir}"
    data.train_batch_size="${batch_size}"
    data.val_batch_size="${batch_size}"
    data.shuffle=false
    data.num_workers=1
    data.train_epoch_sample_fraction=1.0
    trainer.devices=1
    trainer.strategy=auto
    trainer.limit_train_batches=1
    trainer.limit_val_batches=1
    trainer.val_check_interval=null
    trainer.check_val_every_n_epoch=10
    trainer.log_every_n_steps=1
    trainer.max_epochs=180
    model.model_config.n_rollout_closed_val=16
    model.model_config.scorer_scene_num="${batch_size}"
    model.model_config.n_batch_sim_agents_metric=1
    model.model_config.self_forced.estimator_warmup_steps=0
    model.model_config.lr=1e-5
    model.model_config.self_forced.estimator_lr=1e-5
    model.model_config.self_forced.unfrozen_range=velocity_head_only
    model.model_config.self_forced.estimator_updates_per_step=3
    model.model_config.self_forced.n_anchors=4
    model.model_config.self_forced.anchor_stride=4
    model.model_config.self_forced.dmd_beta=1
    model.model_config.self_forced.n_rollouts=4
    model.model_config.self_forced.gradient_clip_val=10.0
    model.model_config.self_forced.sampling.sample_steps=16
    model.model_config.self_forced.sampling.noise_scale=1.0
    model.model_config.self_forced.sampling.random_terminal_step.enabled=false
    logger.wandb.entity=se99an
    logger.wandb.project=clsft-catk
    logger.wandb.group="${WANDB_GROUP}"
  )

  echo "[watcher] launching B=${batch_size}"
  echo "[watcher] scene_dir=${scene_dir}"
  echo "[watcher] tfr_dir=${tfr_dir}"

  setsid env LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 \
    "${PY}" -m src.run "${common_args[@]}" task_name="${task_old}" model.model_config.self_forced.adam_beta1=0.9 model.model_config.self_forced.adam_beta2=0.999 model.model_config.self_forced.adam_eps=1.0e-8 > "${log_old}" 2>&1 < /dev/null &
  local pid_old=$!

  setsid env LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online CUDA_VISIBLE_DEVICES=3 \
    "${PY}" -m src.run "${common_args[@]}" task_name="${task_new}" model.model_config.self_forced.adam_beta1=0.0 model.model_config.self_forced.adam_beta2=0.999 model.model_config.self_forced.adam_eps=1.0e-8 > "${log_new}" 2>&1 < /dev/null &
  local pid_new=$!

  echo "[watcher] B=${batch_size} old_pid=${pid_old} task=${task_old} log=${log_old}"
  echo "[watcher] B=${batch_size} new_pid=${pid_new} task=${task_new} log=${log_new}"
  wait_for_pids "B=${batch_size}" "${pid_old}" "${pid_new}"
}

# Current corrected B=4 runs.
wait_for_pids "B=4" 2581516 2581517
launch_pair 8
launch_pair 16
echo "[watcher] done"
