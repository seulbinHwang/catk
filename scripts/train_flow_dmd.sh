#!/bin/sh
# Self-Forcing DMD fine-tuning launcher (현재 repo 모델/inference + OCSC_clean DMD 알고리즘).
# experiment=flow_dmd 는 self_forced.distribution_matching_objective=dmd 로 동작하며,
# generator loss 는 OCSC self_forcing_dmd 와 bit-동치 (smart_flow.py).
#
# 사용:
#   bash scripts/train_flow_dmd.sh                       # fresh finetune (action=finetune)
#   ACTION=fit CKPT_PATH=<dmd_run_ckpt> bash scripts/train_flow_dmd.sh   # resume
#   DMD_BETA=0.7 bash scripts/train_flow_dmd.sh          # diversity↑ (CPD↑)
#
# 주의: 모든 학습/평가는 GPU 2,3 만 사용 (repo 정책).
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# repo 정책: GPU 2,3 만 사용. single GPU 면 CUDA_VISIBLE_DEVICES=2 또는 3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

MY_EXPERIMENT="${MY_EXPERIMENT:-flow_dmd}"
MY_TASK_NAME="${MY_TASK_NAME:-${MY_EXPERIMENT}-debug}"
ACTION="${ACTION:-finetune}"
CKPT_PATH="${CKPT_PATH:-logs/pretrained/pretrained.ckpt}"

CATK_CONDA_ENV="${CATK_CONDA_ENV:-catk}"
. "$(dirname "$0")/_activate_conda.sh"

# DMD 핵심 knob (self_forced.* 로 override). β=1 vanilla / <1 diversity↑ / >1 sharpening.
DMD_BETA="${DMD_BETA:-1.0}"
LR="${LR:-2.0e-6}"

# device 수는 CUDA_VISIBLE_DEVICES 개수에 맞춤 (콤마 개수 + 1).
N_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"

python \
  -m src.run \
  experiment="${MY_EXPERIMENT}" \
  action="${ACTION}" \
  task_name="${MY_TASK_NAME}" \
  ckpt_path="${CKPT_PATH}" \
  trainer.devices="${N_DEVICES}" \
  model.model_config.lr="${LR}" \
  model.model_config.self_forced.dmd_beta="${DMD_BETA}" \
  ${EXTRA_ARGS}

echo "train_flow_dmd.sh done! (experiment=${MY_EXPERIMENT}, action=${ACTION}, beta=${DMD_BETA})"
