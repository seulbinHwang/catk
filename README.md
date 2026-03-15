# CAT-K Flow Matching

이 저장소는 **flow matching 학습/추론/평가 전용**으로 정리된 버전입니다.
이 문서는 **Waymo 2025 Sim Agents 평가와 제출**만 설명합니다. 이 저장소의 평가는 2025 Sim Agents 기준만 사용합니다.

## 1. 핵심 포인트

- 학습/추론 본체는 `smart_flow` 계열만 사용합니다.
- closed-loop local 평가는 **Waymo 공식 2025 Sim Agents scorer**를 그대로 사용합니다.
- local metric namespace는 모두 `val_closed/sim_agents_2025/*` 로 기록됩니다.
- scenario mean raw metric은 `val_closed/sim_agents_2025_mean/*` 로 기록됩니다.
- submission export도 2025 Sim Agents 기준으로 저장되며 출력 폴더 이름은 `sim_agents_2025_submission/` 입니다.

## 2. 관련 파일

- `src/smart/model/smart_flow.py`
- `src/smart/metrics/sim_agents_metrics.py`
- `src/smart/metrics/sim_agents_submission.py`
- `src/utils/sim_agents_utils.py`
- `configs/model/smart_flow.yaml`
- `configs/experiment/pre_bc_flow.yaml`
- `configs/experiment/local_val_flow.yaml`
- `configs/experiment/sim_agents_sub_flow.yaml`
- `configs/run.yaml`
- `scripts/train_flow.sh`
- `scripts/local_val_flow.sh`
- `scripts/sim_agents_sub_flow.sh`

## 3. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.x`
- `ffmpeg`

예시:

```bash
conda create -n catk python=3.11.9 -y
conda activate catk

pip install --upgrade pip
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```

`ffmpeg` 는 visualization 용으로 필요합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

W&B를 쓸 경우:

```bash
wandb login
export WANDB_PROJECT=SMART-FLOW
export WANDB_ENTITY=<your_entity>
```

### 3.1 중요

이 저장소는 시작 시점에 **공식 2025 Sim Agents scorer** 와 **traffic light violation 관련 2025 필드** 가 실제로 있는지 바로 확인합니다.
따라서 예전 Waymo 패키지를 설치하면 validation 시작 전에 명확하게 실패합니다.

## 4. WOMD 데이터 준비

원본 TFRecord 는 아래 구조를 기준으로 둡니다.

```text
$RAW_ROOT/
├── training/
├── validation/
└── testing/
```

예시:

```bash
export RAW_ROOT=/workspace/womd_v1_3/scenario
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

토큰 파일은 저장소에 포함되어 있으므로 따로 받을 필요가 없습니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

## 5. 캐시 생성

학습과 추론은 scenario 별 `.pkl` 캐시를 사용합니다.
canonical 경로는 `src.data_preprocess` 입니다.

### 5.1 training

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split training \
  --num_workers 56
```

### 5.2 validation

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split validation \
  --num_workers 56
```

### 5.3 testing

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split testing \
  --num_workers 56
```

완료 후 구조는 대략 아래와 같습니다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/` 에는 scenario 별 `.pkl` 이 저장됩니다.
- `validation_tfrecords_splitted/` 는 validation cache 생성 시 자동 생성됩니다.
- **공식 local 2025 score 계산은 `validation_tfrecords_splitted/` 가 있어야만 가능합니다.**
- test split 은 local numeric metric 을 계산하지 않고 submission export 만 합니다.

## 6. 6x H100에서 Flow Matching 학습

기본 학습 설정은 `configs/experiment/pre_bc_flow.yaml` 입니다.

- `model=smart_flow`
- `precision=bf16-mixed`
- `max_epochs=64`
- `train_batch_size=12 per GPU`
- `val_batch_size=4`
- `test_batch_size=4`
- `num_workers=10`
- `lr=5e-4`
- `lr_warmup_steps=2`
- best checkpoint monitor: `val_closed/sim_agents_2025/realism_meta_metric`

권장 실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_h1006
```

### 6.1 학습 중 W&B에 기록되는 주요 항목

- `train/loss`
- `train/ADE2s`
- `train/FDE2s`
- `val_denoise/loss`
- `val_open/ADE2s`
- `val_open/FDE2s`
- `val_closed/sim_agents_2025/realism_meta_metric`
- `val_closed/sim_agents_2025/kinematic_metrics`
- `val_closed/sim_agents_2025/interactive_metrics`
- `val_closed/sim_agents_2025/map_based_metrics`
- `val_closed/sim_agents_2025/simulated_collision_rate`
- `val_closed/sim_agents_2025/simulated_offroad_rate`
- `val_closed/sim_agents_2025/simulated_traffic_light_violation_rate`
- `val_closed/sim_agents_2025/minADE_best_of_<n_rollout_closed_val>`
- `val_closed/sim_agents_2025_mean/*`

## 7. Local validation: Waymo 2025 Sim Agents score

`configs/experiment/local_val_flow.yaml` 은 validation split 에서 closed-loop rollout 을 수행하고,
**Waymo 공식 2025 Sim Agents metric** 을 계산합니다.

single GPU 예시:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=local_val_flow \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_2025_validate
```

6 GPU DDP 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=local_val_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_2025_validate_ddp
```

이 경로는 아래를 한 번에 수행합니다.

- validation split inference
- closed-loop rollout
- `val_closed/sim_agents_2025/*`
- `val_closed/sim_agents_2025_mean/*`

### 7.1 open-loop만 보고 싶을 때

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=local_val_flow \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_open_val \
  model.model_config.val_open_loop=true \
  model.model_config.val_closed_loop=false
```

## 8. Waymo 2025 Sim Agents submission export

`configs/experiment/sim_agents_sub_flow.yaml` 은 **2025 Sim Agents submission export** 용 설정입니다.
local metric 을 계산하지 않고 rollout proto 와 tar.gz 를 저장합니다.

채워야 하는 항목:

- `model.model_config.sim_agents_submission.method_name`
- `model.model_config.sim_agents_submission.authors`
- `model.model_config.sim_agents_submission.affiliation`
- `model.model_config.sim_agents_submission.description`
- `model.model_config.sim_agents_submission.method_link`
- `model.model_config.sim_agents_submission.account_name`

### 8.1 validation split 을 submission 형식으로 저장

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_2025_validate_export \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

### 8.2 test split submission export

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=sim_agents_sub_flow \
  action=test \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_2025_test_export \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

출력:

- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission/`
- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission.tar.gz`

## 9. 자주 보는 설정 키

- `model.model_config.val_open_loop`
- `model.model_config.val_closed_loop`
- `model.model_config.n_rollout_closed_val`
- `model.model_config.n_batch_sim_agents_metric`
- `model.model_config.n_vis_batch`
- `model.model_config.n_vis_scenario`
- `model.model_config.n_vis_rollout`
- `model.model_config.delete_local_videos_after_wandb_upload`
- `model.model_config.sim_agents_submission.*`
- `callbacks.model_checkpoint.monitor`

## 10. 기본 스크립트

```bash
bash scripts/train_flow.sh
bash scripts/local_val_flow.sh
bash scripts/sim_agents_sub_flow.sh
```

`scripts/sim_agents_sub_flow.sh` 에서 `ACTION=validate` 또는 `ACTION=test` 를 바꾸면 됩니다.

## 11. 동작 원리 한 줄 정리

- local score: validation TFRecord + rollout 을 받아 **Waymo 공식 2025 Sim Agents scorer** 를 호출
- submission export: rollout 을 `SimAgentsChallengeSubmission` 으로 묶어 shard 와 tar.gz 저장

즉, 이 저장소에서 2025 metric 변경점인 **traffic light violation** 과 공식 scorer 안의 **2025 TTC 경로** 는 모두 저장소 자체 구현이 아니라 **Waymo 공식 scorer 경로** 를 그대로 따라갑니다.