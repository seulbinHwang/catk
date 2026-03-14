# CAT-K Flow Matching

이 문서는 `test_new` 브랜치의 구조를 참고해, **flow matching 경로만** 따로 정리한 README입니다.  
`smart_flow` 계열만 다루며, token-classification SMART, closed-loop fine-tuning, ego GMM 경로는 설명하지 않습니다.

- 기존 SMART의 map/context trunk를 그대로 재사용하고, agent 쪽만 flow decoder로 바꿔 scene-context 품질을 유지합니다.
- `FlowTokenProcessor`가 14-slot context pack과 13개 anchor를 만들어 2초 미래를 연속값으로 supervision 합니다.
- `HierarchicalFlowDecoder`와 `FlowODE`가 local normalized future를 직접 복원해 discrete token id보다 trajectory geometry를 더 부드럽게 모델링합니다.
- closed-loop inference는 0.5초씩 commit 하며 `pred_traj_10hz`, `pred_head_10hz`, `pred_z_10hz`를 바로 내보내 WOSAC 경로와 바로 연결됩니다.
- a2a relation에 coarse relative motion을 추가해 상호작용 정보를 강화하면서도 추가 정규화 규칙을 최소화합니다.

## 1. 관련 파일

이 README에서 다루는 핵심 파일은 아래입니다.

- `src/smart/model/smart_flow.py`
- `src/smart/modules/smart_flow_decoder.py`
- `src/smart/modules/flow_agent_decoder.py`
- `src/smart/modules/flow_local_decoder.py`
- `src/smart/tokens/flow_token_processor.py`
- `src/smart/metrics/flow_metrics.py`
- `configs/model/smart_flow.yaml`
- `configs/experiment/pre_bc_flow.yaml`
- `configs/experiment/local_val_flow.yaml`
- `configs/experiment/wosac_sub_flow.yaml`

## 2. 환경 설치

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
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

`ffmpeg`는 visualization용으로 필요합니다.

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

## 3. WOMD 데이터 다운로드

이 경로는 **WOMD scenario TFRecord**를 기준으로 합니다.

원하는 위치에 아래 구조가 되도록 준비합니다.

```text
$RAW_ROOT/
├── training/
├── validation/
└── testing/
```

예시 경로:

```bash
export RAW_ROOT=/workspace/womd_v1_3/scenario
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

토큰 파일은 저장소에 이미 포함되어 있으므로 별도 다운로드가 필요 없습니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

## 4. 캐시 생성

학습과 평가는 원본 TFRecord가 아니라 시나리오별 `.pkl` 캐시를 사용합니다.  
canonical 경로는 `src.data_preprocess`를 직접 호출하는 것입니다.

### 4.1 training 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split training \
  --num_workers 56
```

### 4.2 validation 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split validation \
  --num_workers 56
```

### 4.3 testing 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split testing \
  --num_workers 56
```

캐시가 끝나면 대략 아래처럼 생깁니다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/`에는 시나리오별 `.pkl`이 저장됩니다.
- `validation_tfrecords_splitted/`는 `validation` 캐시 생성 시 자동 생성됩니다.
- `validation_tfrecords_splitted/`는 local evaluation, WOSAC metric 계산, mp4 visualization에 필요합니다.

## 5. 6x H100에서 Flow Matching 학습

이 경로의 기본 학습 설정은 `configs/experiment/pre_bc_flow.yaml`입니다.

- `model=smart_flow`
- `precision=bf16-mixed`
- `max_epochs=64`
- `train_batch_size=12 per GPU`
- `val_batch_size=4`
- `test_batch_size=4`
- `num_workers=10`
- `lr=5e-4`
- `lr_warmup_steps=2`

H100 6장 기준 권장 실행:

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

메모리가 부족하면 아래처럼 train batch를 줄이면 됩니다.

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
  task_name=flow_pretrain_bs8 \
  data.train_batch_size=8
```

학습 중 W&B에는 기본적으로 아래 metric이 기록됩니다.

- `train/loss`
- `train/ADE2s`
- `train/FDE2s`
- `val_open/loss`
- `val_open/ADE2s`
- `val_open/FDE2s`
- `val_closed/ADE`
- `val_closed/wosac/*`

## 6. 평가와 추론

### 6.1 Validation set closed-loop 평가

`configs/experiment/local_val_flow.yaml`은 validation split에서 closed-loop rollout을 수행하고, WOSAC metric을 계산합니다.  
가장 단순한 사용법은 single GPU 평가입니다.

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
  task_name=flow_local_val
```

이 명령은 아래를 한 번에 수행합니다.

- validation split inference
- closed-loop rollout
- `val_closed/ADE`
- `val_closed/wosac/*`

### 6.2 Validation set에서 open-loop만 보고 싶을 때

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

### 6.3 6 GPU로 validation inference를 병렬화하고 싶을 때

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
  task_name=flow_local_val_ddp
```

## 7. WOSAC 제출 파일 생성

`configs/experiment/wosac_sub_flow.yaml`은 WOSAC submission export용 설정입니다.  
제출 전 아래 항목은 반드시 채워야 합니다.

- `ckpt_path`
- `model.model_config.wosac_submission.method_name`
- `authors`
- `affiliation`
- `description`
- `method_link`
- `account_name`

### 7.1 validation split으로 submission 형식 점검

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=wosac_sub_flow \
  action=validate \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_wosac_validate \
  model.model_config.wosac_submission.method_name="SMART-flow-7M" \
  model.model_config.wosac_submission.authors=[Anonymous] \
  model.model_config.wosac_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.wosac_submission.description="YOUR_DESCRIPTION" \
  model.model_config.wosac_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.wosac_submission.account_name="YOUR_ACCOUNT_NAME"
```

### 7.2 test split submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=wosac_sub_flow \
  action=test \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_wosac_test \
  model.model_config.wosac_submission.method_name="SMART-flow-7M" \
  model.model_config.wosac_submission.authors=[Anonymous] \
  model.model_config.wosac_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.wosac_submission.description="YOUR_DESCRIPTION" \
  model.model_config.wosac_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.wosac_submission.account_name="YOUR_ACCOUNT_NAME"
```

출력물:

- `logs/<task_name>/runs/<timestamp>/wosac_submission/`
- `logs/<task_name>/runs/<timestamp>/wosac_submission.tar.gz`

## 8. Visualization

visualization은 closed-loop validation 중에 같이 생성합니다.  
기본 `local_val_flow.yaml`은 비디오를 끄고 있으므로 아래 override를 줘야 합니다.

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
  task_name=flow_local_val_vis \
  model.model_config.n_vis_batch=2 \
  model.model_config.n_vis_scenario=5 \
  model.model_config.n_vis_rollout=5
```

비디오 저장 위치:

```text
logs/<task_name>/runs/<timestamp>/videos/
```

생성되는 파일:

- `gt.mp4`
- `rollout_00.mp4`
- `rollout_01.mp4`
- ...

W&B logger를 켜 둔 경우 같은 mp4가 W&B에도 함께 업로드됩니다.

## 9. 빠른 체크리스트

학습 전:

- `training/` 캐시 존재
- `validation/` 캐시 존재
- `validation_tfrecords_splitted/` 존재
- `paths.cache_root="$CACHE_ROOT"` 확인

WOSAC test submission 전:

- `testing/` 캐시 존재
- `ckpt_path` 확인
- submission metadata 6개 필드 확인

## 10. 자주 쓰는 명령 모음

### 캐시 생성

```bash
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split training --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split validation --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split testing --num_workers 56
```

### 6x H100 학습

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=pre_bc_flow trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" task_name=flow_pretrain_h1006
```

### validation 평가

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run experiment=local_val_flow trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_local_val
```

### test submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=wosac_sub_flow action=test trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_wosac_test
```
