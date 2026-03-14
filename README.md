# CAT-K Flow

이 저장소는 Waymo Open Motion Dataset(WOMD) scenario 데이터를 입력으로 사용해 아래 순서로 실행하는 것을 기준으로 정리되어 있습니다.

1. WOMD scenario 데이터 준비
2. pkl 캐시 생성 또는 다운로드
3. open-loop 학습
4. short closed-loop fine-tuning
5. 추론 및 로컬 평가
6. mp4 저장
7. WOSAC 제출 파일 생성

기본으로 사용하는 실험 설정은 아래 4개입니다.

- open-loop pretraining: `configs/experiment/flow_pretrain_h1006.yaml`
- short closed-loop fine-tuning: `configs/experiment/flow_clsft_h1006.yaml`
- local evaluation: `configs/experiment/flow_local_val.yaml`
- WOSAC submission export: `configs/experiment/flow_wosac_sub.yaml`

토큰 파일은 저장소에 이미 포함되어 있으므로 별도 다운로드가 필요하지 않습니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

## 1. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.1`
- `ffmpeg` 설치 완료 상태

이 저장소의 스크립트는 conda 경로나 env 이름을 하드코딩하지 않습니다. 현재 쉘에서 이미 활성화된 Python 환경의 `python` / `torchrun` 을 그대로 사용합니다.

```bash
conda create -n catk python=3.11.9 -y
conda activate catk

pip install --upgrade pip
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

`ffmpeg` 는 시스템 패키지로 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 2. W&B 준비

학습 스크립트 `scripts/train_flow_h1006.sh` 와 `scripts/finetune_flow_h1006.sh` 는 기본적으로 W&B online 로 실행됩니다.

기본값:

- `WANDB_PROJECT=SMART-FLOW`
- `WANDB_ENTITY=jksg01019-naver-labs`
- `WANDB_MODE=online`

다른 계정을 쓸 경우 실행 전에 바꾸면 됩니다.

```bash
export WANDB_ENTITY=<your_wandb_entity>
wandb login
```

온라인으로 학습하면 W&B에는 아래 정보가 저장됩니다.

- 학습/검증 metric
- 실행 config 와 주요 hyperparameter
- 모델 checkpoint artifact
- 평가 시 생성한 비디오가 있으면 비디오 로그

open-loop pretraining 과 short closed-loop fine-tuning 은 checkpoint를 artifact 로 업로드합니다. 각 run 에 대해 아래 alias 를 바로 사용할 수 있습니다.

- `best`
- `latest`

학습이 끝나면 콘솔에 ready-to-copy artifact ref 가 출력되고, 같은 값이 아래 두 곳에도 저장됩니다.

- W&B summary: `artifact/run_path`, `artifact/best_ckpt_ref`, `artifact/latest_ckpt_ref`
- 로컬 파일: `logs/<task_name>/runs/<...>/artifact_refs.txt`

오프라인으로 돌리고 싶으면 아래처럼 바꾸면 됩니다.

```bash
export WANDB_MODE=offline
```

## 3. WOMD scenario 데이터 준비

이 저장소는 Waymo Open Motion Dataset의 scenario proto TFRecord 를 사용합니다.

- 다운로드 페이지: `https://waymo.com/open/download`
- Motion dataset 설명: `https://waymo.com/open/data/motion/`

예시 디렉터리 구조:

```text
/path/to/womd/scenario/
├── training/
├── validation/
└── testing/
```

이 README 에서는 아래 두 변수를 사용합니다.

```bash
export RAW_ROOT=/path/to/womd/scenario
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

`training` 과 `validation` 캐시는 학습과 로컬 평가에 필요합니다. `testing` 캐시는 최종 WOSAC test submission 을 만들 때 필요합니다.

## 4. pkl 캐시 준비

학습과 평가에는 TFRecord 원본이 아니라 시나리오별 `.pkl` 캐시가 필요합니다.

캐시를 준비하는 방법은 두 가지입니다.

1. 직접 전처리해서 생성
2. Nubes 에 저장된 캐시를 다운로드

### 4.1 전처리로 캐시 생성

split 별로 한 번씩 실행합니다.

```bash
python -m src.data_preprocess \
  --split training \
  --num_workers 56 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"

python -m src.data_preprocess \
  --split validation \
  --num_workers 56 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"

python -m src.data_preprocess \
  --split testing \
  --num_workers 56 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"
```

동일 작업을 스크립트로 실행할 수도 있습니다.

```bash
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh training
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh validation
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh testing
```

전처리가 끝나면 캐시는 대략 아래 구조가 됩니다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/` 에 시나리오별 `.pkl` 파일이 생성됩니다.
- `validation_tfrecords_splitted/` 는 `validation` 전처리 시 자동 생성됩니다.
- 로컬 평가와 mp4 저장은 이 `validation_tfrecords_splitted/` 를 사용하므로 반드시 필요합니다.

### 4.2 Nubes 에서 캐시 다운로드

이미 만들어진 pkl 캐시를 쓰고 싶다면 `scripts/download_smart_cache_from_nubes.sh` 를 사용할 수 있습니다.

기본 사용법:

```bash
bash scripts/download_smart_cache_from_nubes.sh <remote_dir> <local_dir>
```

예시:

```bash
bash scripts/download_smart_cache_from_nubes.sh \
  labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
  "$CACHE_ROOT"
```

또는 환경변수로 넘겨도 됩니다.

```bash
REMOTE_DIR=labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
LOCAL_DIR="$CACHE_ROOT" \
bash scripts/download_smart_cache_from_nubes.sh
```

주의:

- `nubescli` 가 PATH 에 있어야 합니다.
- 스크립트는 `nubescli dir-download -s` 를 사용하므로, 이미 있는 로컬 파일은 유지하고 없는 파일만 다운로드합니다.
- 실행 중에는 원격 전체 파일 수 대비 현재 `LOCAL_DIR` 아래에 존재하는 파일 수를 기준으로 진행률을 1분마다 출력합니다.
- 진행률 로그에는 `percent`, `elapsed`, `eta` 가 함께 표시됩니다.
- 스크립트는 현재 프로세스에 할당된 CPU 수를 자동 감지해 다운로드 동시성을 정합니다. 예를 들어 32 CPU 환경에서는 기본적으로 `NUBES_JOBS=24` 를 사용합니다.
- 필요하면 `NUBES_JOBS`, `NUM_CPUS`, `CPUSET`, `PROGRESS_INTERVAL_SEC` 를 환경변수로 override 할 수 있습니다.
- `CACHE_ROOT` 로 사용할 경로를 그대로 `LOCAL_DIR` 로 넘기는 것이 가장 단순합니다.

## 5. Open-Loop Pretraining

먼저 open-loop pretraining 으로 초기 체크포인트를 만듭니다.

직접 실행:

```bash
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_pretrain_h1006 \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_h1006
```

스크립트 실행:

```bash
CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 bash scripts/train_flow_h1006.sh
```

이 실험은 기본적으로 6 GPU 기준입니다. GPU 수가 다르면 `NPROC_PER_NODE`, `trainer.devices`, 필요시 batch size 를 함께 조정하십시오.

batch size 를 바꿔가며 실험하려면 아래처럼 override 하면 됩니다.

```bash
torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_pretrain_h1006 \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_bs8 \
  data.train_batch_size=8 \
  data.val_batch_size=8
```

예를 들어 `12 -> 10 -> 8 -> 6` 순서로 줄여보면 됩니다.
메모리 여유를 보면서 조절하려면 별도 터미널에서 `watch -n 1 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'` 를 같이 보면 됩니다.

학습 결과는 아래에 저장됩니다.

- 로그 루트: `logs/flow_pretrain_h1006/runs/<YYYY-MM-DD>_<HH-MM-SS>/`
- 체크포인트: `logs/flow_pretrain_h1006/runs/<...>/checkpoints/`
- artifact ref 메모: `logs/flow_pretrain_h1006/runs/<...>/artifact_refs.txt`

다음 단계에는 보통 아래 둘 중 하나를 사용합니다.

- 로컬 체크포인트: `last.ckpt` 또는 원하는 epoch 체크포인트
- W&B artifact ref: `entity/project/model-<run_id>:best` 또는 `:latest`

## 6. Short Closed-Loop Fine-Tuning

short closed-loop fine-tuning 은 open-loop pretraining 결과를 입력으로 사용합니다. 로컬 checkpoint 와 W&B artifact 둘 다 사용할 수 있습니다.

### 6.1 로컬 checkpoint 로 fine-tuning

```bash
export PRETRAIN_CKPT=/absolute/path/to/open_loop/checkpoints/last.ckpt

CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 \
bash scripts/finetune_flow_h1006.sh "$PRETRAIN_CKPT"
```

직접 실행:

```bash
torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_clsft_h1006 \
  ckpt_path="$PRETRAIN_CKPT" \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_clsft_h1006
```

### 6.2 W&B artifact 로 fine-tuning

```bash
export PRETRAIN_ARTIFACT='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best'

CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 \
FLOW_CKPT_ARTIFACT="$PRETRAIN_ARTIFACT" \
bash scripts/finetune_flow_h1006.sh
```

직접 실행:

```bash
torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_clsft_h1006 \
  ckpt_artifact='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best' \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_clsft_h1006
```

fine-tuning 결과 체크포인트도 동일하게 로컬 파일 또는 W&B artifact 로 다음 단계에서 사용할 수 있습니다.

## 7. 추론과 로컬 평가

이 저장소에서 별도 inference-only 엔트리포인트는 없습니다. 보통 아래 두 실행이 추론 역할을 합니다.

- `flow_local_val`: validation split 에서 closed-loop rollout 생성 + metric 계산
- `flow_wosac_sub`: validation 또는 test split 에서 rollout 생성 + 제출 파일 생성

즉, 로컬 평가나 WOSAC 제출 생성 명령이 곧 추론 명령입니다.

### 7.1 로컬 checkpoint 로 평가

```bash
export FT_CKPT=/absolute/path/to/flow_clsft_h1006/checkpoints/last.ckpt

CACHE_ROOT="$CACHE_ROOT" TRAINER_DEVICES=1 \
bash scripts/local_val_flow.sh "$FT_CKPT"
```

직접 실행:

```bash
python -m src.run \
  experiment=flow_local_val \
  action=validate \
  ckpt_path="$FT_CKPT" \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_local_val
```

### 7.2 W&B artifact 로 평가

```bash
export FT_ARTIFACT='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best'

CACHE_ROOT="$CACHE_ROOT" TRAINER_DEVICES=1 \
FLOW_CKPT_ARTIFACT="$FT_ARTIFACT" \
bash scripts/local_val_flow.sh
```

직접 실행:

```bash
python -m src.run \
  experiment=flow_local_val \
  action=validate \
  ckpt_artifact='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best' \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_local_val
```

기본 `flow_local_val` 설정은 아래를 계산합니다.

- `val_closed/ADE`
- WOSAC realism / kinematic / interactive / map-based metrics

## 8. mp4 저장

mp4 저장은 `flow_local_val` 에서 시각화 관련 override 를 켠 경우에만 동작합니다. `ffmpeg` 가 반드시 설치되어 있어야 합니다.

```bash
python -m src.run \
  experiment=flow_local_val \
  action=validate \
  ckpt_path="$FT_CKPT" \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.limit_val_batches=1 \
  data.val_batch_size=1 \
  data.num_workers=0 \
  data.pin_memory=false \
  data.persistent_workers=false \
  paths.cache_root="$CACHE_ROOT" \
  model.model_config.n_rollout_closed_val=2 \
  model.model_config.n_batch_wosac_metric=1 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=1 \
  model.model_config.n_vis_rollout=2 \
  task_name=flow_local_val_video
```

영상은 아래 경로에 저장됩니다.

- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/gt.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_00.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_01.mp4`

주의:

- `model.model_config.n_vis_rollout <= model.model_config.n_rollout_closed_val`
- `model.model_config.n_vis_scenario <= data.val_batch_size`
- 기본 `flow_local_val` 은 `n_vis_batch=0` 이므로 override 없이는 mp4 가 저장되지 않습니다.

## 9. WOSAC 제출 파일 생성

제출 전에 `configs/experiment/flow_wosac_sub.yaml` 의 아래 메타데이터를 실제 값으로 수정하십시오.

- `authors`
- `affiliation`
- `description`
- `method_link`
- `account_name`

### 9.1 validation split 으로 submission 샘플 생성

로컬 checkpoint 사용:

```bash
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" validate
```

W&B artifact 사용:

```bash
CACHE_ROOT="$CACHE_ROOT" \
ACTION=validate \
FLOW_CKPT_ARTIFACT='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best' \
bash scripts/wosac_sub_flow.sh
```

직접 실행:

```bash
python -m src.run \
  experiment=flow_wosac_sub \
  action=validate \
  ckpt_path="$FT_CKPT" \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_wosac_sub_validate
```

### 9.2 test split 으로 최종 submission 생성

로컬 checkpoint 사용:

```bash
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" test
```

W&B artifact 사용:

```bash
CACHE_ROOT="$CACHE_ROOT" \
ACTION=test \
FLOW_CKPT_ARTIFACT='jksg01019-naver-labs/SMART-FLOW/model-<run_id>:best' \
bash scripts/wosac_sub_flow.sh
```

직접 실행:

```bash
python -m src.run \
  experiment=flow_wosac_sub \
  action=test \
  ckpt_path="$FT_CKPT" \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_wosac_sub_test
```

출력 위치:

- shard binproto: `logs/<task_name>/runs/<...>/wosac_submission/`
- 최종 압축 파일: `logs/<task_name>/runs/<...>/wosac_submission.tar.gz`

## 10. 자주 막히는 지점

- `paths.cache_root` 를 지정하지 않으면 기본값 `/scratch/cache/SMART` 를 읽습니다.
- `validation_tfrecords_splitted/` 가 없으면 로컬 평가와 mp4 저장이 실패합니다.
- `ffmpeg` 가 없으면 mp4 저장이 실패합니다.
- W&B artifact 를 쓰는 평가 머신에서도 `wandb login` 또는 `WANDB_API_KEY` 가 필요합니다.
- `scripts/download_smart_cache_from_nubes.sh` 는 기존 파일은 유지하고 누락 파일만 받습니다. 다만 원격 전체 파일 수를 기준으로 진행률을 계산하므로, 원격 목록 조회 시간이 처음에 한 번 필요합니다.

## 11. 최소 실행 순서 요약

처음부터 끝까지 가장 일반적인 순서는 아래입니다.

1. 환경을 만들고 의존성을 설치합니다.
2. WOMD scenario 데이터를 `training/validation/testing` 구조로 준비합니다.
3. pkl 캐시를 전처리로 만들거나, Nubes 에서 다운로드합니다.
4. `flow_pretrain_h1006` 로 open-loop pretraining 을 수행합니다.
5. pretraining checkpoint 로 `flow_clsft_h1006` short closed-loop fine-tuning 을 수행합니다.
6. fine-tuned checkpoint 로 `flow_local_val` 을 실행해 validation metric 을 확인합니다.
7. 필요하면 같은 validation 경로에서 시각화 override 를 켜서 mp4 를 저장합니다.
8. 제출이 필요하면 `flow_wosac_sub` 로 validation/test submission 파일을 생성합니다.
