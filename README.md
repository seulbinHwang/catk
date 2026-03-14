# CAT-K Flow Matching

- 이 저장소의 모델은 SMART의 scene-shared token encoder를 유지하면서, agent 예측만 flow matching head로 바꾼 multi-agent motion forecasting 구조다.
- 각 agent는 0.5초 간격 anchor 13개에서 2초 길이의 연속 미래 `(x, y, cos(delta_yaw), sin(delta_yaw))`를 직접 예측한다.
- `FutureConditioner`가 `noised future + tau`를 작은 조건 벡터로 바꿔 anchor query에만 주입하므로, backbone은 유지하면서 연속 타깃 학습이 가능하다.
- `StructuredFlowHead`가 20 step 미래를 한 번에 출력해 시간축 구조를 보존하고, discrete next-token loss 없이 단일 flow loss로 학습한다.
- closed-loop rollout은 매번 2초를 생성하되 처음 0.5초만 commit하고 다시 예측하는 방식이라 8초 horizon과 WOSAC 인터페이스를 안정적으로 유지한다.

이 저장소는 Waymo Open Motion Dataset(WOMD) scenario TFRecord를 캐시한 뒤, flow matching 기반 SMART 모델을 학습하고, closed-loop validation, WOSAC metric, submission export, visualization까지 수행하는 용도로 사용한다. 토큰 파일은 저장소에 이미 포함되어 있으므로 추가 다운로드가 필요 없다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

## 1. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.x`
- `ffmpeg` 설치 완료 상태

```bash
conda create -n catk python=3.11.9 -y
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2

pip install --upgrade pip
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

기본 logger는 W&B다. 필요하면 실행 전에 아래를 설정한다.

```bash
export WANDB_PROJECT=SMART-FLOW
export WANDB_ENTITY=jksg01019-naver-labs
```

오프라인으로 실행하고 싶으면 아래를 추가한다.

```bash
export WANDB_MODE=offline
```

## 2. 데이터 다운로드와 캐시 준비

이 저장소는 Waymo Open Motion Dataset의 scenario split을 입력으로 사용한다.

- 다운로드 페이지: `https://waymo.com/open/download`
- Motion dataset 설명: `https://waymo.com/open/data/motion/`

원본 데이터는 아래 구조로 준비하면 된다.

```text
$RAW_ROOT/
├── training/
├── validation/
└── testing/
```

예시 환경 변수:

```bash
export RAW_ROOT=/path/to/womd/scenario
export CACHE_ROOT=/path/to/SMART_cache
```

### 2.1 캐시 생성

학습과 평가는 원본 TFRecord가 아니라 split별 `.pkl` 캐시를 읽는다. 가장 간단한 방법은 `scripts/cache_womd.sh`를 split별로 한 번씩 실행하는 것이다.

```bash
INPUT_DIR="$RAW_ROOT" OUTPUT_DIR="$CACHE_ROOT" NUM_WORKERS=12 bash scripts/cache_womd.sh training
INPUT_DIR="$RAW_ROOT" OUTPUT_DIR="$CACHE_ROOT" NUM_WORKERS=12 bash scripts/cache_womd.sh validation
INPUT_DIR="$RAW_ROOT" OUTPUT_DIR="$CACHE_ROOT" NUM_WORKERS=12 bash scripts/cache_womd.sh testing
```

직접 실행하고 싶으면 아래와 같이 호출할 수 있다.

```bash
python -m src.data_preprocess \
  --split training \
  --num_workers 12 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"

python -m src.data_preprocess \
  --split validation \
  --num_workers 12 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"

python -m src.data_preprocess \
  --split testing \
  --num_workers 12 \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT"
```

캐시가 준비되면 구조는 대략 아래와 같다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/`에는 시나리오별 `.pkl`이 저장된다.
- `validation_tfrecords_splitted/`는 `validation` 캐시 생성 시 함께 만들어진다.
- local validation, WOSAC metric 계산, visualization은 `validation_tfrecords_splitted/`를 사용하므로 반드시 필요하다.

### 2.2 `train.sh`의 자동 캐시 동작

`scripts/train.sh`는 `training`과 `validation` 캐시가 없으면 자동으로 캐시 생성을 시도한다. 기본 탐색 순서는 아래와 같다.

- raw data: `${RAW_DATA_ROOT}` -> `/workspace/womd_v1_3/scenario` -> `/scratch/data/womd/uncompressed/scenario` -> `~/womd_v1_3/scenario`
- cache root: `${SMART_CACHE_ROOT}` -> `/workspace/womd_v1_3/SMART_cache` -> `/scratch/cache/SMART` -> `~/womd_v1_3/cache/SMART`

명시적으로 경로를 고정하고 싶으면 `RAW_DATA_ROOT`, `SMART_CACHE_ROOT`를 직접 주는 편이 가장 안전하다.

## 3. H100 GPU 6장 기준 학습

이 저장소의 기본 flow matching 학습 실험은 `experiment=pre_bc`다. H100 6장 기준 권장값은 아래와 같다.

- precision: `bf16-mixed`
- DDP: `trainer=ddp`, `trainer.devices=6`
- train batch size: `10` per GPU
- val/test batch size: `4`
- num workers: `10`
- epochs: `64`

### 3.1 권장 실행: `scripts/train.sh`

```bash
export RAW_DATA_ROOT="$RAW_ROOT"
export SMART_CACHE_ROOT="$CACHE_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

bash scripts/train.sh task_name=flow_pretrain_h1006
```

이 스크립트는 visible GPU 수를 읽어 `torchrun --standalone --nproc_per_node=6`, `trainer=ddp`, batch size, precision, cache path를 자동으로 맞춘다.

### 3.2 직접 실행: `torchrun`

캐시가 이미 준비되어 있으면 아래 명령으로 바로 학습할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 -m src.run \
  experiment=pre_bc \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_h1006
```

batch size나 worker 수를 바꾸고 싶으면 override만 추가하면 된다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 -m src.run \
  experiment=pre_bc \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_h1006_bs8 \
  data.train_batch_size=8 \
  data.val_batch_size=4 \
  data.num_workers=8
```

중간부터 이어서 학습하려면 `ckpt_path`를 넘긴다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --standalone --nproc_per_node=6 -m src.run \
  experiment=pre_bc \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_h1006_resume \
  ckpt_path=/absolute/path/to/last.ckpt
```

출력은 Hydra 기준 아래 경로에 저장된다.

- 로그 루트: `logs/<task_name>/runs/<YYYY-MM-DD>_<HH-MM-SS>/`
- 체크포인트: `logs/<task_name>/runs/<...>/checkpoints/`
- 실행 로그: `logs/<task_name>/runs/<...>/<task_name>.log`

## 4. 평가와 추론

이 저장소에는 별도 inference-only 엔트리포인트가 없다. closed-loop rollout을 생성하는 `validate` 또는 `test` 실행 자체가 추론 역할을 한다.

- validation split에서 metric과 rollout을 보고 싶으면 `experiment=local_val`
- submission용 rollout을 만들고 싶으면 `experiment=wosac_sub`

### 4.1 Local validation

아래 명령은 validation split에서 closed-loop rollout을 수행하고, `val_closed/ADE`와 WOSAC metric을 계산한다.

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/absolute/path/to/model.ckpt \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_local_val
```

기본 `local_val` 설정:

- open-loop metric 비활성화
- closed-loop rollout `32`회 샘플링
- validation batch size `4`
- WOSAC metric 계산용 batch `100`

결과 metric은 Hydra output과 W&B에 함께 기록된다.

## 5. Visualization

비디오 저장은 `local_val`에서 시각화 관련 override를 켰을 때만 동작한다. `ffmpeg`가 반드시 설치되어 있어야 한다.

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/absolute/path/to/model.ckpt \
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

출력 예시:

- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/gt.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_00.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_01.mp4`

주의:

- `model.model_config.n_vis_rollout <= model.model_config.n_rollout_closed_val`
- `model.model_config.n_vis_scenario <= data.val_batch_size`로 두는 것이 가장 안전하다
- `validation_tfrecords_splitted/`가 없으면 visualization이 동작하지 않는다

## 6. WOSAC submission

WOSAC 제출 파일은 `experiment=wosac_sub`로 생성한다. 먼저 `configs/experiment/wosac_sub.yaml`의 아래 메타데이터를 실제 값으로 채운다.

- `authors`
- `affiliation`
- `description`
- `method_link`
- `account_name`

### 6.1 Validation split으로 submission 샘플 생성

```bash
python -m src.run \
  experiment=wosac_sub \
  action=validate \
  ckpt_path=/absolute/path/to/model.ckpt \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_wosac_sub_validate
```

### 6.2 Test split으로 최종 submission 생성

```bash
python -m src.run \
  experiment=wosac_sub \
  action=test \
  ckpt_path=/absolute/path/to/model.ckpt \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_wosac_sub_test
```

출력 위치:

- shard binproto: `logs/<task_name>/runs/<...>/wosac_submission/`
- 최종 압축 파일: `logs/<task_name>/runs/<...>/wosac_submission.tar.gz`

## 7. 자주 확인할 점

- `paths.cache_root`를 지정하지 않으면 기본값은 `data/cache/SMART`다.
- `validation_tfrecords_splitted/`가 없으면 local validation, WOSAC metric, visualization이 실패한다.
- 기본 logger는 W&B이므로, 계정을 쓰지 않을 경우 `WANDB_MODE=offline`을 주는 편이 편하다.
- 6x H100에서 메모리가 빠듯하면 가장 먼저 `data.train_batch_size`를 줄여보면 된다.
- visualization은 batch 크기와 시나리오 수를 작게 두는 것이 안정적이다.

## 8. 가장 짧은 실행 순서

처음부터 끝까지 가장 일반적인 흐름은 아래와 같다.

1. 환경을 만들고 의존성을 설치한다.
2. WOMD scenario 데이터를 `training/validation/testing` 구조로 다운로드한다.
3. `scripts/cache_womd.sh`로 `training`, `validation`, `testing` 캐시를 만든다.
4. `scripts/train.sh` 또는 `torchrun`으로 `experiment=pre_bc` 학습을 수행한다.
5. 완성된 checkpoint로 `experiment=local_val`을 실행해 validation metric과 rollout을 확인한다.
6. 필요하면 visualization override를 켜서 mp4를 저장한다.
7. 제출이 필요하면 `experiment=wosac_sub`로 validation/test submission 파일을 생성한다.
