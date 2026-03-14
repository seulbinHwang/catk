# SMART-flow 7M pre-BC

이 저장소는 `brand_new` 브랜치를 바탕으로, 기존 SMART-tiny 7M의 **scene-shared 병렬 구조**와 **WOSAC 출력 인터페이스**는 유지하면서, agent 예측 부분만 **flow matching 기반 2초 연속 미래 예측**으로 바꾼 버전이다.

이 브랜치의 범위는 딱 하나다.

- **SMART-flow 7M pre-BC 학습**
- **closed-loop validation / WOSAC submission 유지**
- **GMM 기반 경로와 CAT-K fine-tuning 경로는 사용하지 않음**

학습 목표는 각 agent-anchor마다 2초 길이 10Hz 미래 하나를 맞추는 것이다. 표현은 아래 4개 값으로 통일했다.

- `x_local`
- `y_local`
- `cos(delta_yaw)`
- `sin(delta_yaw)`

loss는 하나만 쓴다. `x, y`는 loss 계산 때만 `20`으로 나누고, `cos/sin`은 그대로 쓴다.

---

## 1. 핵심 동작 방식

### 1-1. 무엇이 바뀌었는가

기존 `brand_new`는 NTP 방식으로 다음 token을 맞췄다. 이 버전은 마지막 token classification head를 제거하고, 대신 아래 구조를 사용한다.

1. coarse token 기반 scene encoder는 유지한다.
2. 각 valid anchor에 대해 2초 미래 GT를 local 좌표로 만든다.
3. 학습 때는 clean future에 랜덤 noise를 섞는다.
4. `future conditioner(noised future + tau)`가 작은 조건 벡터를 만든다.
5. 이 조건 벡터를 anchor query 쪽에만 넣는다.
6. structured flow head가 20 step의 velocity를 예측한다.
7. loss는 masked MSE 하나만 사용한다.

### 1-2. 어떤 anchor를 쓰는가

anchor는 10Hz 전체 step을 다 쓰지 않고, 기존 SMART의 scene-level 병렬 구조를 살리기 위해 **0.5초 간격**으로 잡는다.

- 현재 설정: `num_historical_steps=11`, `num_future_steps=80`
- valid anchor step: `10, 15, 20, ..., 70`
- 총 **13개 anchor**

각 anchor는 자기 시점부터 앞으로 20 step(=2초, 10Hz)을 예측한다.

### 1-3. closed-loop rollout은 어떻게 하는가

추론은 매번 2초 미래를 생성하지만, 실제 상태 업데이트는 **첫 0.5초만 commit**한다.

즉, 아래처럼 반복한다.

1. 현재 state에서 2초 미래 생성
2. 첫 0.5초만 사용
3. 그 상태를 새 current state로 갱신
4. 다시 2초 미래 생성

이 과정을 16번 반복하면 8초 미래 80 step이 채워진다.

---

## 2. 설치

### 2-1. conda 환경

기존 README와 같은 방식으로 시작하면 된다.

```bash
conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

### 2-2. 선택 사항: Docker

기존 repo처럼 Docker를 써도 된다. 원래 repo 설명대로 Docker 쪽이 더 빠를 수 있다.

---

## 3. 데이터 준비

### 3-1. Waymo Open Motion Dataset 다운로드

이 코드는 **Waymo Open Motion Dataset v1.2.1**을 기준으로 쓴다.

Waymo 페이지에서 아래 3개 split을 준비해야 한다.

- `training`
- `validation`
- `testing`

압축을 푼 뒤 예시는 아래처럼 잡으면 된다.

```text
/scratch/data/womd/uncompressed/scenario/
  training/
  validation/
  testing/
```

### 3-2. 기본 경로와 자동 감지

지금 스크립트는 아래 순서로 경로를 자동 감지한다.

- raw WOMD root: `/workspace/womd_v1_3/scenario` -> `/scratch/data/womd/uncompressed/scenario` -> `~/womd_v1_3/scenario`
- cache root: `/workspace/womd_v1_3/SMART_cache` -> `/scratch/cache/SMART` -> `~/womd_v1_3/cache/SMART`

현재 H100 서버에서는 캐시가 이미 아래에 있다고 가정하면 된다.

```text
/workspace/womd_v1_3/SMART_cache/
  training/
  validation/
  testing/
  validation_tfrecords_splitted/
```

다른 경로를 쓰고 싶으면 환경 변수로 override 하면 된다.

```bash
INPUT_DIR=/your/raw/scenario/root OUTPUT_DIR=/your/cache/root bash scripts/cache_womd.sh training
SMART_CACHE_ROOT=/your/cache/root bash scripts/train.sh
```

### 3-3. 전처리 실행

가장 쉬운 방법은 `scripts/cache_womd.sh`를 split별로 돌리는 것이다.

기본값은 위 자동 감지 규칙을 따른다. split만 넘기면 된다.

```bash
bash scripts/cache_womd.sh training
bash scripts/cache_womd.sh validation
bash scripts/cache_womd.sh testing
```

worker 수를 바꾸고 싶으면 환경 변수만 주면 된다.

```bash
NUM_WORKERS=12 bash scripts/cache_womd.sh training
```

### 3-4. 전처리 후 확인할 것

아래 경로가 채워졌는지 확인한다.

- `${paths.cache_root}/training`
- `${paths.cache_root}/validation`
- `${paths.cache_root}/testing`
- `${paths.cache_root}/validation_tfrecords_splitted`

학습과 검증은 pickle 캐시를 읽고, WOSAC metric 계산 쪽은 validation tfrecord split 경로를 같이 쓴다.

---

## 4. wandb 설정

이 버전은 `test_new` 브랜치의 개인 wandb 설정을 그대로 따른다.

기본값은 아래다.

- `entity: jksg01019-naver-labs`
- `project: SMART-FLOW`

기본값 그대로 쓰려면 아무 것도 안 해도 된다.

직접 바꾸고 싶으면 실행 전에 환경 변수만 넣으면 된다.

```bash
export WANDB_ENTITY=jksg01019-naver-labs
export WANDB_PROJECT=SMART-FLOW
```

로그는 task 이름 아래로 저장된다.

---

## 5. SMART-flow 7M pre-BC 학습

### 5-1. 기본 학습 설정

H100 6장 기준 기본값은 아래로 맞춰 두었다.

- `precision: bf16-mixed`
- `lr: 5e-4`
- `max_epochs: 64`
- `train_batch_size: 12 per GPU`
- `val_batch_size: 4`
- `test_batch_size: 4`
- `num_workers: 10`
- `accumulate_grad_batches: 1`

메모리가 빠듯하면 가장 먼저 할 일은 **train batch를 12에서 10으로 낮추는 것**이다.

### 5-2. 6x H100에서 학습 시작

H100 서버에서는 아래 한 줄이 기본 실행법이다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train.sh
```

이 스크립트는 visible GPU 수를 읽어서 자동으로:

- `torchrun --nproc_per_node=6`
- `trainer=ddp`
- `precision=bf16-mixed`
- `train_batch_size=12 per GPU`
- `val_batch_size=4`
- `test_batch_size=4`
- `data.num_workers=10`

으로 맞춘다.

이미 `/workspace/womd_v1_3/SMART_cache` 아래에 캐시가 있으면 그 캐시를 그대로 쓰고, 없을 때만 raw WOMD를 찾아서 `training`과 `validation` 전처리를 자동으로 수행한다.

### 5-3. train batch를 10으로 낮추고 싶을 때

아래처럼 override를 추가하면 된다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train.sh \
  task_name=smart_flow_7m_pre_bc_bs10 \
  data.train_batch_size=10
```

### 5-4. 중간부터 이어서 학습할 때

`ckpt_path`를 넣으면 된다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train.sh \
  task_name=smart_flow_7m_pre_bc_resume \
  ckpt_path=/path/to/last.ckpt
```

### 5-5. 출력 위치

Hydra 출력 폴더 아래에 체크포인트와 로그가 모인다.

기본 로그 루트는 `configs/paths/default.yaml` 기준으로 `logs/` 아래다.

---

## 6. 로컬 검증

### 6-1. 기본 로컬 검증

체크포인트 경로를 `configs/experiment/local_val.yaml`의 `ckpt_path`에 넣는다.

```yaml
ckpt_path: YOUR_MODEL.ckpt
```

그 다음 실행한다.

```bash
bash scripts/local_val.sh
```

기본 설정은 아래다.

- `val_open_loop: false`
- `val_closed_loop: true`
- `n_rollout_closed_val: 32`
- `sample_steps: 4`
- `sample_method: euler`

### 6-2. 빠른 smoke test

로컬에서 가볍게만 보고 싶으면 rollout 수를 줄이면 된다.

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.n_rollout_closed_val=4 \
  trainer.devices=1 \
  trainer.strategy=auto
```

### 6-3. open-loop loss까지 같이 보고 싶을 때

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.val_open_loop=true
```

---

## 7. WOSAC submission 파일 만들기

### 7-1. submission 메타 정보 수정

`configs/experiment/wosac_sub.yaml` 안의 아래 항목을 실제 값으로 바꾼다.

```yaml
authors: [Anonymous]
affiliation: YOUR_AFFILIATION
description: YOUR_DESCRIPTION
method_link: YOUR_METHOD_LINK
account_name: YOUR_ACCOUNT_NAME
```

### 7-2. validation split submission 만들기

```bash
bash scripts/wosac_sub.sh
```

기본 스크립트는 `ACTION=validate`로 되어 있다.

### 7-3. test split submission 만들기

`scripts/wosac_sub.sh` 안의 값을 바꾸거나, 직접 아래처럼 실행한다.

```bash
python -m src.run \
  experiment=wosac_sub \
  action=test \
  ckpt_path=/path/to/model.ckpt \
  task_name=smart_flow_7m_wosac_test
```

### 7-4. 업로드 파일 위치

실행이 끝나면 `logs/` 아래 Hydra 출력 폴더 안에 `wosac_submission.tar.gz`가 생성된다.

이 파일을 Waymo WOSAC leaderboard 제출 페이지에 업로드하면 된다.

---

## 8. 시각화

이 repo는 validation 단계에서 자동으로 rollout 비디오를 저장할 수 있다.

### 8-1. 비디오 저장 켜기

아래 값을 0보다 크게 주면 된다.

- `model.model_config.n_vis_batch`
- `model.model_config.n_vis_scenario`
- `model.model_config.n_vis_rollout`

예시는 아래다.

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=2 \
  model.model_config.n_vis_rollout=4 \
  trainer.devices=1 \
  trainer.strategy=auto
```

### 8-2. 저장 위치

비디오는 각 실행 폴더 아래 `videos/`에 저장된다.

예시:

```text
logs/.../videos/batch_00-scenario_00/
```

wandb를 켜 둔 경우, 저장된 비디오는 logger를 통해 같이 올라간다.

---

## 9. 자주 바꾸는 설정

### 9-1. solver step 수 바꾸기

기본은 4 step이다.

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.validation_rollout_sampling.sample_steps=6
```

### 9-2. heun으로 바꾸기

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.validation_rollout_sampling.sample_method=heun
```

### 9-3. rollout 수 바꾸기

```bash
python -m src.run \
  experiment=local_val \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  model.model_config.n_rollout_closed_val=64
```

---

## 10. 이 브랜치에서 의도적으로 제외한 것

이 버전은 아래 범위를 일부러 제외했다.

- GMM 기반 ego policy
- CAT-K fine-tuning
- token overlap loss
- token classification loss

즉, 이 버전은 **SMART-flow 7M pre-BC + closed-loop validation + WOSAC submission**만 유지한다.

---

## 11. 추천 실행 순서

처음 보는 사람이 그대로 따라 하려면 아래 순서로 하면 된다.

1. conda 환경 생성 및 패키지 설치
2. Waymo dataset 다운로드
3. `configs/paths/default.yaml`의 `cache_root` 확인
4. `scripts/cache_womd.sh`로 `training / validation / testing` 캐시 생성
5. `bash scripts/train.sh`로 pre-BC 학습
6. `bash scripts/local_val.sh`로 로컬 검증
7. 시각화가 필요하면 `n_vis_*` 값을 켜서 다시 검증
8. `bash scripts/wosac_sub.sh`로 submission 파일 생성
9. `wosac_submission.tar.gz` 업로드

---

## 12. 참고

이 버전은 원래 `brand_new`의 SMART-tiny 7M 구조를 바탕으로 하며,
네트워크 폭과 깊이는 유지하고 head 예산만 flow matching 쪽으로 다시 배치했다.
