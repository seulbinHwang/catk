# CAT-K Flow

이 README는 이 저장소를 처음 받는 사람이 Waymo Open Motion Dataset(WOMD) scenario 데이터 준비부터 open-loop 학습, short closed-loop fine-tuning, 추론, 평가, mp4 저장, WOSAC 제출 파일 생성까지 그대로 따라할 수 있게 정리한 실행 가이드입니다.

이 저장소에서 기본으로 사용하는 실험 설정은 아래 4개입니다.

- open-loop pretraining: `configs/experiment/flow_pretrain_h1006.yaml`
- short closed-loop fine-tuning: `configs/experiment/flow_clsft_h1006.yaml`
- local evaluation: `configs/experiment/flow_local_val.yaml`
- WOSAC submission export: `configs/experiment/flow_wosac_sub.yaml`

토큰 파일은 저장소에 이미 포함되어 있으므로 별도로 다운로드할 필요가 없습니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

중요:

- 아래 명령은 이 README를 기준으로 직접 실행하는 방식을 표준으로 설명합니다.
- `scripts/*.sh` 는 conda 경로나 env 이름을 하드코딩하지 않습니다. 현재 쉘에서 이미 활성화된 Python 환경을 그대로 사용합니다.
- 학습/평가 실행 시 캐시 루트는 `paths.cache_root` 로 결정됩니다. 기본값은 `configs/paths/default.yaml` 의 `/scratch/cache/SMART` 이므로, 다른 경로를 쓸 경우 아래 예시처럼 매번 `paths.cache_root=...` 를 넘기는 편이 가장 안전합니다.

## 1. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.1`
- `ffmpeg` 설치 완료 상태

저장소 루트에서 아래 순서로 설치합니다.

```bash
conda create -n catk python=3.11.9 -y
conda activate catk

pip install --upgrade pip
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

`ffmpeg` 는 시스템 패키지로 설치해야 합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

W&B를 온라인으로 쓰지 않을 계획이면 아래 설정을 먼저 해두는 것이 편합니다.

```bash
export WANDB_MODE=offline
```

온라인 로깅을 쓰려면 `configs/logger/wandb.yaml` 의 `entity` 값을 실제 계정으로 바꾸십시오.

### 1.1 W&B 상세 사용법

이 저장소의 W&B 기본 설정은 [configs/logger/wandb.yaml](/home/user/PycharmProjects/catk/configs/logger/wandb.yaml) 에 있습니다.

- logger type: `lightning.pytorch.loggers.wandb.WandbLogger`
- 기본 project: `clsft-catk`
- run name: `task_name` 값
- 기본 resume 정책: `resume: allow`
- 코드에서 `WandbLogger` 사용 시 `watch(model, log="all")` 이 자동 실행됩니다.

또한 실행 종료 시 [src/run.py](/home/user/PycharmProjects/catk/src/run.py) 에서 `wandb.finish()` 를 호출합니다.

온라인 로깅을 정확히 쓰는 순서:

1. W&B 로그인

```bash
wandb login
```

2. 실행 시 온라인 모드 명시

스크립트 실행(권장):

```bash
WANDB_OFFLINE=False WANDB_ENTITY=<your_wandb_entity> CACHE_ROOT="$CACHE_ROOT" bash scripts/train_flow_h1006.sh
```

직접 실행:

```bash
python -m src.run \
  experiment=flow_local_val \
  action=validate \
  ckpt_path=/absolute/path/to/model.ckpt \
  logger.wandb.offline=False \
  logger.wandb.entity=<your_wandb_entity> \
  logger.wandb.project=clsft-catk \
  task_name=flow_local_val_online
```

오프라인 로깅:

- 스크립트 기본값은 `WANDB_OFFLINE=True` 입니다.
- 직접 실행에서는 `logger.wandb.offline=True` 를 넣으면 됩니다.
- 오프라인 런 파일은 각 hydra output dir 아래 `wandb/` 폴더에 저장됩니다.

오프라인 결과를 나중에 업로드:

```bash
wandb sync logs/<task_name>/runs/<YYYY-MM-DD>_<HH-MM-SS>/wandb/offline-run-*
```

중단된 W&B run 재개:

```bash
python -m src.run \
  experiment=flow_pretrain_h1006 \
  action=fit \
  ckpt_path=/absolute/path/to/last.ckpt \
  logger.wandb.offline=False \
  logger.wandb.entity=<your_wandb_entity> \
  logger.wandb.id=<existing_wandb_run_id> \
  logger.wandb.resume=must \
  task_name=flow_pretrain_h1006_resume
```

참고:

- 체크포인트 재개(`ckpt_path`)와 W&B run 재개(`logger.wandb.id`, `logger.wandb.resume`)는 별개입니다. 둘 다 넣어야 학습 상태와 W&B 런이 동시에 이어집니다.
- 평가 시 mp4를 생성하면 모델 코드에서 `self.logger.log_video(...)` 를 호출해 W&B에 비디오를 같이 업로드합니다.

## 2. Waymo 데이터 다운로드

이 저장소는 Waymo Open Motion Dataset의 scenario proto TFRecord를 사용합니다. 공식 다운로드 페이지에서 Motion dataset의 scenario 데이터를 받아 아래 구조로 정리하십시오.

- 공식 다운로드 페이지: `https://waymo.com/open/download`
- Motion dataset 설명: `https://waymo.com/open/data/motion/`

예시 디렉터리 구조:

```text
/path/to/womd/scenario/
├── training/
├── validation/
└── testing/
```

이 README에서는 아래 두 변수를 사용합니다.

```bash
export RAW_ROOT=/path/to/womd/scenario
export CACHE_ROOT=/path/to/SMART_cache
```

`training` 과 `validation` 캐시는 학습과 로컬 평가에 필요합니다. `testing` 캐시는 최종 WOSAC test submission 을 만들 때만 필요합니다.

## 3. 데이터 전처리와 캐시 생성

전처리는 split 별로 한 번씩 실행합니다.

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

동일 작업을 스크립트로 실행하려면:

```bash
RAW_ROOT=/path/to/womd/scenario CACHE_ROOT=/path/to/SMART_cache bash scripts/cache_womd.sh training
RAW_ROOT=/path/to/womd/scenario CACHE_ROOT=/path/to/SMART_cache bash scripts/cache_womd.sh validation
RAW_ROOT=/path/to/womd/scenario CACHE_ROOT=/path/to/SMART_cache bash scripts/cache_womd.sh testing
```

전처리가 끝나면 캐시는 아래처럼 생겨야 합니다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/` 안에는 시나리오별 `.pkl` 캐시가 생깁니다.
- `validation_tfrecords_splitted/` 는 `validation` 전처리 때 자동 생성됩니다.
- 로컬 평가와 mp4 저장은 원본 validation TFRecord 를 시나리오 단위로 다시 읽기 때문에 `validation_tfrecords_splitted/` 가 반드시 있어야 합니다.

## 4. Open-Loop Pretraining

먼저 open-loop pretraining 으로 초기 체크포인트를 만듭니다.

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
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  task_name=flow_pretrain_h1006
```

동일 작업을 스크립트로 실행하려면:

```bash
CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 bash scripts/train_flow_h1006.sh
```

이 설정은 기본적으로 6 GPU 기준입니다. GPU 수가 다르면 `--nproc_per_node`, `trainer.devices`, 그리고 필요시 `data.train_batch_size` / `data.val_batch_size` / `data.test_batch_size` 를 함께 조정하십시오.

출력 위치:

- 로그 루트: `logs/flow_pretrain_h1006/runs/<YYYY-MM-DD>_<HH-MM-SS>/`
- 체크포인트: `logs/flow_pretrain_h1006/runs/<...>/checkpoints/`
- 보통 다음 단계에는 `last.ckpt` 또는 원하는 epoch 체크포인트를 사용합니다.

## 5. Short Closed-Loop Fine-Tuning

open-loop 체크포인트를 만든 뒤 short closed-loop fine-tuning 을 수행합니다.

```bash
export PRETRAIN_CKPT=/absolute/path/to/open_loop/checkpoints/last.ckpt

torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_clsft_h1006 \
  ckpt_path="$PRETRAIN_CKPT" \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  task_name=flow_clsft_h1006
```

동일 작업을 스크립트로 실행하려면:

```bash
CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 bash scripts/finetune_flow_h1006.sh "$PRETRAIN_CKPT"
```

fine-tuning 역시 기본값은 6 GPU 기준입니다. GPU 수가 다르면 `--nproc_per_node`, `trainer.devices`, `data.*_batch_size`, `trainer.accumulate_grad_batches` 를 같이 조정해야 합니다.

fine-tuning 결과 체크포인트는 다음 단계의 추론/평가/제출 생성에 사용합니다.

```bash
export FT_CKPT=/absolute/path/to/flow_clsft_h1006/checkpoints/last.ckpt
```

## 6. 추론

이 저장소에는 별도의 "inference only" 엔트리포인트가 있지 않습니다. 대신 아래 두 방식이 추론 역할을 합니다.

- validation split 에서 rollout 을 생성하면서 지표를 계산하려면 `flow_local_val`
- validation 또는 test split 에서 rollout 을 생성해 WOSAC 제출 파일로 저장하려면 `flow_wosac_sub`

즉, 보통은 아래 7단계와 9단계 명령이 곧 추론 명령입니다.

## 7. 로컬 평가

validation split 에서 closed-loop rollout 을 만들고, ADE 및 WOSAC 계열 지표를 계산합니다.

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
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  task_name=flow_local_val
```

동일 작업을 스크립트로 실행하려면:

```bash
CACHE_ROOT="$CACHE_ROOT" TRAINER_DEVICES=1 bash scripts/local_val_flow.sh "$FT_CKPT"
```

기본 `flow_local_val` 설정은 비디오를 저장하지 않습니다. 대신 다음 항목들을 계산합니다.

- `val_closed/ADE`
- WOSAC realism / kinematic / interactive / map-based metrics

## 8. 평가 결과를 mp4로 저장하기

mp4 저장은 validation 시각화 경로를 켜야만 동작합니다. `ffmpeg` 가 설치되어 있어야 하며, 아래처럼 시각화 관련 옵션을 override 하십시오.

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
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  model.model_config.n_rollout_closed_val=2 \
  model.model_config.n_batch_wosac_metric=1 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=1 \
  model.model_config.n_vis_rollout=2 \
  task_name=flow_local_val_video
```

저장 위치:

- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/gt.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_00.mp4`
- `logs/flow_local_val_video/runs/<...>/videos/batch_00-scenario_00/rollout_01.mp4`

주의:

- `model.model_config.n_vis_rollout` 는 `model.model_config.n_rollout_closed_val` 보다 크면 안 됩니다.
- `model.model_config.n_vis_scenario` 는 `data.val_batch_size` 보다 크면 안 됩니다.
- 평가 기본 설정인 `flow_local_val` 은 `n_vis_batch=0` 이므로, override 를 주지 않으면 mp4 가 저장되지 않습니다.

## 9. WOSAC 제출 파일 생성

제출 파일을 만들기 전에 먼저 `configs/experiment/flow_wosac_sub.yaml` 의 메타데이터를 실제 값으로 수정하십시오.

- `authors`
- `affiliation`
- `description`
- `method_link`
- `account_name`

### 9.1 validation split 으로 submission 샘플 생성

```bash
python -m src.run \
  experiment=flow_wosac_sub \
  action=validate \
  ckpt_path="$FT_CKPT" \
  paths.cache_root="$CACHE_ROOT" \
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  task_name=flow_wosac_sub_validate
```

동일 작업을 스크립트로 실행하려면:

```bash
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" validate
```

### 9.2 test split 으로 최종 submission 생성

```bash
python -m src.run \
  experiment=flow_wosac_sub \
  action=test \
  ckpt_path="$FT_CKPT" \
  paths.cache_root="$CACHE_ROOT" \
  logger.wandb.offline=True \
  logger.wandb.entity=null \
  task_name=flow_wosac_sub_test
```

동일 작업을 스크립트로 실행하려면:

```bash
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" test
```

출력 위치:

- shard binproto: `logs/<task_name>/runs/<...>/wosac_submission/`
- 최종 압축 파일: `logs/<task_name>/runs/<...>/wosac_submission.tar.gz`

## 10. 자주 막히는 지점

- `paths.cache_root` 를 빼먹으면 기본값 `/scratch/cache/SMART` 를 읽습니다.
- `validation_tfrecords_splitted/` 가 없으면 로컬 평가의 WOSAC metric 계산과 mp4 저장이 실패합니다.
- `ffmpeg` 가 없으면 mp4 저장이 실패합니다.
- W&B 온라인 로깅을 쓸 때 `entity` 를 실제 값으로 바꾸지 않으면 실행 초기에 막힐 수 있습니다.

## 11. 최소 실행 순서 요약

처음부터 끝까지 가장 일반적인 순서는 아래입니다.

1. 환경을 만들고 의존성을 설치합니다.
2. WOMD scenario 데이터를 `training/validation/testing` 구조로 내려받습니다.
3. `training`, `validation`, `testing` split 을 캐시로 전처리합니다.
4. `flow_pretrain_h1006` 로 open-loop pretraining 을 수행합니다.
5. pretraining 체크포인트를 넣어 `flow_clsft_h1006` 로 short closed-loop fine-tuning 을 수행합니다.
6. fine-tuned 체크포인트로 `flow_local_val` 을 실행해 validation 지표를 확인합니다.
7. 필요하면 같은 validation 경로에서 시각화 override 를 켜서 mp4 를 저장합니다.
8. 제출이 필요하면 `flow_wosac_sub` 로 validation/test submission 파일을 생성합니다.
