# CAT-K test_new flow patch

이 패치는 `seulbinHwang/catk`의 `test_new` 브랜치에서 현재 들어가 있는 무거운 flow decoder를 제거하고,
`test2` 브랜치 최신 커밋의 flow 아키텍처를 같은 경로에 그대로 반영하기 위한 파일 묶음입니다.

핵심 목표는 아래 4가지입니다.

1. `test2`의 네트워크 구조를 `test_new`에 그대로 옮긴다.
2. `test_new`의 과한 다중 attention 반복 구조를 제거한다.
3. 기존 `test_new`의 실행 습관은 최대한 유지한다.
4. H100 6장 기준으로 바로 학습할 수 있게 config와 실행 순서를 함께 제공한다.

## 1. 바뀌는 네트워크 구조

이 패치가 반영하는 구조는 아래와 같습니다.

- RoadNet(`map_decoder.py`)은 그대로 둡니다.
- 과거 6개 slot은 기존 SMART agent token embedding을 그대로 씁니다.
- 현재 시점의 연속 상태를 별도 current anchor token으로 추가합니다.
- 2.0초 미래를 `4 x 6 x 4` segment로 나눠 예측합니다.
- flow head는 한 layer 안에서 아래 순서만 사용합니다.
  - future temporal attention
  - history-to-future attention
  - map-to-future attention
  - future agent-to-agent attention
- ODE 적분은 4-step midpoint를 사용합니다.
- short closed-loop fine-tuning은 0.5초씩 4번만 unroll 합니다.
- warm-start inference는 쓰지 않습니다.

즉, `test_new`에서 있던 아래 구조를 제거하는 방향입니다.

- 과거 정보끼리 attention 반복
- map-to-agent 과거 attention 반복
- 과거 agent-to-agent 반복
- flow head에서 미래끼리 / map-to-future / future a2a 를 다시 중복 수행하는 구조

## 2. 어떤 파일을 덮어쓰면 되나

패치 루트에서 아래 명령으로 한 번에 적용할 수 있습니다.

```bash
bash APPLY_PATCH.sh /path/to/catk
```

수동으로 복사하려면 `OVERWRITE_FILES.txt` 목록의 파일을 저장소 같은 경로에 덮어쓰면 됩니다.

## 3. 어떤 파일을 지워야 하나

아래 두 파일은 이제 쓰지 않으므로 지우는 것을 권장합니다.

```text
src/smart/modules/agent_flow_decoder.py
src/smart/metrics/flow_loss.py
```

자동으로 지우려면 역시 `bash APPLY_PATCH.sh /path/to/catk` 를 쓰면 됩니다.

## 4. H100 6장 기준 권장 학습 설정

이 패치의 기본 실험 파일은 아래 4개입니다.

- open-loop pretraining: `configs/experiment/flow_pretrain_h1006.yaml`
- short closed-loop fine-tuning: `configs/experiment/flow_clsft_h1006.yaml`
- local validation: `configs/experiment/flow_local_val.yaml`
- WOSAC submission export: `configs/experiment/flow_wosac_sub.yaml`

권장값은 아래와 같습니다.

### 4-1. Open-loop pretraining

- model: `configs/model/flow_smart.yaml`
- gpu: 6 x H100
- per-device batch: `8`
- grad accumulation: `2`
- effective batch: `96`
- epochs: `64`
- lr: `5e-4`
- precision: `bf16-mixed`
- gradient clip: `0.5`

### 4-2. Short closed-loop fine-tuning

- experiment: `flow_clsft_h1006`
- gpu: 6 x H100
- per-device batch: `4`
- grad accumulation: `2`
- effective batch: `48`
- epochs: `8`
- lr: `5e-5`
- precision: `bf16-mixed`
- gradient clip: `0.5`
- closed-loop unroll: `4`

## 5. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.1`
- `ffmpeg` 설치 완료 상태

```bash
conda create -n catk python=3.11.9 -y
conda activate catk

pip install --upgrade pip
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 6. 데이터 준비

토큰 파일은 저장소에 이미 들어 있다고 가정합니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

Waymo Open Motion Dataset scenario 데이터는 아래처럼 준비합니다.

```text
/path/to/womd/scenario/
├── training/
├── validation/
└── testing/
```

이 README에서는 아래 두 변수를 사용합니다.

```bash
export RAW_ROOT=/path/to/womd/scenario
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

### 6-1. pkl 캐시 생성

```bash
python -m src.data_preprocess   --split training   --num_workers 56   --input_dir "$RAW_ROOT"   --output_dir "$CACHE_ROOT"

python -m src.data_preprocess   --split validation   --num_workers 56   --input_dir "$RAW_ROOT"   --output_dir "$CACHE_ROOT"

python -m src.data_preprocess   --split testing   --num_workers 56   --input_dir "$RAW_ROOT"   --output_dir "$CACHE_ROOT"
```

또는 기존 저장소 스크립트를 그대로 써도 됩니다.

```bash
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh training
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh validation
RAW_ROOT="$RAW_ROOT" CACHE_ROOT="$CACHE_ROOT" NUM_WORKERS=56 bash scripts/cache_womd.sh testing
```

## 7. 실행 순서

## 7-1. 패치 적용

```bash
cd /path/to/patch_root
bash APPLY_PATCH.sh /path/to/catk
```

## 7-2. Open-loop pretraining

```bash
cd /path/to/catk
CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 bash scripts/train_flow_h1006.sh
```

직접 실행은 아래와 같습니다.

```bash
torchrun   --nproc_per_node=6   -m src.run   experiment=flow_pretrain_h1006   trainer.devices=6   paths.cache_root="$CACHE_ROOT"   task_name=flow_pretrain_h1006
```

## 7-3. Short closed-loop fine-tuning

먼저 open-loop checkpoint를 준비합니다.

```bash
export PRETRAIN_CKPT=/absolute/path/to/open_loop/checkpoints/last.ckpt
```

그다음 실행합니다.

```bash
cd /path/to/catk
CACHE_ROOT="$CACHE_ROOT" NPROC_PER_NODE=6 TRAINER_DEVICES=6 bash scripts/finetune_flow_h1006.sh "$PRETRAIN_CKPT"
```

직접 실행은 아래와 같습니다.

```bash
torchrun   --nproc_per_node=6   -m src.run   experiment=flow_clsft_h1006   ckpt_path="$PRETRAIN_CKPT"   trainer.devices=6   paths.cache_root="$CACHE_ROOT"   task_name=flow_clsft_h1006
```

## 7-4. Local validation

```bash
export FT_CKPT=/absolute/path/to/flow_clsft_h1006/checkpoints/last.ckpt
```

```bash
cd /path/to/catk
CACHE_ROOT="$CACHE_ROOT" TRAINER_DEVICES=1 bash scripts/local_val_flow.sh "$FT_CKPT"
```

직접 실행은 아래와 같습니다.

```bash
python -m src.run   experiment=flow_local_val   action=validate   ckpt_path="$FT_CKPT"   trainer=default   trainer.accelerator=gpu   trainer.devices=1   trainer.strategy=auto   paths.cache_root="$CACHE_ROOT"   task_name=flow_local_val
```

## 7-5. mp4 저장

```bash
python -m src.run   experiment=flow_local_val   action=validate   ckpt_path="$FT_CKPT"   trainer=default   trainer.accelerator=gpu   trainer.devices=1   trainer.strategy=auto   trainer.limit_val_batches=1   data.val_batch_size=1   data.num_workers=0   data.pin_memory=false   data.persistent_workers=false   paths.cache_root="$CACHE_ROOT"   model.model_config.n_rollout_closed_val=2   model.model_config.n_batch_wosac_metric=1   model.model_config.n_vis_batch=1   model.model_config.n_vis_scenario=1   model.model_config.n_vis_rollout=2   task_name=flow_local_val_video
```

## 7-6. WOSAC submission 생성

먼저 `configs/experiment/flow_wosac_sub.yaml` 의 아래 값을 실제 값으로 바꿉니다.

- `authors`
- `affiliation`
- `description`
- `method_link`
- `account_name`

validation split 샘플 생성:

```bash
cd /path/to/catk
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" validate
```

test split 최종 submission 생성:

```bash
cd /path/to/catk
CACHE_ROOT="$CACHE_ROOT" bash scripts/wosac_sub_flow.sh "$FT_CKPT" test
```

직접 실행은 아래와 같습니다.

```bash
python -m src.run   experiment=flow_wosac_sub   action=test   ckpt_path="$FT_CKPT"   paths.cache_root="$CACHE_ROOT"   task_name=flow_wosac_sub_test
```

출력 위치:

- shard binproto: `logs/<task_name>/runs/<...>/wosac_submission/`
- 최종 압축 파일: `logs/<task_name>/runs/<...>/wosac_submission.tar.gz`

## 8. 체크포인트 호환성

- 기존 `test_new`의 무거운 flow decoder checkpoint와는 head shape가 달라 strict load가 안 맞는 것이 정상입니다.
- 먼저 `flow_pretrain_h1006` 로 open-loop checkpoint를 새로 만들고,
  그 다음 `flow_clsft_h1006` 에 넣는 순서를 권장합니다.

## 9. 이 패치로 덮어쓴 실제 파일 목록

```text
README.md
configs/model/flow_smart.yaml
configs/experiment/flow_pretrain_h1006.yaml
configs/experiment/flow_clsft_h1006.yaml
configs/experiment/flow_local_val.yaml
configs/experiment/flow_wosac_sub.yaml
scripts/train_flow_h1006.sh
scripts/finetune_flow_h1006.sh
scripts/local_val_flow.sh
scripts/wosac_sub_flow.sh
src/smart/model/smart.py
src/smart/modules/agent_decoder.py
src/smart/modules/smart_decoder.py
src/smart/metrics/__init__.py
src/smart/metrics/flow_matching.py
src/smart/utils/finetune.py
src/smart/utils/flow_traj.py
```
