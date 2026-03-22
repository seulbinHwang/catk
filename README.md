# CAT-K Flow Matching

이 저장소는 **flow matching 학습/추론/평가 전용**으로 정리된 버전입니다.  
기본 실행 경로와 문서, 스크립트는 모두 `smart_flow` 계열만 사용하며 CrossEntropy 기반 next-token 경로는 제거했습니다.  
현재 closed-loop local 평가와 제출 export는 **WOSAC 2025 / Waymo 2025 Sim Agents 기준**만 사용합니다.

- 기존 SMART의 map/context trunk를 그대로 재사용하고, agent 쪽만 flow decoder로 바꿔 scene-context 품질을 유지합니다.
- `FlowTokenProcessor`가 14-slot context pack과 13개 anchor를 만들어 2초 미래를 연속값으로 supervision 합니다.
- `HierarchicalFlowDecoder`와 `FlowODE`가 local normalized future를 직접 복원해 discrete token id보다 trajectory geometry를 더 부드럽게 모델링합니다.
- closed-loop inference는 0.5초씩 commit 하며 `pred_traj_10hz`, `pred_head_10hz`, `pred_z_10hz`를 바로 내보내 2025 Sim Agents rollout proto와 바로 연결됩니다.
- closed-loop local 평가는 `SimAgentsMetrics`가 Waymo 공식 2025 scorer를 그대로 호출해 `val_closed/sim_agents_2025/*`와 `val_closed/sim_agents_2025_mean/*`를 기록합니다.
- submission export는 `SimAgentsSubmission`이 2025 submission shard와 `sim_agents_2025_submission.tar.gz`를 생성합니다.
- 설치 시점에 official 2025 scorer와 `traffic_light_violation` 관련 2025 필드가 실제로 있는지 바로 검증합니다.


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
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
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

### 2.1 2025 scorer 관련 주의사항

이 저장소는 시작 시점에 아래를 바로 확인합니다.

- Waymo 공식 2025 Sim Agents scorer를 실제로 불러올 수 있는지
- `traffic_light_violation_likelihood`, `simulated_traffic_light_violation_rate` 같은 2025 전용 필드가 실제 protobuf에 있는지

즉, 예전 Waymo 패키지를 설치하면 validation 시작 전에 명확하게 실패합니다.  
README 기준으로는 `waymo-open-dataset-tf-2-12-0==1.6.7` 이상을 써야 합니다.

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
export CACHE_ROOT=/mnt/nuplan/womd_v1_3/SMART_cache
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
- `validation_tfrecords_splitted/`는 local evaluation, 2025 Sim Agents metric 계산, mp4 visualization에 필요합니다.

### 4.4 Nubes 에서 캐시 다운로드

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

## 5. 6x H100에서 Flow Matching 학습

이 경로의 기본 학습 설정은 `configs/experiment/pre_bc_flow.yaml`입니다.

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

### 5.1 학습 설정을 거칠게 이해하는 법

- 기본 진입점은 `configs/run.yaml`이고, 여기서 `data/model/callbacks/logger/trainer/paths/hydra`를 조합합니다.
- `experiment=pre_bc_flow`는 `configs/experiment/pre_bc_flow.yaml`을 읽어 학습용 하이퍼파라미터를 덮어씁니다.
- `trainer=ddp`는 `configs/trainer/ddp.yaml`을 읽어 DDP 관련 옵션을 덮어씁니다.
- `task_name=...`는 실험 이름이자 저장 폴더 이름입니다. 결과는 대략 `logs/<task_name>/runs/<timestamp>/` 아래에 생깁니다.
- CLI override가 가장 우선입니다. 즉, 같은 파라미터라도 커맨드에 직접 적은 값이 최종 적용됩니다.

예시:

```bash
torchrun ... -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  task_name=flow_pretrain_h1006
```

### 5.2 Validation 주기와 val_open / val_closed 바꾸기

- 학습 중 validation은 `trainer.check_val_every_n_epoch` 마다 실행됩니다.
- `model.model_config.val_open_loop=true/false`로 open-loop validation on/off를 바꿉니다.
- `model.model_config.val_closed_loop=true/false`로 closed-loop validation on/off를 바꿉니다.
- validation 양 자체는 `trainer.limit_val_batches`로 줄이거나 늘릴 수 있습니다.
- `model.model_config.n_rollout_closed_val`는 `val_closed_loop`에서 scene당 몇 번 rollout sampling할지 정합니다. 현재 `pre_bc_flow` 기본값은 `32`입니다.
- `model.model_config.n_batch_sim_agents_metric`는 validation 중 공식 2025 scorer를 실제로 돌릴 앞쪽 batch 수입니다. `smart_flow` 기본값은 `10`, `local_val_flow`는 `100`, `sim_agents_sub_flow`는 `0`입니다.
- `trainer.limit_val_batches`는 validation에 실제로 사용할 batch 양입니다. `0.1`이면 전체 validation batch의 10%, `1.0`이면 전체, 정수 `20`이면 앞 20 batch만 평가합니다.
- `data.val_batch_size`는 validation batch당 scene 수입니다. 키우면 validation은 빨라질 수 있지만 GPU memory 사용량도 같이 늘어납니다.
- 공식 2025 scorer 기준 총 채점 scene 수는 대략 `min(실행한 val batch 수, n_batch_sim_agents_metric) x val_batch_size` 입니다.
- closed-loop rollout 총 수는 대략 `(실행한 val batch 수) x val_batch_size x n_rollout_closed_val` 입니다.

예시:

```bash
# 매 epoch마다 validation
... trainer.check_val_every_n_epoch=1

# 5 epoch마다 validation
... trainer.check_val_every_n_epoch=5

# val_open만 실행
... model.model_config.val_open_loop=true model.model_config.val_closed_loop=false

# val_closed만 실행
... model.model_config.val_open_loop=false model.model_config.val_closed_loop=true

# val_closed에서 scene당 rollout 64회
... model.model_config.n_rollout_closed_val=64

# training validation에서 공식 2025 scorer를 앞 20 batch에만 적용
... model.model_config.n_batch_sim_agents_metric=20

# validation을 전체 val set에 대해 수행
... trainer.limit_val_batches=1.0

# validation batch size를 4 -> 2로 줄이기
... data.val_batch_size=2
```

### 5.3 Checkpoint 저장 규칙 바꾸기

- monitored checkpoint 저장 시도는 validation이 도는 시점에 함께 일어납니다. 현재 `pre_bc_flow`는 `check_val_every_n_epoch=8` 이라 기본적으로 8 epoch마다 평가됩니다.
- 현재 기본 기준은 `callbacks.model_checkpoint.monitor=val_closed/sim_agents_2025/realism_meta_metric`, `mode=max`, `save_top_k=1` 입니다. 즉, `realism_meta_metric`이 가장 높은 checkpoint 1개를 유지합니다.
- 저장 위치는 `callbacks.model_checkpoint.dirpath=${paths.output_dir}/checkpoints` 이고, 실제 경로는 `logs/<task_name>/runs/<timestamp>/checkpoints/` 입니다.
- 파일명 규칙은 `callbacks.model_checkpoint.filename="epoch_{epoch:03d}"` 이라 `epoch_002.ckpt` 같은 이름이 됩니다.
- `save_last=link` 이라 `last.ckpt`도 함께 생기며, 저장된 checkpoint를 가리키는 링크로 유지됩니다.
- 별도로 `callbacks.epoch_last_checkpoint.filename=epoch_last.ckpt` 가 매 train epoch의 마지막 batch 직후 현재 상태를 같은 파일에 덮어써 저장합니다. validation이 있는 epoch에서는 validation 시작 전에 먼저 저장되고, validation이 없는 epoch에서도 최신 epoch 기준 checkpoint 1개를 유지합니다.
- validation 중간에 코드가 죽었으면 같은 `epoch_last.ckpt` 로 재개할 때 해당 epoch의 train loop를 다시 돌지 않고, 완료하지 못한 fit-time validation부터 다시 시작하도록 상태를 함께 기록합니다.
- 기본 `logger=wandb` 설정은 `logger.wandb.log_model=all` 이라 저장되는 checkpoint를 W&B model artifact로도 함께 올립니다. 단, `logger.wandb.offline=True` 이거나 `WANDB_MODE=offline|dryrun|disabled` 면 업로드는 자동으로 꺼지고 로컬 checkpoint만 남습니다.
- `epoch_last.ckpt` 는 별도 W&B artifact(`epoch-last-<run_id>`)로도 업로드되며, alias는 항상 `latest`, `epoch_last` 로 갱신됩니다.

자주 바꾸는 파라미터:

- `callbacks.model_checkpoint.monitor`: 어떤 metric으로 best를 고를지
- `callbacks.model_checkpoint.mode=min|max`: metric이 작을수록 좋은지, 클수록 좋은지
- `callbacks.model_checkpoint.save_top_k`: best checkpoint를 몇 개 남길지
- `callbacks.model_checkpoint.filename`: 저장 파일명 패턴
- `callbacks.model_checkpoint.dirpath`: 저장 폴더
- `callbacks.model_checkpoint.save_last=true|link|false`: `last.ckpt`를 어떻게 둘지

예시:

```bash
# val_open/ADE2s가 가장 낮은 checkpoint 3개 저장
... callbacks.model_checkpoint.monitor=val_open/ADE2s \
    callbacks.model_checkpoint.mode=min \
    callbacks.model_checkpoint.save_top_k=3

# checkpoint 파일명을 바꾸기
... callbacks.model_checkpoint.filename='epoch_{epoch:03d}_step_{step}'
```

### 5.4 중단된 학습 재개하기

- 학습 재개 여부는 `task_name`이 아니라 `ckpt_path`로 결정됩니다. 같은 설정으로 다시 실행하면서 이전 run의 checkpoint만 넘기면 됩니다.
- 이 레포는 `trainer.fit(..., ckpt_path=...)`로 재개하므로 model weight뿐 아니라 optimizer, lr scheduler, epoch, global step도 함께 이어집니다.
- monitored checkpoint 기준으로 재개하려면 `logs/<task_name>/runs/<timestamp>/checkpoints/last.ckpt` 가 가장 단순합니다.
- 정확히 가장 최근 train epoch 상태에서 재개하려면 `logs/<task_name>/runs/<timestamp>/checkpoints/epoch_last.ckpt` 를 쓰면 됩니다.
- 현재 `pre_bc_flow` 기본값은 validation이 `8` epoch마다 돌아 monitored checkpoint는 그 시점에만 갱신되지만, `epoch_last.ckpt` 는 매 epoch train loop가 끝나는 즉시 먼저 갱신됩니다.
- validation 도중 크래시가 난 경우에는 `epoch_last.ckpt` 를 다시 넘기면 그 epoch의 validation부터 먼저 다시 시작한 뒤 다음 epoch 학습으로 넘어갑니다.

예시:

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
  task_name=flow_pretrain_h1006 \
  ckpt_path=/path/to/previous_run/checkpoints/last.ckpt
```

다른 PC에서 재개할 때는 그 PC에서 접근 가능한 checkpoint 경로를 `ckpt_path`로 주고, 그 PC의 캐시 위치에 맞게 `paths.cache_root`만 맞춰주면 됩니다. 새로 실행한 쪽의 output dir은 항상 새 timestamp 폴더로 생기므로 기존 run 폴더를 덮어쓰지 않습니다.

### 5.5 `val_closed_loop` 비디오 저장하기

- `pre_bc_flow` 기본값은 `n_vis_batch=0`, `n_vis_scenario=0`, `n_vis_rollout=0` 이라서 `val_closed_loop`가 돌아도 mp4는 저장하지 않습니다.
- 전제: `model.model_config.val_closed_loop=true`

꼭 필요한 파라미터는 아래와 같습니다.

- `model.model_config.n_vis_batch`: validation에서 비디오를 남길 앞쪽 batch 수. 보통 `1~2`부터 시작합니다.
- `model.model_config.n_vis_scenario`: 각 batch에서 저장할 scenario 수. 보통 `1~2`부터 시작하고, 현재 batch 크기 이하로 두면 됩니다.
- `model.model_config.n_vis_rollout`: 각 scenario에서 저장할 rollout 영상 수. 보통 `1~2`부터 시작하고, `n_rollout_closed_val` 이하로 두면 됩니다.
- `model.model_config.delete_local_videos_after_wandb_upload=true|false`: `wandb`에 비디오를 넘긴 뒤 `logs/.../videos/` 아래 원본 mp4를 지울지 결정합니다. `wandb` logger를 쓰지 않으면 지우지 않습니다.
- 저장 위치는 `logs/<task_name>/runs/<timestamp>/videos/batch_XX-scenario_YY/` 이고, 각 폴더 아래에 `gt.mp4`, `rollout_00.mp4`, `rollout_01.mp4`, ... 형태로 생깁니다. `gt.mp4`는 GT, `rollout_XX.mp4`는 sampled closed-loop rollout입니다. 단, `delete_local_videos_after_wandb_upload=true`면 upload 직후 이 원본 mp4는 자동 삭제될 수 있습니다.
- `logger=wandb` 상태면 생성된 mp4가 W&B에도 같이 기록됩니다. `logger.wandb.offline=True`면 먼저 로컬 `wandb/`에 저장되고, 이후 `wandb sync`로 올리면 됩니다.

예시:

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
  task_name=flow_pretrain_h1006 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=2 \
  model.model_config.n_vis_rollout=2 \
  model.model_config.delete_local_videos_after_wandb_upload=true
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
- `train/ADEyaw2s`
- `train/FDEyaw2s`
- `val_open/ADE2s`
- `val_open/FDE2s`
- `val_closed/sim_agents_2025/*`
- `val_closed/sim_agents_2025_mean/*`
- `val_closed/sim_agents_2025/minADE_best_of_<n_rollout_closed_val>`

추가로 CUDA OOM 위험도 확인용으로 아래 memory metric이 기록됩니다.

- `worst_peak_reserved_pct`: train batch 1개 기준의 실시간 지표입니다. 각 rank가 자기 GPU의 peak reserved memory 비율(%)을 계산한 뒤, rank 간 `max`로 합친 값입니다. 즉, "그 step에서 가장 위험했던 GPU"를 보여줍니다. W&B에는 20 step 간격으로 샘플링되어 기록됩니다.
- `worst_peak_reserved_pct_epoch_max`: 한 epoch 동안 관측된 `worst_peak_reserved_pct`들 중 최대값입니다. OOM 위험 판단은 이 값을 가장 우선해서 보면 됩니다.

해석 기준은 우선 `worst_peak_reserved_pct_epoch_max`에 적용해서 보면 됩니다. 학습 중 실시간 추세를 볼 때는 `worst_peak_reserved_pct`를 같은 기준으로 봐도 되지만, 최종 판단은 `epoch_max` 기준으로 하는 편이 안전합니다.

- `85%` 미만: 대체로 안정적
- `85% ~ 92%`: 여유가 줄어드는 구간
- `92% ~ 96%`: OOM 고위험 구간
- `97%` 이상: batch 구성이나 입력 길이 스파이크에 따라 바로 OOM이 날 수 있음

추가로 epoch마다 아래 W&B 그래프도 갱신됩니다.

- `training_progress_vs_runtime`: x축은 지금까지 누적된 실제 학습 실행 시간(hours), y축은 전체 epoch 기준 진행률(%)입니다. checkpoint로 학습을 이어서 재개한 경우 이전 runtime도 누적해서 그립니다.

## 6. 평가와 추론

### 6.1 Validation set closed-loop 평가

`configs/experiment/local_val_flow.yaml`은 validation split에서 closed-loop rollout을 수행하고, Waymo 공식 2025 Sim Agents metric을 계산합니다.  
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
- `val_closed/sim_agents_2025/*`
- `val_closed/sim_agents_2025_mean/*`
- `val_closed/sim_agents_2025/minADE_best_of_32`

주의:

- `local_val_flow` 기본값은 `trainer.limit_val_batches=60` 이라 빠른 local check용입니다.
- 전체 validation set을 돌리고 싶으면 `trainer.limit_val_batches=1.0` 을 추가하면 됩니다.
- 현재 `local_val_flow`는 `model.model_config.n_batch_sim_agents_metric=100` 이라 실행한 validation batch 전체에 대해 공식 scorer를 돌립니다.

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

## 7. WOSAC 2025 제출 파일 생성

`configs/experiment/sim_agents_sub_flow.yaml`은 WOSAC 2025 / Waymo 2025 Sim Agents submission export용 설정입니다.  
예전 `wosac_sub_flow` 대신 이 config를 사용합니다.  
제출 전 아래 항목은 반드시 채워야 합니다.

- `ckpt_path`
- `model.model_config.sim_agents_submission.method_name`
- `model.model_config.sim_agents_submission.authors`
- `model.model_config.sim_agents_submission.affiliation`
- `model.model_config.sim_agents_submission.description`
- `model.model_config.sim_agents_submission.method_link`
- `model.model_config.sim_agents_submission.account_name`

### 7.1 validation split으로 submission 형식 점검

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
  task_name=flow_sim_agents_validate \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

### 7.2 test split submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=test \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_test \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

출력물:

- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission/`
- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission.tar.gz`

## 8. Visualization

학습 중 `val_closed_loop` 비디오 저장 방법은 위 `5.4 val_closed_loop 비디오 저장하기`를 참고하면 됩니다.  
checkpoint로 validation visualization만 따로 보고 싶으면 아래처럼 `local_val_flow`를 쓰면 됩니다.

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
  model.model_config.n_vis_rollout=5 \
  model.model_config.delete_local_videos_after_wandb_upload=true
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
- Waymo 2025 scorer 환경 확인

WOSAC 2025 test submission 전:

- `testing/` 캐시 존재
- `ckpt_path` 확인
- submission metadata 6개 필드 확인
- `experiment=sim_agents_sub_flow` 확인

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=sim_agents_sub_flow action=test trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_test
```
