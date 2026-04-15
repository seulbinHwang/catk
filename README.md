# CAT-K Flow Matching

이 저장소는 **flow matching 학습/추론/평가 전용**으로 정리된 버전입니다.  
기본 실행 경로와 문서, 스크립트는 모두 `smart_flow` 계열만 사용하며 CrossEntropy 기반 next-token 경로는 제거했습니다.  
현재 closed-loop local 평가와 제출 export는 **WOSAC 2025 / Waymo 2025 Sim Agents 기준**만 사용합니다.

- 기존 SMART의 map/context trunk를 그대로 재사용하고, agent 쪽만 flow decoder로 바꿔 scene-context 품질을 유지합니다.
- `FlowTokenProcessor`는 14-slot context pack과 13개 anchor를 만들되, 
- **context 위치/방향과 flow target 원점은 token-restored 상태가 아니라 실제 coarse 상태**를 사용합니다.
- agent coarse token id는 **마지막 점 1개가 아니라 0.5초 전체 6개 점 사각형 경로**를 기준으로 매칭합니다.
- `trajectory_token_veh/ped/cyc` 임베딩은 마지막 contour 1개 대신 
- **`agent_token_all_*` 전체 chunk(6 x 4 x 2)** 를 그대로 펼쳐 사용합니다.
- `HierarchicalFlowDecoder`와 `FlowODE`가 local normalized future를 직접 복원해 discrete token id보다 trajectory geometry를 더 부드럽게 모델링합니다.
- closed-loop inference는 0.5초씩 commit 하며 `pred_traj_10hz`, `pred_head_10hz`, `pred_z_10hz`를 바로 내보내 2025 Sim Agents rollout proto와 바로 연결됩니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm` 이 기본값이며, 
- 이때 외부로 내보내는 `pred_traj_10hz`, `pred_head_10hz`는 raw FM 출력 그대로 유지합니다.
- `model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk` 를 쓰면 
- `retokenize`로 고른 token의 0.5초 chunk를 **외부 rollout 10Hz 출력에만** 반영합니다.
- 내부 closed-loop context는 계속 실제 FM commit 상태를 유지합니다.
- `model.model_config.decoder.use_stop_motion=true` 를 켜면 current + 0.1/0.2/0.3/0.4/0.5초
  6점 경로를 motion token으로 다시 보고, **stop token** 과 일치하는 agent의 다음 0.5초 chunk를
  완전히 고정합니다. 이 stop gate는 vehicle / pedestrian / bicycle 모두에 적용됩니다.
- 이 stop-motion 토큰 매칭은 **실제 actor box 크기 대신 class별 고정 토큰 박스**를 사용합니다.
  vehicle은 `2.0 x 4.8`, pedestrian은 `1.0 x 1.0`, bicycle은 `1.0 x 2.0` 입니다.
- `model.model_config.decoder.use_lqr=true` 를 켜면 stop gate를 통과한 vehicle / bicycle에만
  curvature-domain LQR + kinematic bicycle commit bridge를 적용합니다.
- closed-loop local 평가는 `SimAgentsMetrics`가 Waymo 공식 2025 scorer를 그대로 호출해 `val_closed/sim_agents_2025/*`와 `val_closed/sim_agents_2025_mean/*`를 기록합니다.
- submission export는 `SimAgentsSubmission`이 2025 submission shard와 `sim_agents_2025_submission.tar.gz`를 생성합니다.
- 설치 시점에 official 2025 scorer와 `traffic_light_violation` 관련 2025 필드가 실제로 있는지 바로 검증합니다.

### Closed-loop Retokenize Rule

- `retokenize` 자체는 **현재 실제 coarse 상태 + 이번 0.5초 raw FM commit 5점**을 합친 6개 점 경로를 기준으로 
- 다음 token id를 다시 고릅니다.
- `pos_window`, `head_window`, `coarse_pos/head`, 그리고 다음 step motion feature는 
- 모두 **token bank 복원값이 아니라 실제 FM commit의 마지막 상태** 기준으로 갱신합니다.
- 기본값 `raw_fm` 에서는 `pred_traj_10hz`, `pred_head_10hz`를 raw FM 출력 그대로 유지합니다. 
- 따라서 WOSAC metric, submission proto, video visualization은 
- post-process된 token endpoint가 아니라 네트워크가 직접 낸 10Hz trajectory를 봅니다.
- `matched_token_chunk` 에서는 같은 6점 경로 매칭으로 고른 token chunk가 외부 rollout에도 반영됩니다. 
- 다만 내부 closed-loop context는 계속 실제 상태를 유지합니다.
- `use_lqr=true` 를 켠 경우에도 `retokenize`와 내부 문맥 갱신은
  항상 실행된 5개 fine 상태를 기준으로 이뤄집니다.
- 같은 모드에서 `matched_token_chunk`를 써도 vehicle / bicycle의 외부 10Hz 출력은
  token chunk로 다시 덮지 않고 실제 실행 chunk를 유지합니다. pedestrian만 기존 방식대로
  token chunk export를 유지합니다.

### Closed-loop LQR / Stop-motion Switches

- `use_stop_motion=true` 이면 **raw FM 2초 미래 중 앞 5점**만 사용해 stop 여부를 먼저 판단합니다.
- 입력은 현재 pose + 0.1/0.2/0.3/0.4/0.5초 pose의 6점 경로이고, 토큰 매칭은 항상 **고정 class box**로 합니다.
- stop token 과 일치하면 다음 0.5초 5점의 `x / y / yaw` 를 현재 상태와 완전히 같게 고정합니다.
  즉, 작은 떨림을 허용하지 않습니다.
- 이 stop gate는 `use_lqr=false` 여도 동작합니다.
- `use_lqr=true` 이면 stop gate를 통과한 vehicle / bicycle에만 제어용 참조를 만들고,
  1초 horizon의 longitudinal / lateral LQR를 0.1초마다 다시 풀어 다음 0.5초 5점을 실제 실행합니다.
- 이때 제어는 steering angle 이 아니라 **curvature-domain kinematic bicycle** 로 수행합니다.
  wheelbase 추정은 쓰지 않습니다.
- LQR 참조 생성은 이제 `과거+미래`를 한 번에 smooth fitting 하지 않습니다.
  먼저 과거 0.5초 실제 history만으로 현재 `speed / accel / curvature / curvature-rate`를 추정하고,
  그다음 미래 FM trajectory로부터 future speed / curvature reference를 따로 만듭니다.
- 이때 speed는 edge 길이 norm이 아니라 **각 edge 이동을 현재 heading의 forward axis에 투영한 signed speed**
  로 추정합니다. 즉, 후진 구간이면 음수 속도를 그대로 유지합니다.
- curvature reference도 같은 signed speed를 기준으로 만들며, 매우 저속일 때만 최소 speed magnitude를 써서
  수치적으로 안정화합니다.
- 이 초기 past history(`rollout_init_fine_*_history`)는 rollout 시작 직전의 raw 10Hz 최근 6개 상태를 씁니다.
  과거 길이가 부족하면 `pos / head`는 맨 앞 상태를 반복해 길이를 맞추고, 부족한 prefix의 `valid`는 `False`로 둬
  LQR 초기 상태 추정이 패딩을 실제 관측으로 오해하지 않게 합니다.
- LQR future reference의 시작 경계조건은 이제 항상 previous+current 2개 edge prefix로 고정합니다.
  즉 현재 `speed / curvature`뿐 아니라 `accel / curvature-rate` 연속성까지 함께 반영합니다.
- `model.model_config.decoder.lqr_commit.clip_longitudinal_command=true/false` 로
  저속 예외 처리 뒤 종방향 목표 가속도 clamp를 켜거나 끌 수 있습니다.
- `model.model_config.decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=true/false` 로
  현재 속도/동역학 한계 기반 횡방향 projection과 조향 지연 뒤 최종 곡률 상태 재-clip을 함께 켜거나 끌 수 있습니다.
- `use_lqr=true` 일 때 vehicle / bicycle speed state는 전진은 기존 `v_max`를 쓰되,
  후진은 차종별 별도 제한을 씁니다. 현재 vehicle은 `-1.5m/s`, bicycle은 `-0.5m/s`까지 허용합니다.
- 저속 예외 처리 자체는 전진/후진 모두 `abs(speed)` 기준으로 판단합니다.
- LQR가 켜져 있어도 pedestrian은 token/raw branch를 유지합니다.
- `matched_token_chunk` 를 써도 vehicle / bicycle이 LQR를 탄 경우 외부 10Hz 출력은
  token chunk가 아니라 **실제로 실행된 5점**을 유지합니다.

예시:

```bash
# 기존 raw FM closed-loop
python train.py \
  model.model_config.decoder.closed_loop_rollout_mode=raw_fm

# stop-motion gate만 사용
python train.py \
  model.model_config.decoder.use_stop_motion=true \
  model.model_config.decoder.use_lqr=false

# stop-motion + vehicle/bicycle LQR bridge 사용
python train.py \
  model.model_config.decoder.use_stop_motion=true \
  model.model_config.decoder.use_lqr=true

# LQR future reference는 previous+current prefix를 항상 사용
python train.py \
  model.model_config.decoder.use_stop_motion=true \
  model.model_config.decoder.use_lqr=true
```


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

python -m pip install --upgrade pip
python -m pip install -r install/requirements.txt
python -m pip install torch_geometric
python -m pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```

`ffmpeg`는 visualization용으로 필요합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Waymo/WOSAC 자동 제출을 서버에서 사용할 경우, headless browser 런타임 라이브러리도 필요합니다.
최소한 Ubuntu 기준으로는 아래 패키지를 권장합니다.

```bash
apt-get update
apt-get install -y \
  libnss3 \
  libnspr4 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxrandr2 \
  libgbm1 \
  libasound2
```

루트 권한이 없으면 conda env 안에서 아래를 먼저 설치해도 됩니다.

```bash
conda install -y -c conda-forge nss nspr
```

W&B를 쓸 경우:

```bash
wandb login
export WANDB_PROJECT=SMART-FLOW
export WANDB_ENTITY=<your_entity>
```

### 2.1 RTX 5090 / SDPA fallback

- `HierarchicalFlowDecoder`의 local attention은 기본적으로 PyTorch SDPA 고속 커널을 그대로 사용합니다.
- 다만 RTX 5090 같은 Blackwell(`sm_120`) 계열에서는 validation 중 `MultiheadAttention`가
  CUDA 커널 오류로 크래시하는 경우가 있어, 이 저장소는 해당 GPU에서만 자동으로 `math` SDPA 경로로
  내려가도록 되어 있습니다.
- 즉, 일반 GPU에서는 기존 고속 경로를 유지하고, 문제가 알려진 GPU/오류 패턴에서만 안전 경로를 사용합니다.
- 별도 실행 옵션 없이 자동으로 적용됩니다.

### 2.2 2025 scorer 관련 주의사항

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
  task_name=flow_semi_continuous_pretrain_h1006
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
  task_name=flow_semi_continuous_pretrain_h1006
```

### 5.1.1 학습 agent 선택을 validation/추론과 같게 맞추기

기본값은 `data.train_use_eval_agent_selection=false` 입니다.

- `false`면 기존과 같습니다. 학습 입력 agent는 ego 기준 150m 안만 남기고, 학습 대상은 ego/예측 특별 대상과 ego 기준 100m 안이면서 미래 유효 길이가 충분한 agent 중 최대 `data.train_max_num`개를 사용합니다.
- `true`면 학습에서도 validation/추론용 transform을 그대로 사용합니다. 따라서 별도의 150m 입력 제한과 `train_mask` / `train_max_num` 제한을 추가하지 않습니다. 이 경우 학습 입력 agent와 학습 대상 anchor가 validation/추론과 같은 기준으로 정해집니다.
- 이 설정은 pretrain, Flow Matching range fine-tuning, DRaFT fine-tuning에 동일하게 적용됩니다.

예시:

```bash
# pretrain에서 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true
```

### 5.2 Validation 주기와 val_open / val_closed 바꾸기

- 학습 중 validation은 `trainer.check_val_every_n_epoch` 마다 실행됩니다.
- `model.model_config.val_open_loop=true/false`로 open-loop validation on/off를 바꿉니다.
- `model.model_config.val_closed_loop=true/false`로 closed-loop validation on/off를 바꿉니다.
- validation 양 자체는 `trainer.limit_val_batches`로 줄이거나 늘릴 수 있습니다.
- `model.model_config.n_rollout_closed_val`는 `val_closed_loop`에서 scene당 몇 번 rollout sampling할지 정합니다. 현재 `pre_bc_flow` 기본값은 `32`입니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm|matched_token_chunk`로 closed-loop에서 실제로 export/score/video에 쓰는 10Hz rollout 표현을 고릅니다. 기본값은 `raw_fm`이며, `matched_token_chunk`도 내부 문맥 상태 자체는 실제 FM commit을 유지합니다.
- `model.model_config.decoder.use_stop_motion=true/false`로 stop-motion gate를 켜거나 끕니다.
- `model.model_config.decoder.use_lqr=true/false`로 vehicle / bicycle용 dynamics-aware feasible commit bridge를 켜거나 끕니다.
- `use_lqr=true`면 2초 미래를 바로 commit하지 않고, 다음 0.5초 commit window만 실제로 실행합니다.
- future speed / curvature reference의 시작 경계조건은 항상
  previous+current prefix를 고정해 accel / curvature-rate 연속성까지 함께 반영합니다.
- LQR reference의 speed는 heading forward axis projection으로 만든 signed speed라서,
  vehicle / bicycle 후진 구간도 음수 속도로 그대로 추정/제어합니다.
- 실제 LQR commit speed clamp의 후진 하한은 차종별로 다르며, 현재 vehicle은 `-1.5m/s`,
  bicycle은 `-0.5m/s`까지 허용합니다.
- `model.model_config.decoder.lqr_commit.clip_longitudinal_command=true/false`는
  저속 예외 처리 뒤 종방향 목표 가속도 clamp만 제어합니다.
- `model.model_config.decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=true/false`는
  횡방향 동역학 projection과 조향 지연 뒤 최종 곡률 상태 재-clip을 함께 제어합니다.
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

# matched token chunk를 실제 closed-loop rollout/video/score 출력에만 사용
... model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

# stop-motion gate 적용
... model.model_config.decoder.use_stop_motion=true

# stop-motion + vehicle / bicycle dynamics-aware feasible commit bridge 적용
... model.model_config.decoder.use_stop_motion=true \
    model.model_config.decoder.use_lqr=true

# use_lqr + matched token chunk를 함께 쓸 때도
# vehicle / bicycle export는 실행된 5점 chunk를 유지하고 pedestrian만 token chunk를 씁니다.
... model.model_config.decoder.use_lqr=true \
    model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

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
  task_name=flow_semi_continuous_pretrain_h1006 \
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
- `model.model_config.vis_ghost_gt=true|false`: rollout 비디오에서 미래 GT agent를 연한 ghost overlay로 같이 그릴지 정합니다. `false`면 `rollout_XX.mp4`에서는 이 연한 GT overlay를 숨기고 sampled rollout만 보입니다. `gt.mp4` 자체는 그대로 저장됩니다.
- `model.model_config.vis_flow_2s_preview=true|false`: rollout 비디오에서 각 0.5초 closed-loop step마다 네트워크가 raw로 생성한 2초 / 20점 future를 overlay로 그릴지 정합니다. `true`면 `rollout_XX.mp4`에서 현재 decision block에 해당하는 raw 20점 궤적이 함께 보입니다.
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
  task_name=flow_semi_continuous_pretrain_h1006 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=2 \
  model.model_config.n_vis_rollout=2 \
  model.model_config.vis_ghost_gt=false \
  model.model_config.vis_flow_2s_preview=true \
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

### 5.6 6x H100에서 Flow Matching 학습 범위를 넓혀 fine-tuning

`configs/experiment/finetune_flow_range.yaml`은
**기존 flow checkpoint를 pure Flow Matching loss로 이어서, 학습 범위만 넓혀 새 fine-tuning run을 시작하는 설정**입니다.

핵심은 `data.train_use_eval_agent_selection=true` 입니다.
이 값이 켜지면 학습에서도 validation/추론과 같은 transform을 그대로 써서
기존 학습 경로의 150m 입력 제한과 `train_mask` / `train_max_num` 제한 없이
더 넓은 agent/anchor 범위로 FM loss를 다시 학습합니다.

가장 단순한 6 GPU 실행 예시는 아래와 같습니다.

```bash
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=finetune_flow_range \
  action=finetune \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_range_finetune_h1006
```

중요한 차이:

- 이 경로는 `experiment=pre_bc_flow` + `data.train_use_eval_agent_selection=true`를 매번 길게 적지 않도록 묶어둔 preset입니다.
- `model.model_config.draft.enabled=false` 상태라서 DRaFT physics regularizer는 전혀 쓰지 않습니다. 즉, **pure FM fine-tuning** 입니다.
- 첫 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`만 읽고 새 optimizer / lr scheduler / epoch / global step으로 다시 시작합니다.
- 따라서 pretrained checkpoint에서 새 FM fine-tuning run을 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기` 방식대로 `action=fit` + 이 fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 써야 합니다.
- `data.train_use_eval_agent_selection=true`일 때는 `WaymoTargetBuilderVal()`을 학습 transform으로 쓰므로 `data.train_max_num`은 실제로 사용되지 않습니다.

`finetune_flow_range` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `32`
- train batch size: `48`
- val batch size: `16`
- validation 주기: `16` epoch마다
- `data.train_use_eval_agent_selection=true`

메모리 관련 주의:

- 이 fine-tuning은 기존 pretrain보다 한 batch 안에 들어오는 agent 수와 학습 대상 anchor 수가 늘 수 있으므로 GPU memory 사용량이 더 커질 수 있습니다.
- 그래서 6x H100 기본 train batch size를 `26 -> 20`으로 낮춰 둔 preset입니다.
- 그래도 OOM이 나면 가장 먼저 `data.train_batch_size`를 `16`, `12`처럼 더 줄이는 편이 안전합니다.

자주 바꾸는 override 예시는 아래와 같습니다.

```bash
# 메모리가 빠듯하면 batch를 더 줄이기
... data.train_batch_size=16

# fine-tuning learning rate를 더 낮추기
... model.model_config.lr=1e-4

# validation을 매 epoch마다 수행
... trainer.check_val_every_n_epoch=1

# 전체 validation set으로 보기
... trainer.limit_val_batches=1.0
```

학습 범위를 "validation/추론과 완전히 같은 기준"으로 넓히는 것이 아니라,
기존 train 규칙 안에서 학습 대상 수만 늘리고 싶다면 아래처럼 하면 됩니다.

```bash
... data.train_use_eval_agent_selection=false data.train_max_num=48
```

다만 이 경우에도 150m 입력 제한과 ego 기준 100m 학습 대상 제한은 그대로 남습니다.

### 5.7 6x H100에서 DRaFT fine-tuning

`configs/experiment/finetune_draft_flow.yaml`을 쓰면
**기존 flow checkpoint 위에 DRaFT physics penalty를 얹는 fine-tuning**을 바로 시작할 수 있습니다.
LQR DRaFT를 바로 쓰고 싶다면 [`configs/experiment/finetune_lqr_draft_flow.yaml`](/Users/user/PycharmProjects/catk/configs/experiment/finetune_lqr_draft_flow.yaml)을 사용하면 됩니다.
이 경로는 pretrain을 이어서 resume하는 용도가 아니라,
**이미 학습된 checkpoint의 weight만 읽어서 새 fine-tuning run을 시작하는 용도**입니다.

가장 단순한 6 GPU 실행 예시는 아래와 같습니다.

```bash
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun   --standalone   --nproc_per_node=6   -m src.run   experiment=finetune_draft_flow   action=finetune   trainer=ddp   trainer.devices=6   paths.cache_root="$CACHE_ROOT"   ckpt_path="$PRETRAIN_CKPT"   task_name=flow_semi_continuous_finetune_h1006
```

LQR DRaFT를 바로 시작할 때는 experiment만 바꿔 아래처럼 실행하면 됩니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun   --standalone   --nproc_per_node=6   -m src.run   experiment=finetune_lqr_draft_flow   action=finetune   trainer=ddp   trainer.devices=6   paths.cache_root="$CACHE_ROOT"   ckpt_path="$PRETRAIN_CKPT"   task_name=flow_semi_continuous_lqr_finetune_h1006
```

중요한 차이:

- 첫 fine-tuning 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`를 `strict=False`로 읽은 뒤 `trainer.fit(...)`을 새로 시작합니다.
- 즉, optimizer / lr scheduler / epoch / global step은 이어받지 않습니다.
- 반대로 `action=fit`에 `ckpt_path=...`를 주면 **resume training**으로 동작합니다. 이 경우 이전 run의 optimizer 상태까지 이어받습니다.
- 따라서 pretrained checkpoint에서 fine-tuning을 처음 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기`
- 방식대로 `action=fit` + fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 쓰면 됩니다.

fine-tuning에서 실제로 trainable인 모듈은 아래와 같습니다.

- 기본적으로 encoder 전체를 먼저 freeze합니다.
- 그 다음 `agent_encoder.flow_decoder.step_refiner`만 unfreeze합니다.
- 추가로 `agent_encoder.flow_decoder.velocity_head`만 unfreeze합니다.
- 즉 fine-tuning에서는 map encoder, agent embedding, attention layers는 그대로 frozen 상태를 유지합니다.

`finetune_draft_flow` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `32`
- train batch size: `48`
- val batch size: `16`
- validation 주기: `16` epoch마다
- `model.model_config.draft.penalty_type=physics` 가 기본값입니다.

LQR DRaFT를 바로 시작하려면 `finetune_lqr_draft_flow`를 쓰면 됩니다. 이 설정은 위 physics 기본값에 더해 아래를 미리 켜 둡니다.

- `model.model_config.draft.penalty_type=lqr`
- `model.model_config.decoder.use_lqr=true`
- `model.model_config.decoder.use_stop_motion=false`
- `model.model_config.decoder.lqr_commit.clip_longitudinal_command=false`
- `model.model_config.decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=false`

`configs/experiment/finetune_lqr_draft_flow.yaml`은 이 값을 위처럼 미리 덮어쓰도록 만들어 두었습니다.
다만 필요하면 이 조건과 다르게 둬도 `draft.penalty_type=lqr` 자체는 에러 없이 사용할 수 있습니다.

loss와 로그는 아래처럼 보면 됩니다.

- `train/loss`는 최종 학습 loss입니다.
- `train/loss_fm`는 원래 flow matching loss입니다.
- `train/draft_weight`는 `start_epoch` 이후 `ramp_epochs` 동안 선형으로 증가해 `max_weight`까지 올라갑니다.
- `draft.penalty_type=physics`이면 기존과 동일하게 `train/loss_phys`를 기록합니다.
- physics 모드의 최종 합성은 `train/loss = train/loss_fm + train/draft_weight * 0.005 * train/loss_phys` 입니다.
- `draft.penalty_type=lqr`이면 `train/loss_lqr_exec`를 기록합니다.
- lqr 모드의 최종 합성은 `train/loss = train/loss_fm + train/draft_weight * train/loss_lqr_exec` 입니다.
- lqr penalty는 실행된 첫 0.5초 5개 점을 현재 flow target과 같은 local normalized `[x, y, cos, sin]` 표현으로 바꾼 뒤,
  **가중치 없는 MSE**로 GT 첫 0.5초와 비교합니다.
- lqr 모드의 세부 로그는 `draft_lqr/commit_mse`, `draft_lqr/commit_pos_ade_m`, `draft_lqr/commit_pos_fde_m`,
  `draft_lqr/commit_yaw_ade_deg`, `draft_lqr/commit_yaw_fde_deg`, `draft_lqr/active_anchor_count` 입니다.
- physics 모드는 기존과 같이 `raw_feaisble_gap/*`, `gt_feasible_gap/*` 세부 지표를 유지합니다.
- 기본 구현은 trainer가 `bf16-mixed`여도 physics / lqr penalty 계산 구간만 fp32 subregion에서 수행할 수 있습니다.
  physics는 `model.model_config.draft.physics.force_fp32=true/false`,
  lqr는 `model.model_config.draft.lqr.force_fp32=true/false` 로 조절합니다.

자주 바꾸는 override 예시는 아래와 같습니다.

```bash
# fine-tuning에서도 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true

# physics penalty 유지
... model.model_config.draft.penalty_type=physics

# lqr penalty를 바로 쓰려면 experiment만 바꾸기
... experiment=finetune_lqr_draft_flow

# physics experiment 위에서 lqr penalty/runtime 경로를 직접 맞추기
... model.model_config.draft.penalty_type=lqr \
    model.model_config.decoder.use_lqr=true \
    model.model_config.decoder.use_stop_motion=false \
    model.model_config.decoder.lqr_commit.clip_longitudinal_command=false \
    model.model_config.decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=false

# penalty 가중치를 더 강하게
... model.model_config.draft.max_weight=0.1

# ramp를 2 epoch 동안만 수행
... model.model_config.draft.ramp_epochs=2

# physics 모드에서 GT보다 더 나쁜 만큼만 벌주지 않고 절대 penalty 자체를 사용
... model.model_config.draft.gt_excess_only=false

# physics penalty도 mixed precision으로 그대로 계산
... model.model_config.draft.physics.force_fp32=false

# lqr penalty도 mixed precision으로 그대로 계산
... model.model_config.draft.lqr.force_fp32=false

# 샘플러 역전파를 마지막 2 step에만 남겨 메모리 사용량 줄이기
... model.model_config.draft.sampling.backprop_last_k=2

# validation을 매 epoch마다 수행
... trainer.check_val_every_n_epoch=1
```

checkpoint 선택은 보통 아래처럼 하면 됩니다.

- pretrain run의 best 성능 checkpoint를 쓰려면 `epoch_XXX.ckpt`
- 가장 마지막 저장 상태를 쓰려면 `last.ckpt`
- validation 직전까지 포함한 가장 최근 train epoch 상태를 쓰려면 `epoch_last.ckpt`

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

`configs/experiment/sim_agents_sub_flow.yaml`은 **Waymo/WOSAC에 올릴 제출 파일을 만드는 설정**입니다.
점수를 계산하는 설정이 아니라, 최종 제출용 `tar.gz`를 만드는 설정이라고 생각하면 됩니다.

헷갈리기 쉬운 차이는 아래처럼 보면 됩니다.

- `local_val_flow`: validation 점수를 보고 싶을 때
- `sim_agents_sub_flow`: 제출 파일을 만들고 싶을 때
- `action=validate`: validation split으로 제출 형식이 잘 나오는지 미리 확인할 때
- `action=test`: test split으로 최종 제출 파일을 만들 때

`sim_agents_sub_flow`는 기본적으로 아래처럼 동작합니다.

- 제출 파일 생성 모드로 실행됩니다.
- 로컬 점수는 계산하지 않습니다.
- validation/test split 전체를 읽도록 기본값이 잡혀 있습니다.

실행 전에 아래 값은 꼭 채워 주세요.

- `ckpt_path`
- `model.model_config.sim_agents_submission.method_name`
- `model.model_config.sim_agents_submission.authors`
- `model.model_config.sim_agents_submission.affiliation`
- `submission.description` 또는 `model.model_config.sim_agents_submission.description`
- `model.model_config.sim_agents_submission.method_link`
- `model.model_config.sim_agents_submission.account_name`

`ckpt_path`에는 보통 아래 중 하나를 넣으면 됩니다.

- 가장 최근 학습 상태를 쓰려면 `last.ckpt` 또는 `epoch_last.ckpt`
- 가장 성능이 좋았던 checkpoint를 쓰려면 `epoch_XXX.ckpt`

### 7.1 validation split으로 제출 형식 먼저 확인하기

`action=validate`는 validation 데이터를 읽어서 제출 파일이 잘 만들어지는지 확인하는 용도입니다.
점수를 계산하는 명령은 아니므로, validation 점수도 함께 보고 싶다면 `local_val_flow`를 따로 한 번 더 실행해야 합니다.

빠르게 1 GPU로 형식만 확인하고 싶다면:

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
  submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

### 7.2 validation split 전체를 6 GPU로 제출 파일 만들기

validation split 전체를 6 GPU로 나눠서 빠르게 처리하고 싶다면 아래 명령을 쓰면 됩니다.
실행이 끝나면 validation 기준 제출 파일 `tar.gz`가 만들어집니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_val_ddp6_step_16 \
  trainer.limit_val_batches=1.0 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME" \
  paths.log_dir=/workspace/exp_logs
```

이 명령에서 중요한 옵션만 보면 아래와 같습니다.

- `action=validate`: validation split을 사용합니다.
- `trainer=ddp`, `trainer.devices=6`: GPU 6장을 함께 사용합니다.
- `trainer.limit_val_batches=1.0`: validation split 전체를 끝까지 읽습니다.
- `model.model_config.val_open_loop=false`: open-loop 계산은 생략합니다.
- `model.model_config.val_closed_loop=true`: 제출 파일 생성에 필요한 closed-loop rollout은 유지합니다.
- `paths.log_dir=/workspace/exp_logs`: 로그를 저장할 위치입니다.

### 7.3 test split으로 최종 제출 파일 만들기

실제로 Waymo/WOSAC에 올릴 test split 결과를 만들 때는 `action=test`를 사용합니다.
validation 예시와 비교하면 핵심 차이는 `action=test` 하나입니다.

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
  paths.log_dir=/workspace/exp_logs
```

실행이 끝나면 아래 파일이 생성됩니다.

- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission/`
- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission.tar.gz`

validation export와 test export는 저장 위치와 파일 형식이 같습니다.
차이는 validation 데이터를 읽었는지, test 데이터를 읽었는지만 다릅니다.

알아둘 점:

- `sim_agents_sub_flow`는 제출 파일 생성용이라 로컬 점수는 계산하지 않습니다.
- 점수와 제출 파일이 둘 다 필요하면 `local_val_flow`와 `sim_agents_sub_flow`를 각각 한 번씩 실행해야 합니다.
- 특별한 이유가 없으면 `n_rollout_closed_val=32`는 그대로 두는 편이 안전합니다.
- 메모리가 부족하면 `data.val_batch_size` 또는 `data.test_batch_size`를 `4 -> 2 -> 1` 순서로 줄여 보세요.
- validation split export는 형식 확인용으로 좋고, 실제 업로드는 보통 test split에서 만든 `tar.gz`를 사용합니다.

### 7.4 SSH 서버에서 Waymo 사이트로 자동 업로드

SSH 서버에서도 제출 파일을 만든 뒤 바로 Waymo 사이트에 업로드할 수 있습니다.
다만 Google 로그인은 한 번 필요하므로, **GUI가 있는 PC에서 로그인 상태를 저장한 뒤**
서버에서는 그 JSON 내용을 그대로 붙여넣는 방식으로 쓰는 편이 가장 안전합니다.
같은 파일을 서버 저장소에 오래 남겨 둘 필요는 없습니다.

로그인 상태 파일의 기본 위치는 아래와 같습니다.

```text
secrets/waymo/waymo_storage_state.json
```

이 파일은 로그인된 상태를 그대로 담고 있으므로 비밀번호처럼 조심해서 다뤄야 합니다.
공개 저장소에는 올리지 않는 편이 안전합니다.
현재 `.gitignore`에는 `secrets/waymo/waymo_storage_state.json` 과
`secrets/waymo/playwright_profile/` 이 포함되어 있습니다.

준비:

```bash
python -m pip install -r install/requirements.txt
python -m playwright install chromium
```

환경에 `python` 명령이 없으면 아래 예시의 `python`을 전부 `python3`로 바꿔서 실행하면 됩니다.

1. GUI가 있는 PC에서 로그인 상태를 저장합니다.

```bash
python scripts/waymo_save_storage_state.py --browser-channel chrome
```

기본 저장 위치는 `secrets/waymo/waymo_storage_state.json` 입니다.  
로그인이 잘 안 되면 Playwright 기본 Chromium보다 설치된 Chrome이나 Edge를 쓰는 편이 더 안정적입니다.
그래서 GUI PC에서는 `--browser-channel chrome` 또는 `--browser-channel msedge`를 권장합니다.
이 스크립트는 저장 직전에 Sim Agents 페이지를 다시 확인해서 실제 업로드 폼이 보이는지 검증합니다.
즉, Google 로그인만 된 상태가 아니라 **`Submit to Validation Set` / `Submit to Test Set` 업로드 박스가 실제로 보여야** 저장이 완료됩니다.
Waymo가 `Review rules`를 보여주면 그 자리에서 약관 동의를 한 번 마친 뒤 다시 저장해야 합니다.
스크립트는 SSH/headless 업로드에 필요한 `waymo.com`의 localStorage
(`datasetChallengeTermsAgreementAccepted=true`)도 함께 `waymo_storage_state.json`에 넣어 둡니다.

추가로 기억할 점:

- 브라우저 프로필은 실행할 때마다 임시로 만들고, 종료하면 정리합니다.
- `--user-data-dir`를 직접 줄 때는 Playwright 전용의 빈 폴더를 쓰는 편이 안전합니다.
- 평소 쓰는 기본 Chrome 프로필 폴더를 그대로 넣는 건 권장하지 않습니다.
- 예전에 만든 프로필을 재사용하다가 브라우저가 바로 꺼지면 `--user-data-dir` 없이 다시 실행해 보세요.
- 서버에 이 파일을 꼭 복사해 둘 필요는 없습니다. 아래 자동 업로드 명령을 실행하면,
  서버에 파일이 없을 때 rank 0 프로세스가 시작 직후 터미널에 JSON 붙여넣기를 요청합니다.
- 서버에도 파일을 두고 싶다면 `waymo_submission.storage_state_path` 경로에 배치하면 되고,
  그 경우에는 붙여넣기 프롬프트 없이 기존 파일을 그대로 사용합니다.

2. 서버에서 자동 업로드를 켠 상태로 validation 또는 test를 실행합니다.

validation 예시는 아래와 같습니다.  
서버에 `secrets/waymo/waymo_storage_state.json` 파일이 없으면, 이 명령은 시작 직후
rank 0에서 로컬 파일 내용 전체를 붙여넣으라고 묻습니다. pretty-printed JSON을 그대로 붙여넣고
마지막 `}` 뒤에서 Enter를 한 번 더 치면 검증이 바로 이어집니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_val_ddp6_step_16 \
  trainer.limit_val_batches=1.0 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME" \
  waymo_submission.enabled=true \
  waymo_submission.poll_submission_status=false \
  paths.log_dir=/workspace/exp_logs
```

이때 입력된 JSON은 `/tmp` 아래의 임시 파일로만 저장되고, 프로세스 종료 시 자동으로 삭제됩니다.
즉, 서버 저장소 안에 `waymo_storage_state.json`을 따로 커밋하거나 유지하지 않아도 됩니다.

핵심 옵션은 아래만 기억하면 됩니다.

- `waymo_submission.enabled=true`: 자동 업로드를 켭니다.
- `waymo_submission.storage_state_path`: 로그인 상태 파일 경로입니다. 기본값은 `secrets/waymo/waymo_storage_state.json` 입니다.
  이 파일이 서버에 있으면 그대로 쓰고, 없으면 실행 시작 시 JSON 붙여넣기를 요청합니다.
- `waymo_submission.poll_submission_status=false`: 업로드 후 점수 페이지를 계속 확인하지는 않습니다.

추가 참고:

- validation 실행에서는 `waymo_submission.enabled=true`만 주면 업로드까지 진행됩니다.
- `torchrun` DDP에서도 rank 0만 한 번 입력을 받고, 나머지 rank는 그 입력이 끝날 때까지 대기합니다.
- 서버에서 기본으로 headless Chromium을 사용합니다.
- 서버에 설치된 Chrome을 쓰고 싶으면 `waymo_submission.browser_channel=chrome` 또는 `waymo_submission.browser_executable_path=/path/to/chrome`를 지정하면 됩니다.
- 현재 코드는 Chromium launch 전에 `CONDA_PREFIX/lib`를 자동으로 `LD_LIBRARY_PATH` 앞에 추가하고,
  Playwright bundled browser 외에도 system Chrome과 `~/.cache/ms-playwright/chromium-*/chrome-linux/chrome`
  경로를 자동 탐색해 순서대로 재시도합니다.
- 브라우저가 서버 라이브러리 부족 등으로 launch에 실패하면, 현재 코드는 저장된 `waymo_storage_state.json` 쿠키를 사용해 Waymo 업로드 API로 자동 fallback 합니다.
- 저장한 상태 파일이 불완전하면 업로드 단계에서 `Review rules` 또는 로그인 게이트가 잡히도록 에러 메시지가 분명하게 나옵니다.
  이 경우에는 GUI PC에서 `python scripts/waymo_save_storage_state.py --browser-channel chrome`를 다시 실행하고,
  Sim Agents 페이지에 실제 업로드 폼이 보이는 상태에서 저장한 파일로 교체하면 됩니다.
- 로그인 만료나 페이지 구조 변경으로 실패하면 `logs/<task_name>/runs/<timestamp>/waymo_submission_debug/` 아래에 디버그 파일이 남습니다.
- 점수 페이지까지 자동 확인하고 싶으면 `waymo_submission.poll_submission_status=true`를 줄 수 있지만, UI 변경에 영향을 받을 수 있어 기본값은 `false`입니다.

test 자동 제출은 실수 방지를 위해 기본으로 꺼져 있습니다.
Waymo test set은 계정당 30일에 3번만 제출할 수 있으므로, test 업로드를 할 때는 아래 옵션을 추가로 넣어야 합니다.

```bash
... action=test \
    waymo_submission.enabled=true \
    waymo_submission.submit_test=true
```

즉, `waymo_submission.enabled=true`만으로는 test 제출이 올라가지 않습니다.

## 8. Visualization

학습 중 `val_closed_loop` 비디오 저장 방법은 위 `5.5 val_closed_loop 비디오 저장하기`를 참고하면 됩니다.  
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
  model.model_config.vis_ghost_gt=false \
  model.model_config.vis_flow_2s_preview=true \
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

WOSAC 2025 validation submission export 전:

- `validation/` 캐시 존재
- `validation_tfrecords_splitted/` 존재
- `ckpt_path` 확인
- submission metadata 6개 필드 확인
- `experiment=sim_agents_sub_flow action=validate` 확인
- `trainer.limit_val_batches=1.0` 확인

## 10. 자주 쓰는 명령 모음

### 캐시 생성

```bash
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split training --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split validation --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split testing --num_workers 56
```

### 6x H100 학습

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=pre_bc_flow trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" task_name=flow_semi_continuous_pretrain_h1006
```

### validation 평가

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run experiment=local_val_flow trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_local_val
```

### test submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=sim_agents_sub_flow action=test trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_test
```

### validation submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=sim_agents_sub_flow action=validate trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_val_ddp6 trainer.limit_val_batches=1.0 model.model_config.val_open_loop=false model.model_config.val_closed_loop=true
```
