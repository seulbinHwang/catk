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
- closed-loop inference에서 `model.model_config.decoder.use_stop_motion=true` 를 켜면 current + 0.1/0.2/0.3/0.4/0.5초
  6점 경로를 motion token으로 다시 보고, **stop token** 과 일치하는 agent의 다음 0.5초 chunk를
  완전히 고정합니다. 이 stop gate는 vehicle / pedestrian / bicycle 모두에 적용됩니다.
- 이 stop-motion 토큰 매칭은 **실제 actor box 크기 대신 class별 고정 토큰 박스**를 사용합니다.
  vehicle은 `2.0 x 4.8`, pedestrian은 `1.0 x 1.0`, bicycle은 `1.0 x 2.0` 입니다.
- `model.model_config.decoder.use_lqr=true` 를 켜면 stop gate를 통과한 vehicle / bicycle에만
  curvature-domain LQR + kinematic bicycle commit bridge를 적용합니다. 이 모드에서는 2초 FM
  미래를 preview로 보되, 실제 반영은 항상 다음 0.5초 / 5점만 실행합니다.
- LQR bridge는 최근 실제 10Hz 6점 history로 현재 speed / yaw-rate / curvature를 잡고,
  `draft_physics.py`의 차종별 속도, 가감속, yaw-rate, 횡가속, 최소 선회 반경 제한을 같이 씁니다.
- wheelbase가 없는 WOMD multi-agent 특성을 고려해 steering angle 대신 **curvature를 제어 입력**
  으로 쓰는 kinematic bicycle 계열 적분을 쓰며, class별 envelope로 곡률과 곡률 변화율을 한 번 더
  clip 합니다.
- DRaFT physics 경로에는 NaN 방지 가드가 들어 있습니다.
- heading 2-vector와 pedestrian velocity 2-vector는 raw `atan2` 대신 safe angle 복원으로 처리해
  `(0, 0)` 또는 near-zero vector backward에서 gradient NaN이 나지 않도록 막습니다.
- `sample_open_loop_future` 결과나 physics loss 출력이 non-finite면 해당 batch의 draft loss를 0으로
  처리해 flow decoder 전체를 오염시키지 않게 합니다.
- 학습 중에는 non-finite parameter, `fm_loss`, `total_loss`, gradient를 fail-fast로 감지해
  NaN checkpoint가 조용히 저장되지 않도록 즉시 중단합니다.
- closed-loop local 평가는 `SimAgentsMetrics`가 Waymo 공식 2025 scorer를 그대로 호출해 `val_closed/sim_agents_2025/*`와 `val_closed/sim_agents_2025_mean/*`를 기록합니다.
- submission export는 `SimAgentsSubmission`이 2025 submission shard와 `sim_agents_2025_submission.tar.gz`를 생성합니다.
- 설치 시점에 official 2025 scorer와 `traffic_light_violation` 관련 2025 필드가 실제로 있는지 바로 검증합니다.


### Dynamic Traffic-Light Staleness Feature

- 교통 신호는 더 이상 정적 map token feature로 섞지 않습니다.
- 현재 관측된 신호 상태는 agent가 주변 lane을 읽는 단계에서만 사용합니다.
- 이때 신호 상태와 함께 `예측 기준 시점 - 신호 관측 시점` 시간 차를 넣습니다.
- 시간 차는 `[-1초, +6초]` 범위로 clip 한 뒤 `6초` 로 나눠 정규화합니다.
- 따라서 모델 입력 의미는 `이 lane은 빨간불이다` 가 아니라 `이 lane은 Δt초 전에 빨간불로 관측됐다` 입니다.
- pretrain과 closed-loop 추론 모두 실제 미래 신호를 입력하지 않고, 현재 관측 신호와 경과 시간만 사용합니다.
- 이 변경은 map encoder의 정적 traffic-light embedding을 제거하고, agent-lane relation 쪽에 동적 traffic-light embedding과 시간 차 scalar를 추가하므로 기존 pretrained checkpoint와 호환되지 않습니다. 새 pretrain을 기준으로 사용합니다.

### Motion Missingness Feature

- flow encoder의 motion feature는 이제 `motion value = 0` 과 별도 `motion_valid` 입력을 함께 씁니다.
- 첫 context step 또는 invalid/valid 경계처럼 이전 coarse motion을 정의할 수 없는 경우에는 motion 값을 `0`으로 두고 `motion_valid=0`으로 표시합니다.
- 실제로 유효한 연속 coarse step에서 정지한 agent는 motion 값이 `0`이어도 `motion_valid=1`이므로 missing motion과 구분됩니다.
- flow 전용 agent-agent relation의 relative motion에도 `rel_motion_valid`를 추가해, 관계 feature 안에서도 알 수 없는 상대 motion과 실제 0 상대 motion을 분리합니다.
- 이 설계는 `x_a_emb`와 flow `r_a2a_emb` 입력 차원을 바꾸므로, 예전 pretrained checkpoint와의 호환 경로는 제공하지 않습니다. 새 pretrain을 기준으로 사용합니다.

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


### Self-forced Sampling Policy

- `model.model_config.self_forced.sampling.random_terminal_step.policy=paper_uniform` 은 기존 동작과 같습니다.
  실행할 denoising step 수를 `min_executed_steps..sample_steps` 범위에서 균등하게 고릅니다.
- `model.model_config.self_forced.sampling.random_terminal_step.policy=all` 은 closed-loop rollout에서
  항상 `model.model_config.self_forced.sampling.sample_steps` 전체 denoising을 실행합니다.
- `policy=all` 일 때 gradient는 마지막 `model.model_config.self_forced.sampling.backprop_last_k` 개
  denoising step에만 남깁니다. 이 값을 생략하면 기본값은 `8` 입니다.

예시:

```bash
... model.model_config.self_forced.sampling.random_terminal_step.policy=all \
    model.model_config.self_forced.sampling.backprop_last_k=8
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
  task_name=flow_semi_continuous_pretrain_h1006
```

`pre_bc_flow` 기본 `data.train_batch_size=28` 는 6x H100 80GB 에서 OOM 없이 throughput 을 최대로 끌어올리도록 실측으로 맞춘 값입니다.

- 측정 조건: 커스텀 Lightning callback 으로 per-step `peak_reserved` 와 `sec/step` 을 DDP 6-GPU 에서 직접 측정.
- `train_batch_size=20` (이전 기본): 500-step 기준 step 당 0.92s, peak reserved 약 49% -> 여유는 많지만 throughput 손해.
- `train_batch_size=28` (현재 기본): 500-step 기준 step 당 1.21s, peak reserved 최대 약 83% -> baseline 대비 epoch 당 약 6.5% 단축 (`H100x6` 기준 64 epoch 환산 약 4시간 절약).
- `train_batch_size=30`: 500-step 기준 peak reserved 약 88% 까지 올라 OOM margin 이 얇습니다.
- `train_batch_size=32`: 실측에서 71 step 만에 OOM 으로 학습이 죽었습니다.
- 따라서 6x H100 80GB 에서는 `28` 이상으로 올리지 않는 것을 권장합니다. 더 작은 GPU 에서는 아래 예시처럼 override 로 낮춰 쓰면 됩니다.

`flow_window_steps=80` 으로 학습할 때는 6x H100 80GB 에서 `data.train_batch_size=14` 를 쓰세요.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1006 \
  model.model_config.decoder.flow_window_steps=80 \
  data.train_batch_size=14
```

`flow_window_steps=80` + 6x H100 80GB 조합은 `configs/experiment/pre_bc_flow_6_h100.yaml` preset으로 묶어 두었습니다. 이 preset은 `flow_window_steps=80`, `data.train_batch_size=18` 고정값을 쓰며, activation checkpointing이 켜져 있는 상태에서 500-step probe로 OOM 없이 안정 (rank 0 peak 약 87%) 인 것을 실측한 값입니다. bs=19/20은 각각 step 121 / step 430에서 OOM 났습니다. 1 epoch 예상 시간은 약 2.2시간 (global batch 108, step당 약 1.76s, epoch 당 ~4509 step).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow_6_h100 \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1006_fw80
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

### 5.1.2 미래 GT 유효 길이 기반 학습 target 선택

학습 target 선택은 아래 단일 옵션으로 고릅니다.

```bash
model.model_config.token_processor.use_prefix_valid_future_loss_mask=false  # 기존 방식
model.model_config.token_processor.use_prefix_valid_future_loss_mask=true   # prefix-valid 방식
```

- `false`이면 기존과 같습니다. 현재 anchor 뒤 `decoder.flow_window_steps` 전체 미래가 모두 유효한 agent-anchor만 학습합니다.
- `true`이면 현재 anchor 뒤 가장 가까운 미래부터 시작해서, 처음 끊기기 전까지 연속으로 유효한 구간만 학습합니다. 이 구간에만 loss가 들어갑니다.
- full-valid sample은 `true`에서도 그대로 전체 미래 loss를 받습니다. 새로 추가되는 것은 partial-valid sample뿐입니다.
- 이 옵션은 `FlowTokenProcessor`에서 학습 target을 만들 때 적용되므로 pretrain, 일반 fine tuning, DRaFT fine tuning, self-forced fine tuning에서 같은 방식으로 동작합니다.
- README 기준 cache를 그대로 만들었다면 cache 재생성은 필요 없습니다. pkl cache 자체에서 partial-valid agent/anchor를 직접 삭제한 경우에만 cache를 다시 만들어야 합니다.

기존 pretrained checkpoint를 prefix-valid 목표로 이어서 학습할 때는 `action=finetune`을 씁니다. 이 방식은 모델 weight만 불러오고 optimizer / scheduler는 새로 시작합니다. 모델 전체를 학습하려면 `model.model_config.finetune.enabled=false`를 유지합니다.

#### H100 4GPU 단일 pod prefix-valid fine tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_h100_4 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_h100_4
```

#### A100 4GPU x 2node prefix-valid fine tuning

각 node에서 같은 command를 실행하되 `--node_rank`만 다르게 둡니다.

```bash
# node 0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=<node0-address> \
  --master_port=29500 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_a100_4x2 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_a100_4x2

# node 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr=<node0-address> \
  --master_port=29500 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_a100_4x2 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_a100_4x2
```

두 preset 모두 `max_epochs=16`, `lr=1e-4`, `lr_warmup_steps=1`, `gradient_clip_val=1.0`, `val_open_loop=true`, `val_closed_loop=true`를 사용합니다. `decoder.flow_window_steps`는 checkpoint와 같은 값을 써야 합니다. 2초 pretrained checkpoint면 기본값 `20`을 그대로 둡니다.

#### testa/testaa A100x4x2 FW30 prefix-valid fine tuning

`flow_semi_continuous_pretrain_fw30_3s_h100x4x2_bs26_self_forcing_w_road_20260505_222745` 계열처럼 `decoder.flow_window_steps=30`으로 학습된 checkpoint는 fine tuning도 같은 horizon으로 실행해야 합니다. `testa`, `testaa` 2개 A100 4GPU pod에서는 아래 launcher를 씁니다.

```bash
python scripts/launch_finetune_flow_prefix_valid_a100x4x2_fw30_static_pods.py --replace
```

이 launcher는 W&B artifact `jksg01019-naver-labs/SMART-FLOW/epoch-last-swkp98ig:v64`에서 `epoch_last.ckpt`를 받아 각 pod의 `/workspace/fw_30_pretrain/epoch_last.ckpt`에 저장한 뒤 시작합니다. `decoder.flow_window_steps=30`, `token_processor.use_prefix_valid_future_loss_mask=true`, `lr=1e-4`, `train_batch_size=26`이 기본값입니다.

CUDA OOM이 발생하면 전체 multi-node job을 정리한 뒤 최신 `epoch_last.ckpt`를 rank 0에서 확정하고 peer pod로 동기화한 다음 `train_batch_size`를 `2`씩 낮춰 재개합니다. 기본 fallback은 `26 -> 24 -> 22 -> ... -> 2`입니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-prefix-valid-a100x4x2-fw30
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t catk-prefix-valid-a100x4x2-fw30
```

### 5.1.3 Kinematic control-space Flow Matching

기본값은 기존 pose-space Flow Matching입니다.

```bash
model.model_config.token_processor.use_kinematic_control_flow=false
```

`true`로 바꾸면 Flow Matching clean target이 기존 `(x, y, cos(yaw), sin(yaw))` pose 표현에서 `[delta_s, delta_n, delta_yaw]` 제어값 표현으로 바뀝니다.

```bash
torchrun ... -m src.run \
  experiment=pre_bc_flow \
  model.model_config.token_processor.use_kinematic_control_flow=true \
  task_name=flow_control_space_pretrain
```

- GT control label은 기존 cache 안의 GT pose에서 batch 생성 시점에 on-the-fly로 만듭니다. 별도 target cache는 필요 없습니다.
- vehicle / cyclist는 `delta_n=0`인 wheelbase-free non-holonomic decoder를 사용하고, pedestrian은 `delta_s`, `delta_n`을 모두 쓰는 holonomic decoder를 사용합니다.
- label 생성은 decoder-consistent rolling projection 방식입니다. 매 step마다 raw GT 현재 pose가 아니라 직전 control을 kinematic decoder에 통과시킨 pose를 다음 inverse의 현재 pose로 씁니다.
- control-space 정규화는 위치 이동량에는 공통 `control_pos_scale_m=1.0`을 쓰고, yaw에는 config로 관리되는 agent type별 scale을 씁니다. 기본 preset은 `control_vehicle_yaw_scale_rad=0.025`, `control_cyclist_yaw_scale_rad=0.06`, `control_pedestrian_yaw_scale_rad=0.20`입니다. control-space target 생성과 복원 경로에는 항상 `agent_type`이 필요합니다. metric/rollout용 pose-space 복원은 기존 규약대로 위치를 `x/20`, `y/20`으로 정규화합니다.
- control-space 학습에서는 GT pose를 control label로 만든 뒤 다시 pose로 복원했을 때, loss에 들어가는 미래 step 기준 최대 위치 오차가 `control_round_trip_max_position_error_m`보다 큰 anchor를 학습에서 제외합니다. 기본값은 `5.0m`이며, 평가 경로에는 적용하지 않습니다.
- 추가 trajectory loss, x0 loss, open-loop draft loss, 속도/가속도/yaw-rate 제약 loss는 이 옵션에서 새로 추가하지 않습니다. 학습 loss는 control-space Flow Matching loss 하나입니다.
- validation / rollout / metric 경로에서는 control 예측을 기존 pose-space 표현으로 복원해 기존 open-loop metric과 closed-loop rollout을 그대로 계산합니다.

pose-space checkpoint와 control-space checkpoint는 Flow decoder 입출력 차원이 다르므로 서로 섞어 resume하지 않는 것을 권장합니다. 기존 pose-space pretrain weight를 control-space 실험의 초기값으로 재사용하려면 Flow decoder head/encoder 차원 차이를 어떻게 처리할지 별도 migration 정책이 필요합니다.

#### hsb-npc-training/hsb-npc-training2 H100x4x2 control-space pretrain

`hsb-npc-training`, `hsb-npc-training2` 두 H100x4 pod를 묶어 control-space Flow Matching pretrain을 돌릴 때는 아래 launcher를 씁니다.

```bash
python scripts/launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py --replace
```

이 launcher는 `configs/experiment/pre_bc_flow_control_2x4_h100.yaml`을 사용합니다. 해당 preset은 `pre_bc_flow_2x4_h100`의 2-node H100 학습 설정을 유지하면서 `pre_bc_flow_control_4_h100.yaml`과 같은 control-space 설정을 켭니다.

```yaml
model:
  model_config:
    token_processor:
      use_kinematic_control_flow: true
      control_pos_scale_m: 1.0
      control_vehicle_yaw_scale_rad: 0.025
      control_pedestrian_yaw_scale_rad: 0.20
      control_cyclist_yaw_scale_rad: 0.06
      control_round_trip_max_position_error_m: 5.0
```

기본 실험 이름은 `flow_control_space_pretrain_h100x4x2_bs26`이고, tmux session 이름은 `catk-control-pretrain-h100x4x2`입니다. 기본 `train_batch_size`는 `26`입니다. CUDA OOM이 발생하면 전체 multi-node job을 정리한 뒤 rank 0의 최신 `epoch_last.ckpt`를 기준 checkpoint로 확정하고 peer pod로 동기화한 다음 `train_batch_size`를 `2`씩 낮춰 재개합니다. 기본 fallback은 `26 -> 24 -> 22 -> ... -> 2`입니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-control-pretrain-h100x4x2
kubectl exec -it -n p-pnc hsb-npc-training2 -c main -- tmux attach -t catk-control-pretrain-h100x4x2
```

실행 전에 실제 kubectl 명령을 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py --dry-run
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py --stop
```

### 5.2 Validation 주기와 val_open / val_closed 바꾸기

- 학습 중 validation은 `trainer.check_val_every_n_epoch` 마다 실행됩니다.
- `model.model_config.val_open_loop=true/false`로 open-loop validation on/off를 바꿉니다.
- `model.model_config.val_closed_loop=true/false`로 closed-loop validation on/off를 바꿉니다.
- validation 양 자체는 `trainer.limit_val_batches`로 줄이거나 늘릴 수 있습니다.
- `model.model_config.n_rollout_closed_val`는 `val_closed_loop`에서 scene당 몇 번 rollout sampling할지 정합니다. 현재 `pre_bc_flow` 기본값은 `32`입니다.
- `model.model_config.decoder.flow_window_steps`는 flow matching이 한 번에 생성하는 10Hz 미래 길이입니다. 기본값은 `20` step, 즉 `2초`입니다.
- `5`의 배수여야 하며 `decoder.num_future_steps`보다 클 수 없습니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm|matched_token_chunk`로 closed-loop에서 실제로 export/score/video에 쓰는 10Hz rollout 표현을 고릅니다. 기본값은 `raw_fm`이며, `matched_token_chunk`도 내부 문맥 상태 자체는 실제 FM commit을 유지합니다.
- `model.model_config.decoder.use_stop_motion=true/false`로 validation / test / submission inference의 stop-motion gate를 켜거나 끕니다. Self-forced 학습 rollout의 stop-motion 여부는 `model.model_config.self_forced.use_stop_motion`으로 별도 제어합니다.
- `model.model_config.decoder.use_lqr=true/false`로 vehicle / bicycle용 curvature-LQR commit
  bridge를 켜거나 끕니다. 기본값은 `false` 입니다.
- `use_lqr=true`면 2초 미래를 바로 commit하지 않고, 다음 0.5초 commit window만 실제로 실행합니다.
- `use_stop_motion=true`면 stop token 과 일치하는 agent 의 다음 0.5초 5점을 현재 상태로 완전 고정합니다.
- `use_lqr=true`는 stop gate를 통과한 vehicle / bicycle 에만 적용됩니다. pedestrian 은 항상
  token / raw branch 를 유지합니다.
- `model.model_config.n_batch_sim_agents_metric`는 validation 중 공식 2025 scorer를 실제로 돌릴 앞쪽 batch 수입니다. `smart_flow` 기본값은 `10`, `local_val_flow`는 `100`, `sim_agents_sub_flow`는 `0`입니다. 단, `model.model_config.scorer_scene_num`이 양의 정수이면 이 값은 validation 시작 시 자동으로 덮어써집니다.
- `model.model_config.scorer_scene_num`는 GPU 개수와 validation batch size가 달라도 공식 scorer에 들어가는 scene 규모를 비슷하게 맞추기 위한 기준값입니다. 기본값은 `960` 입니다. 실제 적용식은 `n_batch_sim_agents_metric = max(1, ceil(ceil(scorer_scene_num / world_size) / val_batch_size))` 입니다. `null` 또는 `0` 으로 두면 자동 덮어쓰기를 끄고 명시한 `n_batch_sim_agents_metric` 값을 그대로 씁니다.
- `trainer.limit_val_batches`는 validation에 실제로 사용할 batch 양입니다. `0.1`이면 전체 validation batch의 10%, `1.0`이면 전체, 정수 `20`이면 앞 20 batch만 평가합니다.
- `data.val_batch_size`는 validation batch당 scene 수입니다. 키우면 validation은 빨라질 수 있지만 GPU memory 사용량도 같이 늘어납니다. `scorer_scene_num` 자동 덮어쓰기가 켜져 있으면 이 값이 `n_batch_sim_agents_metric` 계산식의 분모가 됩니다.
- 공식 2025 scorer 기준 총 채점 scene 수는 `scorer_scene_num`이 켜져 있으면 대략 `n_batch_sim_agents_metric x val_batch_size x world_size` 입니다. batch 단위로만 자르므로 요청값보다 조금 커질 수 있습니다. 끈 경우에는 대략 `min(실행한 val batch 수, n_batch_sim_agents_metric) x val_batch_size x world_size` 입니다.
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

# self-forced 학습 rollout에서만 stop-motion gate 적용
... model.model_config.self_forced.use_stop_motion=true

# stop-motion + vehicle / bicycle curvature-LQR commit bridge 적용
... model.model_config.decoder.use_stop_motion=true \
    model.model_config.decoder.use_lqr=true

# use_lqr + matched token chunk를 함께 쓸 때도
# vehicle / bicycle export는 실행된 5점 chunk를 유지하고 pedestrian만 token chunk를 씁니다.
... model.model_config.decoder.use_lqr=true \
    model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

# training validation에서 공식 2025 scorer를 앞 20 batch에만 적용
# scorer_scene_num 자동 덮어쓰기를 끈 경우에만 의미가 있습니다.
... model.model_config.scorer_scene_num=null \
    model.model_config.n_batch_sim_agents_metric=20

# 공식 2025 scorer 채점 규모를 GPU 수와 무관하게 대략 1920 scene으로 맞추기
... model.model_config.scorer_scene_num=1920

# scorer_scene_num 자동 덮어쓰기 끄기
... model.model_config.scorer_scene_num=null

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
- `model.model_config.vis_flow_preview=true|false`: rollout 비디오에서 각 0.5초 closed-loop step마다 네트워크가 raw로 생성한 future를 overlay로 그릴지 정합니다. 길이는 `model.model_config.decoder.flow_window_steps`를 따릅니다. 기존 `vis_flow_2s_preview`도 호환됩니다.
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
  model.model_config.vis_flow_preview=true \
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

open-loop metric suffix의 `2s`는 기본 horizon 기준이며, `model.model_config.decoder.flow_window_steps`를 바꾸면
`1s`, `1p5s`, `3s` 같은 suffix로 자동 변경됩니다.

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
- `model.model_config.draft.enabled=false` 상태라서 DRaFT inverse feasibility regularizer는 전혀 쓰지 않습니다.
- 즉, **pure FM fine-tuning** 입니다.
- 첫 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`만 읽고 새 optimizer / lr scheduler / epoch / global step으로 다시 시작합니다.
- 따라서 pretrained checkpoint에서 새 FM fine-tuning run을 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기` 방식대로 `action=fit` + 이 fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 써야 합니다.
- `data.train_use_eval_agent_selection=true`일 때는 `WaymoTargetBuilderVal()`을 학습 transform으로 쓰므로 `data.train_max_num`은 실제로 사용되지 않습니다.

`finetune_flow_range` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `16`
- train batch size: `20`
- val batch size: `16`
- validation 주기: `4` epoch마다
- `data.train_use_eval_agent_selection=true`

메모리 관련 주의:

- 이 fine-tuning은 기존 pretrain보다 한 batch 안에 들어오는 agent 수와 학습 대상 anchor 수가 늘 수 있으므로 GPU memory 사용량이 더 커질 수 있습니다.
- 그래서 6x H100 pretrain 기본값(`train_batch_size=28`)보다 보수적으로 `train_batch_size=20`을 쓰는 preset입니다.
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

`configs/experiment/finetune_draft_flow.yaml`을 써서
**기존 flow checkpoint 위에 DRaFT inverse feasibility regularizer를 얹는 fine-tuning**을 바로 시작할 수 있습니다.
이 경로는 pretrain을 이어서 resume하는 용도가 아니라,
**이미 학습된 checkpoint의 weight만 읽어서 새 fine-tuning run을 시작하는 용도**입니다.

가장 단순한 6 GPU 실행 예시는 아래와 같습니다.

```bash
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=finetune_draft_flow \
  action=finetune \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_semi_continuous_finetune_h1006
```

중요한 차이:

- 첫 fine-tuning 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`를 `strict=False`로 읽은 뒤 `trainer.fit(...)`을 새로 시작합니다. 단, 현재 fine-tuning에서 `requires_grad=True` 인 파라미터가 checkpoint에 없으면 실행을 중단합니다.
- 즉, optimizer / lr scheduler / epoch / global step은 이어받지 않습니다.
- 반대로 `action=fit`에 `ckpt_path=...`를 주면 **resume training**으로 동작합니다. 이 경우 이전 run의 optimizer 상태까지 이어받습니다.
- 따라서 pretrained checkpoint에서 fine-tuning을 처음 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기`
- 방식대로 `action=fit` + fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 쓰면 됩니다.

fine-tuning에서 실제로 trainable인 모듈은 아래와 같습니다.

- 기본적으로 encoder 전체를 먼저 freeze합니다.
- `finetune_draft_flow` preset은 `train_full_flow_decoder_only=true`라서
- `agent_encoder.flow_decoder` 전체를 다시 unfreeze합니다.
- 즉 fine-tuning에서는 map encoder, agent embedding, attention layers는 그대로 frozen 상태를 유지하고,
- flow decoder 전체만 trainable 상태로 둡니다.

`finetune_draft_flow` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `32`
- train batch size: `48` per GPU
- effective global train batch size: `288` with 6 GPUs
- val batch size: `16`
- validation 주기: `16` epoch마다

loss와 로그는 아래처럼 보면 됩니다.

- `train/loss`는 최종 학습 loss입니다.
- `train/loss_fm`는 원래 flow matching loss입니다.
- `train/loss_phys`와 `train/loss_if`는 같은 값이고, 새 inverse feasibility penalty `L_if`를 뜻합니다.
- 실제 학습식은 `train/loss = train/loss_fm + train/draft_weight * 0.005 * train/loss_if` 입니다.
- `train/draft_weight`는 `start_epoch` 이후 `ramp_epochs` 동안 선형으로 증가해 `max_weight`까지 올라갑니다.
- 현재 설정은 `max_weight=0.1`이고, 실제 scale `0.005`는 코드에 고정으로 들어갑니다.
- 따라서 기본 설정의 physics loss 최대 가중치는 `0.1 * 0.005 = 0.0005`입니다.
- 기본 구현은 trainer가 `bf16-mixed`여도 inverse feasibility 계산 구간만 fp32 subregion에서 수행합니다.
- DRaFT physics sample은 FM anchor loss용 train-mode forward를 재사용하지 않고,
- 생성 모델을 eval mode로 잠깐 바꾼 상태에서 gradient를 유지한 채 다시 만듭니다.
- 따라서 dropout과 history drop이 섞인 학습용 trajectory가 아니라
- validation/test와 같은 deterministic inference trajectory를 physics loss로 보정합니다.
- 차량 / 자전거는 유효한 미래 예측 구간을 다시
- `forward speed`, `curvature`, `steering angle`, `steering rate`, `forward acceleration`으로 바꿔 penalty를 계산합니다.
- wheelbase는 agent box length에 각각 `0.60`, `0.85`를 곱해서 만듭니다.
- 사람은 steering state를 두지 않고, 2차원 속도와 2차원 가속도만으로 hard / soft 항을 계산합니다.
- heading은 속도가 `0.5 m/s`보다 클 때만 약하게 봅니다.
- `flow_window_steps`가 20이든 30이든 같은 regularizer를 쓰며, 실제 GT 미래가 유효한 step만 loss 평균에 들어갑니다.
- 첫 제어량은 모두 `prev_control`을 사용합니다.
- 차량 / 자전거는 `v_pre`와 `delta_pre`를 복원해서 첫 가속도와 첫 steering rate를 만들고,
- 사람은 `prev_control[..., :2]`를 `prev_control[..., 2]`의 yaw-rate로 현재 anchor-local 좌표계에 회전한 뒤 첫 2차원 가속도 계산에 씁니다.
- hard 항은 속도, 가속도, steering angle, steering rate, lateral acceleration 제한을 넘는 만큼 `relu(z)^2`로 계산합니다.
- `soft_limit_ratio < 1.0`이면 실제 한계를 넘기 전부터 완충 penalty가 시작됩니다.
- `topk_violation_k`는 agent별로 큰 위반 시점도 함께 보게 해서, 짧은 순간의 큰 물리 위반이 전체 평균에 묻히는 것을 줄입니다.
- `commit_loss_weight`는 closed-loop에서 다음 상태로 직접 반영되는 앞 0.5초, 즉 앞 5개 미래 점의 상대 가중치를 높입니다. 전체 scale은 시간축 평균 분모로 다시 정규화합니다.
- `use_slip_penalty=true`이면 차량 / 자전거의 heading 방향과 실제 이동 방향 차이도 hard 항에 포함해 옆미끄러짐을 직접 줄입니다.
- soft 항은 jerk에 가까운 거칠기 값입니다. 기본값에서는 **GT roughness보다 큰 만큼만** loss에 반영하고,
  `model.model_config.draft.physics.compare_softness_to_gt=false` 로 두면
  GT 비교 없이 prediction roughness 자체를 그대로 반영합니다.
- 그래서 `train/loss_phys_raw`와 `train/loss_if_raw`는 GT 비교 전의 raw prediction 기준 값입니다.
- 최종 `L_if`는 agent 전체 평균이 아니라, **batch 안에 실제로 존재하는 class별 평균을 먼저 구한 뒤 다시 class 평균**을 내는 방식입니다.
- 그래서 vehicle이 많아도 pedestrian / bicycle 항이 묻히지 않습니다.
- class별 세부 loss는 `draft_component/*`에 기록됩니다.
- 현재는 `vehicle_hard`, `vehicle_soft`, `vehicle_total`, `bicycle_*`, `pedestrian_hard`, `pedestrian_soft`, `pedestrian_head`, `pedestrian_total`을 봐두면 됩니다.
- 실제 단위 평균값은 `draft_actual_pred/*`, GT 기준값은 `draft_actual_gt/*`에 기록됩니다.
- 현재는 `speed_excess_mps`, `accel_excess_mps2`, `steer_excess_deg`, `steer_rate_excess_degps`, `lat_accel_excess_mps2`, `slip_beta_excess_deg`, `heading_error_deg`를 남깁니다.

현재 inverse feasibility 기본 하이퍼파라미터는 아래와 같습니다.

- 공통: `soft_weight=0.25`
- aggregation: `soft_limit_ratio=0.8`, `topk_violation_k=20`, `commit_loss_weight=2.0`
- no-slip: 기본은 `use_slip_penalty=false`이고, 차량 `beta_max=0.27 rad`, 자전거 `beta_max=0.70 rad`입니다.
- DRaFT loss ablation: `model.model_config.draft.loss_enabled=false`로 같은 fine-tuning preset에서 inverse feasibility loss만 끌 수 있습니다.
- vehicle: `v_max=35.0`, `a_max=8.0`, `a_lat_max=4.2`, `wheelbase_scale=0.60`, `steer_max=0.55 rad`, `steer_rate_max=0.8 rad/s`
- bicycle: `v_max=22.0`, `a_max=5.5`, `a_lat_max=4.4`, `wheelbase_scale=0.85`, `steer_max=0.90 rad`, `steer_rate_max=1.4 rad/s`
- pedestrian: `v_max=5.0`, `a_max=4.7`, `heading_speed_threshold=0.5 m/s`, `heading_weight=0.05`

자주 바꾸는 override 예시는 아래와 같습니다.

```bash
# fine-tuning에서도 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true

# gamma_draft를 더 빨리/강하게 올리기
... model.model_config.draft.max_weight=1.0     model.model_config.draft.ramp_epochs=2

# inverse feasibility도 mixed precision으로 그대로 계산
... model.model_config.draft.physics.force_fp32=false

# soft roughness를 GT와 비교하지 않고 raw prediction 기준으로 사용
... model.model_config.draft.physics.compare_softness_to_gt=false

# inverse feasibility loss만 꺼서 같은 DRaFT fine-tuning 파이프라인에서 ablation
... model.model_config.draft.loss_enabled=false

# 차량/자전거 옆미끄러짐 penalty 켜기
... model.model_config.draft.physics.use_slip_penalty=true

# hard limit 주변 완충 구간과 큰 위반 시점 강조 조정
... model.model_config.draft.physics.soft_limit_ratio=0.9     model.model_config.draft.physics.topk_violation_k=10

# 실제 commit되는 앞 0.5초 구간을 더 강하게 보기
... model.model_config.draft.physics.commit_loss_weight=3.0

# 차량 steering rate 제한을 더 느슨하게
... model.model_config.draft.physics.vehicle_steer_rate_max_radps=1.0

# 사람 heading 항을 더 약하게
... model.model_config.draft.physics.pedestrian_heading_weight=0.02

# 샘플러 역전파를 마지막 2 step에만 남겨 메모리 사용량 줄이기
... model.model_config.draft.sampling.backprop_last_k=2

# validation을 매 epoch마다 수행
... trainer.check_val_every_n_epoch=1
```

checkpoint 선택은 보통 아래처럼 하면 됩니다.

- pretrain run의 best 성능 checkpoint를 쓰려면 `epoch_XXX.ckpt`
- 가장 마지막 저장 상태를 쓰려면 `last.ckpt`
- validation 직전까지 포함한 가장 최근 train epoch 상태를 쓰려면 `epoch_last.ckpt`

### 5.8 4x A100 80GB 에서 DRaFT fine-tuning

6x H100 이 아닌 **4x A100 80GB (SXM4)** 박스에서 같은 DRaFT fine-tuning 을 돌리고 싶을 때 쓰는 별도 preset 입니다.

- preset 파일: `configs/experiment/finetune_draft_flow_a100x4.yaml`
- 자세한 실행 방법 / 하이퍼파라미터 선택 이유 / OOM 디버깅 순서: [`docs/A100x4_finetune_draft_flow_README.md`](docs/A100x4_finetune_draft_flow_README.md)

요약만 보면 아래와 같습니다.

- `train_batch_size=36` (실측 max), `accumulate_grad_batches=2`, `trainer.devices=4` -> effective global batch **`288`** (6xH100 preset `288` 과 정확히 동일, 따라서 lr 도 그대로 `2e-4`).
- `max_epochs(=32)`, `check_val_every_n_epoch(=16)` 은 6xH100 preset 과 동일.
- `val_batch_size=8` 로 줄이고 `n_rollout_closed_val=16` / `n_batch_sim_agents_metric=10` 은 유지해서 정기 eval 이 OOM 없이 돕니다.
- **bs 상한의 원인은 메모리가 아닙니다**. A100 (sm_80) 의 flash / memory-efficient SDPA kernel 이 `ChunkStepRefiner` 의 self-attention 에서 큰 batch 일 때 `invalid configuration argument` 로 터지는 kernel grid-dim 한계입니다. bs=36 일 때 peak 48 GiB / 80 GiB 로 VRAM 은 남아돕니다.
- 위 crash 를 완전히 없애기 위해 **`src/smart/modules/flow_local_decoder.py` 의 `ChunkStepRefiner` self-attention 만 math-SDPA kernel 로 강제하는 소폭 패치**를 포함했습니다. 실측 결과 bs=36 에서 500 step 이상 안정 + step time 도 오히려 약 20% 단축. 상세: [`docs/A100x4_finetune_draft_flow_README.md`](docs/A100x4_finetune_draft_flow_README.md) 5장.
- 같은 이유로 **`HalfSecondChunkMixerBlock` 의 self-attention** 에도 동일한 math-SDPA wrapper 를 적용했습니다. self-forced fine-tuning 처럼 chunk_mixers 의 backward graph 가 살아있는 학습 경로에서, H100 의 fast/mem-efficient SDPA kernel 이 backward 용 placeholder 메모리를 uninitialized 로 두면 saved tensor 가 NaN bit pattern 으로 박혀 grad 를 오염시킬 수 있어 미리 차단합니다.
- 실행 예시:

```bash
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m src.run \
  experiment=finetune_draft_flow_a100x4 \
  action=finetune \
  trainer=ddp \
  trainer.devices=4 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_semi_continuous_finetune_inv_best_a_100_a100x4
```

### 5.8.1 5x V100 32GB x3 static pods에서 DRaFT fine-tuning

`fv`, `fvv`, `fvvv`, `fvvvv`, `fvvvvv` 처럼 **V100 32GB 3장짜리 pod 5개**를 묶어
DRaFT inverse feasibility fine-tuning을 돌릴 때 쓰는 preset과 launcher입니다.

- preset 파일: `configs/experiment/finetune_draft_flow_v100x3x5.yaml`
- launcher 파일: `scripts/launch_finetune_draft_v100x3x5_static_pods.py`
- 총 DDP rank: `5 nodes * 3 GPUs = 15`
- physics 설정: `soft_limit_ratio=0.9`, `topk_violation_k=20`, `commit_loss_weight=1.0`, `use_slip_penalty=true`
- DRaFT weight: `draft.max_weight=0.15`
- 기본 per-GPU `train_batch_size=24`, `val_batch_size=2`

이 launcher는 pod를 만들거나 지우지 않습니다. 이미 떠 있는 pod 안에 `kubectl exec`로 들어가 tmux 세션만 시작하거나 종료합니다.
지금 다른 실험이 돌고 있으면 `--replace`나 `--stop`을 쓰지 마세요.

실행 전 dry-run:

```bash
python scripts/launch_finetune_draft_v100x3x5_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --dry-run
```

실제 시작:

```bash
python scripts/launch_finetune_draft_v100x3x5_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --task-name flow_finetune_draft_v100x3x5_bs24_soft09_topk20_commit1_slip
```

checkpoint가 아직 pod 안에 없고 W&B artifact에서 받아야 한다면 아래처럼 full name을 넘기면 됩니다.

```bash
python scripts/launch_finetune_draft_v100x3x5_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --wandb-artifact entity/project/artifact-name:v1 \
  --artifact-download-dir /workspace/path/to/new_checkpoint/artifact \
  --task-name flow_finetune_draft_v100x3x5_bs24_soft09_topk20_commit1_slip
```

이미 떠 있는 이 launcher의 tmux 세션만 종료:

```bash
python scripts/launch_finetune_draft_v100x3x5_static_pods.py --stop
```

batch size 판단:

- `64 / 80 * 32 = 25.6`은 **A100 80GB에서 per-GPU batch 64가 안정적이었다면, per-GPU 메모리 대부분이 batch에 선형으로 비례한다**고 가정한 1차 추정입니다.
- 이 가정은 DRaFT fine-tuning처럼 activation이 큰 학습에서는 꽤 유용하지만, 완전한 보장은 아닙니다. V100은 A100과 kernel/workspace, fp16/bf16 특성, fragmentation, batch별 agent 수 분산이 다릅니다.
- 그래서 `25.6`을 그대로 쓰는 대신 `24`로 시작하는 판단은 합리적인 보수값입니다. 다만 실측 전에는 “안전한 시작점”이지 “항상 성공 보장값”은 아닙니다.
- OOM이 나면 같은 launcher에 `--train-batch-size 22` 또는 `20`을 넘겨 다시 시작하세요.

### 5.8.2 8x V100 32GB x4 static pods에서 DRaFT fine-tuning

`testsv`, `testsvv`, `testsvvv`, `testsvvvv`, `sv`, `svv`, `svvv`, `svvvv` 처럼
**V100 32GB 4장짜리 pod 8개**를 묶어 새 checkpoint에서 DRaFT inverse feasibility fine-tuning을
시작할 때 쓰는 preset과 launcher입니다.

- preset 파일: `configs/experiment/finetune_draft_flow_v100x4x8.yaml`
- launcher 파일: `scripts/launch_finetune_draft_v100x4x8_static_pods.py`
- 총 DDP rank: `8 nodes * 4 GPUs = 32`
- physics 설정: `soft_limit_ratio=1.0`, `topk_violation_k=20`, `commit_loss_weight=1.0`, `use_slip_penalty=true`
- DRaFT weight: `draft.max_weight=0.15`
- 기본 per-GPU `train_batch_size=24`, `val_batch_size=2`

실행 전 dry-run:

```bash
python scripts/launch_finetune_draft_v100x4x8_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --dry-run
```

실제 시작:

```bash
python scripts/launch_finetune_draft_v100x4x8_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --task-name flow_finetune_draft_v100x4x8_bs24_soft10_topk20_commit1_slip
```

checkpoint가 아직 pod 안에 없고 W&B artifact에서 받아야 한다면 full name을 같이 넘기면 됩니다.
rank 0 pod가 checkpoint를 받은 뒤 나머지 pod들이 같은 파일을 검증하며 동기화합니다.

```bash
python scripts/launch_finetune_draft_v100x4x8_static_pods.py \
  --ckpt-path /workspace/path/to/new_checkpoint/epoch_last.ckpt \
  --wandb-artifact entity/project/artifact-name:v1 \
  --artifact-download-dir /workspace/path/to/new_checkpoint/artifact \
  --task-name flow_finetune_draft_v100x4x8_bs24_soft10_topk20_commit1_slip
```

이미 떠 있는 이 launcher의 tmux 세션만 종료:

```bash
python scripts/launch_finetune_draft_v100x4x8_static_pods.py --stop
```

batch size 판단:

- `64 / 320 * 128 = 25.6`은 A100x4 pod의 총 320 GiB에서 per-GPU batch 64가 완주했고,
  V100x4 pod의 총 128 GiB로 같은 모델/데이터/precision을 옮긴다는 가정의 1차 추정입니다.
- 실제 학습은 GPU별 activation, attention workspace, batch별 agent 수 분산, V100/A100 kernel 차이에 영향을 받으므로
  완전한 보장은 아닙니다.
- 그래도 `25.6`을 그대로 쓰지 않고 `24`로 내리는 판단은 합리적인 보수 시작점입니다.
  OOM이 나면 같은 launcher에 `--train-batch-size 22` 또는 `20`을 넘겨 다시 시작하세요.

### 5.9 4x H100 80GB 에서 Flow Matching pretrain

6x H100 이 아닌 **4x H100 80GB** 박스에서 `pre_bc_flow` 와 동일한 pretrain 을 돌리고 싶을 때 쓰는 별도 preset 입니다.

- preset 파일: `configs/experiment/pre_bc_flow_4_h100.yaml`
- 베이스: `configs/experiment/pre_bc_flow.yaml` (6x H100 preset)

요약만 보면 아래와 같습니다.

- `flow_window_steps=20` 을 preset 자체에서 고정합니다. 이 horizon 에 맞춰 아래 batch size 상한을 실측했기 때문에 모델 default 가 바뀌더라도 4x H100 메모리 프로파일이 유지됩니다.
- `train_batch_size=52` 가 기본값입니다. 커밋 `b12e653` 에서 추가된 `AttentionLayer` activation recomputation 이 기본으로 켜진 상태에서 4x H100 80GB 로 실측한 상한입니다. `trainer.devices=4`, `accumulate_grad_batches=1` -> effective global batch **`208`**.
- `lr=2.667e-4` 는 이전 per-GPU bs=20 (global 80) 기준으로 맞춰둔 값입니다. 새 global batch 208 에 선형 LR scaling rule 을 적용하려면 `model.model_config.lr=6.933e-4` (= `4e-4 * 208/120`) 로 CLI override 하세요. optimizer 동작을 무언 중에 바꾸지 않기 위해 default 는 기존 값을 유지합니다.
- `max_epochs(=64)`, `check_val_every_n_epoch(=8)`, `limit_val_batches(=0.1)`, `val_batch_size(=16)`, `n_rollout_closed_val(=16)` 은 6xH100 preset 과 동일합니다.
- `flow_window_steps=20`, 4x H100 80GB 에서 `AttentionLayer` activation recomputation 이 켜진 상태로 실측한 per-GPU 메모리 수치입니다.
  - `bs=40`: peak reserved 약 80.2%
  - `bs=48`: peak reserved 약 85.3%
  - **`bs=52` (현재 기본값): peak reserved 약 90.1%, 여유 약 10%**
  - `bs=56`: peak reserved 약 94.0% (200 step 안정) - throughput 을 더 짜내고 싶을 때 override 용.
  - `bs=60`: 96.8% 에서 OOM.
- activation recomputation 이 꺼진 이전 코드 (b12e653 이전) 에서는 같은 설정 상한이 약 `bs=28` 이었습니다. 두 경우 모두 동일하게 4x H100 전부를 사용하는 DDP 기준입니다.
- 1 epoch wall-clock 은 steady-state step time 기준 약 `87 min` (bs=52, 0.44 it/s, 약 2342 steps/epoch). 이전 `bs=20` 설정의 약 `95 min` 대비 epoch 당 약 8% 단축되며, recomputation 으로 인해 step time 이 늘지만 batch 가 커지면서 throughput 이 개선됩니다.

실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m src.run \
  experiment=pre_bc_flow_4_h100 \
  trainer=ddp \
  trainer.devices=4 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1004_fw20
```

이전 global batch `120` 을 정확히 유지하고 싶으면 `data.train_batch_size=30` 으로 override 하세요 (4*30=120). 단 bs=30 은 activation recomputation 이 꺼진 상태에서는 peak ~95% 로 위험하므로, b12e653 이후 코드에서만 권장합니다.

```bash
... data.train_batch_size=30
```

### 5.10 H100 4장짜리 pod 2개로 multi-node Flow Matching pretrain

이미 떠 있는 `hsb-npc-training`, `hsb-npc-training2` pod를 그대로 살려 두고, 각 pod의 H100 4장씩 총 8 GPU로 하나의 pretrain run을 돌릴 때 쓰는 경로입니다. **이 경로는 pod를 새로 만들거나 지우거나 재시작하지 않습니다.** launcher는 `kubectl exec`로 기존 running pod 안에 들어가 tmux 세션과 `torchrun`만 시작합니다.

- preset 파일: `configs/experiment/pre_bc_flow_2x4_h100.yaml`
- pod launcher: `scripts/launch_h100x4_multinode_pretrain_tmux.py`
- pod 내부 실행 wrapper: `scripts/h100x4_multinode_pretrain.sh`
- OOM fallback launcher: `scripts/h100x4_multinode_pretrain_with_oom_retry.sh`
- 기본 구성: `NNODES=2`, `NPROC_PER_NODE=4`, `trainer.num_nodes=2`, `trainer.devices=4`
- 기본 pod별 `CACHE_ROOT`: `hsb-npc-training=/mnt/nuplan/womd_v1_3/SMART_cache`, `hsb-npc-training2=/workspace/womd_v1_3/SMART_cache`
- 기본 per-GPU batch: `data.train_batch_size=26`
- 기본 effective global batch: `26 * 8 GPUs = 208`
- 기본 lr: `5e-4` (이전 H100x4x2 `train_batch_size=20` 설정의 lr을 그대로 유지; batch 증가에 대해 추가 선형 scaling 하지 않음)
- 기본 horizon: `flow_window_steps=20`
- validation 중 공식 scorer가 오래 걸려도 DDP가 조기 timeout 나지 않도록 `trainer=ddp`의 process group timeout은 4시간입니다.
- `pre_bc_flow_2x4_h100` preset은 `TQDMProgressBar(refresh_rate=1)`와 `trainer.enable_progress_bar=true`를 명시합니다. launcher 기본 pod 순서에서는 `hsb-npc-training`이 node rank 0/global rank 0이므로, `check_val_every_n_epoch=32`로 fit-time validation이 시작될 때 validation tqdm 진행률은 `hsb-npc-training`의 `catk-h100x4-pretrain` tmux 주 pane에 표시됩니다. `hsb-npc-training2`는 non-zero rank라 같은 progress bar를 중복 출력하지 않는 것이 정상입니다.
- launcher는 각 pod 안에 쓰는 env 파일에 pod별 `CACHE_ROOT`를 따로 기록합니다. 두 pod가 같은 mount path를 공유하는 경우에만 `--cache-root <PATH>`로 전체 override를 쓰고, pod별 경로를 바꿔야 하면 `--pod-cache-root POD=PATH`를 반복해서 넘깁니다.
- 이 기본값은 H100x4x2의 추가 GPU 수와 H100 메모리 여유를 throughput으로 쓰되, optimizer lr은 이전 H100x4x2 설정과 동일하게 유지하는 선택입니다. 이전 보수적 설정과 같은 per-GPU batch를 유지해야 하는 ablation이면 `--train-batch-size 20`을 쓰고, 기존 6xH100과 global batch까지 맞춰야 하는 ablation이면 `--train-batch-size 15 --learning-rate 4e-4`를 쓰세요.

로컬에서 kubectl이 되는 터미널에서 이 repo checkout으로 이동해 아래를 실행하면, master 주소는 `hsb-npc-training`의 Pod IP로 자동 설정되고 두 pod에 같은 tmux session이 만들어집니다. 새 pretrain을 처음부터 시작하는 경로이므로 `--ckpt-path`는 넘기지 않습니다.

```bash
python scripts/launch_h100x4_multinode_pretrain_tmux.py \
  --namespace p-pnc \
  --pods hsb-npc-training hsb-npc-training2 \
  --container main \
  --project-root /mnt/nuplan/projects/catk \
  --branch self_forcing_bugfix \
  --task-name flow_semi_continuous_pretrain_h100x4x2 \
  --replace
```

긴 pretrain을 OOM fallback과 함께 돌릴 때는 아래 shell wrapper를 권장합니다. 첫 시도는 `data.train_batch_size=26`으로 시작하고, attempt 로그에서 `CUDA out of memory` / `OutOfMemoryError` 계열 마커가 잡히면 batch를 `2`씩 낮춘 뒤 같은 `TASK_NAME`의 최신 `epoch_last.ckpt`를 찾아 `ckpt_path`로 넘겨 resume합니다. 기본 `MIN_BS=20`은 이전에 안정적으로 돌린 per-GPU batch 20을 안전 하한으로 둔 값이며, 더 낮게 내려가야 하면 환경변수로 바꿀 수 있습니다. 이 wrapper도 pod를 새로 만들거나 재시작하지 않고, attempt마다 기존 tmux session만 `--replace`로 교체합니다.

```bash
TASK_NAME=flow_semi_continuous_pretrain_h100x4x2_bs26 \
bash scripts/h100x4_multinode_pretrain_with_oom_retry.sh
```

주요 override:

```bash
INITIAL_BS=26 \
OOM_STEP=2 \
MIN_BS=20 \
TASK_NAME=flow_semi_continuous_pretrain_h100x4x2_bs26 \
BRANCH=self_forcing_bugfix \
bash scripts/h100x4_multinode_pretrain_with_oom_retry.sh
```

retry wrapper의 로컬 attempt 로그는 `logs/_h100x4_multinode_pretrain_oom_retry/<TASK_NAME>/attempt_XXX_bsYY.log`에 저장됩니다. resume 기준 checkpoint는 pod 안의 `logs/<TASK_NAME>/runs/*/checkpoints/epoch_last.ckpt` 중 최신 파일입니다. 학습 epoch 중간에 OOM이 나면 마지막으로 저장된 `epoch_last.ckpt`, 즉 보통 마지막 완료 epoch부터 이어가고, validation 직전/중간 OOM이면 validation 직전에 저장된 pending checkpoint부터 이어갑니다.

실행 후 접속:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-h100x4-pretrain
kubectl exec -it -n p-pnc hsb-npc-training2 -c main -- tmux attach -t catk-h100x4-pretrain
```

중단도 pod가 아니라 tmux session만 종료합니다.

```bash
python scripts/launch_h100x4_multinode_pretrain_tmux.py \
  --namespace p-pnc \
  --pods hsb-npc-training hsb-npc-training2 \
  --stop
```

짧은 smoke run은 아래처럼 전체 batch/epoch를 제한해서 rendezvous와 dataloader만 먼저 확인합니다. preset 기본값이 이미 `data.train_batch_size=26`이므로 별도 batch override는 필요 없습니다.

```bash
python scripts/launch_h100x4_multinode_pretrain_tmux.py \
  --namespace p-pnc \
  --pods hsb-npc-training hsb-npc-training2 \
  --task-name flow_pretrain_h100x4x2_smoke \
  --limit-train-batches 20 \
  --limit-val-batches 0 \
  --max-epochs 1 \
  --replace
```

manual launch가 필요하면 각 pod 안에서 같은 repo로 이동해 아래 환경변수를 다르게 주고 같은 wrapper를 실행합니다. rank 0은 `hsb-npc-training`, rank 1은 `hsb-npc-training2`입니다.

```bash
# hsb-npc-training
export NNODES=2 NPROC_PER_NODE=4 NODE_RANK=0
export MASTER_ADDR=<hsb-npc-training Pod IP>
export MASTER_PORT=29511
export CACHE_ROOT=/mnt/nuplan/womd_v1_3/SMART_cache
export TASK_NAME=flow_semi_continuous_pretrain_h100x4x2
bash scripts/h100x4_multinode_pretrain.sh

# hsb-npc-training2
export NNODES=2 NPROC_PER_NODE=4 NODE_RANK=1
export MASTER_ADDR=<hsb-npc-training Pod IP>
export MASTER_PORT=29511
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
export TASK_NAME=flow_semi_continuous_pretrain_h100x4x2
bash scripts/h100x4_multinode_pretrain.sh
```

batch를 더 공격적으로 키우고 싶으면 launcher 인자로 override 합니다. 기본 정책은 batch를 바꿔도 lr을 자동 선형 scaling하지 않는 것입니다. lr까지 바꿔야 하는 별도 ablation에서만 `--learning-rate`를 명시하세요.

```bash
python scripts/launch_h100x4_multinode_pretrain_tmux.py \
  --pods hsb-npc-training hsb-npc-training2 \
  --train-batch-size 28 \
  --task-name flow_pretrain_h100x4x2_bs28 \
  --replace
```

#### H100x4 1대 + A100x4 2대 mixed prefix-valid FW30 pretrain

`wo-pvc-800`의 H100 4장과 `testa`, `testaa`의 A100 각 4장, 
총 3개 pod / 12 GPU를 하나의 DDP pretrain으로 묶을 때는 아래 preset과 wrapper를 사용합니다. 
이 경로도 pod를 새로 만들거나 지우지 않고, 기존 pod 안의 tmux session과 학습 process만 관리합니다.

```text
configs/experiment/pre_bc_flow_mixed_h100x4_a100x4x2_prefix_valid.yaml
scripts/launch_mixed_h100x4_a100x4x2_prefix_valid_pretrain.py
scripts/launch_mixed_h100x4_a100x4x2_prefix_valid_pretrain_with_oom_retry.sh
```

기본 설정:

- pod 순서: `wo-pvc-800 testa testaa`
- `trainer.num_nodes=3`, `trainer.devices=4` -> 총 12 DDP ranks
- `decoder.flow_window_steps=30`
- `model.model_config.token_processor.use_prefix_valid_future_loss_mask=true`
- `data.train_batch_size=26`, effective global batch `26 * 12 = 312`
- `model.model_config.lr=5e-4`

batch size 판단은 가장 작은 GPU 메모리에 맞춥니다. 여기서는 `wo-pvc-800` H100과 `testa/testaa` A100이 모두 80GB급이므로, H100 FW30에서 실측 최적이었던 per-GPU `26`을 그대로 시작점으로 둡니다. A100이 H100보다 느려 전체 step time은 A100 쪽에 맞춰질 수 있지만, DDP는 rank별 batch를 하나로 맞춰야 하므로 메모리 기준 batch는 80GB 공통 상한을 따르는 것이 맞습니다.

lr은 더 보수적으로 잡았습니다. global batch는 기존 H100x4x2 `208`에서 mixed 12GPU `312`로 커지므로 단순 선형 scaling이면 `5e-4 * 312/208 = 7.5e-4`가 됩니다. 하지만 mixed hardware 첫 run이고 prefix-valid target 선택도 달라지므로, 기본값은 검증된 `5e-4`를 유지합니다. 선형 scaling ablation을 따로 보고 싶을 때만 `--learning-rate 7.5e-4`로 override하세요.

일회 실행:

```bash
python scripts/launch_mixed_h100x4_a100x4x2_prefix_valid_pretrain.py --replace
```

CUDA OOM fallback까지 켜고 실행하려면 아래 wrapper를 권장합니다. 첫 시도는 `train_batch_size=26`이고, OOM marker가 어느 pod 로그에서든 관측되면 전체 tmux/torchrun group을 정리한 뒤 최신 `epoch_last.ckpt`를 기준으로 `26 -> 24 -> 22 -> ... -> 2` 순서로 batch를 2씩 낮춰 재개합니다. 너무 낮은 batch까지 자동으로 내려가는 것을 막고 싶으면 `MIN_BS=20`처럼 하한을 올려 실행하세요.

H100과 A100을 섞는 이 경로에서는 NCCL이 H100 전용 NVLS/CUMEM 경로를 고르지 않도록 `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`을 기본으로 둡니다. 세 pod의 통신 인터페이스는 `eth0`으로 고정됩니다.

```bash
bash scripts/launch_mixed_h100x4_a100x4x2_prefix_valid_pretrain_with_oom_retry.sh
```

주요 override:

```bash
INITIAL_BS=26 \
OOM_STEP=2 \
MIN_BS=20 \
LEARNING_RATE=5.0e-4 \
TASK_NAME=flow_pretrain_prefix_valid_fw30_mixed_h100x4_a100x4x2_bs26 \
bash scripts/launch_mixed_h100x4_a100x4x2_prefix_valid_pretrain_with_oom_retry.sh
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc wo-pvc-800 -c main -- tmux attach -t catk-pretrain-mixed-h100-a100-prefix-fw30
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-pretrain-mixed-h100-a100-prefix-fw30
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t catk-pretrain-mixed-h100-a100-prefix-fw30
```

### 5.11 6x H100에서 Self-Forced NPFM fine-tuning

- preset 파일: `configs/experiment/self_forced_npfm_h100_6.yaml`
- H100 preset은 Generator lr `4e-6`, generated estimator optimizer lr `8e-7` (`4e-6 / 5`), `weight=1.0`, `anchor_weight=0.1`, `use_anchor_flow_matching_loss=false`, `estimator_updates_per_step=5`, `path_step_size=0.05`, `unfrozen_range=except_map_encoder`, sampling = Euler 32-step / `noise_scale=1.0` / random terminal denoising step을 기본으로 둡니다.
- Clean-DMD guidance 기본값 `clean_dmd_normalizer_eps=1.0e-3`, `clean_dmd_tau_low=0.02`, `clean_dmd_tau_high=0.98` 을 함께 둡니다.
- Generator EMA 기본값은 `ema_weight=0.99`, `ema_start_step=50` 입니다. EMA는 online Generator update 직후에만 갱신되고, generated estimator에는 적용하지 않습니다.
- Generated estimator warmup 기본값은 `estimator_warmup_epochs=1` 입니다. self-forcing 시작 후 첫 1 epoch 동안은 현재 Generator의 self-rollout으로 generated estimator만 먼저 학습하고, Generator update와 EMA update는 건너뜁니다.
- 4x/6x H100 self-forced preset과 OOM retry script는 모두 첫 시도 `data.train_batch_size=36` 을 기본으로 둡니다.
- self-forced preset은 각 epoch마다 train dataset의 50%만 새로 랜덤 샘플링해 학습합니다. 비율은 `data.train_epoch_sample_fraction` 으로 바꾸며, `1.0` 으로 두면 전체 train dataset을 사용합니다.
- self-forced fine-tuning에서는 Generator optimizer와 generated estimator optimizer 모두 LR scheduler를 쓰지 않습니다. Generator lr은 `model.model_config.lr` 로 설정하고, generated estimator optimizer lr은 별도 config 없이 `model.model_config.lr / model.model_config.self_forced.estimator_updates_per_step` 으로 계산합니다. 따라서 self-forced preset에는 `lr_warmup_steps` / `lr_min_ratio` override를 두지 않습니다.
- H100x6 차이: `defaults` 에서 `override /trainer: ddp` 를 박아 두고 `trainer.devices=6` 을 고정 → preset 만 줘도 6 GPU DDP 가 가동됩니다 (베이스 `self_forced_npfm.yaml` 은 trainer 를 override 하지 않아 single-process 로 떨어집니다).
- 새 self-forced fine-tuning 시작을 위해 preset 이 `action=finetune` 을 기본으로 고정합니다. 따라서 `ckpt_path` 는 optimizer/epoch 를 resume하지 않고 pretrained weight만 로드합니다.
- 전제: `ckpt_path` 에는 같은 `flow_window_steps` 로 pretrain 된 Generator checkpoint 를 넣습니다. 모델 default 는 `flow_window_steps=20` (2초) 이고, ckpt 가 2초 horizon 으로 pretrain 된 경우 override 하지 않는 편이 안전합니다.

실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=self_forced_npfm_h100_6 \
  action=finetune \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_self_forced_h1006 \
  ckpt_path=/path/to/2s_pretrain_epoch_last.ckpt
```

중단된 self-forced run을 이어서 학습할 때만 `action=fit ckpt_path=/path/to/self_forced_run/last.ckpt` 를 사용하세요. 이 경우에는 Lightning이 optimizer, epoch, global step까지 함께 복원합니다. checkpoint 안에 `self_forced_target_teacher`, `self_forced_generated_estimator`, `self_forced_generator_ema` state가 있으면, fit 시작 hook은 보조 모델과 EMA를 현재 Generator weight로 다시 덮어쓰지 않고 checkpoint의 `F_rho` / `F_psi` / EMA 상태를 보존합니다.

보호 장치도 있습니다. self-forced가 켜진 상태에서 `action=finetune` 에 self-forced checkpoint를 넣으면 실행이 중단됩니다. 반대로 `action=fit` 에 self-forced 보조 state가 없는 pretrained checkpoint를 넣어도 중단됩니다. 즉, pretrained Generator에서 처음 시작할 때는 `action=finetune`, self-forced run을 이어갈 때는 `action=fit` 으로 분리해야 합니다.

Self-forced H100 preset은 self-forced rollout에서 `sample_steps=32`를 유지하되, 학습 중에는 DDP 전체 rank가 같은 random terminal denoising step `s` 하나를 공유합니다. rank0에서 뽑은 `s`를 모든 rank로 broadcast하므로, 모든 rank의 scenario/agent와 0.5초 commit block은 같은 `s`를 쓰며, 실제 실행 step 수는 `K = sample_steps + 1 - s` 입니다. 따라서 이전처럼 scenario마다 다른 `s`를 뽑고 `torch.unique(K)` 그룹마다 sampler를 다시 호출하지 않습니다. 0.5초 block마다 `FlowODE.generate(..., terminal_step=K, return_terminal_clean=True)`를 한 번만 호출해 terminal clean estimate를 commit합니다.

- `model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch` 가 기본값입니다. 이 값은 DDP 전체 rank 공유 `s` fast path를 뜻합니다.
- `policy=paper_uniform` 은 실제 실행 denoising step `K` 를 `[min_executed_steps, sample_steps]` 범위에서 균등 샘플링합니다. 기본 `min_executed_steps=24` 이므로 `sample_steps=32` 에서는 `K=24..32` 만 사용합니다.
- `policy=all` 은 random terminal step을 샘플링하지 않고 항상 `sample_steps` 전체 denoising을 실행합니다. 이때 `sampling.backprop_last_k` 개 마지막 step에만 gradient를 남기며, 값을 생략하면 기본값은 `8` 입니다.
- terminal step 이전 denoising은 gradient 없이 계산하고, terminal clean estimate를 만드는 마지막 호출 하나만 gradient를 유지합니다.
- 선택된 `s`는 self-rollout을 어디서 끊고 commit할지만 정합니다.
- `F_psi` 업데이트와 clean-DMD guidance 계산의 noising `tau` 는 flow ODE의 전체 tau 구간에서 독립적으로 다시 샘플링합니다.

속도 실험용 기본 실행은 아래처럼 두면 됩니다.

```bash
python -m src.run experiment=self_forced_npfm_h100_6 \
    model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch \
    model.model_config.self_forced.sampling.random_terminal_step.policy=paper_uniform
```

#### 8-node x 4 V100 static pod self-forced 실행

이미 떠 있는 V100 4장짜리 pod 8개를 묶어 32-rank self-forced fine-tuning을 돌릴 때는 아래 preset과 launcher를 사용합니다. launcher는 pod를 새로 만들거나 지우거나 재시작하지 않고, `kubectl exec`로 각 pod 안에 tmux 세션과 `torchrun`만 시작합니다.

```text
configs/experiment/self_forced_npfm_v100x4x8.yaml
scripts/launch_self_forced_v100x4x8_static_pods.py
```

대상 pod 순서는 기본값으로 고정되어 있습니다.

```text
testsv testsvv testsvvv testsvvvv sv svv svvv svvvv
```

기본 실험 설정:

- `trainer.num_nodes=8`, `trainer.devices=4` -> 총 32 DDP ranks
- V100용 `trainer.precision=16-mixed`
- `trainer.max_epochs=12`, `trainer.check_val_every_n_epoch=2`
- `model.model_config.lr=1.0e-6`
- `model.model_config.self_forced.estimator_warmup_epochs=0`
- `model.model_config.self_forced.use_stop_motion=false`
- `data.train_batch_size=2`, OOM 시 launcher가 OOM status를 즉시 공유해 모든 pod의 local `torchrun`과 남은 `task_name=...` 학습 rank를 정리하고 `2 -> 1` 순서로 함께 낮춤
- `data.val_batch_size=2`, `model.model_config.scorer_scene_num=320`

pretrained checkpoint는 W&B artifact에서 자동으로 내려받습니다.

```text
artifact: jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64
target:   /workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt
```

실제 실행 전에 로컬에서 렌더링만 확인하려면 `--dry-run`을 사용합니다. 이 모드는 pod 안에 아무 것도 만들지 않습니다.

```bash
python scripts/launch_self_forced_v100x4x8_static_pods.py \
  --dry-run \
  --master-addr <testsv-pod-ip>
```

실행:

```bash
python scripts/launch_self_forced_v100x4x8_static_pods.py --replace
```

smoke test를 돌릴 때는 task 이름을 따로 주는 편이 안전합니다.

```bash
python scripts/launch_self_forced_v100x4x8_static_pods.py \
  --replace \
  --limit-train-batches 20 \
  --limit-val-batches 0 \
  --max-epochs 1 \
  --task-name flow_self_forced_v100x4x8_stopfalse_warmup0_lr1e-6_bs2_smoke
```

attach:

```bash
kubectl exec -it -n p-pnc testsv -c main -- tmux attach -t catk-sf-v100x4x8-stopfalse-warmup0
```

중지:

```bash
python scripts/launch_self_forced_v100x4x8_static_pods.py --stop
```

Validation/test/submission inference의 stop-motion까지 끄고 싶을 때만 아래 override를 추가합니다. 기본 launcher 설정은 self-forced 학습 rollout의 stop-motion만 끕니다.

```bash
python scripts/launch_self_forced_v100x4x8_static_pods.py \
  --replace \
  --decoder-use-stop-motion false
```

#### 2-node x 4 A100 static pod self-forced 실행

A100 80GB 4장짜리 pod 2개(`testa`, `testaa`)를 묶어 8-rank self-forced fine-tuning을 돌릴 때는 아래 preset과 launcher를 사용합니다. launcher는 pod를 새로 만들거나 지우거나 재시작하지 않고, `kubectl exec`로 각 pod 안에 tmux 세션과 `torchrun`만 시작합니다.

```text
configs/experiment/self_forced_npfm_a100x4x2.yaml
scripts/launch_self_forced_a100x4x2_static_pods.py
```

기본 실험 설정:

- 대상 pod: `testa testaa`
- `trainer.num_nodes=2`, `trainer.devices=4` -> 총 8 DDP ranks
- A100용 `trainer.precision=bf16-mixed`
- `model.model_config.lr=1.0e-6`
- `model.model_config.self_forced.estimator_warmup_epochs=1`
- `model.model_config.self_forced.use_stop_motion=false`
- `data.train_batch_size=22`, OOM 시 launcher가 모든 pod의 attempt status를 모아 `22 -> 20 -> 18 -> ...` 순서로 함께 낮춤
- `data.val_batch_size=8`, `model.model_config.scorer_scene_num=320`

pretrained checkpoint는 W&B artifact에서 자동으로 내려받습니다.

```text
artifact: jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64
target:   /workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt
```

실제 실행 전에 로컬에서 렌더링만 확인하려면 `--dry-run`을 사용합니다. 이 모드는 pod 안에 아무 것도 만들지 않습니다.

```bash
python scripts/launch_self_forced_a100x4x2_static_pods.py \
  --dry-run \
  --master-addr <testa-pod-ip>
```

실행:

```bash
python scripts/launch_self_forced_a100x4x2_static_pods.py --replace
```

attach:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-sf-a100x4x2-stopfalse-warmup1
```

중지:

```bash
python scripts/launch_self_forced_a100x4x2_static_pods.py --stop
```

#### 1-node x 4 H100 wo-pvc-800 self-forced 실행

H100 4장짜리 `wo-pvc-800` pod 하나에서 self-forced fine-tuning을 돌릴 때는 아래 preset과 launcher를 사용합니다. launcher는 pod를 새로 만들거나 지우거나 재시작하지 않고, `kubectl exec`로 해당 pod 안에 tmux 세션과 `torchrun --standalone --nproc_per_node=4`만 시작합니다.

```text
configs/experiment/self_forced_npfm_h100x4_wo_pvc_800.yaml
scripts/launch_self_forced_h100x4_wo_pvc_800.py
```

기본 실험 설정:

- 대상 pod: `wo-pvc-800`
- `trainer.num_nodes=1`, `trainer.devices=4` -> 총 4 DDP ranks
- H100용 `trainer.precision=bf16-mixed`
- `trainer.max_epochs=10`, `trainer.check_val_every_n_epoch=2`
- `model.model_config.lr=1.0e-6`
- `model.model_config.self_forced.estimator_warmup_epochs=4`
- `model.model_config.self_forced.use_stop_motion=false`
- `data.train_batch_size=28`
- OOM 시 launcher가 호출하는 retry wrapper가 `28 -> 26 -> 24 -> ...` 순서로 batch size를 낮춰 재시도

pretrained checkpoint는 W&B artifact에서 자동으로 내려받습니다.

```text
artifact: jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64
target:   /workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt
```

실제 실행 전에 로컬에서 렌더링만 확인하려면 `--dry-run`을 사용합니다. 이 모드는 pod 안에 아무 것도 만들지 않습니다.

```bash
python scripts/launch_self_forced_h100x4_wo_pvc_800.py --dry-run
```

실행:

```bash
python scripts/launch_self_forced_h100x4_wo_pvc_800.py --replace
```

smoke test를 돌릴 때는 task 이름을 따로 주는 편이 안전합니다.

```bash
python scripts/launch_self_forced_h100x4_wo_pvc_800.py \
  --replace \
  --limit-train-batches 20 \
  --limit-val-batches 0 \
  --max-epochs 1 \
  --task-name flow_self_forced_h100x4_wo_pvc_800_stopfalse_warmup4_lr1e-6_bs28_smoke
```

attach:

```bash
kubectl exec -it -n p-pnc wo-pvc-800 -c main -- tmux attach -t catk-sf-h100x4-wo-pvc-800
```

중지:

```bash
python scripts/launch_self_forced_h100x4_wo_pvc_800.py --stop
```

Validation/test/submission inference의 stop-motion까지 끄고 싶을 때만 아래 override를 추가합니다. 기본 launcher 설정은 self-forced 학습 rollout의 stop-motion만 끕니다.

```bash
python scripts/launch_self_forced_h100x4_wo_pvc_800.py \
  --replace \
  --decoder-use-stop-motion false
```

#### 1-node x 4 H100 hsb-npc-training self-forced 실행

H100 4장짜리 `hsb-npc-training` pod 하나에서 self-forced fine-tuning을 돌릴 때는 아래 preset과 launcher를 사용합니다. `wo-pvc-800` launcher와 같은 단일 pod OOM fallback 구조를 쓰지만, 기본 pod/cache/task/session을 `hsb-npc-training` 전용으로 분리했습니다.

```text
configs/experiment/self_forced_npfm_h100x4_hsb_npc_training.yaml
scripts/launch_self_forced_h100x4_hsb_npc_training.py
```

기본 실험 설정:

- 대상 pod: `hsb-npc-training`
- `trainer.num_nodes=1`, `trainer.devices=4` -> 총 4 DDP ranks
- H100용 `trainer.precision=bf16-mixed`
- `trainer.max_epochs=10`, `trainer.check_val_every_n_epoch=2`
- `model.model_config.lr=1.0e-6`
- `model.model_config.self_forced.estimator_warmup_epochs=0`
- `model.model_config.self_forced.use_stop_motion=false`
- 기본 cache root: `/mnt/nuplan/womd_v1_3/SMART_cache`
- `data.train_batch_size=28`
- OOM 시 retry wrapper가 `28 -> 26 -> 24 -> ...` 순서로 batch size를 낮추고 local `epoch_last.ckpt`로 재개

pretrained checkpoint는 W&B artifact에서 자동으로 내려받습니다.

```text
artifact: jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64
target:   /workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt
```

지금 pod에 다른 실험이 돌고 있으면 실제 실행 명령을 쓰지 말고, 먼저 dry-run만 확인하세요. `--dry-run`은 pod 안에 아무 것도 만들지 않습니다.

```bash
python scripts/launch_self_forced_h100x4_hsb_npc_training.py --dry-run
```

실행:

```bash
python scripts/launch_self_forced_h100x4_hsb_npc_training.py --replace
```

attach:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-sf-h100x4-hsb-npc-training
```

중지:

```bash
python scripts/launch_self_forced_h100x4_hsb_npc_training.py --stop
```

#### 1-node x 4 H100 hsb-npc-training2 SiD self-forced 실행

H100 4장짜리 `hsb-npc-training2` pod 하나에서 SiD objective로 self-forced fine-tuning을 돌릴 때는 아래 preset과 launcher를 사용합니다. 이 launcher는 기존 pod를 만들거나 지우거나 재시작하지 않고, `kubectl exec`로 이미 떠 있는 pod 안에 tmux 세션과 학습 프로세스만 띄웁니다. pod에 다른 실험이 돌고 있으면 실제 실행 대신 dry-run만 확인하세요.

```text
configs/experiment/self_forced_npfm_sid_h100x4_hsb_npc_training2.yaml
scripts/launch_self_forced_sid_h100x4_hsb_npc_training2.py
```

기본 실험 설정:

- 대상 pod: `hsb-npc-training2`
- `trainer.num_nodes=1`, `trainer.devices=4` -> 총 4 DDP ranks
- H100용 `trainer.precision=bf16-mixed`
- `trainer.max_epochs=10`, `trainer.check_val_every_n_epoch=2`
- `model.model_config.lr=1.0e-6`
- `model.model_config.self_forced.distribution_matching_objective=sid`
- `model.model_config.self_forced.estimator_warmup_epochs=1`
- `model.model_config.self_forced.use_stop_motion=false`
- 기본 cache root: `/workspace/womd_v1_3/SMART_cache`
- `data.train_batch_size=28`
- OOM 시 retry wrapper가 `28 -> 26 -> 24 -> ...` 순서로 batch size를 낮추고 local `epoch_last.ckpt`로 재개

pretrained checkpoint는 W&B artifact에서 자동으로 내려받습니다.

```text
artifact: jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64
target:   /workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/v64/epoch_last.ckpt
```

dry-run:

```bash
python scripts/launch_self_forced_sid_h100x4_hsb_npc_training2.py --dry-run
```

실행:

```bash
python scripts/launch_self_forced_sid_h100x4_hsb_npc_training2.py --replace
```

attach:

```bash
kubectl exec -it -n p-pnc hsb-npc-training2 -c main -- tmux attach -t catk-sf-sid-h100x4-hsb-npc-training2
```

중지:

```bash
python scripts/launch_self_forced_sid_h100x4_hsb_npc_training2.py --stop
```

#### CUDA OOM 자동 fallback 으로 무중단 재개

긴 self-forced fine-tuning 도중 어쩌다 OOM 이 한 번 떨어지면 (heavy batch + self-rollout 메모리 스파이크), 학습이 죽고 그동안 진행한 epoch 들이 의미 없어질 수 있습니다. `scripts/self_forced_h100_4_with_oom_retry.sh` 와 `scripts/self_forced_h100_6_with_oom_retry.sh` 는 이 시나리오를 자동 처리합니다:

- 첫 시도는 `PRETRAIN_CKPT` 에 지정한 2초 horizon pretrained Generator ckpt 로 `action=finetune`
- 학습 도중 OOM 으로 죽으면 attempt log 에서 `OutOfMemoryError` / `CUDA out of memory` 마커를 감지해 `data.train_batch_size` 를 `OOM_STEP` (기본 2) 만큼 낮춤
- 다음 시도부터는 `logs/<TASK_NAME>/runs/*/checkpoints/epoch_last.ckpt` 중 최신 self-forced ckpt 를 골라 `action=fit` 으로 **마지막 완료 epoch 끝부터 재개** (optimizer / epoch / global step / `F_rho` / `F_psi` / Generator EMA 모두 복원)
- `bs` 가 `MIN_BS` (기본 2) 아래로 내려가거나 OOM 이외의 실패가 나면 즉시 중단

실행 예시:

```bash
PRETRAIN_CKPT=/mnt/nuplan/projects/catk/downloads/wandb_ckpts/flow_semi_continuous_finetune_inv_euler_32_a100x4/run_sjan8kmh/v32/epoch_last.ckpt \
bash scripts/self_forced_h100_6_with_oom_retry.sh
```

기본 동작을 바꿀 수 있는 환경변수 (모두 optional):

| 변수 | 기본값 | 설명 |
|---|---|---|
| `INITIAL_BS` | `36` | 첫 시도 `data.train_batch_size` (preset 기본값) |
| `OOM_STEP` | `2` | OOM 한 번당 줄일 batch 크기 |
| `MIN_BS` | `2` | 이 값 미만으로 내려가면 중단 |
| `TASK_NAME` | `flow_semi_continuous_self_forced_h1006` | checkpoint / log 위치 결정 |
| `CACHE_ROOT` | `/mnt/nuplan/womd_v1_3/SMART_cache` | WOMD SMART cache 경로 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5` | 사용할 GPU |
| `NPROC_PER_NODE` | `6` | DDP rank 수 |
| `EXPERIMENT` | `self_forced_npfm_h100_6` | hydra experiment 이름 |
| `RANDOM_TERMINAL_SCOPE` | unset | random terminal scope override, normally `global_batch` |
| `EMA_WEIGHT` | unset | Generator EMA decay override |
| `EMA_START_STEP` | unset | Generator EMA 시작 generator update 수 override |
| `CLEAN_DMD_NORMALIZER_EPS` | unset | Clean-DMD direction 정규화 분모 최소값 override |
| `CLEAN_DMD_TAU_LOW` | unset | Clean-DMD guidance noising tau 하한 override |
| `CLEAN_DMD_TAU_HIGH` | unset | Clean-DMD guidance noising tau 상한 override |

각 시도의 로그는 `logs/_self_forced_oom_retry/<TASK_NAME>/attempt_NNN_bsBB.log` 로 분리 저장되어, 어느 시도에서 OOM 이 났는지 / 어디서 다음 시도가 이어 받았는지 사후 추적 가능합니다.

self-forced fine-tuning에서 학습할 파라미터 범위는 `unfrozen_range` 로 정합니다. 기본값 `except_map_encoder` 는 map encoder만 고정하고 나머지 Generator / generated estimator 파라미터를 학습 대상으로 둡니다. 더 보수적으로 보려면 마지막 궤적 생성부만 여는 `full_flow_decoder` 를 먼저 시도하세요.

```bash
... model.model_config.self_forced.unfrozen_range=full_flow_decoder
```

### WOSAC-CPD / WOSAC-CES Distribution Metrics

closed-loop validation과 Sim Agents submission export에서는 모델이 실제로 만든 10Hz rollout으로 아래 metric을 계산합니다.

- `val_closed/WOSAC-CPD/value`: 같은 scenario 안 rollout끼리의 조건부 다양성입니다. 높을수록 rollout들이 서로 다릅니다.
- `val_closed/WOSAC-CES/value`: validation GT가 있을 때만 계산되는 Energy Score 계열 metric입니다. 낮을수록 좋습니다.
- `test/WOSAC-CPD/value`: test submission export에서 계산되는 CPD입니다. test set은 GT 미래를 제공하지 않으므로 CES는 기록하지 않습니다.
- `*/WOSAC-CPD/DPR`: `model.model_config.wosac_cpd_reference`에 flow-pretrain CPD를 넣었을 때만 기록됩니다. 값은 `현재 CPD / 기준 CPD` 입니다.

이 metric들은 학습 step에서는 계산하지 않고, validation/test closed-loop rollout이 만들어진 뒤에만 계산합니다. `n_rollout_closed_val`이 16이면 이미 생성된 16개 rollout만 사용하고 별도 rollout을 추가 생성하지 않습니다.

### Self-forced Generated Estimator Warmup

- `model.model_config.self_forced.estimator_warmup_epochs=1` 이 기본값입니다.
- 이 기간에는 online Generator를 업데이트하지 않고, 현재 Generator가 만든 self-rollout으로 generated estimator만 먼저 학습합니다.
- warmup 중 self-rollout은 `torch.no_grad()`로 생성하고, Generator optimizer step과 EMA update는 실행하지 않습니다.
- warmup이 끝나면 기존 self-forcing 경로를 그대로 사용합니다. DMD/SiD 선택, block detach, `unfrozen_range` 설정은 그대로 유지됩니다.
- 끄고 싶으면 `estimator_warmup_epochs=0` 으로 설정합니다.

예시:

```bash
... model.model_config.self_forced.estimator_warmup_epochs=0
```

`precision=bf16-mixed` 에서도 self-forced forward / rollout / loss 계산은 그대로 mixed precision 으로 실행합니다. 다만 manual optimization 경로의 backward 진입점만 autocast-disabled boundary 밖에서 실행해, `F_psi` 업데이트와 Generator 업데이트 중 PyTorch autocast dtype promote 경로가 backward graph 를 다시 분류하다가 `Unexpected floating ScalarType in at::autocast::prioritize` 로 멈추는 상황을 피합니다. 현재 self-forced preset 기본값은 `estimator_updates_per_step=5` 이며, 이 값을 늘리면 그 횟수만큼 `F_psi` manual backward가 반복됩니다. 무거운 forward 계산은 bf16-mixed 를 유지하므로 전체 학습을 fp32 로 낮추는 방식보다 속도 손실이 작습니다.

self-forced training rollout 은 gradient 를 유지하므로, temporal edge index 처럼 feature indexing 에 이미 사용된 tensor 를 rollout 중간에 in-place 로 바꾸면 backward 에서 version counter 오류가 날 수 있습니다. 구현은 `build_temporal_edge()` 가 반환한 원본 `edge_index_t` 를 보존하고, current-agent attention 에 필요한 destination row 만 별도 `edge_index_t_current` tensor 로 remap 합니다. inference 경로는 no-grad 라서 이 문제가 잘 보이지 않지만, training rollout 에서는 이 원본 보존 규칙이 필요합니다.

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
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
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
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
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
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
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
  model.model_config.vis_flow_preview=true \
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

### 4x H100 학습

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m src.run experiment=pre_bc_flow_4_h100 trainer=ddp trainer.devices=4 paths.cache_root="$CACHE_ROOT" task_name=flow_semi_continuous_pretrain_h1004
```

### 2-node 4x H100 학습

```bash
python scripts/launch_h100x4_multinode_pretrain_tmux.py --pods hsb-npc-training hsb-npc-training2 --task-name flow_semi_continuous_pretrain_h100x4x2 --replace
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

## Self-Forced N-Second Path-Flow Matching fine-tuning

이 브랜치에는 **Self-Forced N-Second Path-Flow Matching (SF-NPFM)** 이라는 선택적 학습 경로가 추가되어 있습니다. 해당 기능은 `model.model_config.self_forced` 로 제어하며, `configs/model/smart_flow.yaml` 기본값에서는 꺼져 있습니다.

horizon 은 8초로 고정되어 있지 않고, `model.model_config.decoder.flow_window_steps` 값에 연동됩니다. 10 Hz 기준으로 fine-tuning horizon 은 다음과 같이 계산됩니다.

```text
N 초 = flow_window_steps / 10
K commit block 수 = flow_window_steps / 5
```

기본값인 `flow_window_steps: 20` 에서는 SF-NPFM 이 2초짜리 self-rollout 을 0.5초 commit/update block 4 개로 구성해 실행합니다. WOSAC 제출용 rollout 은 여전히 8초 inference loop 를 그대로 쓰지만, fine-tuning objective 자체는 pretrain 에서 사용한 flow window 안쪽에만 머뭅니다.

### 추가되는 구성 요소

- `F_rho`: fresh fine-tuning 시작 시점에 pretrained `SMARTFlowDecoder` 를 복사해 만드는 frozen target path-flow teacher 입니다.
- `F_psi`: `F_rho` 와 같은 pretrained decoder weight 로 초기화한 generated path-flow estimator 이며, detached committed self-rollout 위에서 online 으로 업데이트됩니다.
- Generator EMA: online Generator의 update를 평균낸 frozen copy입니다. `ema_start_step=50` 번째 generator update에서 현재 online Generator를 복사해 시작하고, 그 뒤 update마다 `ema_weight=0.99` 비율로 이전 EMA를 유지합니다. 학습 rollout과 gradient update는 항상 online Generator로 하고, validation / checkpoint 선택 / test submission은 EMA가 준비된 뒤 EMA Generator를 사용합니다.
- self-forced checkpoint resume 에서는 checkpoint에 저장된 `F_rho` / `F_psi` / Generator EMA state를 그대로 보존합니다. 즉, resume 직후 fit 시작 hook이 두 보조 모델이나 EMA를 현재 Generator weight로 다시 덮어쓰지 않습니다.
- guidance 방향을 계산할 때는 `F_rho` 와 비교용 `F_psi` 를 항상 eval mode로 둡니다. 그래서 dropout/history drop 같은 train-mode 랜덤성이 기준 방향에 섞이지 않습니다. `F_psi` 는 detached generated path에 fit되는 online update 구간에서만 train mode로 전환됩니다.
- committed self-rollout 을 만들 때는 현재 Generator를 eval mode로 잠깐 전환하되 autograd는 유지합니다. 따라서 dropout/history drop 없이 실제 inference 조건의 trajectory를 만들고, 그 trajectory를 통해 `sf_loss` gradient는 그대로 Generator로 흐릅니다.
- inference 와 동일한 0.5초 commit/update 규칙을 쓰되 `flow_window_steps / 5` block 만큼만 도는 differentiable training rollout 경로. 학습 중에는 DDP 전체 rank가 random terminal step `s` 를 하나 공유하고, 모든 rank의 scenario/agent와 0.5초 commit block이 같은 `s` 를 씁니다. 실제 실행 step 수는 `K = sample_steps + 1 - s` 이며, terminal 이전 step은 no-grad로 계산하고 terminal clean estimate를 만드는 마지막 step 하나만 gradient를 유지합니다.
- self-forced 학습 rollout의 stop-motion gate는 `model.model_config.self_forced.use_stop_motion` 으로 제어합니다. Validation / test / submission inference의 `model.model_config.decoder.use_stop_motion` 과 분리되어 있으므로, inference ablation을 위해 decoder 설정을 바꿔도 학습 rollout 규칙은 자동으로 바뀌지 않습니다.
- random terminal step `s` 는 self-rollout 의 실행 길이와 commit 지점만 정합니다. Generated estimator `F_psi` 학습과 generator direction 계산에서 쓰는 flow noising `tau` 는 rollout 의 `s` 와 독립적으로 전체 tau 구간에서 새로 샘플링합니다.
- generator direction은 raw score/path 이동량을 그대로 쓰지 않고,
- 같은 noisy path에서 `F_rho` 와 `F_psi` 가 각각 추정한 clean path 차이를 사용합니다.
- `teacher clean path - generated clean path` 를
- agent별 `현재 closed-loop path` 와 `teacher clean path` 사이의 평균 거리로 정규화한 뒤,
- `path_step_size` 를 곱해 target path를 만듭니다.
- Clean-DMD guidance의 기본 noising 구간은 `clean_dmd_tau_low=0.02`, `clean_dmd_tau_high=0.98` 입니다.
- 정규화 분모는 `clean_dmd_normalizer_eps=1.0e-3` 으로 최소값을 둬서 target path가 과하게 튀는 상황을 줄입니다.
- committed self-rollout 에 대해서만 걸리는 control-space physics regularization (선택 사항). `model.model_config.self_forced.use_control_space_physics_regularization` 로 제어합니다.
- 약한 open-loop flow-matching anchor. `model.model_config.self_forced.use_anchor_flow_matching_loss=false` 로 두면 `anchor_weight` 값과 무관하게 self-forced active step에서 training-mode open-loop forward와 FM loss 계산 자체를 생략합니다. `true` 일 때만 `model.model_config.self_forced.anchor_weight` 로 total loss 반영 강도를 제어합니다. anchor FM 을 끈 상태에서 어떤 rank 의 committed self-rollout 까지 비어있는 (모든 agent 가 invalid anchor0) 드문 경우에는, encoder 파라미터 합에 0 을 곱한 zero-loss 로 backward 만 한 번 돌려 DDP all-reduce 참여를 보장하고 optimizer step 은 건너뜁니다. 이 가드가 없으면 그 rank 만 backward 를 호출하지 않아 다른 rank 의 NCCL all-reduce 가 NCCL_TIMEOUT 까지 hang 합니다.
- 선택적 trainable range. `model.model_config.self_forced.unfrozen_range=except_map_encoder` 가 기본값이며, map encoder만 고정하고 나머지 Generator / generated estimator 파라미터는 학습합니다. `middle` 은 마지막 agent 문맥 블록과 flow decoder만 열고, `full_flow_decoder` 는 마지막 궤적 생성부만 엽니다.
- epoch별 train subset sampling. self-forced preset은 `data.train_epoch_sample_fraction=0.5` 를 기본으로 두어 매 epoch 전체 train dataset의 50%만 새로 랜덤 샘플링해 학습합니다. DDP에서는 모든 rank가 같은 전역 subset을 공유한 뒤 rank별로 나눠 받습니다. `1.0` 으로 override하면 기존처럼 전체 train dataset을 씁니다.
- Generator EMA는 Generator에만 적용합니다. `F_psi` 는 현재 online Generator가 만든 분포를 따라가야 하므로 EMA를 두지 않고, `F_rho` 는 pretrained 기준점이라 계속 frozen 상태로 둡니다.
- bf16-mixed 안전 backward boundary. self-forced 경로의 forward 와 loss 계산은 mixed precision 으로 유지하되, `manual_backward` 호출 순간만 autocast 를 끄고 scalar loss 를 fp32 로 넘깁니다. 이는 manual optimization 에서 반복 backward 를 수행할 때 PyTorch autocast promote 규칙이 backward graph 의 dtype 을 다시 분류하다가 실패하는 문제를 피하기 위한 경계입니다.
- autograd-safe temporal edge remap. training rollout 에서는 temporal relation embedding 계산에 쓴 원본 `edge_index_t` 를 in-place 수정하지 않고, current-agent attention 용 remapped edge index 를 새 tensor 로 만들어 사용합니다.
- autograd-safe geometry helpers. agent encoder / flow agent decoder 의 edge feature (relative position norm, relative angle) 는 정지 또는 중첩된 agent 가 만드는 영벡터에 대해 backward 가 정의되지 않습니다 (`torch.norm` 의 `x/||x||` 가 `0/0`, `atan2(0, 0)` 의 `1/(y²+x²)` 가 `1/0`). self-forced rollout 처럼 이 feature 들이 살아있는 backward graph 의 일부가 되는 경로에서 한 번이라도 영벡터가 들어오면 NaN gradient 가 encoder weight 까지 흘러 학습이 첫 step 에서 죽습니다. 이를 막기 위해 `safe_norm_2d` helper 가 `(sum(x²) + eps).sqrt()` 형태로 norm 의 backward 분모를 strictly positive 로 유지하고, `angle_between_2d_vectors` 는 상대 벡터가 0일 때 기준 heading 방향을 대체값으로 써서 상대각 0 의미를 보존합니다. flow heading 복원도 `safe_angle_from_2d_vector` 로 통일해 heading vector 가 `[0, 0]` 일 때 `atan2(0, 0)` backward 가 생기지 않게 했습니다. self-forced generator backward 에서 non-finite 가 재발하면 `committed_path_norm`, `path_delta`, `target_path_norm` 요약과 첫 non-finite gradient 이름을 함께 출력하되, 정상 step 에서는 큰 텐서를 스캔하지 않습니다.

### Self-Forced random terminal denoising

Self-forced fine-tuning은 학습 중 `self_forced.sampling.sample_steps` 값을 줄이지 않고도 평균 sampler 호출 수를 줄일 수 있습니다. `sample_steps=32`는 전체 denoising grid로 유지하고, 학습 rollout마다 DDP 전체 rank가 terminal denoising step `s` 하나를 공유합니다. rank0에서 뽑은 `s`를 모든 rank로 broadcast하므로, 모든 rank 안의 scenario/agent와 0.5초 commit block들은 같은 `s`를 사용합니다.

학습 rollout에서는 `K = sample_steps + 1 - s` step까지만 진행한 뒤, 중간 noisy state를 commit하지 않고 terminal step에서 예측한 clean estimate를 2초 preview로 사용합니다. 그 preview 중 앞 0.5초만 기존 commit bridge로 반영합니다. terminal 이전 step은 gradient 없이 계산하고, terminal clean estimate를 만든 step 하나에만 gradient를 남깁니다. 이전 구현처럼 `torch.unique(K)` 로 terminal step별 agent group을 나눠 sampler를 여러 번 호출하지 않고, 0.5초 block마다 DDP 전체 rank가 공유한 `K` 로 `FlowODE.generate(..., terminal_step=K, return_terminal_clean=True)`를 한 번만 호출합니다. 다음 block의 context/cache로 들어가는 상태는 detach하여 미래 block loss가 이전 block 내부로 역전파되지 않게 합니다.

Generated Path-Flow Estimator와 generator direction 계산은 random-s 정보를 noising 구간으로 재사용하지 않습니다.
rollout에서 선택된 `s`는 terminal clean estimate를 만들 실행 step 수 `K`와 commit 지점만 정하며,
packed committed path를 만든 뒤에는 `s`별 `[tau_low, tau_high]` 를 전달하지 않습니다.
`F_psi` 학습은 flow ODE의 기본 전체 tau 구간에서 새 tau를 샘플링합니다.
Clean-DMD direction 계산은 `clean_dmd_tau_low` / `clean_dmd_tau_high` 구간에서 새 tau를 샘플링하되,
direction 계산 안에서는 `F_rho`와 `F_psi`가 항상 같은 noisy path와 같은 tau를 봅니다.

DDP에서는 step 시간이 가장 늦게 끝난 rank에 맞춰지므로, rank마다 서로 다른 `s`를 뽑으면 짧은 `K`를 뽑은 rank가 긴 `K`를 뽑은 rank를 기다리게 됩니다. `scope=global_batch`는 이 대기 손실을 줄이기 위해 모든 rank가 같은 `K`를 쓰게 합니다. 단일 GPU 또는 torch.distributed가 초기화되지 않은 실행에서는 같은 설정이 자동으로 일반 batch 공유 방식처럼 동작합니다.

```yaml
model:
  model_config:
    self_forced:
      use_anchor_flow_matching_loss: false
      use_stop_motion: false
      sampling:
        sample_steps: 32
        sample_method: euler
        noise_scale: 1.0
        backprop_last_k: 16
        random_terminal_step:
          enabled: true
          scope: global_batch
          policy: paper_uniform
          min_executed_steps: 24
      ema_weight: 0.99
      ema_start_step: 50
data:
  train_epoch_sample_fraction: 0.5
```

최종 inference 모델은 fine-tuning 된 Generator의 EMA copy입니다. EMA가 아직 준비되지 않은 early checkpoint나 old checkpoint에서는 online Generator로 fallback합니다. `F_rho` 와 `F_psi` 는 학습 시점 보조 모델이며 submission export 에는 사용하지 않습니다.

### Fine-tuning 설정 예시

바로 쓰거나 수정해서 쓸 수 있는 설정 파일이 아래 경로에 있습니다.

```text
configs/experiment/self_forced_npfm.yaml
```

현재 설정의 `decoder.flow_window_steps` 와 같은 값으로 학습된 pretrained checkpoint 를 함께 넘겨 실행합니다.

```bash
python -m src.run experiment=self_forced_npfm action=finetune ckpt_path=/path/to/pretrained.ckpt
```

이미 self-forced로 학습 중이던 checkpoint를 이어서 학습할 때는 `action=finetune` 이 아니라 `action=fit ckpt_path=/path/to/self_forced_run/last.ckpt` 를 사용합니다. 실행 코드는 checkpoint 안의 `F_rho` / `F_psi` 보조 state 유무를 보고 두 경로가 섞이면 조기에 에러를 냅니다.

이 구현은 WOSAC RMM 을 reward 나 optimization objective 로 사용하지 않습니다. 기존 closed-loop 평가 경로와 동일하게, RMM 은 validation / 리포팅 용도로만 쓸 수 있습니다.

### 중요한 일관성 규칙

fine-tuning 에 쓰는 rollout 과 inference 에 쓰는 rollout 은 기본 commit/update 의미가 어긋나지 않아야 합니다. 다만 stop-motion gate는 목적이 다른 두 config로 분리합니다.

- `model.model_config.decoder.use_stop_motion`: validation / test / submission inference 전용입니다.
- `model.model_config.self_forced.use_stop_motion`: self-forced closed-loop training rollout 전용입니다.

따라서 inference ablation을 위해 `decoder.use_stop_motion` 을 바꿔도 self-forced 학습 rollout은 바뀌지 않습니다. 학습 rollout에서도 같은 stop-motion gate를 쓰고 싶을 때만 `self_forced.use_stop_motion=true` 를 명시하세요.

### Self-forced Strict DMD Update Separation

- self-forcing DMD에서 Generator update와 generated estimator update를 더 강하게 분리합니다.
- Generator update에서는 target teacher와 generated estimator를 평가자로만 사용하고, 두 보조 모델에 gradient가 생기면 즉시 오류를 냅니다.
- generated estimator update에서는 현재 Generator가 만든 detached closed-loop path만 학습 대상으로 사용해야 하며, Generator에 gradient가 생기면 즉시 오류를 냅니다.
- 이 update 중에는 detached clone으로 만든 path/noisy target만 `F_psi` 입력으로 쓰고, tokenized map/agent context와 anchor mask도 detached view로 넘깁니다. 그래서 rollout context에 Generator graph가 남아 있어도 estimator backward가 online Generator로 되돌아가지 않습니다.
- online Generator와 frozen teacher의 parameter gradient 누적도 update 동안 임시로 꺼 둡니다.
- update 경계마다 이전 단계의 gradient를 명확히 비워서, DMD 방향이 optimizer 간에 섞이지 않게 했습니다.
- Clean-DMD guidance는 기존처럼 teacher/generated clean path 추정 차이를 agent별 teacher 기준 거리로 정규화합니다.
- `use_anchor_flow_matching_loss=false`, `use_control_space_physics_regularization=false` 설정은 그대로 유지됩니다.

### Self-forced SiD-lite Update

- `model.model_config.self_forced.distribution_matching_objective=dmd` 는 기존 방식입니다.
  frozen teacher / generated estimator 차이로 방향을 만들고,
  `committed_path_norm + path_step_size * path_delta` 를 detached target으로 둔 뒤 MSE를 겁니다.
- `model.model_config.self_forced.distribution_matching_objective=sid` 는 SiD-lite 방식입니다.
  closed-loop self-rollout, frozen teacher, generated estimator, estimator update, EMA 구조는 그대로 두고,
  generator update만 `X`, `R`, `F` 관계식으로 계산합니다.
- SiD-lite에서 `X` 는 Generator가 실제로 실행한 path,
  `R` 은 frozen teacher의 clean path 예측,
  `F` 는 generated estimator의 clean path 예측입니다.
- SiD-lite loss는 `path_step_size` 를 사용하지 않습니다.
  `sid_alpha` 기본값은 `1.0`, `sid_normalizer_eps` 기본값은 `1.0e-3` 입니다.
- 바로 실행하려면 아래 preset을 사용할 수 있습니다.

```bash
... experiment=self_forced_npfm_sid \
    model.model_config.self_forced.distribution_matching_objective=sid
```

H100 preset은 `experiment=self_forced_npfm_sid_h100_4` 또는
`experiment=self_forced_npfm_sid_h100_6` 를 사용하면 됩니다.

### Self-forced Block-boundary Gradient Detach

- `model.model_config.self_forced.detach_block_transition=false` 는 기존 동작입니다.
  2초 self-rollout 안의 0.5초 block 사이 gradient 연결을 유지합니다.
- `model.model_config.self_forced.detach_block_transition=true` 는 전체 rollout 길이는 그대로 유지하되,
  매 0.5초 commit 이후 다음 block 입력 상태의 gradient만 끊습니다.
- 이 설정은 self-forcing 학습용 `training_rollout_from_cache` 경로에만 적용됩니다.
  validation / submission rollout 경로는 그대로 둡니다.
- 이미 생성된 `pred_*` / `committed_*` 출력은 끊지 않습니다.
  그래서 각 0.5초 실행 결과는 자기 loss로 직접 학습되고, 뒤쪽 loss가 앞쪽 실행 상태를 거꾸로 조작하는 경로만 막습니다.

예시:

```bash
... model.model_config.self_forced.detach_block_transition=true
```

### Self-forced Trainable Range

- `model.model_config.self_forced.unfrozen_range=except_map_encoder` 가 기본값입니다.
  기존 self-forcing의 `freeze_map_encoder=true` 와 같은 의도이며, map encoder만 고정하고 나머지 generator / generated estimator는 학습할 수 있게 둡니다.
- `unfrozen_range=middle` 은 map encoder와 대부분의 agent 문맥부를 고정하고,
  `agent_encoder.flow_decoder` 와 마지막 temporal / map-to-agent / agent-to-agent 문맥 블록만 엽니다.
  즉 `except_map_encoder` 보다 더 보수적이고, `full_flow_decoder` 보다는 덜 보수적인 중간 설정입니다.
- `unfrozen_range=full_flow_decoder` 는 draft fine-tuning의 `train_full_flow_decoder_only=true` 와 같은 의도입니다.
  지도/장면/상호작용 해석부는 pretrained 상태로 보존하고, 자기 rollout 분포 차이는 마지막 궤적 생성부가 흡수하게 합니다.
- self-forcing용 기존 `freeze_map_encoder` 설정은 제거했습니다. 같은 동작이 필요하면 `unfrozen_range=except_map_encoder` 를 사용합니다.

예시:

```bash
... model.model_config.self_forced.unfrozen_range=full_flow_decoder
```

<!-- CATK_ROAD_FINE_TUNING_SECTION -->
## RoaD closed-loop fine-tuning

이 학습법은 self-forcing과 무관한 독립 fine-tuning 경로입니다. 시작점은 Flow Matching pretrained checkpoint이며, 매 epoch마다 현재 모델로 closed-loop RoaD cache를 새로 만들고 그 epoch 학습에만 사용한 뒤 삭제합니다.

### 핵심 설정

| 항목 | 값 |
|---|---:|
| action | `road_finetune` |
| 시작 checkpoint | Flow Matching pretrained checkpoint |
| fine-tuning epoch | 32 |
| max learning rate | `5e-5` |
| 원본 WOMD training scenario 수 | 486,995 |
| scenario당 RoaD rollout 수 | 3 |
| epoch마다 생성되는 RoaD cache 수 | 1,460,985 |
| RoaD 1 epoch에서 실제 학습 sample 수 | 486,995 |
| candidate 수 | 64 |
| sampling temperature | 0.8 |
| closed-loop commit 단위 | 0.5초, 즉 5 step |
| 후보 선택 기준 | 첫 20 step의 사각형 4개 꼭지점 평균 거리 |
| data update frequency | always |
| 사용 완료 cache | epoch 종료 직후 삭제 |

### 실행 방법

먼저 기존 방식대로 WOMD training set을 `.pkl` cache로 만들어둡니다. 기본 원본 cache 경로는 `${paths.cache_root}/training`입니다.

```bash
bash scripts/road_flow_finetune.sh /path/to/flow_pretrained.ckpt
```

Hydra override를 직접 쓰는 경우는 다음과 같습니다.

```bash
python src/run.py \
  experiment=road_flow \
  ckpt_path=/path/to/flow_pretrained.ckpt
```

분산 학습 예시는 다음과 같습니다.

```bash
python src/run.py \
  experiment=road_flow \
  ckpt_path=/path/to/flow_pretrained.ckpt \
  trainer.devices=4 \
  trainer.num_nodes=1
```

원본 cache 경로를 바꾸려면 다음처럼 지정합니다.

```bash
python src/run.py \
  experiment=road_flow \
  ckpt_path=/path/to/flow_pretrained.ckpt \
  road.source_train_raw_dir=/data/womd_cache/training
```

### 동작 순서

1. epoch 0 시작 전, 현재 모델로 원본 WOMD training cache를 순회합니다.
2. 각 scenario마다 RoaD rollout 3개를 생성합니다.
3. 각 0.5초 block마다 현재 closed-loop scene에서 후보 64개를 temperature 0.8로 새로 만들고, GT future와 사각형 4개 꼭지점 평균 거리가 가장 작은 후보를 agent별로 고릅니다.
4. 선택된 후보의 앞 0.5초만 scene에 반영하고, 이 과정을 16번 반복해 8초 future를 만듭니다.
5. 선택된 future를 기존 WOMD `.pkl` schema와 같은 RoaD `.pkl` cache로 저장합니다.
6. 생성된 3N개 cache 중 scenario마다 하나만 균등하게 골라 selected cache 폴더를 만듭니다.
7. selected cache N개만 사용해 1 epoch 학습합니다. 따라서 optimizer update 수는 기존 CAT-K fine-tuning 1 epoch와 같습니다.
8. epoch 종료 후 다음 epoch용 RoaD cache를 최신 모델로 다시 만들고, 이미 사용한 이전 epoch cache는 삭제합니다.
9. 이 과정을 32 epoch 반복합니다.

### 저장 구조

기본 저장 위치는 `${paths.output_dir}/road_cache`입니다.

```text
road_cache/
  epoch_000/
    all/
      variant_00/  # N개
      variant_01/  # N개
      variant_02/  # N개
    selected/      # 실제 학습에 쓰는 N개
```

`selected/`는 hardlink를 먼저 시도하고, 파일 시스템이 지원하지 않으면 복사합니다. 한 epoch 학습이 끝나면 해당 epoch 폴더는 삭제됩니다.

### 주의사항

`trainer.reload_dataloaders_every_n_epochs=1`이 필요합니다. 새 epoch마다 selected cache 폴더가 바뀌기 때문입니다. `road_flow` 실험 설정에는 이 값이 이미 들어 있습니다.

`road.candidate_micro_batch_size`는 기본 4입니다. K=64 후보를 한 번에 모두 만들지 않고 작은 묶음으로 나누어 생성하므로, GPU 메모리가 부족하면 1 또는 2로 낮추면 됩니다.

### Type-aware Control Yaw Normalization

- control-space Flow Matching에서 position scale은 계속 `1.0m` 공통값을 씁니다.
- yaw scale은 agent type별로 적용합니다: vehicle `0.025rad`, cyclist `0.06rad`, pedestrian `0.20rad`.
- 이 변경은 정규화/역정규화 좌표계만 바꾸며, decoder 구조와 loss 구성은 바꾸지 않습니다.
- 학습 target 생성, metric용 pose 복원, round-trip error 계산 모두 같은 type-aware yaw scale을 씁니다.
