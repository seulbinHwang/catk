# CAT-K Flow Matching

이 저장소는 **flow matching 학습/추론/평가 전용**으로 정리된 버전입니다.
기본 실행 경로와 문서, 스크립트는 모두 `smart_flow` 계열만 사용하며 CrossEntropy 기반 next-token 경로는 제거했습니다.
현재 closed-loop local 평가는 **TrajTok Fast WOSAC 2025 metric**을 사용하고, 제출 export는 **Waymo 2025 Sim Agents** 형식을 사용합니다.

- 기존 SMART의 map/context trunk를 그대로 재사용하고, agent 쪽만 flow decoder로 바꿔 scene-context 품질을 유지합니다.
- `FlowTokenProcessor`는 18-token context pack과 16개 NTP-aligned anchor를 만들고,
  tail anchor는 존재하는 future prefix에만 Flow loss를 줍니다.
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
- stop-motion gate는 전 경로에서 사용하지 않습니다. `decoder.use_stop_motion` 또는
  `self_forced.use_stop_motion` 값을 외부에서 true로 넘겨도 내부 rollout은 항상 false로 동작합니다.
- `model.model_config.decoder.use_lqr=true` 를 켜면 vehicle / bicycle에만
  curvature-domain LQR + kinematic bicycle commit bridge를 적용합니다. 이 모드에서는 2초 FM
  미래를 preview로 보되, 실제 반영은 항상 다음 0.5초 / 5점만 실행합니다.
- control-space Flow에서는 LQR가 raw control tensor를 직접 pose 미래로 오해하지 않도록,
  `use_holonomic_model_only=true/false` 설정에 맞춰 2초 control output을 pose-space reference로
  먼저 복원한 뒤 LQR reference로 사용합니다.
- LQR bridge는 최근 실제 10Hz 6점 history로 현재 speed / yaw-rate / curvature를 잡고,
  차종별 속도, 가감속, yaw-rate, 횡가속, 최소 선회 반경 제한을 같이 씁니다.
- wheelbase가 없는 WOMD multi-agent 특성을 고려해 steering angle 대신 **curvature를 제어 입력**
  으로 쓰는 kinematic bicycle 계열 적분을 쓰며, class별 envelope로 곡률과 곡률 변화율을 한 번 더
  clip 합니다.
- heading 2-vector와 pedestrian velocity 2-vector는 raw `atan2` 대신 safe angle 복원으로 처리해
  `(0, 0)` 또는 near-zero vector backward에서 gradient NaN이 나지 않도록 막습니다.
- 학습 중에는 `fm_loss`와 `total_loss` 같은 scalar loss의 non-finite를 fail-fast로 감지합니다.
  정상 step의 속도를 위해 전체 parameter/gradient tensor를 매 step 스캔하지는 않습니다.
- closed-loop local 평가는 `SimAgentsMetrics`가 vendored TrajTok Fast WOSAC 2025 evaluator를 호출해 `val_closed/sim_agents_2025/*`와 `val_closed/sim_agents_2025_mean/*`를 기록합니다.
- submission export는 `SimAgentsSubmission`이 2025 submission shard와 `sim_agents_2025_submission.tar.gz`를 생성합니다.
- 설치 시점에 2025 Sim Agents proto와 `traffic_light_violation` 관련 2025 필드가 실제로 있는지 바로 검증합니다.

### Map-Agent Radius Edge Sorting

`SMARTAgentEncoder.build_map2agent_edge`는 map polyline token에서 agent context token으로 가는 radius edge를 만듭니다. 이때 `torch_cluster.radius`도 `radius_graph`처럼 batch index가 단조 비감소 순서로 들어온다는 silent 가정을 갖습니다.

Flow decoder의 multi-step context 경로는 agent 쪽 batch를 `tokenized_agent["batch"].repeat(n_step)`으로 만들기 때문에 step 사이에서 scene 번호가 다시 작아질 수 있습니다. 이 값을 그대로 넘기면 cross-scene edge가 생기지는 않더라도, 같은 scene 안의 map-agent edge가 조용히 누락되어 일부 agent/time-step이 자기 scene의 지도 정보를 받지 못합니다.

그래서 `build_map2agent_edge`는 `radius` 호출 직전에 agent 쪽과 map 쪽 batch를 모두 안정 정렬하고, 반환된 edge index를 원래 node 순서로 되돌립니다. downstream relation feature와 attention 입력 shape은 그대로 유지되며, 학습과 closed-loop inference 양쪽에서 의도한 map context를 빠짐없이 받게 하는 correctness fix입니다.

회귀 테스트 `test_map2agent_edge_no_silent_drop_cpu` / `_gpu`는 production 패킹과 같은 `batch.repeat(n_step)` 입력에서 brute-force로 계산한 기대 edge 수와 실제 생성 edge 수가 일치하는지 확인합니다.

### Fast WOSAC Metric

- TrajTok의 `wosac_fast_eval_tool.fast_sim_agents_metrics` 구현을 `src/smart/metrics/wosac_fast_eval_tool/` 아래에 vendoring했습니다.
- production closed-loop metric 경로는 더 이상 Waymo 공식 TensorFlow scorer를 호출하지 않고, torch tensor rollout을 Fast WOSAC 입력 형태 `[n_rollout, n_agents, n_step, (x,y,z,yaw)]`로 직접 변환해 평가합니다.
- 공식 scorer와의 수치 차이는 아래 스크립트로 검증합니다. local split validation TFRecord 3개, 32 rollout 기준 최대 절대오차는 `3.5762786865234375e-07`로 `1e-6` 기준을 통과했습니다.

```bash
conda run -n catk python tools/compare_fast_wosac_metric.py \
  --num-scenarios 3 \
  --threshold 1e-6 \
  --device cpu \
  --json-output artifacts/fast_wosac_compare_3scenarios.json
```

### 정적 교통 신호 Map Feature

- `f6e96cf8`의 동적 traffic-light staleness 입력은 `semi_control_stable`에서 사용하지 않습니다.
- 교통 신호는 다시 map token의 정적 categorical feature로만 들어갑니다.
- `SMARTMapDecoder`는 cache의 `light_type`을 map point embedding에 더하고, encoder 출력에는 별도 `light_type` metadata를 넘기지 않습니다.
- agent-lane relation은 `distance / bearing / relative heading` 3D 기하 정보만 사용합니다.
- `prediction_time - observed_light_time` scalar를 만들지 않으므로, pretrain / CAT-K fine-tuning / RoaD fine-tuning / closed-loop validation / WOSAC 제출 경로 모두 rollout block 진행 시간에 따라 신호 입력이 바뀌지 않습니다.
- 현재 관측 신호 상태 자체는 cache에 저장된 현재 map token feature로만 소비하며, 미래 traffic-light 상태를 입력하지 않습니다.
- 이 구조는 동적 stale relation feature를 제거하고 static map-token `light_type` embedding을 복원하므로, 동적 stale 입력을 전제로 학습한 checkpoint와는 구조적으로 호환되지 않습니다. 이 브랜치 최신 pretrain을 기준으로 사용합니다.

### Motion Missingness Feature

- flow encoder의 motion feature는 이제 `motion value = 0` 과 별도 `motion_valid` 입력을 함께 씁니다.
- 첫 context step 또는 invalid/valid 경계처럼 이전 coarse motion을 정의할 수 없는 경우에는 motion 값을 `0`으로 두고 `motion_valid=0`으로 표시합니다.
- 실제로 유효한 연속 coarse step에서 정지한 agent는 motion 값이 `0`이어도 `motion_valid=1`이므로 missing motion과 구분됩니다.
- agent-agent relation은 edge마다 relative motion을 다시 붙이지 않고 `distance / bearing / relative heading` 3D 기하 정보만 사용합니다.
- 대신 a2a radius graph를 만들 때 유효하지 않은 agent state를 neighbor 후보에서 먼저 제외합니다.
- 따라서 missingness 정보는 agent node feature에만 있고, 주변 agent의 missingness는 sender/receiver hidden state를 통해 전달됩니다. 이 구조는 motion missingness 의미를 유지하면서 edge 수에 비례하던 relative-motion embedding 비용을 제거합니다.
- 이 설계는 `x_a_emb` 입력 차원을 바꾸므로, 예전 pretrained checkpoint와의 호환 경로는 제공하지 않습니다. 또한 6D a2a relation을 쓰던 중간 checkpoint도 현재 3D a2a relation 구조와 shape이 맞지 않습니다. 새 pretrain을 기준으로 사용합니다.

### Graph Attention FP32 Aggregation

`bf16-mixed` 학습에서는 q/k/v projection, FFN 같은 dense 연산은 mixed precision 이득을
유지하되, PyG `MessagePassing` 기반 graph attention aggregation만 fp32로 계산할 수
있습니다. `CATK_ATTENTION_GRAPH_FP32=1`이면 `AttentionLayer`가 sparse softmax,
message 생성, scatter aggregation 구간을 autocast 밖에서 fp32로 실행한 뒤 원래 dtype으로
되돌립니다. 모델 구조, 파라미터 수, edge set, loss target은 바뀌지 않습니다.

이 경로는 A100에서 bf16 sparse graph attention backward가 느려지는 현상을 완충하기 위한
실행 경로입니다. `scripts/h100x4_multinode_pretrain.sh`는 기본으로
`CATK_ATTENTION_GRAPH_FP32=1`을 설정합니다. 필요하면 실행 전에
`CATK_ATTENTION_GRAPH_FP32=0`을 명시해 기존 pure autocast 경로로 되돌릴 수 있습니다.

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
이 스크립트는 기본적으로 Nubes 다운로드를 `-j 96` 병렬 작업으로 수행합니다. 다른 병렬도를 쓰려면 `NUBES_JOBS` 환경변수로 override합니다.

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

기본값은 `data.train_use_eval_agent_selection=true` 입니다.

- `false`면 기존과 같습니다. 학습 입력 agent는 ego 기준 150m 안만 남기고, 학습 대상은 ego/예측 특별 대상과 ego 기준 100m 안이면서 미래 유효 길이가 충분한 agent 중 최대 `data.train_max_num`개를 사용합니다.
- `true`면 학습에서도 validation/추론용 transform을 그대로 사용합니다. 따라서 별도의 150m 입력 제한과 `train_mask` / `train_max_num` 제한을 추가하지 않습니다. 이 경우 학습 입력 agent와 학습 대상 anchor가 validation/추론과 같은 기준으로 정해집니다.
- 이 설정은 pretrain과 Flow Matching range fine-tuning에 동일하게 적용됩니다.

예시:

```bash
# pretrain에서 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true
```

### 5.1.2 NTP-Aligned Tail-Prefix Flow Supervision

Flow pretrain의 기본 target coverage는 SMART NTP와 맞춘 `18-token / 16-anchor` 구조입니다.

- encoder 입력 coarse context는 raw step `5, 10, ..., 90`에 해당하는 18개 slot을 사용합니다.
- Flow 학습 anchor는 raw step `10, 15, ..., 85`의 16개 현재 상태입니다.
- temporal attention은 과거에서 현재 방향으로만 연결되므로, 각 anchor hidden은 자기 시점까지의 context만 봅니다. 마지막 raw step `90` context는 encoder 입력에는 들어가지만 target anchor로는 쓰지 않습니다.
- `use_prefix_valid_future_loss_mask=true`가 기본값입니다. 이때 raw step `75/80/85` tail anchor는 각각 `15/10/5`개 fine-step prefix만 loss와 future-step decoding attention에 포함합니다.
- loss는 anchor 평균이 아니라 실제 유효 fine-step 기준으로 평균냅니다. 따라서 0.5초짜리 tail anchor가 2초짜리 anchor와 같은 weight를 받지 않습니다.

전체 agent가 끝까지 유효한 2초 horizon 기준 supervision 수는 기존 `13 x 20 = 260` fine-step에서 `13 x 20 + 15 + 10 + 5 = 290` fine-step으로 늘어납니다. 추가되는 3개 anchor는 closed-loop rollout 후반부 query state이므로, pretrain에서 직접 보는 현재 상태 coverage가 넓어집니다.

학습 target 선택은 아래 옵션으로 고릅니다.

```bash
model.model_config.token_processor.use_prefix_valid_future_loss_mask=true   # 기본 Tail-Prefix 방식
model.model_config.token_processor.use_prefix_valid_future_loss_mask=false  # full-window anchor만 학습
```

- `true`이면 현재 anchor 뒤 가장 가까운 미래부터 시작해서, 처음 끊기기 전까지 연속으로 유효한 prefix만 학습합니다. 없는 suffix는 target을 억지로 정지/반복으로 만들지 않고, loss와 decoder future-step attention에서 완전히 mask 처리합니다.
- full-valid sample은 `true`에서도 그대로 전체 미래 loss를 받습니다. 새로 추가되는 것은 partial-valid tail sample뿐입니다.
- `false`이면 `18-token / 16-anchor` slot shape은 유지하지만, 현재 anchor 뒤 `decoder.flow_window_steps` 전체 미래가 모두 유효한 agent-anchor만 학습합니다. 기본 WOMD 91-step horizon에서는 raw step `75/80/85` tail anchor가 제외되어 실질적으로 기존 13개 full-horizon anchor만 loss를 받습니다.
- 이 옵션은 `FlowTokenProcessor`에서 학습 target을 만들 때 적용되므로 pretrain, 일반 fine tuning, self-forced fine tuning에서 같은 방식으로 동작합니다.
- README 기준 cache를 그대로 만들었다면 cache 재생성은 필요 없습니다. pkl cache 자체에서 partial-valid agent/anchor를 직접 삭제한 경우에만 cache를 다시 만들어야 합니다.

Map/agent trajectory token matching은 항상 deterministic argmin으로 수행합니다. 이전의 `map_token_sampling.num_k/temp`, `agent_token_sampling.num_k/temp` top-k sampling 옵션은 기본값 `num_k=1`에서 쓰이지 않던 경로였으므로 제거했습니다. 따라서 학습, fine tuning, validation, closed-loop inference, WOSAC submission 모두 같은 token matching 규칙을 사용합니다.

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

기본 epoch116 launcher tmux 확인:

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
- `model.model_config.token_processor.use_holonomic_model_only=true`를 켜면 ablation용으로 vehicle / cyclist에도 pedestrian과 같은 holonomic decoder를 적용합니다. 이때 vehicle / cyclist의 `delta_n`도 학습 target과 rollout 복원에 사용됩니다. 기본값 `false`는 기존 agent-type-aware non-holonomic/holonomic 혼합 방식입니다.
- `model.model_config.token_processor.use_rolling_supervision=true`가 기본값이며, label 생성은 decoder-consistent rolling projection 방식입니다. 매 step마다 raw GT 현재 pose가 아니라 직전 control을 kinematic decoder에 통과시킨 pose를 다음 inverse의 현재 pose로 씁니다. `false`로 두면 각 step의 raw GT pose pair만으로 inverse control label을 만듭니다. `use_holonomic_model_only=true`일 때는 holonomic decoder가 raw GT를 정확히 따라가므로 `use_rolling_supervision` 값에 따른 target 차이가 없습니다.
- `model.model_config.token_processor.control_vehicle_no_slip_point_ratio`와 `model.model_config.token_processor.control_cyclist_no_slip_point_ratio`는 vehicle / cyclist의 non-holonomic 제약을 box center가 아니라 box center 뒤쪽의 effective no-slip point에 적용합니다. offset은 각 agent type별로 `ratio * WOMD box length`이며, 최종 metric/rollout pose는 여전히 box center로 복원됩니다. 기본값은 전체 SMART cache training split에서 추정한 `vehicle=0.2289518863`, `cyclist=0.0495847873`입니다. 이 값으로 validation residual median은 vehicle `0.0958m -> 0.0203m`, cyclist `0.0222m -> 0.0213m`로 줄었습니다. 두 값을 모두 `0.0`으로 두면 기존 box-center midpoint arc rule과 같은 동작입니다. pedestrian과 `use_holonomic_model_only=true` 경로에서는 이 값이 적용되지 않습니다. 학습 target 생성과 decoder rollout은 이 token processor 값을 단일 기준으로 공유하므로, decoder 쪽 동일 key만 따로 다르게 override하면 실행 초기에 에러를 냅니다.
- 실제 cache에서 no-slip point ratio를 추정하려면 아래 도구를 씁니다. `CACHE_ROOT`는 README의 cache 생성 절차에서 쓰는 동일한 경로이며, fitting은 training split만 사용하고 validation split은 residual 개선율 확인에만 사용합니다.

```bash
export CACHE_ROOT=/path/to/womd_v1_3/SMART_cache
python tools/estimate_control_no_slip_rho.py \
  --fit-split training \
  --eval-split validation \
  --num-workers 8 \
  --output-json outputs/control_no_slip_rho.json
```

  이 도구는 vehicle / cyclist를 분리해서 0.5초 sliding segment를 만들고, agent별 bounded weighted median을 먼저 구한 뒤 type별 capped weighted median으로 `rho_vehicle`, `rho_cyclist` 후보를 출력합니다. segment filter는 `|p[t+5]-p[t]| >= 0.25m`, `|2L sin(delta_yaw/2)| >= 0.10m` 두 개만 씁니다. 전체 SMART cache에서 `training 486,995`개 파일로 fitting하고 `validation 44,097`개 파일로 residual을 확인한 기본 후보는 `rho_vehicle=0.2289518863`, `rho_cyclist=0.0495847873`입니다. 출력의 `residual` 항목은 `before = b`, `after = b - rho * c`의 validation median absolute residual과 개선율입니다. validation 개선율이 양수인지 먼저 확인한 뒤 실험 config에 반영하세요.
- control-space 정규화는 위치 이동량에는 공통 `control_pos_scale_m=1.0`을 쓰고, yaw에는 config로 관리되는 agent type별 scale을 씁니다. 기본 preset은 `control_vehicle_yaw_scale_rad=0.025`, `control_cyclist_yaw_scale_rad=0.06`, `control_pedestrian_yaw_scale_rad=0.20`입니다. control-space target 생성과 복원 경로에는 항상 `agent_type`이 필요합니다. metric/rollout용 pose-space 복원은 기존 규약대로 위치를 `x/20`, `y/20`으로 정규화합니다.
- control-space 학습에서는 GT pose를 control label로 만든 뒤 다시 pose로 복원했을 때, loss에 들어가는 미래 step 기준 최대 위치 오차가 `control_round_trip_max_position_error_m`보다 큰 anchor를 학습에서 제외합니다. 기본값은 `0.5m`이며, 평가 경로에는 적용하지 않습니다. 기본 no-slip point ratio 기준 전체 training cache 분석에서 `0.5m`는 약 `0.21%`의 anchor만 제거하면서 vehicle round-trip tail을 크게 줄인 값입니다.
- `control_round_trip_max_position_error_m` 값을 데이터 분포에서 고르려면 training cache에 대해 아래 분석 도구를 먼저 돌립니다. 이 값은 anchor별로 “loss에 실제 들어가는 미래 step들의 GT -> control -> pose 복원 위치 오차 중 최대값”을 기준으로 집계하므로, 학습 필터가 보는 값과 같은 의미입니다.

```bash
export CACHE_ROOT=/path/to/womd_v1_3/SMART_cache
python tools/analyze_control_round_trip_error.py \
  --split training \
  --flow-window-steps 20 \
  --thresholds 0.5,1,1.5,2,3,5,10 \
  --num-workers 48 \
  --chunksize 512 \
  --output-json outputs/control_round_trip_training.json
```

  prefix-valid 실험이면 `--use-prefix-valid-future-loss-mask`를 같이 켜고, `use_holonomic_model_only`, `use_rolling_supervision`, `control_vehicle_no_slip_point_ratio`, `control_cyclist_no_slip_point_ratio`를 바꾼 실험이면 도구에도 같은 옵션을 넘겨야 합니다. 이 분석 도구는 파일 내부 anchor 계산을 NumPy로 벡터화하고 worker별 file chunk 단위로 histogram을 합쳐 IPC 비용을 줄입니다. 현재 컨테이너에서는 `--num-workers 48 --chunksize 512`가 training `486,995`개 파일을 20분 안에 끝내는 기준 설정입니다. 출력은 전체/vehicle/pedestrian/cyclist별 anchor max error percentile, step error percentile, threshold별 anchor 제거율을 포함합니다. 기본 추천값은 전체 anchor max error의 p99.5를 `0.25m` 단위로 올림한 값입니다. 실전적으로는 이 추천값과 threshold table을 함께 보고, 정상적인 대다수 anchor를 유지하면서 명백한 non-holonomic projection outlier만 제거하는 값을 고릅니다.
- 추가 trajectory loss, x0 loss, open-loop auxiliary loss, 속도/가속도/yaw-rate 제약 loss는 이 옵션에서 새로 추가하지 않습니다. 학습 loss는 control-space Flow Matching loss 하나입니다.
- validation / rollout / metric 경로에서는 control 예측을 기존 pose-space 표현으로 복원해 기존 open-loop metric과 closed-loop rollout을 그대로 계산합니다.

pose-space checkpoint와 control-space checkpoint는 Flow decoder 입출력 차원이 다르므로 서로 섞어 resume하지 않는 것을 권장합니다. 기존 pose-space pretrain weight를 control-space 실험의 초기값으로 재사용하려면 Flow decoder head/encoder 차원 차이를 어떻게 처리할지 별도 migration 정책이 필요합니다.

#### hsb-npc-training/hsb-npc-training2 H100x4x2 control-space pretrain

`hsb-npc-training`, `hsb-npc-training2` 두 H100x4 pod를 묶어 control-space Flow Matching pretrain을 돌릴 때는 아래 launcher를 씁니다.

```bash
python scripts/launch_pre_bc_flow_control_h100x4x2_hsb_static_pods.py --replace
```

이 launcher는 `configs/experiment/pre_bc_flow_control_2x4_h100.yaml`을 사용합니다. 해당 preset은 `pre_bc_flow_2x4_h100`의 2-node H100 학습 설정을 유지하면서 `pre_bc_flow_control_4_h100.yaml`과 같은 control-space 설정을 켭니다. 기본 lr은 H100x4x2 global batch `208` 기준 `6e-4`입니다.

```yaml
model:
  model_config:
    token_processor:
      use_kinematic_control_flow: true
      use_holonomic_model_only: false
      use_rolling_supervision: true
      use_prefix_valid_future_loss_mask: false
      control_pos_scale_m: 1.0
      control_vehicle_no_slip_point_ratio: 0.2289518863
      control_cyclist_no_slip_point_ratio: 0.0495847873
      control_vehicle_yaw_scale_rad: 0.025
      control_pedestrian_yaw_scale_rad: 0.20
      control_cyclist_yaw_scale_rad: 0.06
      control_round_trip_max_position_error_m: 0.5
```

기본 실험 이름은 `flow_control_space_pretrain_h100x4x2_fullvalid_roundtrip05_lr6e-4_bs26`이고, tmux session 이름은 `catk-control-pretrain-h100x4x2`입니다. 기본 `train_batch_size`는 `26`이라 effective global batch는 `208`이며, 기본 lr은 `6e-4`입니다. `use_prefix_valid_future_loss_mask=false`라 전체 2초 미래가 유효한 anchor만 학습하고, `control_round_trip_max_position_error_m=0.5`로 round-trip 이상치 anchor를 거릅니다. CUDA OOM이 발생하면 전체 multi-node job을 정리한 뒤 rank 0의 최신 `epoch_last.ckpt`를 기준 checkpoint로 확정하고 peer pod로 동기화한 다음 `train_batch_size`를 `2`씩 낮춰 재개합니다. 기본 fallback은 `26 -> 24 -> 22 -> ... -> 2`입니다.

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

#### hsb-npc-training/wo-pvc-2 H100x4+H100x2 prefix-valid default no-slip ratio pretrain

`hsb-npc-training`의 H100 4장과 `wo-pvc-2`의 H100 2장을 묶어 `semi_control_stable` 최신 prefix-valid default no-slip control-space pretrain을 돌릴 때는 아래 launcher를 씁니다. 이 launcher는 기존 running pod 안의 tmux session과 학습 프로세스만 만들거나 교체하며, pod를 새로 만들거나 재시작하지 않습니다.

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py --replace
```

이 launcher는 아래 preset과 static pod 구성을 사용합니다.

```text
configs/experiment/pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip.yaml
scripts/launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pod / GPU | `hsb-npc-training` 4 H100 + `wo-pvc-2` 2 H100 = 총 6 rank |
| cache root | 두 pod 모두 `/workspace/womd_v1_3/SMART_cache` |
| context / anchor | `18-token / 16-anchor` Tail-Prefix supervision |
| Flow target mask | `use_prefix_valid_future_loss_mask=true` |
| control target | kinematic control-space, rolling supervision, default no-slip ratio |
| round-trip filter | `control_round_trip_max_position_error_m=0.5` |
| model parameters | 총 `7,045,051`개, trainable `7,045,051`개 |
| precision | `bf16-mixed` |
| batch / lr | per-rank `train_batch_size=20`, effective global batch `120`, `lr=6e-4` |
| fit-time validation | preset과 launcher override 모두 `check_val_every_n_epoch=16`으로 고정, 64 epoch 중 15/31/47/63 epoch 이후 4회 평가 |
| metadata | `${REMOTE_LOG_DIR}/dataset_metadata/womd_training_memory_balance_h100x6_hsb_wo_pvc2.pt` preflight 생성/검증 |

두 pod의 local GPU 수가 `4 + 2`로 다르기 때문에 homogeneous `torchrun --nproc_per_node=4`를 쓰면 안 됩니다. 이 launcher는 `--manual-rank-offsets` 경로를 사용해 `hsb-npc-training`에 rank `0~3`, `wo-pvc-2`에 rank `4~5`를 직접 배정하고, `HeterogeneousTorchElasticEnvironment` / `HeterogeneousDDPStrategy`로 Lightning의 homogeneous world-size 가정을 완화합니다. sampler, validation sharding, Fast WOSAC scorer는 launcher가 넣은 실제 `WORLD_SIZE=6`을 기준으로 동작하도록 회귀 테스트로 고정합니다.

H100 4+2 batch size probe 결과:

| per-rank batch | probe 결과 | worst peak reserved | 판단 |
|---:|---:|---:|---|
| 22 | OOM | - | 제외 |
| 21 | 12-step 성공 | `80613 / 81559 MiB` | full epoch 마진 부족 |
| 20 | 64-step 성공 | `77739 / 81559 MiB` | 기본값 |

따라서 이 6 H100 조합에서는 `train_batch_size=20`을 기본 시작값으로 둡니다. `bs22`는 `wo-pvc-2`에서 실제 OOM이 났고, `bs21`은 짧은 probe는 통과했지만 peak가 약 `80.6GB / 81.6GB`라 full epoch 안정권으로 보지 않습니다. `bs20`은 64-step probe에서 peak 약 `77.7GB / 81.6GB`로 통과했습니다. training split `486,995`개 / global batch `120` 기준 한 epoch는 약 `4,059` step입니다. launcher의 기본 OOM fallback은 `20 -> 19 -> 18 -> ... -> 12`이며, fallback이 발생하면 최신 rank-0 `epoch_last.ckpt` 또는 `last.ckpt`를 peer pod로 동기화한 뒤 재개합니다. 더 낮은 batch까지 자동 재시도해야 하면 `--min-bs`를 더 낮게 override합니다.

Agent tokenization의 첫 valid 이전 token-step 외삽은 agent별 Python loop 대신 batch mask/index 연산으로 처리합니다. 외삽 규칙은 기존과 같습니다. 첫 valid step을 기준으로 직전 coarse token boundary까지 `vel[first_valid] * 0.1` 간격으로 위치를 뒤쪽으로 채우고, velocity/heading/valid도 같은 prefix 구간에 복사합니다. `t=10`인데 raw step 5가 invalid인 history 보강 예외도 유지합니다. H100 4+2, per-rank `train_batch_size=15`, 6-rank 평균 profile 기준으로 이 변경은 외삽 구간을 `35.29ms -> 0.49ms`로 줄였고, token processor 전체는 `99.70ms -> 65.43ms`, 전체 train step은 `1133.86ms -> 1107.43ms`로 줄었습니다. 기존 loop reference 대비 위치/heading/velocity 오차는 `1e-6` 이하이며 valid mask는 동일합니다.

Agent trajectory token matching도 coarse step을 하나씩 반복하지 않고, 모든 coarse segment query를 `[agent, coarse_step]` 축으로 묶어 처리합니다. global contour는 segment window 전체에 대해 한 번에 만들고, type별 token bank argmin은 전체 valid query를 모은 뒤 chunked deterministic argmin으로 계산합니다. invalid segment는 기존과 같이 최종 `valid_mask=false`, token index/pose/heading 0으로 유지합니다. 기존 loop reference 대비 `valid_mask`, token index는 exact match이고, pose/heading은 `1e-6` 이하로 일치합니다. 이 경로는 token processor 공통 경로라 train, fine tuning, validation, closed-loop rollout, WOSAC submission이 모두 같은 deterministic token matching을 사용합니다.

H100 4+2, per-rank `train_batch_size=15`, 6-rank 평균 profile 기준:

| 구간 | 이전 | batched matching | 변화 |
|---|---:|---:|---:|
| token processor 전체 | `63.91ms` | `44.85ms` | `-29.8%` |
| `tokenize_agent` | `42.55ms` | `23.67ms` | `-44.4%` |
| `match_agent_token` | `29.67ms` | `10.65ms` | `-64.1%` |
| contour 생성 | `6.79ms` | `0.72ms` | `-89.4%` |
| token argmin | `20.06ms` | `9.53ms` | `-52.5%` |
| 전체 train step | `1093.61ms` | `1087.18ms` | `-0.59%` |

실행 전에 실제 환경 변수와 retry wrapper만 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-control-pretrain-h100x4-h100x2-prefix-default-noslip
kubectl exec -it -n p-pnc wo-pvc-2 -c main -- tmux attach -t catk-control-pretrain-h100x4-h100x2-prefix-default-noslip
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py --stop
```

#### hsb-npc-training/wo-pvc-2 H100x4+H100x2 epoch-last artifact Fast-RMM sweep

학습이 끝난 뒤 마지막 여러 epoch 중 RMM이 가장 높은 checkpoint를 고를 때는 아래 launcher를 씁니다. 이 launcher는 W&B `epoch-last-<run_id>` artifact version들을 내려받고, 각 checkpoint에 대해 closed-loop Fast-RMM validation만 실행합니다. pod를 새로 만들거나 재시작하지 않고, 기존 `hsb-npc-training` 4 H100 + `wo-pvc-2` 2 H100 안에 tmux session만 만듭니다.

```bash
python scripts/launch_fast_rmm_epoch_sweep_h100x4_h100x2_static_pods.py --replace
```

기본값은 `x5f9g0ce` pretrain의 마지막 8개 epoch sweep에 맞춰져 있습니다.

| 항목 | 기본값 |
|---|---|
| pods | `hsb-npc-training` 4 H100 + `wo-pvc-2` 2 H100 |
| experiment | `pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip` |
| artifact prefix | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce` |
| epoch versions | `56:v52,57:v53,58:v54,59:v55,60:v56,61:v57,62:v58,63:v60` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| rollout count | `n_rollout_closed_val=32` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=48`, `limit_val_batches=auto -> 6` |
| W&B group | `fast_rmm_epoch_sweep_x5f9g0ce_rmm_only_bs48` |
| tmux session | `fast-rmm-epoch-sweep-h100x4-h100x2` |

`val_batch_size=48`은 6 rank 기준 `48 * 6 * 6 = 1728` scene을 평가합니다. 실제 probe에서 `bs96`은 PyTorch 내부 `nonzero` tensor 크기 한계로 실패했고, `bs70`은 GPU 메모리 peak가 약 `80.6GB / 81.6GB`까지 올라가 안정권으로 보지 않습니다. 그래서 기본값은 RMM scene 수를 유지하면서 OOM 여유가 있는 `bs48`입니다. 이 설정은 open-loop validation을 생략하므로 checkpoint ranking용 RMM sweep에 맞춘 경로입니다.

다른 run에 재사용할 때는 artifact prefix와 epoch-version mapping만 바꿉니다.

```bash
python scripts/launch_fast_rmm_epoch_sweep_h100x4_h100x2_static_pods.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch-versions 56:v52,57:v53,58:v54,59:v55,60:v56,61:v57,62:v58,63:v60 \
  --sweep-name fast_rmm_epoch_sweep_<run_id> \
  --wandb-group fast_rmm_epoch_sweep_<run_id>_rmm_only_bs48 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_epoch_sweep_h100x4_h100x2_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t fast-rmm-epoch-sweep-h100x4-h100x2
kubectl exec -it -n p-pnc wo-pvc-2 -c main -- tmux attach -t fast-rmm-epoch-sweep-h100x4-h100x2
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_epoch_sweep_h100x4_h100x2_static_pods.py --stop
```

#### testa/testaa A100x4x2 epoch-last artifact Fast-RMM sweep

같은 epoch-last artifact sweep을 `testa` 4 A100 + `testaa` 4 A100에서 돌릴 때는 아래 launcher를 씁니다. 동작 방식은 H100 4+2 sweep과 같고, pod를 새로 만들거나 재시작하지 않으며 기존 pod 안에 tmux session만 만듭니다.

```bash
python scripts/launch_fast_rmm_epoch_sweep_a100x4x2_testa_testaa_static_pods.py --replace
```

기본값은 `x5f9g0ce` pretrain의 마지막 8개 epoch sweep에 맞춰져 있습니다.

| 항목 | 기본값 |
|---|---|
| pods | `testa` 4 A100 + `testaa` 4 A100 |
| experiment | `pre_bc_flow_control_a100x4x2_prefix_default_noslip` |
| artifact prefix | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce` |
| epoch versions | `56:v52,57:v53,58:v54,59:v55,60:v56,61:v57,62:v58,63:v60` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| rollout count | `n_rollout_closed_val=32` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=42`, `limit_val_batches=auto -> 5` |
| W&B group | `fast_rmm_epoch_sweep_x5f9g0ce_a100x4x2_rmm_only_bs42` |
| tmux session | `fast-rmm-epoch-sweep-a100x4x2-testa-testaa` |

`val_batch_size=42`는 8 rank 기준 `42 * 8 * 5 = 1680` scene을 평가합니다. 이 경로는 open-loop validation을 생략하고 closed-loop Fast-RMM만 실행하므로, 마지막 epoch들 중 RMM 기준 최고 checkpoint를 고르는 용도입니다.

다른 run에 재사용할 때는 artifact prefix와 epoch-version mapping만 바꿉니다.

```bash
python scripts/launch_fast_rmm_epoch_sweep_a100x4x2_testa_testaa_static_pods.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch-versions 56:v52,57:v53,58:v54,59:v55,60:v56,61:v57,62:v58,63:v60 \
  --sweep-name fast_rmm_epoch_sweep_<run_id>_a100x4x2 \
  --wandb-group fast_rmm_epoch_sweep_<run_id>_a100x4x2_rmm_only_bs42 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_epoch_sweep_a100x4x2_testa_testaa_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t fast-rmm-epoch-sweep-a100x4x2-testa-testaa
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t fast-rmm-epoch-sweep-a100x4x2-testa-testaa
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_epoch_sweep_a100x4x2_testa_testaa_static_pods.py --stop
```

#### testa/testaa A100x4x2 epoch 61 Flow sample-steps Fast-RMM sweep

epoch 61 checkpoint 하나를 고정하고, Flow closed-loop rollout 안의 denoising step 수가 RMM에 미치는 영향을 보려면 아래 launcher를 씁니다. 여기서 바꾸는 값은 rollout 샘플 개수인 `n_rollout_closed_val`이 아니라, 한 rollout을 생성할 때 쓰는 Flow denoising depth인 `model.model_config.validation_rollout_sampling.sample_steps`입니다.

```bash
python scripts/launch_fast_rmm_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --replace
```

기본 설정:

| 항목 | 기본값 |
|---|---|
| pods | `testa` 4 A100 + `testaa` 4 A100 |
| experiment | `pre_bc_flow_control_a100x4x2_prefix_default_noslip` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | `61` |
| swept config | `model.model_config.validation_rollout_sampling.sample_steps` |
| sample steps | 기본 `16` |
| fixed rollout count | `n_rollout_closed_val=32` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=42`, `limit_val_batches=auto -> 5` |
| W&B group | `fast_rmm_sample_steps_sweep_epoch061_x5f9g0ce_a100x4x2_bs42` |
| tmux session | `fast-rmm-sample-steps-sweep-a100x4x2-testa-testaa` |

각 sample step 평가는 별도 DDP validation 프로세스로 순차 실행됩니다. 런처는 NCCL/TCP rendezvous 재사용 충돌을 피하려고 `--master-port`를 시작 포트로 쓰고, sample step index마다 포트를 하나씩 올려 씁니다. 기본값은 `29882..29892`입니다. 각 평가가 끝난 뒤에는 다음 평가 전 기본 15초를 대기합니다.

다른 checkpoint에 재사용할 때는 artifact version을 바꿉니다. 전체 repository 기본 실험은 `sample_steps=16`으로 통일되어 있으며, sweep launcher의 기본값도 `16`입니다.

```bash
python scripts/launch_fast_rmm_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch 61 \
  --artifact-version v57 \
  --sample-steps 16 \
  --sweep-name fast_rmm_sample_steps_sweep_epoch061_<run_id>_a100x4x2 \
  --wandb-group fast_rmm_sample_steps_sweep_epoch061_<run_id>_a100x4x2_bs42 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t fast-rmm-sample-steps-sweep-a100x4x2-testa-testaa
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t fast-rmm-sample-steps-sweep-a100x4x2-testa-testaa
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --stop
```

#### testa/testaa A100x4x2 epoch 61 Flow midpoint sample-steps Fast-RMM sweep

epoch 61 checkpoint 하나를 고정하고, Flow solver를 `midpoint`로 바꾼 상태에서 denoising step 수가 RMM에 미치는 영향을 보려면 아래 launcher를 씁니다. Fast-RMM rollout에서는 `validation_rollout_sampling.sample_method`가 실제 sampling method로 전달되므로, 이 launcher는 `model.model_config.decoder.flow_solver_method=midpoint`와 `model.model_config.validation_rollout_sampling.sample_method=midpoint`를 함께 고정합니다.

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --replace
```

기본 설정:

| 항목 | 기본값 |
|---|---|
| pods | `testa` 4 A100 + `testaa` 4 A100 |
| experiment | `pre_bc_flow_control_a100x4x2_prefix_default_noslip` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | `61` |
| fixed solver | `model.model_config.decoder.flow_solver_method=midpoint`, `model.model_config.validation_rollout_sampling.sample_method=midpoint` |
| swept config | `model.model_config.validation_rollout_sampling.sample_steps` |
| sample steps | `2,4,6,8,10,12,16,20,24,28,32` |
| stop-motion | `model.model_config.decoder.use_stop_motion=false`, `model.model_config.self_forced.use_stop_motion=false` |
| fixed rollout count | `n_rollout_closed_val=32` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=42`, `limit_val_batches=auto -> 5` |
| W&B group | `fast_rmm_midpoint_sample_steps_sweep_epoch061_x5f9g0ce_a100x4x2_bs42` |
| tmux session | `fast-rmm-midpoint-sample-steps-sweep-a100x4x2-testa-testaa` |

각 sample step 평가는 별도 DDP validation 프로세스로 순차 실행됩니다. 기본 master port 범위는 `29930..29940`입니다.

다른 checkpoint에 재사용할 때는 artifact version과 sweep 이름만 바꿉니다.

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch 61 \
  --artifact-version v57 \
  --sample-steps 2,4,6,8,10,12,16,20,24,28,32 \
  --sweep-name fast_rmm_midpoint_sample_steps_sweep_epoch061_<run_id>_a100x4x2 \
  --wandb-group fast_rmm_midpoint_sample_steps_sweep_epoch061_<run_id>_a100x4x2_bs42 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t fast-rmm-midpoint-sample-steps-sweep-a100x4x2-testa-testaa
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t fast-rmm-midpoint-sample-steps-sweep-a100x4x2-testa-testaa
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_a100x4x2_testa_testaa_static_pods.py --stop
```

#### hsb-npc-training-1 H100x6 epoch 61 Flow midpoint sample-steps Fast-RMM sweep

위 midpoint sample-steps sweep과 같은 실험을 `hsb-npc-training-1` 단일 H100x6 pod에서 돌릴 때는 아래 launcher를 씁니다. pod를 새로 만들거나 재시작하지 않고, 기존 pod 안에 tmux session만 만듭니다.

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_h100x6_hsb1_static_pod.py --replace
```

기본 설정:

| 항목 | 기본값 |
|---|---|
| pod | `hsb-npc-training-1` 6 H100 |
| experiment | `pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | `61` |
| fixed solver | `model.model_config.decoder.flow_solver_method=midpoint`, `model.model_config.validation_rollout_sampling.sample_method=midpoint` |
| swept config | `model.model_config.validation_rollout_sampling.sample_steps` |
| sample steps | `2,4,6,8,10,12,16,20,24,28,32` |
| stop-motion | `model.model_config.decoder.use_stop_motion=false`, `model.model_config.self_forced.use_stop_motion=false` |
| fixed rollout count | `n_rollout_closed_val=32` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=56`, `limit_val_batches=auto -> 5` |
| W&B group | `fast_rmm_midpoint_sample_steps_sweep_epoch061_x5f9g0ce_h100x6_hsb1_bs56` |
| tmux session | `fast-rmm-midpoint-sample-steps-sweep-h100x6-hsb1` |

`val_batch_size=56`은 6 rank 기준 `56 * 6 * 5 = 1680` scene을 정확히 평가합니다. 각 sample step 평가는 별도 DDP validation 프로세스로 순차 실행되며, 기본 master port 범위는 `29950..29960`입니다.

다른 checkpoint에 재사용할 때는 artifact version과 sweep 이름만 바꿉니다.

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_h100x6_hsb1_static_pod.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch 61 \
  --artifact-version v57 \
  --sample-steps 2,4,6,8,10,12,16,20,24,28,32 \
  --sweep-name fast_rmm_midpoint_sample_steps_sweep_epoch061_<run_id>_h100x6_hsb1 \
  --wandb-group fast_rmm_midpoint_sample_steps_sweep_epoch061_<run_id>_h100x6_hsb1_bs56 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_h100x6_hsb1_static_pod.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- tmux attach -t fast-rmm-midpoint-sample-steps-sweep-h100x6-hsb1
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_midpoint_sample_steps_sweep_h100x6_hsb1_static_pod.py --stop
```

#### testa/testaa A100x4x2 epoch 61 Flow noise-scale Fast-RMM sweep

epoch 61 checkpoint와 `sample_steps=16`은 고정하고, closed-loop rollout 시작 Gaussian noise 크기만 바꿔 RMM 변화를 보려면 아래 launcher를 씁니다. 실제 rollout에서 쓰이는 값은 `eval_sampling_noise.noise_scale`이 아니라 `model.model_config.validation_rollout_sampling.noise_scale`입니다.

```bash
python scripts/launch_fast_rmm_noise_scale_sweep_a100x4x2_testa_testaa_static_pods.py --replace
```

기본 설정:

| 항목 | 기본값 |
|---|---|
| pods | `testa` 4 A100 + `testaa` 4 A100 |
| experiment | `pre_bc_flow_control_a100x4x2_prefix_default_noslip` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | `61` |
| fixed denoising steps | `model.model_config.validation_rollout_sampling.sample_steps=16` |
| swept config | `model.model_config.validation_rollout_sampling.noise_scale` |
| noise scales | `0.7,0.8,0.9,0.95,1.0,1.05,1.1,1.2,1.3` |
| stop-motion | `model.model_config.decoder.use_stop_motion=false`, `model.model_config.self_forced.use_stop_motion=false` |
| fixed rollout count | `n_rollout_closed_val=32` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| RMM scene target | `scorer_scene_num=1680` |
| val batch / batches | per-rank `val_batch_size=42`, `limit_val_batches=auto -> 5` |
| W&B group | `fast_rmm_noise_scale_sweep_epoch061_x5f9g0ce_a100x4x2_bs42` |
| tmux session | `fast-rmm-noise-scale-sweep-a100x4x2-testa-testaa` |

각 noise scale 평가는 별도 DDP validation 프로세스로 순차 실행됩니다. 런처는 NCCL/TCP rendezvous 재사용 충돌을 피하려고 `--master-port`를 시작 포트로 쓰고, noise scale index마다 포트를 하나씩 올려 씁니다. 기본값은 `29910..29918`입니다.

다른 checkpoint나 noise scale 목록에 재사용할 때는 아래처럼 바꿉니다.

```bash
python scripts/launch_fast_rmm_noise_scale_sweep_a100x4x2_testa_testaa_static_pods.py \
  --artifact-prefix jksg01019-naver-labs/SMART-FLOW/epoch-last-<run_id> \
  --epoch 61 \
  --artifact-version v57 \
  --sample-steps 16 \
  --noise-scales 0.7,0.8,0.9,0.95,1.0,1.05,1.1,1.2,1.3 \
  --sweep-name fast_rmm_noise_scale_sweep_epoch061_<run_id>_a100x4x2 \
  --wandb-group fast_rmm_noise_scale_sweep_epoch061_<run_id>_a100x4x2_bs42 \
  --replace
```

실행 전 dry-run:

```bash
python scripts/launch_fast_rmm_noise_scale_sweep_a100x4x2_testa_testaa_static_pods.py --dry-run --replace
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t fast-rmm-noise-scale-sweep-a100x4x2-testa-testaa
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t fast-rmm-noise-scale-sweep-a100x4x2-testa-testaa
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_fast_rmm_noise_scale_sweep_a100x4x2_testa_testaa_static_pods.py --stop
```

#### Flow closed-loop antithetic rollout noise

Flow closed-loop validation/submission의 기본 Gaussian noise는 `model.model_config.validation_rollout_sampling.antithetic_pairs=true` 입니다. 이 설정은 32개 rollout noise를 모두 독립으로 뽑지 않고, 앞 16개 rollout noise를 뽑은 뒤 뒤 16개 rollout에는 같은 noise의 부호 반전값을 씁니다.

개념적 shape은 아래와 같습니다.

```text
base_noise: [16, N_agent, 95, D]
full_noise: [32, N_agent, 95, D]
full_noise[0:16]  = base_noise
full_noise[16:32] = -base_noise
```

control-space Flow에서는 `D=3` 입니다. 각 0.5초 closed-loop block은 기존과 동일하게 `full_noise[:, active_agents, k*5 : k*5 + 20, :]` 를 2초 initial noise로 잘라 씁니다. 따라서 checkpoint, model parameter, denoising step, solver는 바뀌지 않고 validation/submission sampling noise set만 더 균형 잡히게 됩니다.

끄고 싶으면 아래처럼 override합니다.

```bash
model.model_config.validation_rollout_sampling.antithetic_pairs=false
```

추가 실험 옵션으로 `model.model_config.validation_rollout_sampling.stratified_gaussian_noise=true` 를 켤 수 있습니다. 기본값은 `false` 입니다. 이 옵션은 scenario별 deterministic seed를 넘기는 closed-loop validation/submission 경로 전용이며, `antithetic_pairs=true` 와 함께 쓸 때만 유효합니다. 켜면 16개 base rollout이 각 coordinate에서 Gaussian quantile bin을 더 고르게 덮도록 만듭니다.

개념적 shape은 그대로 유지됩니다.

```text
U_base: [16, N_agent, 95, D]  # 0~1 quantile 구간을 나눠 만든 uniform 값
Z_base: [16, N_agent, 95, D]  # inverse normal CDF(Phi^-1) 적용 후 Gaussian noise
Z_full: [32, N_agent, 95, D]
Z_full[0:16]  = Z_base
Z_full[16:32] = -Z_base
```

실제 구현은 rollout chunk 크기가 바뀌어도 같은 의미를 유지하도록, rollout index별 stratum과 scenario별 permutation seed를 decoder에 넘깁니다. 따라서 OOM fallback 때문에 32 rollout을 `16+16`, `8+8+...`처럼 나눠 실행해도 같은 scenario/rollout 조합은 같은 stratified noise를 받습니다.

Fast-RMM에서 직접 켜려면 아래 override를 추가합니다.

```bash
model.model_config.validation_rollout_sampling.stratified_gaussian_noise=true
```

`flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` 의 W&B 마지막 artifact `epoch-last-x5f9g0ce:v60` 을 `hsb-npc-training-1` H100x6, `use_lqr=false`, `use_stop_motion=false`, `sample_steps=16`, `noise_scale=1.0`, `val_batch_size=56`, `scorer_scene_num=1680` 조건으로 Fast-RMM 비교한 결과는 아래와 같습니다.

| noise 방식 | RMM | CPD | CES |
|---|---:|---:|---:|
| iid Gaussian | 0.781528 | 0.200296 | 0.096098 |
| antithetic pair | 0.781846 | 0.201687 | 0.095312 |
| stratified Gaussian + antithetic pair | 0.781294 | 0.198843 | 0.095803 |

RMM 기준으로 antithetic pair가 더 높았으므로 repository 기본값은 `antithetic_pairs=true` 로 둡니다.
`stratified_gaussian_noise=true` 는 같은 조건의 H100x6 1680-scene Fast-RMM에서 RMM이 낮았으므로 기본값은 `false` 로 유지합니다.

같은 checkpoint와 Fast-RMM 조건에서 `antithetic_pairs=true` 를 고정하고 `noise_scale` 만 바꿔 비교한 결과는 아래와 같습니다.

| antithetic noise_scale | RMM | CPD | CES |
|---:|---:|---:|---:|
| 0.6 | 0.770086 | 0.102039 | 0.099064 |
| 0.7 | 0.774481 | 0.121340 | 0.096547 |
| 0.8 | 0.778329 | 0.143602 | 0.094837 |
| 0.9 | 0.780525 | 0.169850 | 0.094274 |
| 0.96 | 0.781651 | 0.187995 | 0.094656 |
| 0.97 | 0.781664 | 0.191210 | 0.094831 |
| 0.98 | 0.781816 | 0.194725 | 0.095018 |
| 0.99 | 0.781770 | 0.198165 | 0.095108 |
| 1.0 | 0.781846 | 0.201687 | 0.095312 |
| 1.01 | 0.781898 | 0.205243 | 0.095548 |
| 1.02 | 0.781821 | 0.208967 | 0.095836 |
| 1.03 | 0.781781 | 0.212847 | 0.096066 |
| 1.04 | 0.781810 | 0.216404 | 0.096263 |
| 1.05 | 0.781730 | 0.220319 | 0.096584 |

이 sweep에서는 RMM 기준 `noise_scale=1.01` 이 가장 높았습니다. 다만 현재 repository 기본값은 self-forced fine-tuning / validation submission 기본 설정을 일관시키기 위해 `model.model_config.validation_rollout_sampling.noise_scale=1.0` 입니다. `noise_scale=1.01` 은 필요할 때 override로 사용합니다.

같은 checkpoint와 Fast-RMM 조건에서 `antithetic_pairs=true`, `noise_scale=1.01`, `sample_steps=16`, `use_lqr=false`, `use_stop_motion=false` 를 고정하고 `model.model_config.validation_closed_seed` 만 바꾼 seed-bank sweep 결과는 아래와 같습니다. 336 scene quick stage로 후보를 고른 뒤, 상위 후보와 기존 seed 0을 1680 scene full stage에서 다시 평가했습니다.

| validation_closed_seed | RMM | WOSAC-CPD | CES |
|---:|---:|---:|---:|
| 4 | 0.782288 | 0.204221 | 0.095397 |
| 97 | 0.782226 | 0.204136 | 0.095197 |
| 53 | 0.782148 | 0.204055 | 0.095516 |
| 0 | 0.781992 | 0.205177 | 0.095503 |
| 7 | 0.781502 | 0.204455 | 0.095486 |

이 sweep에서는 RMM 기준 `validation_closed_seed=4` 가 가장 높았으므로 repository 기본값은 `model.model_config.validation_closed_seed=4` 로 둡니다. 이 변경은 checkpoint, model parameter, solver, denoising step, noise scale을 바꾸지 않고 closed-loop initial noise의 deterministic seed만 고정합니다.

같은 checkpoint와 Fast-RMM 조건에서 `validation_closed_seed=4`, `antithetic_pairs=true`, `sample_steps=16`, `use_lqr=false`, `use_stop_motion=false` 를 고정하고 `noise_scale` 근방을 더 촘촘히 비교한 결과는 아래와 같습니다.

| noise_scale | RMM | WOSAC-CPD | CES |
|---:|---:|---:|---:|
| 0.995 | 0.782161 | 0.198992 | 0.095206 |
| 0.9975 | 0.782207 | 0.199774 | 0.095229 |
| 1.0 | 0.782208 | 0.200665 | 0.095238 |
| 1.0033 | 0.782214 | 0.201869 | 0.095355 |
| 1.0066 | 0.782251 | 0.203052 | 0.095401 |
| 1.01 | 0.782377 | 0.204144 | 0.095362 |

추가로 같은 조건에서 `noise_scale=1.01` 위쪽 근방을 더 확인한 결과는 아래와 같습니다.

| noise_scale | RMM | WOSAC-CPD | CES |
|---:|---:|---:|---:|
| 1.008 | 0.782219 | 0.203613 | 0.095458 |
| 1.009 | 0.782094 | 0.204003 | 0.095422 |
| 1.01 | 0.782194 | 0.204474 | 0.095495 |
| 1.012 | 0.782332 | 0.204990 | 0.095516 |
| 1.014 | 0.782185 | 0.205982 | 0.095599 |
| 1.016 | 0.782407 | 0.206521 | 0.095705 |

두 fine sweep을 종합하면 RMM 기준 `noise_scale=1.016` 이 가장 높았습니다. 현재 repository 기본값은 self-forced fine-tuning / validation submission 기본 설정을 일관시키기 위해 `model.model_config.validation_rollout_sampling.noise_scale=1.0` 이며, `1.016` 은 해당 sweep 조건을 재현할 때 override로 사용합니다.

#### hsb-npc-training/wo-pvc-2 H100x4+H100x2 epoch 61 Waymo validation 제출

`flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` 학습에서 고른 epoch 61 `epoch_last.ckpt`로 validation split 전체의 Waymo Sim Agents 제출물을 만들고, Waymo 사이트에 자동 업로드하려면 아래 wrapper를 씁니다. 이 wrapper도 기존 `hsb-npc-training` 4 H100 + `wo-pvc-2` 2 H100 pod 안의 tmux session만 만들며, pod를 새로 만들거나 재시작하지 않습니다.

```bash
CKPT_PATH=/path/to/epoch_061/epoch_last.ckpt \
TASK_NAME=flow_control_waymo_val_epoch061_h100x4_h100x2 \
bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission_epoch061.sh
```

일반 checkpoint에 재사용할 때는 generic wrapper를 직접 호출합니다.

```bash
CKPT_PATH=/path/to/model.ckpt \
TASK_NAME=flow_control_waymo_val_custom_h100x4_h100x2 \
bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission.sh
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pods | `hsb-npc-training` 4 H100 + `wo-pvc-2` 2 H100 = 총 6 rank |
| experiment/action | `experiment=sim_agents_sub_flow`, `action=validate` |
| DDP strategy | 4+2 heterogeneous pod 구성을 위해 `HeterogeneousDDPStrategy` / `HeterogeneousTorchElasticEnvironment`를 launcher가 자동 지정 |
| rollout count | `n_rollout_closed_val=32` |
| Flow denoising steps | `model.model_config.validation_rollout_sampling.sample_steps=16`, 필요 시 `WAYMO_FLOW_SAMPLE_STEPS=<N>`으로 override |
| val batch | per-rank `VAL_BATCH_SIZE=48`, 필요 시 `VAL_BATCH_SIZE=24` 또는 `12`로 override |
| cache root | 두 pod 모두 `/workspace/womd_v1_3/SMART_cache` |
| output/log root | 기본 `LOG_DIR=/workspace/exp_logs`; `sim_agents_2025_submission` shard와 `tar.gz`도 이 경로 아래에 저장 |
| tmux session | `catk-flow-waymo-val-submission-h100x4-h100x2` |
| shard 처리 | `CATK_SUBMISSION_STREAM_SHARDS=1`로 `wo-pvc-2` shard를 rank 0으로 streaming 후 tar.gz 생성 |
| 자동 업로드 | `waymo_submission.enabled=true`, validation 제출만 허용, test 제출은 꺼짐 |

`CKPT_PATH`는 rank 0인 `hsb-npc-training`에서 읽을 수 있는 실제 epoch 61 checkpoint 경로여야 합니다. 다른 pod에 같은 파일이 없으면 launcher가 rank 0 checkpoint를 `wo-pvc-2`의 같은 경로로 동기화한 뒤 validation을 시작합니다. W&B artifact에서 내려받은 checkpoint를 쓰는 경우에도 최종 파일 경로를 `CKPT_PATH`에 지정하면 됩니다.

멀티 노드 제출 shard 수집은 실패 중간 파일을 최종 shard로 남기지 않도록 `.part` 임시 파일을 거쳐 완료 후 rename합니다. 네트워크가 끊기면 rank 0은 서버를 바로 종료하지 않고 송신 rank의 재시도를 기다립니다. 기본 재시도 횟수는 `CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS=16`이며 필요하면 늘릴 수 있습니다. `tar.gz` 생성은 `pigz`가 있으면 병렬 gzip을 우선 사용하고, 없으면 Python gzip으로 fallback합니다. 압축 레벨은 기본 `CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL=1`입니다.

rollout은 끝났지만 shard 수집이나 archive/upload 단계만 실패했다면 rollout을 다시 돌리지 않고 아래 finalizer로 복구합니다. 기본값은 `wo-pvc-2`의 `submission-rank04...05-*.binproto` shard를 `hsb-npc-training`의 `sim_agents_2025_submission_rank0_collect/`로 복사한 뒤, rank 0~5 shard가 모두 있는지 확인하고 archive를 만든 다음 업로드합니다.

```bash
python scripts/finalize_flow_control_h100x4_h100x2_waymo_submission.py \
  --run-dir /workspace/exp_logs/<TASK_NAME>/runs/<RUN_ID> \
  --upload
```

이미 archive까지 만들어진 상태에서 업로드만 재시도하려면:

```bash
python scripts/finalize_flow_control_h100x4_h100x2_waymo_submission.py \
  --run-dir /workspace/exp_logs/<TASK_NAME>/runs/<RUN_ID> \
  --skip-copy --skip-archive --upload
```

Waymo 자동 제출에는 로그인 상태 파일이 필요합니다. 기본 위치는 repository root 기준 아래 경로입니다.

```text
secrets/waymo/waymo_storage_state.json
```

다른 경로를 쓰려면:

```bash
WAYMO_STORAGE_STATE_PATH=/path/to/waymo_storage_state.json \
CKPT_PATH=/path/to/epoch_061/epoch_last.ckpt \
bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission_epoch061.sh
```

실행 전 dry-run으로 pod/환경 변수/Hydra override만 확인하려면:

```bash
CKPT_PATH=/tmp/fake_epoch061.ckpt \
bash scripts/start_flow_control_h100x4_h100x2_waymo_val_submission_epoch061.sh --dry-run
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-flow-waymo-val-submission-h100x4-h100x2
kubectl exec -it -n p-pnc wo-pvc-2 -c main -- tmux attach -t catk-flow-waymo-val-submission-h100x4-h100x2
```

#### hsb-npc-training-2/wo-pvc-1 H100x4+H100x2 prefix-valid default no-slip ratio pretrain

`hsb-npc-training-2`의 H100 4장과 `wo-pvc-1`의 H100 2장을 묶어 같은 `semi_control_stable` control-space pretrain을 돌릴 때는 아래 전용 wrapper를 씁니다. 이 wrapper도 기존 running pod 안의 tmux session과 학습 프로세스만 만들거나 교체하며, pod를 새로 만들거나 재시작하지 않습니다.

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_static_pods.py --replace
```

실행 전에 실제 pod, branch, task name, metadata cache 경로, retry wrapper 환경을 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_static_pods.py --dry-run --replace
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pod / GPU | `hsb-npc-training-2` 4 H100 + `wo-pvc-1` 2 H100 = 총 6 rank |
| config | `configs/experiment/pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip.yaml` |
| cache root | 두 pod 모두 `/workspace/womd_v1_3/SMART_cache` |
| task name | `flow_control_space_pretrain_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_lr6e-4_bs18` |
| tmux session | `catk-control-pretrain-h100x4-h100x2-hsb2-wo1-prefix-default-noslip` |
| 시작 batch / lr | per-rank `train_batch_size=18`, effective global batch `108`, `lr=6e-4` |
| OOM fallback | CUDA OOM 감지 시 최신 rank-0 `epoch_last.ckpt` 또는 `last.ckpt`에서 resume하며 `18 -> 17 -> 16 -> ... -> 12` 순서로 1씩 낮춤 |
| metadata | `${REMOTE_LOG_DIR}/dataset_metadata/womd_training_memory_balance_h100x6_hsb2_wo1.pt` preflight 생성/검증 |

OOM fallback은 학습 수식이나 데이터 선택 규칙을 바꾸지 않고, 실패한 시점 이후 최신 저장 checkpoint를 기준으로 재시작합니다. 단, checkpoint는 epoch 마지막 기준으로 저장되므로 OOM 직전 mini-batch까지 완전히 이어지는 것은 아니며, 저장된 최신 `epoch_last.ckpt` 또는 `last.ckpt` 기준으로 resume합니다. `bs=18`이 장기 학습에서 안정적이면 그대로 진행하고, OOM이 실제로 발생한 경우에만 batch를 1씩 낮춥니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-2 -c main -- tmux attach -t catk-control-pretrain-h100x4-h100x2-hsb2-wo1-prefix-default-noslip
kubectl exec -it -n p-pnc wo-pvc-1 -c main -- tmux attach -t catk-control-pretrain-h100x4-h100x2-hsb2-wo1-prefix-default-noslip
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_static_pods.py --stop
```

#### testa/testaa A100x4x2 prefix-valid control-space pretrain

`testa`, `testaa` 두 A100x4 pod를 묶어 control-space Flow Matching pretrain을 돌릴 때는 아래 launcher를 씁니다. H100x4x2 control-space recipe와 같은 global batch `208`, lr `6e-4`, round-trip filter `0.5m`를 쓰되, `use_prefix_valid_future_loss_mask=true`를 켭니다.

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_static_pods.py --replace
```

이 launcher는 `configs/experiment/pre_bc_flow_control_a100x4x2_prefix_valid.yaml`을 사용합니다. 해당 preset은 `pre_bc_flow_control_2x4_h100`을 상속하므로 `train_batch_size=26`, `trainer.num_nodes=2`, `trainer.devices=4`, `precision=bf16-mixed`, `lr=6e-4`, `control_round_trip_max_position_error_m=0.5`는 그대로 유지합니다.

```yaml
model:
  model_config:
    token_processor:
      use_kinematic_control_flow: true
      use_prefix_valid_future_loss_mask: true
      control_round_trip_max_position_error_m: 0.5
```

현재 브랜치의 `use_prefix_valid_future_loss_mask=true`는 가까운 미래부터 처음 끊기기 전까지의 연속 valid prefix 전체를 loss에 반영합니다.

기본 실험 이름은 `flow_control_space_pretrain_a100x4x2_prefix_roundtrip05_lr6e-4_bs26`이고, tmux session 이름은 `catk-control-pretrain-a100x4x2-prefix`입니다. CUDA OOM이 발생하면 전체 multi-node job을 정리한 뒤 rank 0의 최신 `epoch_last.ckpt`를 기준 checkpoint로 확정하고 peer pod로 동기화한 다음 `train_batch_size`를 `2`씩 낮춰 재개합니다. 기본 fallback은 `26 -> 24 -> 22 -> ... -> 2`입니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-control-pretrain-a100x4x2-prefix
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t catk-control-pretrain-a100x4x2-prefix
```

실행 전에 실제 kubectl 명령을 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_static_pods.py --dry-run
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_static_pods.py --stop
```

#### testa/testaa A100x4x2 prefix-valid default no-slip ratio pretrain

`testa`, `testaa`에서 기존 box-center ratio `0.0 / 0.0` 대비 default no-slip point ratio 효과를 보려면 아래 전용 launcher를 씁니다.

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_prefix_default_noslip_static_pods.py --replace
```

이 launcher는 `configs/experiment/pre_bc_flow_control_a100x4x2_prefix_default_noslip.yaml`을 사용합니다. 학습 shape은 A100x4x2 prefix-valid control-space pretrain과 같고, 아래 값만 실험 의도가 드러나도록 명시적으로 고정합니다.

```yaml
model:
  model_config:
    token_processor:
      use_kinematic_control_flow: true
      use_holonomic_model_only: false
      use_rolling_supervision: true
      use_prefix_valid_future_loss_mask: true
      control_vehicle_no_slip_point_ratio: 0.2289518863
      control_cyclist_no_slip_point_ratio: 0.0495847873
      control_round_trip_max_position_error_m: 0.5
```

기본 실험 이름은 `flow_control_space_pretrain_a100x4x2_prefix_default_noslip_roundtrip05_lr6e-4_bs26`이고, tmux session 이름은 `catk-control-pretrain-a100x4x2-prefix-default-noslip`입니다. 실행 전에 실제 kubectl 명령을 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_prefix_default_noslip_static_pods.py --dry-run
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_a100x4x2_prefix_default_noslip_static_pods.py --stop
```

#### V100 47GPU prefix-valid default no-slip ratio pretrain

W&B의 `flow_control_space_pretrain_a100x4x2_prefix_default_noslip_roundtrip05_lr6e-4_bs26`와 objective 설정을 맞추고, hardware만 `sv~svvvv + testsv~testsvvvv + fv~fvvvvv` V100 47GPU fleet로 바꾸려면 아래 전용 launcher를 씁니다.

```bash
python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py --replace
```

이 launcher는 아래 preset을 사용합니다.

```text
configs/experiment/pre_bc_flow_control_v100x47_prefix_default_noslip.yaml
scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py
```

비교 의도:

- A100 run과 같은 control-space / non-holonomic / default no-slip ratio / round-trip filter 설정을 유지합니다.
- `f278261 Add tail-prefix flow pretraining anchors` 이후의 `18-token / 16-anchor` Tail-Prefix supervision을 사용합니다.
- hardware만 A100x4 pod 2개에서 V100 47GPU static fleet로 바꿉니다.

기본 설정:

- pod 순서: `sv svv svvv svvvv testsv testsvv testsvvv testsvvvv fv fvv fvv fvvvv fvvvvv`
- V100x4 pod는 4 rank, V100x3 pod는 3 rank를 띄워 총 47 rank를 사용합니다.
- `model.model_config.lr=6e-4`
- `model.model_config.decoder.flow_window_steps=20`
- `model.model_config.token_processor.use_kinematic_control_flow=true`
- `model.model_config.token_processor.use_holonomic_model_only=false`
- `model.model_config.token_processor.use_rolling_supervision=true`
- `model.model_config.token_processor.use_prefix_valid_future_loss_mask=true`
- `model.model_config.token_processor.control_vehicle_no_slip_point_ratio=0.2289518863`
- `model.model_config.token_processor.control_cyclist_no_slip_point_ratio=0.0495847873`
- `model.model_config.token_processor.control_round_trip_max_position_error_m=0.5`
- `trainer.precision=16-mixed`
- `data.train_batch_size=4`, effective global batch `4 * 47 = 188`
- `data.train_memory_balanced_batches=true`
- `data.train_memory_balance_metadata_cache=${paths.log_dir}/dataset_metadata/womd_training_memory_balance_v1.pt`
- `data.train_memory_balance_build_on_missing=false`
- `trainer.use_distributed_sampler=false`

`data.train_batch_size=4`는 기존 V100x47 안정 설정을 따른 값입니다. A100x4x2 run의 effective global batch는 `26 * 8 = 208`이므로 완전히 같지는 않지만, V100 32GB fleet에서 memory-safe한 기본값을 쓰고 lr은 A100 run과 같은 `6e-4`로 고정합니다.

이 preset은 `train_use_eval_agent_selection=true`로 모든 agent를 살리기 때문에, dense scene 몇 개가 한 rank의 local batch에 같이 들어가면 CUDA peak가 크게 튈 수 있습니다. 그래서 pkl별 `agent_count`, current-step valid agent 수, map polyline 수만 한 번 스캔해 metadata cache로 저장하고, 이후 epoch마다 무거운 scene이 local batch 안에 몰리지 않도록 balanced batch sampler가 index 순서만 재배치합니다. 모델, loss, per-rank batch size, effective global batch `188`은 바뀌지 않고, 학습 objective도 그대로입니다.

`shuffle=true`일 때 balanced batch sampler는 Lightning의 epoch hook을 받아 매 epoch `seed + epoch`으로 새 순서를 만듭니다. 즉 memory-balanced 제약은 유지하되, epoch마다 같은 batch 순서가 반복되지는 않습니다.

metadata cache는 학습 cache 파일을 수정하지 않습니다. metadata build가 필요할 때는 pkl decode가 CPU 병목이므로 `ProcessPoolExecutor`로 여러 프로세스가 cache pkl을 병렬 스캔하고, 결과는 tensor payload로 저장합니다. 같은 cache가 있으면 학습 중에는 pkl을 다시 스캔하지 않습니다. 따라서 steady-state 학습 속도에 추가되는 비용은 epoch 시작 시 작은 tensor 정렬과 index list 생성뿐입니다. Lightning의 기본 distributed sampler가 이 batch sampler를 덮어쓰지 않도록 이 preset은 `trainer.use_distributed_sampler=false`를 명시합니다.

V100x47 production preset은 학습 프로세스 안에서 조용히 긴 metadata build가 들어가는 것을 막기 위해 `data.train_memory_balance_build_on_missing=false`를 사용합니다. 대신 이 전용 launcher가 첫 attempt 전에 모든 학습 pod에서 각 pod의 `CACHE_ROOT` 기준으로 metadata cache를 preflight 생성/검증합니다. 기본 cache 위치는 아래와 같습니다.

```text
$REMOTE_LOG_DIR/dataset_metadata/womd_training_memory_balance_v1.pt
```

따라서 위 launcher command만 실행해도 metadata cache가 없거나 stale인 경우 학습 시작 전에 먼저 만들어집니다. `--memory-metadata-cache-path`를 지정하면 preflight build/validate와 실제 datamodule의 `data.train_memory_balance_metadata_cache` Hydra override가 같은 파일을 보도록 같이 설정됩니다. metadata fingerprint는 pkl basename만 보지 않고 dataloader가 실제로 사용할 SMART cache path list까지 포함하므로, 다른 cache root에서 만든 metadata를 같은 파일명이라는 이유만으로 재사용하지 않습니다. V100 pod들의 `/mnt/nuplan/projects/catk/logs`가 pod-local일 수 있으므로 preflight는 master pod에서 만든 metadata payload를 같은 `CACHE_ROOT`를 쓰는 나머지 pod로 복사한 뒤 각 pod에서 다시 validate합니다. pod별 cache root를 바꾸는 경우에는 `--pod-cache-root POD=PATH` 설정이 각 pod의 실제 학습 경로와 일치해야 합니다.

metadata build 중인 프로세스는 `.lock` 디렉터리 heartbeat를 갱신합니다. 프로세스가 죽어 heartbeat가 끊긴 stale lock은 기본 30초 뒤 자동 회수되므로, 이전 prebuild 실패 때문에 다음 V100x47 실행이 2시간씩 대기하지 않습니다. metadata를 강제로 다시 만들려면 `--force-memory-metadata-rebuild`를 붙입니다. 다른 metadata build가 실제로 실행 중일 때는 쓰지 않습니다. preflight를 명시적으로 건너뛰어야 하는 특수 상황에서는 `--skip-memory-metadata-preflight`를 붙일 수 있지만, 그러면 cache가 없거나 stale일 때 학습 시작 전에 실패합니다. cache가 없는 상태에서 ad-hoc으로 학습 프로세스 안에서 build까지 허용해야 할 때만 `data.train_memory_balance_build_on_missing=true`를 override합니다.

이 전용 launcher는 OOM fallback을 끄는 것이 기본값입니다. CUDA OOM이 어느 pod 로그에서든 관측되면 전체 multi-node job을 정리하고 rank 0의 최신 `epoch_last.ckpt`를 기준 checkpoint로 확정해 peer pod로 동기화한 뒤, `train_batch_size=4`를 유지한 채 같은 설정으로 다시 시작합니다. 즉 기본 `--oom-step=0`이며, `4 -> 2`처럼 batch를 낮추지 않습니다. 다만 같은 batch size에서 반복 OOM이 무한히 이어지는 것을 막기 위해 기본 `--max-same-bs-oom-retries=3` 이후에는 중단합니다.

기본 실험 이름은 `flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs4`이고, tmux session 이름은 `catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix`입니다.

실행 전에 실제 환경 변수와 retry wrapper만 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py --dry-run
```

##### V100 47GPU latest semi_control_stable tail-prefix run

기존 W&B run인 `flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs4`와 설정은 맞추되, `semi_control_stable` 최신 코드로 새 실험을 시작하려면 아래 wrapper를 씁니다. 이 wrapper는 같은 preset과 같은 V100 47GPU fleet를 쓰지만, 기존 run과 log/checkpoint/W&B task name이 섞이지 않도록 기본 task/session 이름만 분리합니다.

```bash
python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_latest_static_pods.py --replace
```

| 항목 | 설정 |
|---|---|
| 실행 스크립트 | `scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_latest_static_pods.py` |
| 내부 base launcher | `scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py` |
| branch | `semi_control_stable` 최신 head. `--git-ref`를 넘기지 않으면 각 pod에서 launch 시점의 branch head를 checkout |
| experiment config | `configs/experiment/pre_bc_flow_control_v100x47_prefix_default_noslip.yaml` |
| 기본 task name | `flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs4_stable_latest` |
| 기존 W&B run과의 차이 | 동일 계열 설정이지만 최신 `semi_control_stable` 코드 기준. 기존 task name을 재사용하지 않아 과거 run resume/log 혼선을 피함 |
| pod / GPU | `sv svv svvv svvvv testsv testsvv testsvvv testsvvvv fv fvv fvvv fvvvv fvvvvv`, 총 V100 47 rank |
| context / anchor | `FLOW_CONTEXT_TOKEN_COUNT=18`, `FLOW_TRAIN_ANCHOR_COUNT=16` |
| Flow target mask | `use_prefix_valid_future_loss_mask=true`: tail anchor의 없는 미래 suffix는 loss와 future-step decoding에서 제외 |
| map-agent edge fix | `bba4a5b` 이후 코드 포함: map-agent `radius` 호출 전 batch 정렬로 same-scene map edge silent drop 방지 |
| batch / lr | per-rank `train_batch_size=4`, effective global batch `188`, `lr=6e-4` |
| bs4 memory guard | `detach_train_metric_clean=true`. Loss/target/anchor/model capacity는 그대로 두고 train metric clean 복원 graph만 분리함. Training activation checkpointing 경로는 production pretrain 속도 기준에서 제거됨 |
| metadata preflight | 13개 V100 pod 전체에서 memory-balanced metadata를 학습 시작 전에 생성/복사/검증. latest wrapper는 다른 V100 run과 cache 파일을 공유하지 않도록 기본 `${REMOTE_LOG_DIR}/dataset_metadata/womd_training_memory_balance_v1_stable_latest.pt`를 사용하고, 같은 path를 `data.train_memory_balance_metadata_cache`에도 넘김. stale `.lock`은 heartbeat 기준 기본 30초 뒤 자동 회수 |
| stale run cleanup | `--replace` 정리 시 stale `torchrun_pgid`가 현재 cleanup shell process group과 충돌하거나 같은 task 프로세스가 없으면 group kill을 건너뛰고 pgid 파일만 회수함 |
| OOM 동작 | 기본 `--oom-step=0`: batch size는 줄이지 않고 checkpoint에서 재개. 같은 batch 반복 OOM은 기본 3회 뒤 중단 |

dry-run으로 실제로 어떤 base launcher를 호출하는지 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_latest_static_pods.py --dry-run --replace
```

실제 장기 학습을 시작할 때는 local retry supervisor가 학습 종료 또는 실패까지 살아 있어야 하므로, 아래처럼 local `tmux` controller로 띄웁니다:

```bash
mkdir -p logs/_manual_launch
tmux new-session -d -s catk-v100x47-stable-latest-controller \
  -c "$PWD" \
  "set -o pipefail; PYTHONUNBUFFERED=1 python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_latest_static_pods.py --replace 2>&1 | tee -a logs/_manual_launch/v100x47_prefix_default_noslip_latest.out; echo controller_exit=\${PIPESTATUS[0]}; exec bash"
```

local controller 로그는 `logs/_manual_launch/v100x47_prefix_default_noslip_latest.out`에 남고, attempt별 통합 로그는 `logs/_h100x4_multinode_pretrain_oom_retry/flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs4_stable_latest/` 아래에 저장됩니다.

latest run tmux 확인:

```bash
kubectl exec -it -n p-pnc sv -c main -- tmux attach -t catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix-stable-latest
kubectl exec -it -n p-pnc fv -c main -- tmux attach -t catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix-stable-latest
```

기존 기본 launcher session의 tmux 확인:

```bash
kubectl exec -it -n p-pnc sv -c main -- tmux attach -t catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix
kubectl exec -it -n p-pnc fv -c main -- tmux attach -t catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py --stop
```

#### V100 47GPU static pod control-space pretrain

`testsv`, `testsvv`, `testsvvv`, `testsvvvv`, `sv`, `svv`, `svvv`, `svvvv`의 V100x4 pod 8개와 `fv`, `fvv`, `fvvv`, `fvvvv`, `fvvvvv`의 V100x3 pod 5개를 묶어 control-space pretrain을 돌릴 때는 아래 preset과 launcher를 씁니다.

```text
configs/experiment/pre_bc_flow_control_v100x47.yaml
scripts/launch_pre_bc_flow_control_v100x47_static_pods.py
```

기본 설정:

- pod 순서: `testsv testsvv testsvvv testsvvvv sv svv svvv svvvv fv fvv fvvv fvvvv fvvvvv`
- 기본 실행 브랜치: `semi_control_stable`
- V100x4 pod는 4 rank, V100x3 pod는 3 rank를 띄워 총 47 rank를 사용합니다.
- `trainer.precision=16-mixed`
- `model.model_config.decoder.flow_window_steps=20`
- `model.model_config.token_processor.use_kinematic_control_flow=true`
- `model.model_config.token_processor.use_prefix_valid_future_loss_mask=true`
- `model.model_config.token_processor.control_round_trip_max_position_error_m=0.5`
- `data.train_batch_size=4`, effective global batch `4 * 47 = 188`
- `model.model_config.lr=6e-4`

현재 `semi_control_stable`의 `use_prefix_valid_future_loss_mask=true`는 가까운 미래부터 처음 끊기기 전까지의 연속 valid prefix 전체를 loss에 반영합니다.

4GPU pod와 3GPU pod를 섞으면 `torchrun --nproc_per_node`의 homogeneous local world size 가정과 Lightning 기본 TorchElastic 검증의 `devices * num_nodes == WORLD_SIZE` 가정이 맞지 않습니다. 그래서 이 launcher는 `--manual-rank-offsets` 경로로 각 pod의 GPU 수를 읽어 `RANK/WORLD_SIZE/LOCAL_RANK`를 직접 배정하고, `HeterogeneousTorchElasticEnvironment`로 homogeneous 검증만 완화합니다.

batch size는 가장 작은 V100 32GB GPU에 맞춰 정합니다. 이전 V100x47 실측에서 더 큰 per-GPU batch는 OOM fallback을 거쳤고, 이번 preset은 요청한 global batch `188`에 맞춰 per-GPU `4`를 기본값으로 둡니다. OOM fallback은 `4 -> 2`입니다.

lr은 H100x4x2 기준 global batch `26 * 8 = 208`에서 쓰던 `6e-4`를 유지합니다. 이번 V100 47GPU 설정은 `4 * 47 = 188`이라 global batch가 충분히 가까워, 불필요한 lr scaling 없이 같은 값을 쓰는 쪽이 비교가 깔끔합니다.

실행:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_static_pods.py --replace
```

CUDA OOM이 어느 pod 로그에서든 관측되면 전체 multi-node job을 정리하고 rank 0의 최신 `epoch_last.ckpt`를 기준 checkpoint로 확정해 peer pod로 동기화한 뒤, `train_batch_size`를 2씩 낮춰 재개합니다.

실행 전에 실제 환경 변수와 retry wrapper만 확인하려면:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_static_pods.py --dry-run
```

tmux 확인:

```bash
kubectl exec -it -n p-pnc testsv -c main -- tmux attach -t catk-control-pretrain-v100x47
kubectl exec -it -n p-pnc fv -c main -- tmux attach -t catk-control-pretrain-v100x47
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_pre_bc_flow_control_v100x47_static_pods.py --stop
```

### 5.2 Validation 주기와 val_open / val_closed 바꾸기

- 학습 중 validation은 `trainer.check_val_every_n_epoch` 마다 실행됩니다.
- self-forced fine-tuning에서 `model.model_config.self_forced.estimator_warmup_epochs>0`
  이면 warmup epoch 끝에는 validation을 실행하지 않습니다. 이때
  `trainer.check_val_every_n_epoch` 주기는 warmup이 끝나고 Generator까지 함께
  학습하는 첫 epoch부터 다시 계산됩니다.
- `model.model_config.val_open_loop=true/false`로 open-loop validation on/off를 바꿉니다.
- `model.model_config.val_closed_loop=true/false`로 closed-loop validation on/off를 바꿉니다.
- validation 양 자체는 `trainer.limit_val_batches`로 줄이거나 늘릴 수 있습니다.
- `model.model_config.n_rollout_closed_val`는 `val_closed_loop`에서 scene당 몇 번 rollout sampling할지 정합니다. 현재 `pre_bc_flow` 기본값은 `32`입니다.
- `model.model_config.decoder.flow_window_steps`는 flow matching이 한 번에 생성하는 10Hz 미래 길이입니다. 기본값은 `20` step, 즉 `2초`입니다.
- `5`의 배수여야 하며 `decoder.num_future_steps`보다 클 수 없습니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm|matched_token_chunk`로 closed-loop에서 실제로 export/score/video에 쓰는 10Hz rollout 표현을 고릅니다. 기본값은 `raw_fm`이며, `matched_token_chunk`도 내부 문맥 상태 자체는 실제 FM commit을 유지합니다.
- stop-motion gate는 branch-wide로 비활성화되어 있습니다. `model.model_config.decoder.use_stop_motion`와 `model.model_config.self_forced.use_stop_motion`은 config/checkpoint 호환용 키로만 남아 있으며, true override를 주더라도 실제 rollout에서는 false로 고정됩니다.
- `model.model_config.decoder.use_lqr=true/false`로 vehicle / bicycle용 curvature-LQR commit
  bridge를 켜거나 끕니다. 기본값은 `false` 입니다.
- `use_lqr=true`면 2초 미래를 바로 commit하지 않고, 다음 0.5초 commit window만 실제로 실행합니다.
- control-space Flow에서는 `use_kinematic_control_flow=true` 와 함께 사용할 수 있으며,
  LQR는 `use_holonomic_model_only` 설정을 따른 pose-space reference를 만든 뒤 vehicle / bicycle에만
  적용됩니다.
- `use_stop_motion`은 항상 false로 처리되므로 stop token 기반 0.5초 고정은 적용되지 않습니다.
- `use_lqr=true`는 vehicle / bicycle 에만 적용됩니다. pedestrian 은 항상
  token / raw branch 를 유지합니다.
- `model.model_config.n_batch_sim_agents_metric`는 validation 중 Fast WOSAC scorer를 실제로 돌릴 앞쪽 batch 수입니다. `smart_flow` 기본값은 `10`, `local_val_flow`는 `100`, `sim_agents_sub_flow`는 `0`입니다. 단, `model.model_config.scorer_scene_num`이 양의 정수이면 이 값은 validation 시작 시 자동으로 덮어써집니다.
- `model.model_config.scorer_scene_num`는 GPU 개수와 validation batch size가 달라도 Fast WOSAC scorer에 들어가는 scene 규모를 비슷하게 맞추기 위한 기준값입니다. 기본값은 `1680` 입니다. 실제 적용식은 `n_batch_sim_agents_metric = max(1, ceil(ceil(scorer_scene_num / world_size) / val_batch_size))` 입니다. `null` 또는 `0` 으로 두면 자동 덮어쓰기를 끄고 명시한 `n_batch_sim_agents_metric` 값을 그대로 씁니다.
- `trainer.limit_val_batches`는 validation에 실제로 사용할 batch 양입니다. `0.1`이면 전체 validation batch의 10%, `1.0`이면 전체, 정수 `20`이면 앞 20 batch만 평가합니다. 다만 `scorer_scene_num`이 양수이고 `limit_val_batches`가 Fast WOSAC 채점에 필요한 batch 수보다 작으면, 실행 시 `limit_val_batches`를 필요한 batch 수까지 자동으로 늘립니다.
- `data.val_batch_size`는 validation batch당 scene 수입니다. 키우면 validation은 빨라질 수 있지만 GPU memory 사용량도 같이 늘어납니다. `scorer_scene_num` 자동 덮어쓰기가 켜져 있으면 이 값이 `n_batch_sim_agents_metric` 계산식의 분모가 됩니다.
- Fast WOSAC scorer 기준 총 채점 scene 수는 `scorer_scene_num`이 켜져 있으면 대략 `n_batch_sim_agents_metric x val_batch_size x world_size` 입니다. batch 단위로만 자르므로 요청값보다 조금 커질 수 있습니다. 끈 경우에는 대략 `min(실행한 val batch 수, n_batch_sim_agents_metric) x val_batch_size x world_size` 입니다.
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

# stop-motion gate는 사용하지 않음
... model.model_config.decoder.use_stop_motion=false

# self-forced 학습 rollout도 stop-motion gate를 사용하지 않음
... model.model_config.self_forced.use_stop_motion=false

# vehicle / bicycle curvature-LQR commit bridge 적용
... model.model_config.decoder.use_stop_motion=false \
    model.model_config.decoder.use_lqr=true

# use_lqr + matched token chunk를 함께 쓸 때도
# vehicle / bicycle export는 실행된 5점 chunk를 유지하고 pedestrian만 token chunk를 씁니다.
... model.model_config.decoder.use_lqr=true \
    model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

`flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` 의 W&B 마지막 artifact `epoch-last-x5f9g0ce:v60` 을 `hsb-npc-training-1` H100x6, `use_stop_motion=false`, `sample_steps=16`, `antithetic_pairs=true`, `noise_scale=1.0`, `val_batch_size=56`, `scorer_scene_num=1680` 조건으로 Fast-RMM 비교한 결과는 아래와 같습니다.

| use_lqr | RMM | CPD | CES |
|---|---:|---:|---:|
| false | 0.781846 | 0.201687 | 0.095312 |
| true | 0.739973 | 0.132768 | 0.138147 |

RMM 기준으로 `use_lqr=false`가 더 높았으므로 repository 기본값은 `model.model_config.decoder.use_lqr=false` 로 유지합니다.

# training validation에서 Fast WOSAC scorer를 앞 20 batch에만 적용
# scorer_scene_num 자동 덮어쓰기를 끈 경우에만 의미가 있습니다.
... model.model_config.scorer_scene_num=null \
    model.model_config.n_batch_sim_agents_metric=20

# Fast WOSAC scorer 채점 규모를 GPU 수와 무관하게 대략 1920 scene으로 맞추기
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

`data.train_use_eval_agent_selection=true`에서는 validation/추론과 같은 agent 선택 규칙을 쓰므로
`data.train_max_num`은 적용되지 않습니다. 메모리가 빠듯하면 학습 대상 수를 다시 제한하는 대신
batch size, memory-balanced sampler, worker 수를 먼저 조정하는 것을 권장합니다.

### 5.9 4x H100 80GB 에서 Flow Matching pretrain

6x H100 이 아닌 **4x H100 80GB** 박스에서 `pre_bc_flow` 와 동일한 pretrain 을 돌리고 싶을 때 쓰는 별도 preset 입니다.

- preset 파일: `configs/experiment/pre_bc_flow_4_h100.yaml`
- 베이스: `configs/experiment/pre_bc_flow.yaml` (6x H100 preset)

요약만 보면 아래와 같습니다.

- `flow_window_steps=20` 을 preset 자체에서 고정합니다. 이 horizon 에 맞춰 아래 batch size 상한을 실측했기 때문에 모델 default 가 바뀌더라도 4x H100 메모리 프로파일이 유지됩니다.
- `train_batch_size=52` 가 기본값입니다. 커밋 `b12e653` 에서 추가된 `AttentionLayer` activation recomputation 이 기본으로 켜진 상태에서 4x H100 80GB 로 실측한 상한입니다. `trainer.devices=4`, `accumulate_grad_batches=1` -> effective global batch **`208`**.
- `lr=2.667e-4` 는 이전 per-GPU bs=20 (global 80) 기준으로 맞춰둔 값입니다. 새 global batch 208 에 선형 LR scaling rule 을 적용하려면 `model.model_config.lr=6.933e-4` (= `4e-4 * 208/120`) 로 CLI override 하세요. optimizer 동작을 무언 중에 바꾸지 않기 위해 default 는 기존 값을 유지합니다.
- `max_epochs(=64)`, `check_val_every_n_epoch(=8)`, `limit_val_batches(=0.1)`, `val_batch_size(=16)`, `n_rollout_closed_val(=32)` 은 6xH100 preset 과 동일합니다.
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
- 기본 lr: `6e-4`
- 기본 horizon: `flow_window_steps=20`
- validation 중 WOSAC scoring이 오래 걸려도 DDP가 조기 timeout 나지 않도록 `trainer=ddp`의 process group timeout은 4시간입니다.
- `pre_bc_flow_2x4_h100` preset은 `TQDMProgressBar(refresh_rate=1)`와 `trainer.enable_progress_bar=true`를 명시합니다. launcher 기본 pod 순서에서는 `hsb-npc-training`이 node rank 0/global rank 0이므로, `check_val_every_n_epoch=32`로 fit-time validation이 시작될 때 validation tqdm 진행률은 `hsb-npc-training`의 `catk-h100x4-pretrain` tmux 주 pane에 표시됩니다. `hsb-npc-training2`는 non-zero rank라 같은 progress bar를 중복 출력하지 않는 것이 정상입니다.
- launcher는 각 pod 안에 쓰는 env 파일에 pod별 `CACHE_ROOT`를 따로 기록합니다. 두 pod가 같은 mount path를 공유하는 경우에만 `--cache-root <PATH>`로 전체 override를 쓰고, pod별 경로를 바꿔야 하면 `--pod-cache-root POD=PATH`를 반복해서 넘깁니다.
- 이 기본값은 H100x4x2 global batch `208` 기준으로 `6e-4`를 사용합니다. 이전 보수적 설정과 같은 per-GPU batch를 유지해야 하는 ablation이면 `--train-batch-size 20`을 쓰고, 기존 6xH100과 global batch까지 맞춰야 하는 ablation이면 `--train-batch-size 15 --learning-rate 4e-4`를 쓰세요.

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

mixed 실험 lr은 별도로 더 보수적으로 잡았습니다. H100x4x2 기준 global batch `208`의 기본 lr은 이제 `6e-4`이고, mixed 12GPU global batch `312`로 단순 선형 scaling하면 `9e-4`가 됩니다. 하지만 mixed hardware 첫 run이고 prefix-valid target 선택도 달라지므로, mixed preset 기본값은 여전히 `5e-4`를 유지합니다. 선형 scaling ablation을 따로 보고 싶을 때만 `--learning-rate 9e-4`로 override하세요.

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
- H100 preset은 Generator lr `1e-6`, generated estimator optimizer lr `1e-6`, `weight=1.0`, `anchor_weight=0.1`, `use_anchor_flow_matching_loss=false`, `estimator_updates_per_step=5`, `unfrozen_range=middle`, `detach_block_transition=true`, sampling = Euler `sample_steps=16` / `backprop_last_k=8` / `noise_scale=1.0` 을 기본으로 둡니다.
- Bounded Clean-DMD guidance 기본값 `clean_dmd_normalizer_eps=0.05`, `dmd_stable_scale_scope=type`, `clean_dmd_tau_low=0.02`, `clean_dmd_tau_high=0.98` 을 함께 둡니다. `clean_dmd_normalizer_eps` 는 stable scale의 최소값입니다.
- Generator EMA 기본값은 `ema_weight=0.99`, `ema_start_step=50` 입니다. EMA는 online Generator update 직후에만 갱신되고, generated estimator에는 적용하지 않습니다.
- Generated estimator warmup 기본값은 `estimator_warmup_epochs=0` 입니다. self-forcing 시작 직후부터 generated estimator 업데이트와 Generator 업데이트를 같은 train step 안에서 수행합니다.
- 4x/6x H100 self-forced preset과 OOM retry script는 모두 첫 시도 `data.train_batch_size=36` 을 기본으로 둡니다.
- self-forced preset은 각 epoch마다 train dataset의 25%만 새로 랜덤 샘플링해 학습합니다. 비율은 `data.train_epoch_sample_fraction` 으로 바꾸며, `1.0` 으로 두면 전체 train dataset을 사용합니다.
- self-forced fine-tuning에서는 Generator optimizer와 generated estimator optimizer 모두 LR scheduler를 쓰지 않습니다. 두 optimizer 모두 같은 `model.model_config.lr` 을 사용합니다. 따라서 self-forced preset에는 `lr_warmup_steps` / `lr_min_ratio` override를 두지 않습니다.
- H100x6 차이: `defaults` 에서 `override /trainer: ddp` 를 박아 두고 `trainer.devices=6` 을 고정 → preset 만 줘도 6 GPU DDP 가 가동됩니다 (베이스 `self_forced_npfm.yaml` 은 trainer 를 override 하지 않아 single-process 로 떨어집니다).
- 새 self-forced fine-tuning 시작을 위해 preset 이 `action=finetune` 을 기본으로 고정합니다. 따라서 `ckpt_path` 는 optimizer/epoch 를 resume하지 않고 pretrained weight만 로드합니다.
- 전제: `ckpt_path` 에는 같은 `flow_window_steps` 로 pretrain 된 Generator checkpoint 를 넣습니다. 모델 default 는 `flow_window_steps=20` (2초) 이고, ckpt 가 2초 horizon 으로 pretrain 된 경우 override 하지 않는 편이 안전합니다.

#### Pose-projected DMD guidance

`use_kinematic_control_flow=true`, `use_holonomic_model_only=false`, `use_rolling_supervision=true`
조건의 self-forced DMD fine-tuning에서는 기본적으로 pose-projected DMD guidance를 사용합니다.

```yaml
model:
  model_config:
    self_forced:
      project_dmd_to_pose_space: true
      dmd_use_stable_scale_filter: true
      dmd_stable_scale_scope: type
      dmd_use_teacher_alignment_filter: false
      dmd_use_trust_region_filter: false
      dmd_use_injection_ramp: false
```

핵심은 DMD 방향을 3축 control 값에서 바로 판단하지 않고, 실제 closed-loop metric이 보는
pose-space에서 먼저 판단한 뒤 최종 target만 다시 rolling control-space로 되돌리는 것입니다.

```text
현재 generator control
-> pose로 복원
-> teacher / generated estimator clean control도 pose로 복원
-> pose-space에서 Clean-DMD 안정화 방향 계산
-> heading cos/sin 재정규화
-> rolling control target으로 역변환
-> 기존 active-control loss로 Generator 학습
```

수식으로는 기존 방식이

```math
d_C = \operatorname{DMD}(C_G,\hat{C}_T,\hat{C}_E)
```

였다면, 현재 기본 방식은

```math
\begin{aligned}
P_C &= D(C_G),\\
P_T &= D(\hat{C}_T),\\
P_E &= D(\hat{C}_E),\\
d_P &= \operatorname{DMD}(P_C,P_T,P_E),\\
P^* &= \operatorname{NormalizeHeading}(P_C + \lambda d_P),\\
C^* &= R(P^*)
\end{aligned}
```

입니다. 여기서 `D`는 normalized rolling control을 pose metric 표현으로 복원하는 변환이고,
`R`은 pose target을 다시 rolling control target으로 바꾸는 변환입니다.

이 옵션은 generated estimator 학습 공간, Generator 출력 차원, pretrain target 정의, non-holonomic
active-control loss를 바꾸지 않습니다. vehicle/cyclist의 lateral control 축은 최종 control loss에서
계속 제외됩니다. 즉 DMD 판단 공간만 closed-loop 평가 공간과 맞추고, 모델 학습 target은 기존
control-space recipe를 유지합니다.

기존 control-space DMD로 되돌려 비교하려면 아래 override를 사용합니다.

```bash
model.model_config.self_forced.project_dmd_to_pose_space=false
```

학습 로그에서는 `train/sf_pose_projected_dmd=1` 이면 pose-projected DMD가 실제로 활성화된
상태입니다. `use_kinematic_control_flow=false` 이거나 flow state가 pose-space인 경우에는 이 옵션을
켜도 자동으로 legacy 경로와 같은 pose-space DMD 판단이 됩니다.

Clean-DMD 방향 안정화 필터는 pose-projected 경로와 direct control-space 경로에 동일하게 적용됩니다.

| config | 기본값 | 의미 |
|---|---:|---|
| `dmd_use_stable_scale_filter` | `true` | \(S=\max(\mathrm{mean}(|G|),\epsilon)\) 로 나눈 \(D_0=R/S\) 방향을 사용합니다. 평균 범위는 `dmd_stable_scale_scope`가 정합니다. |
| `dmd_stable_scale_scope` | `type` | `agent`는 agent별, `type`은 같은 scene 안 같은 agent type별, `scene`은 같은 scene 전체 agent별로 stable scale을 공유합니다. |
| `dmd_use_teacher_alignment_filter` | `false` | \(a=\mathbf{1}[\langle P_T-P_X,R\rangle>0]\) gate를 적용해 teacher 방향과 정렬된 agent만 남깁니다. |
| `dmd_use_trust_region_filter` | `false` | \(D=D_1\min(1,\mathrm{rms}(G)/(\mathrm{rms}(D_1)+\epsilon))\) 로 DMD 방향 크기를 teacher 거리 이하로 제한합니다. |
| `dmd_use_injection_ramp` | `false` | `true`이면 DMD target 주입량을 warmup 종료 후 첫 generator-DMD epoch부터 \(\lambda_e=0.25+0.75\min(1,\frac{\max(0,e-e_0)}{2})\) 로 2 epoch 동안 키웁니다. `false`이면 항상 \(\lambda=1\) 입니다. |

여기서 \(R\) 은 teacher-estimator clean 추정 차이, \(G\) 는 현재 generator와 teacher의 차이입니다.
`project_dmd_to_pose_space=true` 일 때는 위 값들이 pose-space에서 계산되고,
`false` 일 때는 control-space에서 계산됩니다. vehicle/cyclist lateral 축 제외는 별도의 active-control
mask 규칙이라 위 세 필터와 무관하게 기존처럼 최종 control loss 단계에서 유지됩니다.

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

#### hsb-npc-training-1 H100x6 DMD self-forcing launcher

`hsb-npc-training-1` 단일 H100x6 pod에서 epoch 116 pretrained Generator를 시작점으로 DMD-style self-forced fine-tuning을 돌릴 때는 아래 launcher를 씁니다. pod를 새로 만들거나 재시작하지 않고, 기존 pod 안에 tmux session과 OOM retry 학습 프로세스만 만듭니다.

```bash
python scripts/launch_self_forced_dmd_h100x6_hsb1_static_pod.py --replace
```

먼저 짧은 smoke/probe만 확인하려면:

```bash
python scripts/launch_self_forced_dmd_h100x6_hsb1_static_pod.py \
  --replace \
  --task-name flow_self_forced_dmd_h100x6_hsb1_smoke_bs18 \
  --session catk-self-forced-dmd-h100x6-hsb1-smoke \
  --initial-bs 18 \
  --max-epochs 1 \
  --limit-train-batches 2 \
  --limit-val-batches 0
```

기본 launcher 설정:

| 항목 | 값 |
|---|---|
| pod | `hsb-npc-training-1` 단일 H100x6 |
| branch | `semi_control_stable` |
| experiment | `self_forced_npfm_h100_6` |
| default task | `flow_self_forced_dmd_h100x6_hsb1_epoch116_activecontrol_sample16_backprop8_lr1e-6_bs18_oomretry` |
| pretrained checkpoint | `jksg01019-naver-labs/SMART-FLOW/epoch-last-mqfq3u39:v121` |
| checkpoint 의미 | train+validation pretrain run의 epoch 116 Generator |
| action | 첫 시도 `finetune`, OOM 후 재시도는 최신 self-forced `epoch_last.ckpt` 기준 `fit` |
| DMD objective | `model.model_config.self_forced.distribution_matching_objective=dmd` |
| control mode | `use_kinematic_control_flow=true`, `use_holonomic_model_only=false` |
| DMD active axes | pedestrian `[delta_s, delta_n, delta_theta]`, vehicle/cyclist `[delta_s, delta_theta]` |
| no-slip point | vehicle `0.2289518863`, cyclist `0.0495847873` |
| round-trip filter | `control_round_trip_max_position_error_m=0.5` |
| prefix valid mask | `use_prefix_valid_future_loss_mask=true` |
| rolling supervision | `use_rolling_supervision=true` |
| lr | `1.0e-6` |
| generated estimator lr | `1.0e-6` |
| estimator updates | `5` per train step |
| estimator warmup | `0` epoch |
| detach block transition | `true` |
| self-forced sample steps | Euler `sample_steps=16` |
| self-forced backprop | `backprop_last_k=8` |
| random terminal policy | `all` |
| train metric clean | `decoder.detach_train_metric_clean=true` |
| validation rollout sampling | Euler `sample_steps=16`, `antithetic_pairs=true`, `stratified_gaussian_noise=false`, `noise_scale=1.0` |
| validation mode | `val_closed_loop=true`, `val_open_loop=false` |
| train data fraction | `data.train_epoch_sample_fraction=0.25` |
| validation cadence | `check_val_every_n_epoch=1`, `limit_val_batches=0.1` |
| epochs | `16` |
| initial train batch | per-rank `18`, effective global batch `108` |
| OOM fallback | `18 -> 17 -> ...`, latest self-forced checkpoint resume |
| tmux session | `catk-self-forced-dmd-h100x6-hsb1` |

#### hsb-npc-training-1 H100x6 epoch061 DMD self-forcing launcher

`flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20`
pretrain run의 epoch061 Generator checkpoint를 시작점으로 DMD-style self-forced fine-tuning을 돌릴 때는 아래 wrapper를 씁니다.
내부적으로 위 공용 H100x6 launcher를 호출하되, checkpoint artifact와 task/session 이름만 epoch061 전용값으로 고정합니다.

```bash
python scripts/launch_self_forced_dmd_epoch061_h100x6_hsb1_static_pod.py --replace
```

| 항목 | 값 |
|---|---|
| pod | `hsb-npc-training-1` 단일 H100x6 |
| branch | `semi_control_stable` |
| experiment | `self_forced_npfm_h100_6` |
| task | `flow_self_forced_dmd_h100x6_hsb1_epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr1e-6_bs18_frac025_ep16_oomretry` |
| pretrained checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint 의미 | `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` epoch061, artifact metadata epoch 62 / global step 278192 |
| local checkpoint path in pod | `/workspace/flow_self_forced_dmd_h100x6_hsb1_pretrain_epoch061_x5f9g0ce/v57/epoch_061.ckpt` |
| action | 첫 시도 `finetune`, OOM 후 재시도는 최신 self-forced `epoch_last.ckpt` 기준 `fit` |
| DMD objective | active-control DMD, vehicle/cyclist lateral DMD axis disabled |
| control mode | `use_kinematic_control_flow=true`, `use_holonomic_model_only=false` |
| sample steps / solver | Euler `sample_steps=16` |
| backprop | `backprop_last_k=8`, `detach_block_transition=true` |
| validation | `val_closed_loop=true`, `val_open_loop=false`, `check_val_every_n_epoch=1`, `limit_val_batches=0.1` |
| train metric path | `decoder.detach_train_metric_clean=true` |
| train data fraction | `data.train_epoch_sample_fraction=0.25` |
| epochs | `16` |
| lr | Generator `1.0e-6`, generated estimator `1.0e-6` |
| initial train batch | per-rank `18`, effective global batch `108` |
| OOM fallback | `18 -> 17 -> ...`, latest self-forced checkpoint resume |
| tmux session | `catk-self-forced-dmd-epoch061-h100x6-hsb1` |

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-self-forced-dmd-epoch061-h100x6-hsb1
```

2026-06-05 H100x6 full-run probe 기준으로 `bs19` 이상은 첫 train batch 안의 generated-estimator attention 경로에서 CUDA OOM이 났고, `bs18`은 active-control DMD train step을 정상 진행했습니다. full run에서는 rare later-batch memory spike에 대비해 OOM retry를 유지합니다. 이후 더 긴 batch에서 OOM이 발생하면 launcher가 최신 self-forced checkpoint를 기준으로 `bs17`부터 자동 재개합니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-self-forced-dmd-h100x6-hsb1
```

학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_h100x6_hsb1_static_pod.py --stop
```

중단된 self-forced run을 이어서 학습할 때만 `action=fit ckpt_path=/path/to/self_forced_run/last.ckpt` 를 사용하세요. 이 경우에는 Lightning이 optimizer, epoch, global step까지 함께 복원합니다. checkpoint 안에 `self_forced_target_teacher`, `self_forced_generated_estimator`, `self_forced_generator_ema` state가 있으면, fit 시작 hook은 보조 모델과 EMA를 현재 Generator weight로 다시 덮어쓰지 않고 checkpoint의 `F_rho` / `F_psi` / EMA 상태를 보존합니다.

보호 장치도 있습니다. self-forced가 켜진 상태에서 `action=finetune` 에 self-forced checkpoint를 넣으면 실행이 중단됩니다. 반대로 `action=fit` 에 self-forced 보조 state가 없는 pretrained checkpoint를 넣어도 중단됩니다. 즉, pretrained Generator에서 처음 시작할 때는 `action=finetune`, self-forced run을 이어갈 때는 `action=fit` 으로 분리해야 합니다.

Self-forced H100 preset은 self-forced rollout에서 `sample_steps=16`, `random_terminal_step.policy=all`, `backprop_last_k=8`을 사용합니다. 즉 random terminal step을 샘플링하지 않고 항상 16 denoising step을 실행하되, terminal 쪽 마지막 8 step에만 gradient를 남깁니다. 0.5초 block마다 `FlowODE.generate(..., return_terminal_clean=True)`를 호출해 terminal clean estimate를 commit합니다.

- `model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch` 는 DDP 전체 rank 공유 `s` fast path입니다. H100 preset의 현재 기본 `policy=all`에서는 random `s`를 뽑지 않습니다.
- `policy=paper_uniform` 으로 override하면 실제 실행 denoising step `K` 를 `[min_executed_steps, sample_steps]` 범위에서 균등 샘플링합니다. `min_executed_steps=16`, `sample_steps=16` 에서는 결과적으로 `K=16` 을 사용합니다.
- `policy=all` 은 random terminal step을 샘플링하지 않고 항상 `sample_steps` 전체 denoising을 실행합니다. 이때 `sampling.backprop_last_k` 개 마지막 step에만 gradient를 남기며, H100 preset 기본값은 `8` 입니다.
- terminal step 이전 denoising은 gradient 없이 계산하고, terminal clean estimate를 만드는 마지막 호출 하나만 gradient를 유지합니다.
- 선택된 `s`는 self-rollout을 어디서 끊고 commit할지만 정합니다.
- `F_psi` 업데이트와 clean-DMD guidance 계산의 noising `tau` 는 flow ODE의 전체 tau 구간에서 독립적으로 다시 샘플링합니다.

속도 실험용 기본 실행은 아래처럼 두면 됩니다.

```bash
python -m src.run experiment=self_forced_npfm_h100_6 \
    model.model_config.self_forced.sampling.random_terminal_step.scope=global_batch \
    model.model_config.self_forced.sampling.random_terminal_step.policy=all \
    model.model_config.self_forced.sampling.backprop_last_k=8
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
- `data.val_batch_size=2`, `model.model_config.scorer_scene_num=1680`

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
- `data.train_batch_size=20`, OOM 시 launcher가 모든 pod의 attempt status를 모아 `20 -> 19 -> 18 -> ... -> 12` 순서로 함께 낮춤
- `data.val_batch_size=8`, `model.model_config.scorer_scene_num=1680`

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

#### testa/testaa A100x4x2 DMD self-forcing fine-tuning

`testa`, `testaa` 두 A100 4GPU pod를 묶어 `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20`
pretrain run의 epoch 61 checkpoint에서
DMD-style self-forced fine-tuning을 시작할 때는 아래 wrapper를 씁니다. pod를 새로 만들거나
재시작하지 않고, 기존 pod 안에 tmux session과 2-node `torchrun`만 시작합니다.

```bash
python scripts/launch_self_forced_dmd_a100x4x2_testa_static_pods.py --replace
```

짧은 multi-node smoke/probe만 확인하려면:

```bash
python scripts/launch_self_forced_dmd_a100x4x2_testa_static_pods.py \
  --replace \
  --task-name flow_self_forced_dmd_a100x4x2_testa_smoke_bs20 \
  --session catk-self-forced-dmd-a100x4x2-testa-smoke \
  --initial-bs 20 \
  --max-epochs 1 \
  --limit-train-batches 2 \
  --limit-val-batches 0
```

기본 실험 설정:

| 항목 | 값 |
|---|---|
| pods | `testa` 4 A100 + `testaa` 4 A100 |
| branch | `semi_control_stable` |
| experiment | `self_forced_npfm_a100x4x2` |
| default task | `flow_self_forced_dmd_a100x4x2_testa_epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr1e-6_bs20_frac025_ep16_middle_oomretry` |
| pretrained checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint 의미 | `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` epoch 61 Generator, artifact metadata epoch 62 / global step 278192, 원 파일명 `epoch_last.ckpt` |
| local checkpoint path in pod | `/workspace/flow_self_forced_dmd_a100x4x2_testa_pretrain_epoch061_x5f9g0ce/v57/epoch_061.ckpt` |
| action | 첫 시도 `finetune`, OOM 후 재시도는 최신 self-forced `epoch_last.ckpt` 기준 `fit` |
| DDP shape | `trainer.num_nodes=2`, `trainer.devices=4`, 총 8 ranks |
| precision | `bf16-mixed` |
| DMD objective | `model.model_config.self_forced.distribution_matching_objective=dmd` |
| control mode | `use_kinematic_control_flow=true`, `use_holonomic_model_only=false` |
| DMD active axes | pedestrian `[delta_s, delta_n, delta_theta]`, vehicle/cyclist `[delta_s, delta_theta]` |
| no-slip point | vehicle `0.2289518863`, cyclist `0.0495847873` |
| round-trip filter | `control_round_trip_max_position_error_m=0.5` |
| prefix valid mask | `use_prefix_valid_future_loss_mask=true` |
| rolling supervision | `use_rolling_supervision=true` |
| lr | Generator `1.0e-6`, generated estimator `1.0e-6` |
| estimator updates | `5` per train step |
| estimator warmup | `1` epoch |
| frozen map feature cache | `model.model_config.self_forced.cache_frozen_map_features=true` |
| detach block transition | `true` |
| self-forced sample steps | Euler `sample_steps=16` |
| self-forced backprop | `backprop_last_k=8` |
| random terminal policy | `all` |
| stop-motion | self-forced training rollout `false`, validation/inference decoder `false` |
| train metric path | `decoder.detach_train_metric_clean=true` |
| train data fraction | `data.train_epoch_sample_fraction=0.25` |
| validation | `val_closed_loop=true`, `val_open_loop=false`, `limit_val_batches=0.1` |
| epochs | `16` |
| initial train batch | per-rank `20`, effective global batch `160` |
| OOM fallback | `20 -> 19 -> 18 -> ... -> 12`, latest self-forced checkpoint resume |
| val/test batch | per-rank `8` |
| scorer scenes | `1680` |
| tmux session | `catk-self-forced-dmd-a100x4x2-testa` |

2026-06-06 A100x4x2 `unfrozen_range=middle` probe 결과, `bs32` 20 train step,
`bs128` 20 train step, `bs160` 8 train step이 모두 CUDA OOM 없이 종료되었습니다.
관측된 W&B peak reserved는 각각 약 `24.1%`, `74.8%`, `83.8%`였습니다. 다만
기본 launcher는 더 보수적인 비교 실험을 위해 per-rank `bs20`에서 시작합니다.
rare heavy batch에서 OOM이 나면 launcher가 최신 self-forced checkpoint에서
batch를 1씩 낮춰 재개합니다.

시간 감각은 다음처럼 보면 됩니다. 기본 `train_epoch_sample_fraction=0.25`에서
`bs20`은 epoch당 약 `761` train step입니다. 이전 `bs18`,
`train_epoch_sample_fraction=0.25` 측정에서 train-only는 epoch당 약 `142~143분`
수준이었고, `bs20`은 global batch가 조금 더 커서 이보다 약간 짧게 걸릴 가능성이
큽니다. 반면 `bs160`, fraction `0.1`은 실측 기준 train-only epoch당 약 `13분`,
16 epoch 약 `3.5시간`이었습니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- \
  tmux attach -t catk-self-forced-dmd-a100x4x2-testa
kubectl exec -it -n p-pnc testaa -c main -- \
  tmux attach -t catk-self-forced-dmd-a100x4x2-testa
```

학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_a100x4x2_testa_static_pods.py --stop
```

#### testa A100x4 single-pod DMD self-forcing fine-tuning

`testa` 단일 A100 4GPU pod에서 같은 epoch 61 pretrained Generator checkpoint로
DMD-style self-forced fine-tuning을 돌릴 때는 아래 launcher를 씁니다. 이 launcher는
pod를 만들거나 지우거나 재시작하지 않고, `testa` 안에 tmux session과
`torchrun --standalone --nproc_per_node=4`만 시작합니다.

```bash
python scripts/launch_self_forced_dmd_a100x4_testa_static_pod.py --replace
```

짧은 smoke/probe만 확인하려면:

```bash
python scripts/launch_self_forced_dmd_a100x4_testa_static_pod.py \
  --replace \
  --task-name flow_self_forced_dmd_a100x4_testa_smoke_bs144 \
  --session catk-self-forced-dmd-a100x4-testa-smoke \
  --initial-bs 144 \
  --max-epochs 1 \
  --limit-train-batches 20 \
  --limit-val-batches 0
```

기본 실험 설정:

| 항목 | 값 |
|---|---|
| pod | `testa` 4 A100 |
| branch | `semi_control_stable` |
| experiment | `self_forced_npfm_a100x4_testa` |
| default task | `flow_self_forced_dmd_a100x4_testa_epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr1e-6_bs144_frac025_ep16_middle_oomretry` |
| pretrained checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint 의미 | `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` epoch 61 Generator, artifact metadata epoch 62 / global step 278192 |
| local checkpoint path in pod | `/workspace/flow_self_forced_dmd_a100x4_testa_pretrain_epoch061_x5f9g0ce/v57/epoch_061.ckpt` |
| action | 첫 시도 `finetune`, OOM 후 재시도는 최신 self-forced `epoch_last.ckpt` 기준 `fit` |
| DDP shape | `trainer.num_nodes=1`, `trainer.devices=4`, 총 4 ranks |
| precision | `bf16-mixed` |
| DMD objective | active-control DMD |
| DMD active axes | pedestrian `[delta_s, delta_n, delta_theta]`, vehicle/cyclist `[delta_s, delta_theta]` |
| lr | Generator `1.0e-6`, generated estimator `1.0e-6` |
| estimator updates | `5` per train step |
| estimator warmup | `1` epoch |
| trainable range | `unfrozen_range=middle` |
| detach block transition | `true` |
| self-forced sample steps | Euler `sample_steps=16` |
| self-forced backprop | `backprop_last_k=8` |
| random terminal policy | `all` |
| stop-motion | self-forced training rollout `false`, validation/inference decoder `false` |
| train data fraction | `data.train_epoch_sample_fraction=0.25` |
| train batch construction | `data.train_memory_balanced_batches=true`, `trainer.use_distributed_sampler=false` |
| validation | `val_closed_loop=true`, `val_open_loop=false`, `limit_val_batches=0.1`, every 2 epochs |
| epochs | `16` |
| initial train batch | per-rank `144`, effective global batch `576` |
| OOM fallback | `144 -> 128 -> ... -> 64`, latest self-forced checkpoint resume |
| val/test batch | per-rank `8` |
| scorer scenes | `1680` |
| tmux session | `catk-self-forced-dmd-a100x4-testa` |

2026-06-06 기준 기본 시작 batch는 per-rank `144`로 둡니다. 이전 `testa` A100x4
`unfrozen_range=middle`, `train_memory_balanced_batches=true` probe에서는 per-rank
`bs160`이 20-step smoke를 CUDA OOM 없이 통과했고, full-epoch probe도 사용자 요청으로
중단하기 전 `32/191` train step까지 OOM 없이 진행했습니다. 해당 partial probe에서 최대 관측
GPU memory는 약 `74.5 GiB / 80 GiB`였습니다. full epoch 전체를 끝까지 돈 값은 아니므로,
rare heavy batch에 대비해 시작 batch를 `144`로 낮추고 OOM fallback을 함께 둡니다.
memory-balanced sampler는 rank별 batch를 직접 나누므로
Lightning의 자동 distributed sampler는 꺼 둡니다. 이 조합이 빠지면 Lightning이 batch sampler를
다시 감싸려 하면서 실행 전 오류가 납니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- \
  tmux attach -t catk-self-forced-dmd-a100x4-testa
```

학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_a100x4_testa_static_pod.py --stop
```

#### hsb-npc-training-3-1 H100x3 single-pod DMD self-forcing fine-tuning

`hsb-npc-training-3-1` 단일 H100 3GPU pod에서 epoch 61 pretrained Generator checkpoint로
DMD-style self-forced fine-tuning을 돌릴 때는 아래 launcher를 씁니다. 이 launcher는
pod를 만들거나 지우거나 재시작하지 않고, pod 안의 `/tmp/catk_self_forced_dmd_h100x3_hsb31`
클린 체크아웃과 tmux session만 사용합니다. 따라서 pod의 기존 `/mnt/nuplan/projects/catk`
작업트리를 건드리지 않습니다.

```bash
python scripts/launch_self_forced_dmd_h100x3_hsb31_static_pod.py --replace
```

짧은 training + validation smoke만 확인하려면:

```bash
python scripts/launch_self_forced_dmd_h100x3_hsb31_static_pod.py \
  --replace \
  --task-name flow_self_forced_dmd_h100x3_hsb31_smoke_bs144 \
  --session catk-self-forced-dmd-h100x3-hsb31-smoke \
  --max-epochs 1 \
  --check-val-every-n-epoch 1 \
  --limit-train-batches 2 \
  --limit-val-batches 1
```

기본 실험 설정:

| 항목 | 값 |
|---|---|
| pod | `hsb-npc-training-3-1` 3 H100 |
| branch | `semi_control_stable` |
| pod checkout | `/tmp/catk_self_forced_dmd_h100x3_hsb31` |
| experiment | `self_forced_npfm_h100_3_hsb31` |
| default task | `flow_self_forced_dmd_h100x3_hsb31_epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr1e-6_bs144_frac025_ep16_middle` |
| pretrained checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint 의미 | `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20` epoch 61 Generator, artifact metadata epoch 62 / global step 278192 |
| local checkpoint path in pod | `/workspace/flow_self_forced_dmd_h100x3_hsb31_pretrain_epoch061_x5f9g0ce/v57/epoch_061.ckpt` |
| action | 첫 시도 `finetune`, OOM 후 재시도는 최신 self-forced `epoch_last.ckpt` 기준 `fit` |
| DDP shape | `trainer.num_nodes=1`, `trainer.devices=3`, 총 3 ranks |
| precision | `bf16-mixed` |
| DMD objective | active-control DMD |
| DMD active axes | pedestrian `[delta_s, delta_n, delta_theta]`, vehicle/cyclist `[delta_s, delta_theta]` |
| lr | Generator `1.0e-6`, generated estimator `1.0e-6` |
| estimator updates | `5` per train step |
| estimator warmup | `1` epoch |
| trainable range | `unfrozen_range=middle` |
| detach block transition | `true` |
| self-forced sample steps | Euler `sample_steps=16` |
| self-forced backprop | `backprop_last_k=8` |
| random terminal policy | `all` |
| stop-motion | self-forced training rollout `false`, validation/inference decoder `false` |
| train data fraction | `data.train_epoch_sample_fraction=0.25` |
| train batch construction | `data.train_memory_balanced_batches=true`, `trainer.use_distributed_sampler=false` |
| validation | `val_closed_loop=true`, `val_open_loop=false`, `limit_val_batches=0.1`, every 2 epochs |
| epochs | `16` |
| initial train batch | per-rank `144`, effective global batch `432` |
| OOM fallback | 기본값은 `144 -> 128 -> ... -> 16`, latest self-forced checkpoint resume |
| val/test batch | per-rank `8` |
| scorer scenes | `1680` |
| tmux session | `catk-self-forced-dmd-h100x3-hsb31` |

2026-06-06 `hsb-npc-training-3-1` H100x3 검증 결과 및 현재 보수 설정:

| 검증 | 설정 | 결과 |
|---|---|---|
| historical train smoke | `bs160`, 3 ranks, `limit_train_batches=2`, `max_epochs=1` | 성공, `train/loss_epoch=0.07317`, `time/train_epoch_minutes=0.63746`, `worst_peak_reserved_pct_epoch_max=83.45095` |
| validate smoke | 위 smoke의 `epoch_last.ckpt`, `action=validate`, `scorer_scene_num=24`, `limit_val_batches=1` | 성공, `val_closed/sim_agents_2025/realism_meta_metric=0.769756`, `scenario_counter=24` |
| current default | `bs144`, 3 ranks | 이전 `bs160` smoke보다 낮은 보수 시작 batch. OOM 시 `16`씩 낮춰 최신 self-forced checkpoint로 resume |

위 historical smoke 기준 첫 train batch에는 dataloader/DDP warmup이 포함되어 28초, 두 번째
train batch는 약 9초였습니다. current default full run의 epoch당 step 수는 effective global batch
`432`와 `train_epoch_sample_fraction=0.25` 기준 약 284 step 수준으로 잡히므로, 초기
추정은 train-only epoch당 대략 `40~50분`, validation epoch은 추가 시간이 붙습니다.
실제 full run 시작 후에는 첫 1~2 epoch의 `time/train_epoch_minutes`를 기준으로 다시
보정하세요.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-3-1 -c main -- \
  tmux attach -t catk-self-forced-dmd-h100x3-hsb31
```

학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_h100x3_hsb31_static_pod.py --stop
```

#### hsb-npc-training-3-2 H100x3 single-pod DMD self-forcing fine-tuning

`hsb-npc-training-3-2`에서 같은 H100x3 self-forced DMD recipe를 돌릴 때는
아래 launcher를 씁니다. 학습 설정은 `hsb-npc-training-3-1` launcher와 같고,
pod / clean checkout / task name / tmux session / checkpoint cache path만 `hsb32`
전용으로 분리됩니다.

```bash
python scripts/launch_self_forced_dmd_h100x3_hsb32_static_pod.py --replace
```

기본 차이:

| 항목 | 값 |
|---|---|
| pod | `hsb-npc-training-3-2` 3 H100 |
| pod checkout | `/tmp/catk_self_forced_dmd_h100x3_hsb32` |
| experiment | `self_forced_npfm_h100_3_hsb32` |
| default task | `flow_self_forced_dmd_h100x3_hsb32_epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr1e-6_bs144_frac025_ep16_middle` |
| local checkpoint path in pod | `/workspace/flow_self_forced_dmd_h100x3_hsb32_pretrain_epoch061_x5f9g0ce/v57/epoch_061.ckpt` |
| tmux session | `catk-self-forced-dmd-h100x3-hsb32` |

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-3-2 -c main -- \
  tmux attach -t catk-self-forced-dmd-h100x3-hsb32
```

학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_h100x3_hsb32_static_pod.py --stop
```

같은 단일 A100x4 recipe를 `testaa` pod에서 별도 task/session으로 돌리려면 아래 wrapper를
사용합니다. 학습 설정은 위 `testa` launcher와 같고, 기본 pod / task name / tmux session /
pretrain checkpoint cache 경로만 `testaa` 전용으로 분리됩니다.

```bash
python scripts/launch_self_forced_dmd_a100x4_testaa_static_pod.py --replace
```

`testaa` tmux 확인:

```bash
kubectl exec -it -n p-pnc testaa -c main -- \
  tmux attach -t catk-self-forced-dmd-a100x4-testaa
```

`testaa` 학습 프로세스만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_self_forced_dmd_a100x4_testaa_static_pod.py --stop
```

`cache_frozen_map_features=true` 는 self-forced DMD train step 안에서 frozen map encoder의 출력을
재사용합니다. 기본 `unfrozen_range=middle` 에서는 map encoder가 학습되지 않으므로,
generated estimator를 한 step 안에서 여러 번 업데이트할 때 같은 map을 반복 인코딩하지 않아도 됩니다.
이 최적화는 decoder의 map encoder parameter가 실제로 frozen일 때만 켜지고, map encoder를 학습하도록
설정을 바꾸면 자동으로 기존처럼 매번 인코딩합니다. 문제가 의심되면 아래 override로 즉시 끌 수 있습니다.

```bash
python scripts/launch_self_forced_dmd_a100x4x2_testa_static_pods.py \
  --replace \
  --extra-hydra-overrides 'model.model_config.self_forced.cache_frozen_map_features=false'
```

2026-06-06 `testa + testaa` A100x4x2 짧은 probe 결과:

| 조건 | 설정 | 결과 |
|---|---|---:|
| 기존 경로 | `cache_frozen_map_features=false`, bs18, train 12 batch, validation off | `2:27` |
| cached 경로 | `cache_frozen_map_features=true`, bs18, train 12 batch, validation off | `2:22` |

두 probe 모두 `Trainer.fit stopped: max_epochs=1 reached` 까지 정상 종료했습니다. 짧은 측정 기준으로
전체 step time은 약 `3~5%` 개선됐고, train epoch loss는 같은 스케일을 유지했습니다. 이 최적화는 frozen
map encoder의 dropout sample을 한 step 안에서 재사용하므로 bitwise 동일성을 목표로 하지는 않습니다.
수식 구조, 학습 파라미터, edge set, loss target은 바꾸지 않습니다.

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

self-forced fine-tuning에서 학습할 파라미터 범위는 `unfrozen_range` 로 정합니다. 기본값 `middle` 은 map encoder와 대부분의 agent 문맥부를 고정하고, 마지막 agent 문맥 블록과 flow decoder만 학습 대상으로 둡니다. 더 넓게 학습하려면 map encoder만 고정하는 `except_map_encoder` 를 명시하고, 더 보수적으로 보려면 마지막 궤적 생성부만 여는 `full_flow_decoder` 를 시도하세요.

```bash
... model.model_config.self_forced.unfrozen_range=full_flow_decoder
```

### WOSAC-CPD / WOSAC-CES Distribution Metrics

closed-loop validation과 Sim Agents submission export에서는 모델이 실제로 만든 10Hz rollout으로 아래 metric을 계산합니다.

- `val_closed/WOSAC-CPD/value`: 같은 scenario 안 rollout끼리의 조건부 다양성입니다. 높을수록 rollout들이 서로 다릅니다.
- `val_closed/WOSAC-CES/value`: validation GT가 있을 때만 계산되는 Energy Score 계열 metric입니다. 낮을수록 좋습니다.
- `test/WOSAC-CPD/value`: test submission export에서 계산되는 CPD입니다. test set은 GT 미래를 제공하지 않으므로 CES는 기록하지 않습니다.
- `*/WOSAC-CPD/DPR`: `model.model_config.wosac_cpd_reference`에 flow-pretrain CPD를 넣었을 때만 기록됩니다. 값은 `현재 CPD / 기준 CPD` 입니다.

CPD/CES 정규화 scale은 기본적으로 training cache 전체에서 offline 계산한 agent type별 고정값을 사용합니다. 순서는 `vehicle`, `pedestrian`, `cyclist`입니다.

```yaml
model:
  model_config:
    wosac_distribution_type_scale: [22.3461620418, 4.5793447978, 18.5374388830]
```

고정 scale이 있으면 validation/test CPD/CES는 항상 이 값을 우선 사용합니다. 그래서 `val_closed/WOSAC-CPD/value`와 `test/WOSAC-CPD/value`가 같은 normalization 기준을 공유합니다. `model.model_config.wosac_distribution_type_scale=null`로 명시하면 기존 동작처럼 validation GT에서 scale을 fallback 계산합니다. Test split에는 GT future가 없으므로 고정 scale이 없을 때는 test CPD가 raw 단위로 계산됩니다.

이 metric들은 학습 step에서는 계산하지 않고, validation/test closed-loop rollout이 만들어진 뒤에만 계산합니다. `n_rollout_closed_val`이 32이면 이미 생성된 32개 rollout만 사용하고 별도 rollout을 추가 생성하지 않습니다.

### Self-forced Generated Estimator Warmup

- `model.model_config.self_forced.estimator_warmup_epochs=0` 이 기본값입니다.
- 따라서 online Generator warmup skip 없이, 첫 train step부터 generated estimator 업데이트와 Generator 업데이트를 수행합니다.
- `estimator_warmup_epochs>0` 으로 override하면 해당 기간에는 online Generator를 업데이트하지 않고, 현재 Generator가 만든 self-rollout으로 generated estimator만 먼저 학습합니다.
- warmup 중 self-rollout은 `torch.no_grad()`로 생성하고, Generator optimizer step과 EMA update는 실행하지 않습니다.
- warmup epoch 끝에는 validation을 건너뜁니다. 예를 들어
  `estimator_warmup_epochs=1`, `trainer.check_val_every_n_epoch=2` 이면 epoch 0은
  generated estimator warmup만 수행하고 validation 없이 끝납니다. epoch 1부터
  Generator 학습이 시작되며, validation은 generator 학습 epoch 기준 두 번째인
  epoch 2 끝에서 처음 실행됩니다.

예시:

```bash
... model.model_config.self_forced.estimator_warmup_epochs=1
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

- `local_val_flow` 기본값은 `model.model_config.scorer_scene_num=1680` 입니다. 따라서 1GPU / `val_batch_size=4`에서는 Fast WOSAC scorer가 약 1680 scene을 보도록 `limit_val_batches=60`이 실행 중 `420` batch로 늘어납니다. 6GPU / `val_batch_size=4`에서는 rank당 `70` batch가 필요합니다.
- 전체 validation set을 돌리고 싶으면 `trainer.limit_val_batches=1.0` 을 추가하면 됩니다.
- `scorer_scene_num=null` 또는 `0` 으로 끄면 예전처럼 `n_batch_sim_agents_metric`와 `trainer.limit_val_batches`를 직접 조합해 더 작은 quick check를 만들 수 있습니다.

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

#### hsb-npc-training-1 H100x6 epoch 61 Waymo validation 제출

`flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20`
학습에서 Fast-RMM 기준으로 고른 epoch 61 checkpoint를 사용해 validation split 전체의 Waymo Sim Agents 제출물을 만들 때는 아래 launcher를 씁니다.
기본 추론 설정은 `sample_steps=16`, Euler solver, `use_stop_motion=false`, `use_lqr=false`, `antithetic_pairs=true`, `noise_scale=1.0`입니다.

이 스크립트는 기본적으로 **validation leaderboard에 업로드하지 않습니다.** 먼저 validation split rollout, archive 생성, proto 검증만 수행합니다.
실제 Waymo validation 제출을 하려면 `--submit-validation`을 명시합니다.

```bash
# 안전 smoke: validation split 앞 1 batch만 생성하고 tar.gz/proto 구조를 검증합니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --smoke-test \
  --replace
```

```bash
# full validation archive 생성 + archive 검증. 업로드는 하지 않습니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py --replace
```

```bash
# Waymo validation 업로드 UI 접근성만 확인합니다. 파일 첨부/submit 클릭은 하지 않습니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --smoke-test \
  --verify-waymo-ui \
  --replace
```

```bash
# 실제 validation leaderboard 제출.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --submit-validation \
  --replace
```

```bash
# stratified Gaussian + antithetic pair 설정으로 실제 validation leaderboard에 제출합니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --antithetic-pairs true \
  --stratified-gaussian-noise true \
  --noise-scale 1.0 \
  --submit-validation \
  --replace
```

```bash
# validation rollout/archive 생성은 성공했지만 Waymo 업로드만 네트워크 문제로 실패한 경우,
# 기존 archive를 재검증한 뒤 업로드만 다시 시도합니다. full validation은 다시 돌리지 않습니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --upload-existing-archive /mnt/nuplan/projects/catk/logs/flow_agents_7m_waymo_val_epoch061_x5f9g0ce_h100x6_hsb1_sample16_euler_antithetic_stratified_noise1000/runs/20260604_220315_full/sim_agents_2025_submission.tar.gz \
  --antithetic-pairs true \
  --stratified-gaussian-noise true \
  --noise-scale 1.0 \
  --submit-validation \
  --replace
```

```bash
# iid Gaussian noise, 즉 antithetic pair를 끈 설정으로 실제 validation leaderboard에 제출합니다.
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py \
  --antithetic-pairs false \
  --noise-scale 1.0 \
  --submit-validation \
  --replace
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pod | `hsb-npc-training-1` 단일 H100x6 |
| branch | `semi_control_stable` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | 61 |
| action / experiment | `action=validate`, `experiment=sim_agents_sub_flow` |
| rollout count | `model.model_config.n_rollout_closed_val=32` |
| solver / denoising | Euler, `model.model_config.validation_rollout_sampling.sample_steps=16` |
| inference noise | 기본 `antithetic_pairs=true`, `stratified_gaussian_noise=false`, `noise_scale=1.0`, `validation_closed_seed=4`; stratified 제출은 `--stratified-gaussian-noise true`, iid 제출은 `--antithetic-pairs false --noise-scale 1.0` |
| post-process | `use_lqr=false`, `use_stop_motion=false` |
| method name | `Flow Agents 7M` |
| authors / affiliation | `SB H`, `KO O` / `NLK` |
| description | 기본 `flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_tailprefix_roundtrip05_lr6e-4_bs20_epoch061_true_stratified_false_1.0`; stratified 제출은 `..._epoch061_true_stratified_true_1.0`, iid 제출은 `..._epoch061_false_stratified_false_1.0` |
| account | `h.sb@naverlabs.com` |
| default validation batch | per-rank `val_batch_size=48` |
| tmux session | 기본 `catk-flow-waymo-val-submission-epoch061-h100x6-hsb1-antithetic-iidgaussian-noise1000`; stratified 제출은 `catk-flow-waymo-val-submission-epoch061-h100x6-hsb1-antithetic-stratified-noise1000`, iid 제출은 `catk-flow-waymo-val-submission-epoch061-h100x6-hsb1-iid-iidgaussian-noise1000` |

기존 `noise_scale=1.016` 설정을 재현해야 하면 동일 launcher에 `--noise-scale 1.016`을 추가합니다. 이 경우 task/session/description은 noise scale에 맞춰 자동으로 분리됩니다.

스크립트가 생성한 archive는 `scripts/verify_waymo_submission_archive.py`로 자동 검증합니다. 이 검증은 tar member 이름, shard proto parse,
submission metadata, scenario id 중복, scenario당 32개 rollout, trajectory당 80 future step, NaN/Inf 부재를 확인합니다.
full validation 실행에서는 launcher가 `${CACHE_ROOT}/validation`의 `.pkl` 개수를 직접 세고 archive의 scenario 수가 그 값과 정확히 같은지도 검증합니다.
필요하면 `--expected-validation-scenarios 44097`처럼 수동 기대값을 지정해 같은 검증을 강제할 수 있습니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-flow-waymo-val-submission-epoch061-h100x6-hsb1-antithetic-iidgaussian-noise1000
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_waymo_val_submission_epoch061_h100x6_hsb1_static_pod.py --stop
```

#### hsb-npc-training-1 H100x6 epoch 116 Waymo validation 제출

`flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18`
학습에서 Fast-RMM 기준으로 고른 epoch 116 checkpoint를 사용해 validation split 전체의 Waymo Sim Agents 제출물을 만들 때는 아래 launcher를 씁니다.
기본 추론 설정은 `sample_steps=16`, Euler solver, `use_stop_motion=false`, `use_lqr=false`, `antithetic_pairs=true`, `stratified_gaussian_noise=true`, `noise_scale=1.0`입니다.

이 스크립트는 기본적으로 **validation leaderboard에 업로드하지 않습니다.** 먼저 validation split rollout, archive 생성, proto 검증만 수행합니다.
실제 Waymo validation 제출을 하려면 `--submit-validation`을 명시합니다.

```bash
# 안전 smoke: validation split 앞 1 batch만 생성하고 tar.gz/proto 구조를 검증합니다.
python scripts/launch_waymo_val_submission_epoch116_h100x6_hsb1_static_pod.py \
  --smoke-test \
  --replace
```

```bash
# full validation archive 생성 + archive 검증. 업로드는 하지 않습니다.
python scripts/launch_waymo_val_submission_epoch116_h100x6_hsb1_static_pod.py --replace
```

```bash
# 실제 validation leaderboard 제출.
python scripts/launch_waymo_val_submission_epoch116_h100x6_hsb1_static_pod.py \
  --submit-validation \
  --replace
```

```bash
# antithetic pair + iid Gaussian + noise_scale=1.016 설정으로 실제 validation leaderboard에 제출합니다.
python scripts/launch_waymo_val_submission_epoch116_antithetic_noise1016_h100x6_hsb1_static_pod.py \
  --submit-validation \
  --replace
```

```bash
# antithetic pair + iid Gaussian + noise_scale=1.0 설정으로 실제 validation leaderboard에 제출합니다.
python scripts/launch_waymo_val_submission_epoch116_antithetic_noise1000_h100x6_hsb1_static_pod.py \
  --submit-validation \
  --replace
```

```bash
# antithetic pair를 끈 iid Gaussian + noise_scale=1.0 설정으로 실제 validation leaderboard에 제출합니다.
python scripts/launch_waymo_val_submission_epoch116_iid_noise1000_h100x6_hsb1_static_pod.py \
  --submit-validation \
  --replace
```

```bash
# validation rollout/archive 생성은 성공했지만 Waymo 업로드만 네트워크 문제로 실패한 경우,
# 기존 archive를 재검증한 뒤 업로드만 다시 시도합니다. full validation은 다시 돌리지 않습니다.
python scripts/launch_waymo_val_submission_epoch116_h100x6_hsb1_static_pod.py \
  --upload-existing-archive /mnt/nuplan/projects/catk/logs/flow_agents_7m_waymo_val_epoch116_mqfq3u39_h100x6_hsb1_sample16_euler_antithetic_stratified_noise1000/runs/<run_id>/sim_agents_2025_submission.tar.gz \
  --submit-validation \
  --replace
```

```bash
# antithetic pair + iid Gaussian + noise_scale=1.016 archive는 이 명령으로 업로드만 재시도합니다.
python scripts/launch_waymo_val_submission_epoch116_antithetic_noise1016_h100x6_hsb1_static_pod.py \
  --upload-existing-archive /mnt/nuplan/projects/catk/logs/flow_agents_7m_waymo_val_epoch116_mqfq3u39_h100x6_hsb1_sample16_euler_antithetic_iidgaussian_noise1016/runs/<run_id>/sim_agents_2025_submission.tar.gz \
  --submit-validation \
  --replace
```

```bash
# antithetic pair + iid Gaussian + noise_scale=1.0 archive는 이 명령으로 업로드만 재시도합니다.
python scripts/launch_waymo_val_submission_epoch116_antithetic_noise1000_h100x6_hsb1_static_pod.py \
  --upload-existing-archive /mnt/nuplan/projects/catk/logs/flow_agents_7m_waymo_val_epoch116_mqfq3u39_h100x6_hsb1_sample16_euler_antithetic_iidgaussian_noise1000/runs/<run_id>/sim_agents_2025_submission.tar.gz \
  --submit-validation \
  --replace
```

```bash
# antithetic pair를 끈 iid Gaussian + noise_scale=1.0 archive는 이 명령으로 업로드만 재시도합니다.
python scripts/launch_waymo_val_submission_epoch116_iid_noise1000_h100x6_hsb1_static_pod.py \
  --upload-existing-archive /mnt/nuplan/projects/catk/logs/flow_agents_7m_waymo_val_epoch116_mqfq3u39_h100x6_hsb1_sample16_euler_iid_iidgaussian_noise1000/runs/<run_id>/sim_agents_2025_submission.tar.gz \
  --submit-validation \
  --replace
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pod | `hsb-npc-training-1` 단일 H100x6 |
| branch | `semi_control_stable` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-mqfq3u39:v121` |
| checkpoint epoch | 116 |
| action / experiment | `action=validate`, `experiment=sim_agents_sub_flow` |
| rollout count | `model.model_config.n_rollout_closed_val=32` |
| solver / denoising | Euler, `model.model_config.validation_rollout_sampling.sample_steps=16` |
| inference noise | 기본 wrapper는 `antithetic_pairs=true`, `stratified_gaussian_noise=true`, `noise_scale=1.0`, `validation_closed_seed=4`; `launch_waymo_val_submission_epoch116_antithetic_noise1016_h100x6_hsb1_static_pod.py`는 `antithetic_pairs=true`, `stratified_gaussian_noise=false`, `noise_scale=1.016`; `launch_waymo_val_submission_epoch116_antithetic_noise1000_h100x6_hsb1_static_pod.py`는 `antithetic_pairs=true`, `stratified_gaussian_noise=false`, `noise_scale=1.0`; `launch_waymo_val_submission_epoch116_iid_noise1000_h100x6_hsb1_static_pod.py`는 `antithetic_pairs=false`, `stratified_gaussian_noise=false`, `noise_scale=1.0` |
| post-process | `use_lqr=false`, `use_stop_motion=false` |
| method name | `Flow Agents 7M` |
| authors / affiliation | `SB H`, `KO O` / `NLK` |
| description | 기본 wrapper는 `flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18_116_true_stratified_true_1.0`; noise 1.016 wrapper는 `..._116_true_stratified_false_1.016`; antithetic noise 1.0 wrapper는 `..._116_true_stratified_false_1.0`; iid noise 1.0 wrapper는 `..._116_false_stratified_false_1.0` |
| account | `h.sb@naverlabs.com` |
| default validation batch | per-rank `val_batch_size=48` |
| tmux session | 기본 wrapper는 `catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-antithetic-stratified-noise1000`; noise 1.016 wrapper는 `catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-antithetic-iidgaussian-noise1016`; antithetic noise 1.0 wrapper는 `catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-antithetic-iidgaussian-noise1000`; iid noise 1.0 wrapper는 `catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-iid-iidgaussian-noise1000` |

스크립트가 생성한 archive는 `scripts/verify_waymo_submission_archive.py`로 자동 검증합니다. 이 검증은 tar member 이름, shard proto parse,
submission metadata, scenario id 중복, scenario당 32개 rollout, trajectory당 80 future step, NaN/Inf 부재를 확인합니다.
full validation 실행에서는 launcher가 `${CACHE_ROOT}/validation`의 `.pkl` 개수를 직접 세고 archive의 scenario 수가 그 값과 정확히 같은지도 검증합니다.
필요하면 `--expected-validation-scenarios 44097`처럼 수동 기대값을 지정해 같은 검증을 강제할 수 있습니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-antithetic-stratified-noise1000
```

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-antithetic-iidgaussian-noise1000
```

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-flow-waymo-val-submission-epoch116-h100x6-hsb1-iid-iidgaussian-noise1000
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_waymo_val_submission_epoch116_h100x6_hsb1_static_pod.py --stop
python scripts/launch_waymo_val_submission_epoch116_antithetic_noise1000_h100x6_hsb1_static_pod.py --stop
python scripts/launch_waymo_val_submission_epoch116_iid_noise1000_h100x6_hsb1_static_pod.py --stop
```

#### hsb-npc-training-1 H100x6 epoch 116 Waymo test 제출

`flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18`
학습에서 Fast-RMM 기준으로 고른 epoch 116 checkpoint를 사용해 test split 전체 제출물을 만들 때는 아래 launcher를 씁니다.

이 스크립트는 기본적으로 **test leaderboard에 업로드하지 않습니다.** 먼저 test split rollout, archive 생성, proto 검증만 수행합니다.
실제 Waymo test 제출 횟수를 쓰려면 `--submit-test`를 명시해야 합니다.

```bash
# 안전 smoke: test split 앞 1 batch만 생성하고 tar.gz/proto 구조를 검증합니다.
python scripts/launch_waymo_test_submission_h100x6_hsb1_static_pod.py \
  --smoke-test \
  --replace
```

```bash
# full test archive 생성 + archive 검증. 업로드는 하지 않습니다.
python scripts/launch_waymo_test_submission_h100x6_hsb1_static_pod.py --replace
```

```bash
# Waymo test 업로드 UI 접근성만 확인합니다. 파일 첨부/submit 클릭은 하지 않습니다.
python scripts/launch_waymo_test_submission_h100x6_hsb1_static_pod.py \
  --smoke-test \
  --verify-waymo-ui \
  --replace
```

```bash
# 실제 test leaderboard 제출. 이 명령만 test 제출 횟수를 소비할 수 있습니다.
python scripts/launch_waymo_test_submission_h100x6_hsb1_static_pod.py \
  --submit-test \
  --replace
```

기본 설정:

| 항목 | 설정 |
|---|---|
| pod | `hsb-npc-training-1` 단일 H100x6 |
| branch | `semi_control_stable` |
| checkpoint artifact | `jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57` |
| checkpoint epoch | 116 |
| action / experiment | `action=test`, `experiment=sim_agents_sub_flow` |
| rollout count | `model.model_config.n_rollout_closed_val=32` |
| solver / denoising | Euler, `model.model_config.validation_rollout_sampling.sample_steps=16` |
| inference noise | `antithetic_pairs=false`, `noise_scale=1.0`, `validation_closed_seed=4` |
| post-process | `use_lqr=false`, `use_stop_motion=false` |
| method name | `Flow Agents 7M` |
| authors / affiliation | `SB H`, `KO O` / `NLK` |
| description | `flow_control_space_pretrain_h100x6_hsb1_prefix_default_noslip_train_plus_validation_tailprefix_roundtrip05_lr6e-4_bs18_116_false_1.0` |
| account | `h.sb@naverlabs.com` |
| default test batch | per-rank `test_batch_size=48` |
| tmux session | `catk-flow-waymo-test-submission-h100x6-hsb1` |

스크립트가 생성한 archive는 `scripts/verify_waymo_submission_archive.py`로 자동 검증합니다. 이 검증은 tar member 이름, shard proto parse,
submission metadata, scenario id 중복, scenario당 32개 rollout, trajectory당 80 future step, NaN/Inf 부재를 확인합니다.
full test 실행에서는 launcher가 `${CACHE_ROOT}/testing`의 `.pkl` 개수를 직접 세고 archive의 scenario 수가 그 값과 정확히 같은지도 검증합니다.
필요하면 `--expected-test-scenarios 44920`처럼 수동 기대값을 지정해 같은 검증을 강제할 수 있습니다.

tmux 확인:

```bash
kubectl exec -it -n p-pnc hsb-npc-training-1 -c main -- \
  tmux attach -t catk-flow-waymo-test-submission-h100x6-hsb1
```

실험 코드만 멈추고 pod는 그대로 두려면:

```bash
python scripts/launch_waymo_test_submission_h100x6_hsb1_static_pod.py --stop
```

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
Waymo test set은 제출 횟수 제한이 있으므로, test 업로드를 할 때는 아래 옵션을 추가로 넣어야 합니다.

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
- control-space Flow Matching에서는 committed pose rollout을 그대로 `F_psi` / `F_rho` 입력으로 쓰지 않습니다. 실제 실행된 pose trajectory를 첫 anchor 기준 rolling control sequence로 다시 투영한 뒤 generated estimator, teacher, DMD/SiD loss를 모두 3차원 control flow state 위에서 계산합니다. 따라서 rollout은 metric/실행용 pose로 굴러가되, self-forced 분포맞춤 objective는 control-space와 섞이지 않습니다.
- inference 와 동일한 0.5초 commit/update 규칙을 쓰되 `flow_window_steps / 5` block 만큼만 도는 differentiable training rollout 경로. 학습 중에는 DDP 전체 rank가 random terminal step `s` 를 하나 공유하고, 모든 rank의 scenario/agent와 0.5초 commit block이 같은 `s` 를 씁니다. 실제 실행 step 수는 `K = sample_steps + 1 - s` 이며, terminal 이전 step은 no-grad로 계산하고 terminal clean estimate를 만드는 마지막 step 하나만 gradient를 유지합니다.
- stop-motion gate는 self-forced 학습 rollout과 validation / test / submission inference 모두에서 사용하지 않습니다. `decoder.use_stop_motion` 과 `self_forced.use_stop_motion` 은 호환용 config 키로만 남아 있고 실제 동작은 false로 고정됩니다.
- random terminal step `s` 는 self-rollout 의 실행 길이와 commit 지점만 정합니다. Generated estimator `F_psi` 학습과 generator direction 계산에서 쓰는 flow noising `tau` 는 rollout 의 `s` 와 독립적으로 전체 tau 구간에서 새로 샘플링합니다.
- generator direction은 raw score/path 이동량을 그대로 쓰지 않고, 같은 noisy path에서 `F_rho` 와 `F_psi` 가 각각 추정한 clean path 차이를 사용합니다.
- DMD 방향은 active 축의 `teacher clean path - generated clean path` 를 stable scale `max(mean(abs(X-R)), 0.05)` 로 정규화해서 만듭니다. 기본 scope는 `type`이라 같은 scene 안의 같은 agent type은 하나의 scale을 공유합니다.
- agent별 teacher 방향 정렬 gate가 `sum((R-X) * (R-F)) <= 0` 인 DMD 방향을 버립니다.
- agent별 trust-region은 고정 반경을 쓰지 않고 현재 Generator와 teacher 사이의 active RMS 거리 `rms(X-R)` 를 그대로 사용합니다. 따라서 최종 DMD 방향의 active RMS는 항상 `rms(X-R)` 이하입니다.
- DMD target은 self-forced DMD update가 시작된 뒤 2 epoch 동안 `0.25 -> 0.625 -> 1.0` 계수로 완만하게 주입합니다.
- control-space self-forcing에서 `use_holonomic_model_only=false`이면 active-control DMD를 사용합니다. pedestrian은 `[delta_s, delta_n, delta_theta]` 3축을 모두 쓰고, vehicle/cyclist는 실제 non-holonomic decode에 직접 쓰이는 `[delta_s, delta_theta]`만 씁니다. 즉 vehicle/cyclist lateral `delta_n`에는 DMD 방향과 DMD loss gradient를 주지 않습니다.
- DMD 정규화 분모도 같은 active 축만 사용합니다. vehicle/cyclist lateral 오차가 DMD 방향 크기를 키우거나 줄이지 않게 하기 위한 규칙입니다.
- DMD target은 `committed_path_norm + eta * path_delta`를 detached target으로 둡니다. 여기서 `path_delta` 는 teacher-aligned bounded DMD 방향이고, `eta` 는 DMD 시작 후 2 epoch ramp 계수입니다.
- Clean-DMD guidance의 기본 noising 구간은 `clean_dmd_tau_low=0.02`, `clean_dmd_tau_high=0.98` 입니다.
- stable scale은 `clean_dmd_normalizer_eps=0.05` 로 최소값을 둬서 pretrained 근처에서 target path가 과하게 튀는 상황을 줄입니다.
- 약한 open-loop flow-matching anchor. `model.model_config.self_forced.use_anchor_flow_matching_loss=false` 로 두면 `anchor_weight` 값과 무관하게 self-forced active step에서 training-mode open-loop forward와 FM loss 계산 자체를 생략합니다. `true` 일 때만 `model.model_config.self_forced.anchor_weight` 로 total loss 반영 강도를 제어합니다. anchor FM 을 끈 상태에서 어떤 rank 의 committed self-rollout 까지 비어있는 (모든 agent 가 invalid anchor0) 드문 경우에는, encoder 파라미터 합에 0 을 곱한 zero-loss 로 backward 만 한 번 돌려 DDP all-reduce 참여를 보장하고 optimizer step 은 건너뜁니다. 이 가드가 없으면 그 rank 만 backward 를 호출하지 않아 다른 rank 의 NCCL all-reduce 가 NCCL_TIMEOUT 까지 hang 합니다.
- 선택적 trainable range. `model.model_config.self_forced.unfrozen_range=middle` 이 기본값이며, map encoder와 대부분의 agent 문맥부는 고정하고 마지막 agent 문맥 블록과 flow decoder만 학습합니다. `except_map_encoder` 는 map encoder만 고정하고 나머지 Generator / generated estimator 파라미터를 열며, `full_flow_decoder` 는 마지막 궤적 생성부만 엽니다.
- epoch별 train subset sampling. self-forced preset은 `data.train_epoch_sample_fraction=0.25` 를 기본으로 두어 매 epoch 전체 train dataset의 25%만 새로 랜덤 샘플링해 학습합니다. DDP에서는 모든 rank가 같은 전역 subset을 공유한 뒤 rank별로 나눠 받습니다. `1.0` 으로 override하면 기존처럼 전체 train dataset을 씁니다.
- Generator EMA는 Generator에만 적용합니다. `F_psi` 는 현재 online Generator가 만든 분포를 따라가야 하므로 EMA를 두지 않고, `F_rho` 는 pretrained 기준점이라 계속 frozen 상태로 둡니다.
- bf16-mixed 안전 backward boundary. self-forced 경로의 forward 와 loss 계산은 mixed precision 으로 유지하되, `manual_backward` 호출 순간만 autocast 를 끄고 scalar loss 를 fp32 로 넘깁니다. 이는 manual optimization 에서 반복 backward 를 수행할 때 PyTorch autocast promote 규칙이 backward graph 의 dtype 을 다시 분류하다가 실패하는 문제를 피하기 위한 경계입니다.
- autograd-safe temporal edge remap. training rollout 에서는 temporal relation embedding 계산에 쓴 원본 `edge_index_t` 를 in-place 수정하지 않고, current-agent attention 용 remapped edge index 를 새 tensor 로 만들어 사용합니다.
- autograd-safe geometry helpers. agent encoder / flow agent decoder 의 edge feature (relative position norm, relative angle) 는 정지 또는 중첩된 agent 가 만드는 영벡터에 대해 backward 가 정의되지 않습니다 (`torch.norm` 의 `x/||x||` 가 `0/0`, `atan2(0, 0)` 의 `1/(y²+x²)` 가 `1/0`). self-forced rollout 처럼 이 feature 들이 살아있는 backward graph 의 일부가 되는 경로에서 한 번이라도 영벡터가 들어오면 NaN gradient 가 encoder weight 까지 흘러 학습이 첫 step 에서 죽습니다. 이를 막기 위해 `safe_norm_2d` helper 가 `(sum(x²) + eps).sqrt()` 형태로 norm 의 backward 분모를 strictly positive 로 유지하고, `angle_between_2d_vectors` 는 상대 벡터가 0일 때 기준 heading 방향을 대체값으로 써서 상대각 0 의미를 보존합니다. flow heading 복원도 `safe_angle_from_2d_vector` 로 통일해 heading vector 가 `[0, 0]` 일 때 `atan2(0, 0)` backward 가 생기지 않게 했습니다. self-forced generator loss 에서 non-finite 가 재발하면 `committed_path_norm`, `path_delta`, `target_path_norm` 요약을 함께 출력하되, 정상 step 에서는 큰 텐서를 스캔하지 않습니다.

### Self-Forced random terminal denoising

Self-forced fine-tuning은 학습 중 `self_forced.sampling.sample_steps` 값을 줄이지 않고도 평균 sampler 호출 수를 줄일 수 있습니다. 현재 H100 self-forced 기본값은 `sample_steps=16`, `random_terminal_step.policy=all`, `backprop_last_k=8`입니다. 이 기본값에서는 terminal denoising step `s`를 랜덤 샘플링하지 않고, 항상 전체 16 step을 실행한 뒤 마지막 8 step에만 gradient를 남깁니다.

`policy=paper_uniform` 으로 override하면 학습 rollout에서는 `K = sample_steps + 1 - s` step까지만 진행한 뒤, 중간 noisy state를 commit하지 않고 terminal step에서 예측한 clean estimate를 2초 preview로 사용합니다. 그 preview 중 앞 0.5초만 기존 commit bridge로 반영합니다. terminal 이전 step은 gradient 없이 계산하고, terminal clean estimate를 만든 step 하나만 gradient를 유지합니다. 이전 구현처럼 `torch.unique(K)` 로 terminal step별 agent group을 나눠 sampler를 여러 번 호출하지 않고, 0.5초 block마다 DDP 전체 rank가 공유한 `K` 로 `FlowODE.generate(..., terminal_step=K, return_terminal_clean=True)`를 한 번만 호출합니다. 다음 block의 context/cache로 들어가는 상태는 detach하여 미래 block loss가 이전 block 내부로 역전파되지 않게 합니다.

Generated Path-Flow Estimator와 generator direction 계산은 random-s 정보를 noising 구간으로 재사용하지 않습니다.
rollout에서 선택된 `s`는 terminal clean estimate를 만들 실행 step 수 `K`와 commit 지점만 정하며,
packed committed path를 만든 뒤에는 `s`별 `[tau_low, tau_high]` 를 전달하지 않습니다.
`F_psi` 학습은 flow ODE의 기본 전체 tau 구간에서 새 tau를 샘플링합니다.
Clean-DMD direction 계산은 `clean_dmd_tau_low` / `clean_dmd_tau_high` 구간에서 새 tau를 샘플링하되,
direction 계산 안에서는 `F_rho`와 `F_psi`가 항상 같은 noisy path와 같은 tau를 봅니다.

DDP에서는 step 시간이 가장 늦게 끝난 rank에 맞춰지므로, `paper_uniform`에서 rank마다 서로 다른 `s`를 뽑으면 짧은 `K`를 뽑은 rank가 긴 `K`를 뽑은 rank를 기다리게 됩니다. `scope=global_batch`는 이 대기 손실을 줄이기 위해 모든 rank가 같은 `K`를 쓰게 합니다. 단일 GPU 또는 torch.distributed가 초기화되지 않은 실행에서는 같은 설정이 자동으로 일반 batch 공유 방식처럼 동작합니다.

```yaml
model:
  model_config:
    self_forced:
      use_anchor_flow_matching_loss: false
      use_stop_motion: false
      sampling:
        sample_steps: 16
        sample_method: euler
        noise_scale: 1.0
        backprop_last_k: 8
        random_terminal_step:
          enabled: true
          scope: global_batch
          policy: all
          min_executed_steps: 16
      ema_weight: 0.99
      ema_start_step: 50
data:
  train_epoch_sample_fraction: 0.25
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

fine-tuning 에 쓰는 rollout 과 inference 에 쓰는 rollout 은 기본 commit/update 의미가 어긋나지 않아야 합니다. 현재 `semi_control_stable`에서는 stop-motion gate를 모든 경로에서 비활성화합니다.

- `model.model_config.decoder.use_stop_motion`: 호환용 키입니다. true override를 줘도 validation / test / submission inference는 false로 동작합니다.
- `model.model_config.self_forced.use_stop_motion`: 호환용 키입니다. true override를 줘도 self-forced closed-loop training rollout은 false로 동작합니다.

### Self-forced Strict DMD Update Separation

- self-forcing DMD에서 Generator update와 generated estimator update를 더 강하게 분리합니다.
- Generator update에서는 target teacher와 generated estimator를 평가자로만 사용하고, 두 보조 모델에 gradient가 생기면 즉시 오류를 냅니다.
- generated estimator update에서는 현재 Generator가 만든 detached closed-loop path만 학습 대상으로 사용해야 하며, Generator에 gradient가 생기면 즉시 오류를 냅니다.
- 이 update 중에는 detached clone으로 만든 path/noisy target만 `F_psi` 입력으로 쓰고, tokenized map/agent context와 anchor mask도 detached view로 넘깁니다. 그래서 rollout context에 Generator graph가 남아 있어도 estimator backward가 online Generator로 되돌아가지 않습니다.
- online Generator와 frozen teacher의 parameter gradient 누적도 update 동안 임시로 꺼 둡니다.
- update 경계마다 이전 단계의 gradient를 명확히 비워서, DMD 방향이 optimizer 간에 섞이지 않게 했습니다.
- Clean-DMD guidance는 active RMS stable scale, teacher 방향 정렬 gate, agent별 trust-region, 2 epoch DMD target ramp를 함께 적용합니다.
- `use_anchor_flow_matching_loss=false` 설정은 그대로 유지됩니다.

### Self-forced SiD-lite Update

- `model.model_config.self_forced.distribution_matching_objective=dmd` 는 active-control Clean-DMD 방식입니다.
  frozen teacher / generated estimator 차이로 teacher-aligned bounded 방향을 만들고, `committed_path_norm + eta * path_delta` 를 detached target으로 둡니다.
  loss는 agent별 active 축 평균에 `0.5`를 곱해 계산합니다. 그래서 surrogate gradient가 active 축에서 `-eta * path_delta` 방향이 되고, vehicle/cyclist lateral 축은 non-holonomic 실행 결과에 직접 쓰이지 않으므로 제외됩니다.
- `model.model_config.self_forced.distribution_matching_objective=sid` 는 SiD-lite 방식입니다.
  closed-loop self-rollout, frozen teacher, generated estimator, estimator update, EMA 구조는 그대로 두고,
  generator update만 `X`, `R`, `F` 관계식으로 계산합니다.
- SiD-lite에서 `X` 는 Generator가 실제로 실행한 path,
  `R` 은 frozen teacher의 clean path 예측,
  `F` 는 generated estimator의 clean path 예측입니다.
- SiD-lite loss도 `path_step_size` 를 사용하지 않습니다.
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

- `model.model_config.self_forced.unfrozen_range=middle` 이 기본값입니다.
  map encoder와 대부분의 agent 문맥부는 고정하고, 마지막 agent 문맥 블록과 flow decoder만 학습할 수 있게 둡니다.
- `unfrozen_range=middle` 은 map encoder와 대부분의 agent 문맥부를 고정하고,
  `agent_encoder.flow_decoder` 와 마지막 temporal / map-to-agent / agent-to-agent 문맥 블록만 엽니다.
  즉 `except_map_encoder` 보다 더 보수적이고, `full_flow_decoder` 보다는 덜 보수적인 중간 설정입니다.
- `unfrozen_range=full_flow_decoder` 는 마지막 궤적 생성부만 학습 가능하게 두는 설정입니다.
  지도/장면/상호작용 해석부는 pretrained 상태로 보존하고, 자기 rollout 분포 차이는 마지막 궤적 생성부가 흡수하게 합니다.
- self-forcing용 기존 `freeze_map_encoder` 설정은 제거했습니다. map encoder만 고정하고 나머지를 전부 학습하는 기존 동작이 필요하면 `unfrozen_range=except_map_encoder` 를 사용합니다.

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
5. 선택된 future를 기존 WOMD `.pkl` schema와 같은 RoaD `.pkl` cache로 저장합니다. Traffic-light 입력은 원본 현재 관측값에서 만든 정적 map token feature로만 유지하며, RoaD block이 진행돼도 별도 stale time scalar를 만들거나 미래 traffic-light 상태를 보지 않습니다.
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
