# Set-level Self-Forced GAN Fine-tuning

이 패치는 CATK pretrained SMART-flow 모델에 **set-level GAN 기반 closed-loop fine-tuning**을 추가합니다.
목표는 closed-loop fine-tuning으로 covariate shift를 줄이면서, student가 특정 mode 하나로 몰리는 mode collapse / mode seeking을 줄이는 것입니다.

## 핵심 아이디어

```text
real = frozen teacher open-loop rollout set, K=16 sampled from offline cache[32]
fake = student closed-loop rollout set, K=16
D(scene, rollout_set) -> scalar logit
```

학습 target은 다음입니다.

```text
student closed-loop rollout set ≈ teacher open-loop rollout set
```

## 추가 파일

```text
src/smart/model/smart_flow_gan.py
src/smart/modules/self_forced_gan_critic.py
src/smart/modules/self_forced_gan_cache.py
configs/model/smart_flow_gan.yaml
configs/experiment/self_forced_gan_h100_6.yaml
tools/build_self_forced_gan_teacher_cache.py
tools/check_self_forced_gan_critic.py
tools/validate_self_forced_gan_cache.py
```

## Teacher cache 형식

multi-node 학습에서는 teacher real set을 online으로 만들지 말고, 학습 전에 offline cache로 고정합니다.
각 scene 파일은 `.pt` dict로 저장합니다.

```python
{
    "rollout_pose": Tensor,  # [32, 20, N, 4], fp16/bf16/float32
    "agent_id": Tensor,      # [N]
    "agent_type": Tensor,    # [N], optional but recommended
    "valid_mask": Tensor,    # [N], teacher가 실제 생성한 current-valid agent
    "seed": Tensor,          # [32], optional
}
```

`rollout_pose`의 마지막 4개 channel은 아래 순서입니다.

```text
x, y, cos(yaw), sin(yaw)
```

권장 저장 방식은 scene당 1개 `.pt` 파일입니다. `index.json`이 있으면 scenario id에서 파일 경로를 읽고, 없으면 `<scenario_id>.pt`를 찾습니다.

cache 검증:

```bash
PYTHONPATH=. python tools/validate_self_forced_gan_cache.py $TEACHER_GAN_CACHE_ROOT --max-files 100
```

cache 생성:

```bash
PYTHONPATH=. python tools/build_self_forced_gan_teacher_cache.py \
  --ckpt-path <path-to-2s-pretrain.ckpt> \
  --output-root "$TEACHER_GAN_CACHE_ROOT" \
  --split train \
  --rollouts-per-scene 32 \
  --storage-dtype float16 \
  --override paths.cache_root="$CACHE_ROOT"
```

이 builder는 pretrained checkpoint를 frozen teacher로 로드한 뒤, 각 scene의 현재 관측
context에서 2초 open-loop trajectory를 32개 샘플링하여 scene별 `.pt` 파일로 저장합니다.
sampling target에는 future GT를 넣지 않고, 저장된 `valid_mask`는 현재 context에서 teacher
sampling을 실제로 수행한 agent만 표시합니다.

## 기본 학습 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=6 -m src.run \
  experiment=self_forced_gan_h100_6 \
  action=finetune \
  paths.cache_root="$CACHE_ROOT" \
  paths.teacher_gan_cache_root="$TEACHER_GAN_CACHE_ROOT" \
  task_name=sf_gan_k16 \
  ckpt_path=<path-to-2s-pretrain.ckpt>
```

기본값은 다음입니다.

| 항목 | 값 |
|---|---:|
| teacher cache | 32 / scene |
| train real set | 16 / scene |
| train fake set | 16 / scene |
| validation rollout | 32 / scene |
| horizon | 2초 |
| commit | 0.5초 × 4 |
| sampling | 32-step Euler, noise scale 1.0 |
| effective scene batch | 64 |
| D warmup | scene exposure 64k 기준, 500~1500 update |
| D:G ratio | 1:1 |
| student train scope | flow decoder only |
| student LR | 1e-6 |
| D LR | 5e-6 |
| R1/R2 sigma | 0.01 |
| R1/R2 weight | 0.1 / 0.1 |
| EMA | 0.99 |
| EMA start | 50 generator updates 이후 |
| fine-tuning epoch | 6 |
| validation | 매 epoch 끝 |

## Discriminator 구조

새 discriminator는 pretrained scene encoder를 새로 학습하지 않습니다. pretrained generator에서 나온 scene context를 freeze해서 쓰고, 새로 학습하는 critic만 작게 둡니다.

```text
X = [B, 16, 20, N, 4]
agent context = [B, N, 128]
map context = [B, M, 128], map geometry = position/orientation

Frozen Scene Encoder            -> agent [B, N, 128], map [B, M, 128]
Trajectory Encoder              -> time [B, 16, 20, N, 128], endpoint [B, 16, 4, N, 128], agent [B, 16, N, 128]
Scene Condition Fusion          -> [B, 16, N, 128]
Map Compliance Encoder          -> endpoint [B, 16, 4, N, 128], pooled [B, 16, N, 128]
Interaction Encoder             -> endpoint [B, 16, 4, N, 128], pooled [B, 16, N, 128]
Agent Pooling mean/std/max/min  -> [B, 16, 128]
Set Pooling mean/std/max/min    -> [B, 512]
Scalar Head                     -> [B, 1]
```

smoke test:

```bash
PYTHONPATH=. python tools/check_self_forced_gan_critic.py
```

현재 critic의 새 학습 parameter 수는 약 `0.68M`입니다. frozen pretrained scene encoder parameter는 여기에 포함하지 않습니다.

## R1/R2 설명

R1/R2는 discriminator가 너무 예민해지는 것을 막는 안정화 항입니다.
작은 pose perturbation을 넣었을 때 discriminator 점수가 과하게 흔들리면 penalty를 줍니다.

```text
R1: teacher real set 주변 smoothness
R2: student fake set 주변 smoothness
```

기본값은 약하게 둡니다.

```text
position_sigma = 0.01 in local normalized coordinate
yaw_sigma      = 0.01 rad
r1_weight      = 0.1
r2_weight      = 0.1
```

주의: position scale은 global x/y에 직접 나누지 않습니다. 현재 pose 기준 local displacement로 변환한 뒤 agent type별 scale을 적용합니다.

| agent type | scale |
|---|---:|
| vehicle | 22.3461620418 |
| pedestrian | 4.5793447978 |
| cyclist | 18.5374388830 |

이 값은 repo의 `wosac_distribution_type_scale`과 동일합니다. vehicle/pedestrian/cyclist의 정상 이동 범위가 다르기 때문에 같은 거리 오차를 type별로 맞춰 보기 위한 값입니다.

## Warmup 규칙

early stop은 쓰지 않습니다.

```text
warmup_updates = clip(ceil(64000 / effective_scene_batch), 500, 1500)
```

예시:

| effective scene batch | warmup updates |
|---:|---:|
| 32 | 1500 |
| 64 | 1000 |
| 128 | 500 |
| 256 | 500 |

warmup 뒤 확인할 diagnostic:

| 값 | 판단 |
|---|---|
| d_R - d_F = 0.5 ~ 3.0 | 정상 |
| d_R - d_F > 5.0 | D가 너무 강함, D LR 낮춤 |
| d_R - d_F < 0.2 | D가 너무 약함, D LR 또는 cache/architecture 점검 |

## Fine-tuning epoch

pretrain은 64 epoch였지만, GAN fine-tuning은 모델을 처음 배우는 과정이 아닙니다.
pretrained 분포를 망가뜨리지 않으면서 closed-loop 분포만 보정하는 과정이므로 **6 epoch**로 둡니다.
validation은 매 epoch 끝에서 수행합니다.

## 하이퍼파라미터 결정 실험 순서

아래 순서로 진행합니다. 한 번에 너무 많은 값을 열지 않습니다.

### 0. Sanity check

목적: cache shape, critic forward, warmup 동작 확인.

```text
K=16, student LR=1e-6, D LR=5e-6, R1/R2=0.1, 1 epoch only
```

통과 조건:

```text
critic forward OK
train/gan/d_margin 유한
validation이 끝까지 실행됨
```

### 1. 기본 실험

```text
K=16
student LR=1e-6
D LR=5e-6
R1/R2=0.1
6 epochs
```

이 실험이 기준선입니다.

### 2. R1/R2 ablation

R1/R2가 traffic sharp event를 둔감하게 만들 수 있으므로 반드시 봅니다.

| 실험 | R1/R2 |
|---|---:|
| no regularization | 0.0 / 0.0 |
| weak regularization | 0.1 / 0.1 |
| strong regularization | 1.0 / 1.0 |

선택 기준:

```text
WOSAC realism 유지
collision/off-road 악화 10% 이내
student diversity / teacher diversity 0.8~1.2
```

### 3. LR pair ablation

| 실험 | student LR | D LR |
|---|---:|---:|
| 기본 | 1e-6 | 5e-6 |
| 강한 update | 2e-6 | 1e-5 |

`d_R-d_F`가 5 이상으로 유지되면 D LR을 낮춥니다.
student가 거의 변하지 않으면 student LR을 2e-6으로 올립니다.

### 4. K ablation

| 실험 | K |
|---|---:|
| cheap | 8 |
| default | 16 |

K=32는 기본 학습에는 쓰지 않습니다. compute가 충분하면 마지막 0.5~1 epoch polishing으로만 고려합니다.

### 5. 비교 baseline

논문용 최소 비교입니다.

```text
baseline pretrained CATK
self-forced DMD / SiD
mode-aware nearest OL matching
trajectory-level GAN
set-level GAN final
```

## 반드시 볼 metric

```text
WOSAC realism meta-metric
collision / off-road submetric
endpoint diversity ratio
heading-change histogram distance
stop/go ratio distance
teacher-student set distance
```

핵심 주장은 단순 WOSAC 개선이 아닙니다.
`mode coverage가 좋아지고 realism도 유지된다`를 보여줘야 합니다.
