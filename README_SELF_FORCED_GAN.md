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
가장 안전한 운영 방식은 한 노드에서 한 번 만든 cache를 모든 학습 노드가 같은 경로로 공유하거나, byte-identical하게 복사해서 쓰는 것입니다. 각 노드가 같은 scene의 teacher cache를 따로 생성하면 seed 설계상 같은 rollout을 의도하더라도 CUDA/library 비결정성 때문에 미세한 차이가 생길 수 있습니다.
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
  --batch-size 32 \
  --rollout-batch-size 32 \
  --storage-dtype float16 \
  --override paths.cache_root="$CACHE_ROOT"
```

이 builder는 pretrained checkpoint를 frozen teacher로 로드한 뒤, 각 scene의 현재 관측
context에서 2초 open-loop trajectory를 32개 샘플링하여 scene별 `.pt` 파일로 저장합니다.
sampling target에는 future GT를 넣지 않고, 저장된 `valid_mask`는 현재 context에서 teacher
sampling을 실제로 수행한 agent만 표시합니다.

cache 생성이 정상 완료되면 root에 `teacher_cache_manifest.json`을 저장합니다. manifest에는
checkpoint W&B artifact/epoch/global step, checkpoint SHA256, repo commit, split, scene 수,
`rollouts_per_scene`, seed, 저장 dtype, `index.json` entry 수가 기록됩니다. 중간에 끊긴
작업은 같은 cache identity를 담은 `teacher_cache_manifest.expected.json`이 있을 때만
`--skip-existing`로 이어받습니다. manifest가 없거나 현재 checkpoint/cache 설정과 다르면
기존 `.pt` 파일을 신뢰하지 않고 다시 생성합니다.

속도 최적화 경로에서는 scene batch와 rollout batch를 함께 사용합니다. 기존의
scene 1개 x rollout 32개 순차 생성과 같은 teacher/cache 의미를 유지하되, 여러 scene의
context encode와 32개 rollout sampling을 GPU batch로 묶어 처리합니다.

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

## hsb-npc-training-1 H100x6 단일 pod 실행

`hsb-npc-training-1`처럼 H100 80GB 6장이 한 pod에 있는 경우는 V100x4x2 멀티노드와 다르게
노드 간 rendezvous와 teacher cache sync가 필요 없습니다. 전용 launcher는 같은
Set-level Self-Forced GAN objective와 같은 pinned pretrain checkpoint를 쓰되, H100에 맞춰
`self_forced_gan_h100_6` preset, `bf16-mixed`, `nproc_per_node=6`, rank당 train microbatch 1을
기본값으로 둡니다. Optimizer step은 `gradient_accumulation=16`으로 수행합니다.

| 항목 | 값 |
|---|---:|
| node/GPU | 1 node x 6 H100 = 6 ranks |
| precision | `bf16-mixed` |
| train microbatch | rank당 1 scene |
| gradient accumulation | 16 |
| micro-step scene batch | 6 scene / micro-step |
| optimizer-step scene batch | 96 scene / optimizer step |
| teacher/student set | K=16 유지 |
| teacher cache | scene당 32 rollout 유지 |
| teacher cache build | 6 GPU shard, pod 내부 merge |
| cache build batch | GPU당 scene batch 32, rollout batch 32 |
| cache build workers | GPU process당 data workers 2, save workers 8 |
| cache build AMP | `float16` |
| validation rollout | 32 |

cache smoke는 32 scene만 만들어 launcher, checkpoint, teacher cache builder, cache validator를
먼저 확인합니다.

```bash
python scripts/launch_self_forced_gan_h100x6_hsb_npc_training_1_static_pod.py \
  --build-teacher-cache \
  --build-cache-only \
  --parallel-teacher-cache \
  --teacher-cache-max-scenes 32 \
  --teacher-cache-batch-size 32 \
  --teacher-cache-rollout-batch-size 32 \
  --teacher-cache-data-num-workers 2 \
  --teacher-cache-save-workers 8 \
  --teacher-cache-amp-dtype float16 \
  --replace
```

cache smoke가 끝난 뒤 같은 smoke cache root로 1 train batch를 통과시킵니다.

```bash
python scripts/launch_self_forced_gan_h100x6_hsb_npc_training_1_static_pod.py \
  --skip-pretrain-download \
  --teacher-cache-max-scenes 32 \
  --limit-train-batches 1 \
  --disable-validation \
  --max-epochs 1 \
  --extra-hydra-overrides "data.shuffle=false data.train_epoch_sample_fraction=1.0" \
  --task-name sf_gan_k16_h100x6_hsb_npc_training_1_smoke \
  --session catk-sf-gan-h100x6-hsb-npc-training-1-smoke \
  --replace
```

전체 학습은 smoke 옵션인 `--teacher-cache-max-scenes`, `--limit-train-batches`,
`--disable-validation`, `--max-epochs 1`, smoke용 task/session override를 제거하고 실행합니다.

```bash
python scripts/launch_self_forced_gan_h100x6_hsb_npc_training_1_static_pod.py \
  --build-teacher-cache \
  --parallel-teacher-cache \
  --replace
```

H100x6 단일 pod에서 1536 scene smoke cache로 측정한 결과, GPU당 batch를 64/96으로 키우거나
bf16 AMP를 쓰는 것보다 scene batch 32, rollout batch 32, data workers 2, save workers 8,
float16 AMP가 가장 빨랐습니다. 이 설정은 checkpoint, seed, rollout 수, scenario index를 바꾸지
않으므로 cache 의미는 유지하고, DataLoader/저장 병렬도만 H100x6 pod에 맞춰 더 씁니다.

Fine-tuning train batch는 full teacher cache와 실제 H100x6 DDP 경로에서 OOM 경계를 probe한
결과 rank당 1 scene으로 고정합니다. Rank당 3 scene은 첫 backward에서 CUDA OOM이 발생했고,
rank당 2 scene도 8-batch stability probe 중 CUDA OOM이 발생했습니다. Rank당 1 scene은
16 train batch stability probe를 status 0으로 통과했고, 해당 probe의
`worst_peak_reserved_pct_epoch_max`는 33.56%였습니다. 따라서 epoch 전체 안정성을 우선하는
기본값은 `train_batch_size=1`입니다.

이 batch 크기에서 optimizer-step effective batch를 키우기 위해 H100x6 launcher와
`self_forced_gan_h100_6` preset은 `trainer.accumulate_grad_batches=16`을 기본값으로 둡니다.
따라서 한 optimizer step은 `1 scene/rank x 6 ranks x 16 accumulation = 96 scenes`입니다.

중지할 때는 pod를 삭제하지 말고 tmux session과 해당 task process만 종료합니다.

```bash
python scripts/launch_self_forced_gan_h100x6_hsb_npc_training_1_static_pod.py --stop
```

## svv + svvv V100x4x2 실행

`svv`, `svvv`는 각각 V100 32GB 4장이라 H100 preset을 그대로 쓰지 않습니다. 전용 preset은
`self_forced_gan_v100x4x2_svv_svvv`이고, 핵심 차이는 다음입니다.

| 항목 | 값 |
|---|---:|
| node/GPU | 2 nodes x 4 V100 = 8 ranks |
| precision | `16-mixed` |
| train microbatch | rank당 1 scene |
| gradient accumulation | manual optimization 기준 12 microbatch |
| effective train scene batch | 96 scene / optimizer step |
| teacher/student set | K=16 유지 |
| teacher cache | scene당 32 rollout 유지 |
| fake rollout backprop | 마지막 8 step |
| discriminator checkpointing | backward 때 discriminator activation 재계산 |
| validation rollout | 8 |

아래 두 메모리 안정화는 `svv + svvv` V100 멀티노드 환경에서도
Set-level Self-Forced GAN Fine-tuning을 돌리기 위해 적용한 특화 대응입니다.
먼저 rank당 train microbatch를 1까지 줄였고, 그 상태에서도 V100 OOM이 발생해서
학습 의미를 바꾸지 않는 범위에서 순간 CUDA memory peak만 낮췄습니다.
`SMARTFlowGAN`은 Lightning manual optimization을 쓰므로, V100 launcher의 기본
gradient accumulation 12는 `trainer.accumulate_grad_batches`가 아니라
`self_forced_gan.manual_accumulate_grad_batches`로 적용합니다.
V100 32GB에서는 DDP `no_sync`가 non-step microbatch의 순간 메모리 피크를 키울 수 있어,
gradient sync는 매 microbatch 유지하고 optimizer step만 12 microbatch마다 수행합니다.
또한 hard scene의 map radius attention peak를 낮추기 위해 V100 preset은
discriminator map query chunk를 `16 -> 1`로 낮춰 계산합니다.
map sender token도 한 번에 전부 보지 않고 `4096`개 단위 streaming softmax로 처리합니다.
K=16 rollout set도 map attention에서는 rollout 1개 단위로 나눠 같은 score를 계산합니다.
agent-agent interaction query chunk는 상대적으로 작아서 `4`를 유지합니다.
GAN-active 구간에서는 discriminator forward activation을 backward 때 재계산해서,
K=16 student adversarial path와 R1/R2 finite-difference regularization의 peak memory를
V100 32GB 안으로 낮춥니다.

| 항목 | svv + svvv 특화 대응 |
|---|---|
| discriminator map/agent attention | map-agent, agent-agent attention을 전체 agent query에 대해 한 번에 만들지 않고 agent query chunk 단위로 나눠 계산합니다. |
| R1/R2 finite-difference regularization | discriminator loss, R1, R2의 큰 graph를 동시에 들고 있지 않고 같은 loss 항을 순차 backward로 반영해 live graph 크기를 줄입니다. |

pretrain 시작점은 W&B model artifact `epoch-last-sqverrgj:v38`로 고정합니다. 이 artifact는
`flow_control_space_pretrain_h100x6_hsb2_wo1_execctx_prefix_balanced_lr6e-4_bs18_oomretry`
run의 epoch 37, global step 171380 checkpoint입니다. launcher 기본값은 더 이상 최신
`epoch-last-*`를 탐색하지 않고, 이미 내려받은 v38 checkpoint marker가 맞으면 local file을
그대로 사용합니다. 그래서 이후 실행에서 새 W&B latest checkpoint 때문에 teacher cache root가
바뀌거나 cache를 다시 만들려고 하지 않습니다.

먼저 teacher cache가 없으면 한 번 생성합니다. `svv + svvv`에서는 8개 V100을 모두 cache
builder shard로 사용하고, 각 pod가 만든 shard를 pod-to-pod direct stream으로 교환한 뒤
양쪽 pod에서 같은 `index.json`을 merge합니다. 학습 시에는 양쪽 pod가 같은 cache root를
읽습니다.

cache builder는 train split sharding에 `ExactDistributedSampler`를 사용합니다. train scene 수가
8개 shard로 나누어떨어지지 않아도 padding duplicate 없이 각 scene을 정확히 한 번만 생성합니다.
launcher는 양방향 shard 교환, 양쪽 pod의 index merge, 양쪽 pod의 validator를 병렬로 실행합니다.
각 GPU builder는 기본 `--teacher-cache-data-num-workers 0`으로 dataset을 직접 읽습니다. 실측상
V100 pod에서는 dataloader worker를 늘리는 것보다 안정적으로 빨랐습니다. 대신 GPU가 다음 batch를
생성하는 동안 per-scene `torch.save`가 background worker에서 진행되도록
`--teacher-cache-save-workers 4`를 기본으로 사용합니다.

launcher는 기본적으로 checkpoint별 subdir를 자동으로 붙입니다. 기본 pinned checkpoint 기준
cache root는 아래처럼 해석됩니다.

```text
$TEACHER_GAN_CACHE_ROOT/epoch-last-sqverrgj_v38_epoch37_gs171380_seed817_k32_fp16
```

다른 checkpoint를 의도적으로 쓰려면 `--wandb-pretrain-artifact <artifact>`를 명시해야 합니다.
기존처럼 W&B run의 최신 `epoch-last-*`를 다시 탐색하고 싶을 때만
`--wandb-pretrain-artifact ""`를 사용합니다.

`--teacher-cache-max-scenes`를 쓰는 smoke cache는 같은 full cache root를 오염시키지 않도록
key 뒤에 `max32` 같은 suffix가 추가됩니다.

build 전에 양쪽 pod에서 `teacher_cache_manifest.json`과 cache validator를 확인합니다.
manifest가 현재 checkpoint/cache identity와 일치하고 validator가 통과하면 cache build를
통째로 skip합니다. 같은 checkpoint cache를 재사용하려면 기존 cache root를 그대로 두고
동일한 launcher command를 다시 실행하면 됩니다.

```bash
python scripts/launch_self_forced_gan_v100x4x2_svv_svvv_static_pods.py \
  --build-teacher-cache \
  --build-cache-only \
  --parallel-teacher-cache \
  --sync-teacher-cache \
  --teacher-cache-batch-size 32 \
  --teacher-cache-rollout-batch-size 32 \
  --teacher-cache-data-num-workers 0 \
  --teacher-cache-save-workers 4 \
  --replace
```

강제로 다시 만들고 싶으면 `--no-reuse-matching-teacher-cache`를 추가합니다. 이전 실행이
중간에 끊겨 같은 manifest identity의 partial cache만 남아 있을 때는
`--skip-existing-teacher-cache`를 추가하면 이미 완료된 scene 파일만 건너뛰고 이어받습니다.
manifest가 맞지 않는 stale cache에서는 이 옵션을 주더라도 기존 파일을 재사용하지 않습니다.

checkpoint별 subdir 자동 생성을 끄고 직접 지정한 cache root를 그대로 쓰려면
`--no-teacher-cache-keyed-root`를 추가합니다. 이 경우에도 manifest identity 검증은 유지됩니다.

최신 측정은 `svv + svvv`의 V100 8장, `batch_size=32`, `rollout_batch_size=32`,
`data_num_workers=0`, `save_workers=4` 기준입니다. 8192 scene cache build는 shard builder
`146.1s`, 양방향 shard 교환 `4.0s`, 양쪽 index merge `3.8s`, 양쪽 validate `3.7s`,
end-to-end `161.92s`가 걸렸습니다. train split 전체 486,995 scene 기준 단순 외삽 예상은
약 `2.67h`입니다. 이전 최신 측정 `4.27h` 대비 약 `1.60x` 빨라졌습니다.

8192 scene cache의 실측 용량은 `1.526GB`입니다. 이를 train split 전체로 단순 외삽하면 약
`90.7GB` / `84.5GiB`이며, scene별 agent 수 분포 차이를 감안해 cache root에는 최소 90GiB
이상의 여유를 두는 것을 권장합니다.

실행 전 smoke는 cache 일부만 만들어 1 train batch를 통과시키는 방식으로 확인합니다.

```bash
python scripts/launch_self_forced_gan_v100x4x2_svv_svvv_static_pods.py \
  --build-teacher-cache \
  --build-cache-only \
  --parallel-teacher-cache \
  --teacher-cache-max-scenes 32 \
  --sync-teacher-cache \
  --teacher-cache-batch-size 32 \
  --teacher-cache-rollout-batch-size 32 \
  --replace
```

cache smoke가 끝난 뒤 1 train batch를 통과시키려면 이미 만든 smoke cache root를 지정해서
학습 smoke를 별도로 실행합니다.

```bash
python scripts/launch_self_forced_gan_v100x4x2_svv_svvv_static_pods.py \
  --skip-pretrain-download \
  --teacher-cache-root "$TEACHER_GAN_CACHE_ROOT" \
  --limit-train-batches 1 \
  --disable-validation \
  --max-epochs 1 \
  --extra-hydra-overrides "data.shuffle=false data.train_epoch_sample_fraction=1.0" \
  --task-name sf_gan_k16_v100x4x2_svv_svvv_smoke \
  --session catk-sf-gan-v100x4x2-svv-svvv-smoke \
  --replace
```

32-scene smoke는 launcher/script wiring과 V100 memory path를 검증하기 위한 최소 실행입니다.
전체 학습은 train split 전체에 대한 teacher cache를 먼저 만든 뒤 실행해야 합니다.

중지할 때는 pod를 삭제하지 말고 tmux session과 해당 task process만 종료합니다.

```bash
python scripts/launch_self_forced_gan_v100x4x2_svv_svvv_static_pods.py \
  --task-name sf_gan_k16_v100x4x2_svv_svvv \
  --session catk-sf-gan-v100x4x2-svv-svvv \
  --stop
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
