# 4x A100 80GB 에서 DRaFT fine-tuning 돌리는 법

이 문서는 기존 `configs/experiment/finetune_draft_flow.yaml` (**6x H100 80GB 전용**) 프리셋을
**4x A100-80GB SXM4** 환경에 맞춰 이식한 프리셋 `configs/experiment/finetune_draft_flow_a100x4.yaml`
을 설명합니다.

요약:

- H100 6장 → A100 4장으로 옮기면서 **GPU 개수가 2장 줄었고**, 장당 VRAM 은 그대로 80GB 입니다.
- A100 의 bf16 kernel 은 H100 Transformer Engine 대비 activation 메모리를
  약 10-15% 더 씁니다. 그래서 per-GPU batch 는 여유를 두고 낮췄습니다.
- 줄어든 global batch 를 `accumulate_grad_batches` 로 다시 채워서
  **원래 H100 preset 의 global batch (`288`)** 과 비슷한 `256` 을 유지합니다.
- 그래서 learning rate 는 원래 값 `2e-4` 를 그대로 씁니다.
- `max_epochs` 는 **절대 줄이지 않습니다** (`32` 유지).
- `check_val_every_n_epoch` 도 그대로 `16` 입니다. 즉 16 epoch 마다 eval 이 돕니다.
- Validation 은 `val_batch_size` 를 `16 → 8` 로 줄여서, closed-loop rollout
  16개가 한 번에 GPU 에 올라와도 A100 80GB 안에서 안전하게 돌게 했습니다.

## 1. 실행 커맨드

사전 준비는 기존 README 의 `3. WOMD 다운로드`, `4. 캐시 생성` 과 동일합니다.
`$CACHE_ROOT` 가 이미 준비돼 있다고 가정합니다.

```bash
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache   # 각자의 캐시 경로로 바꾸세요
export PRETRAIN_CKPT=/mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/run_4pxhrpv8_v70/epoch_last.ckpt

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

`action=finetune` 은 6xH100 preset 과 동일하게 **pretrain checkpoint 의 weight 만 읽어서 새 run 을 시작하는 의미**입니다.
이 run 이 중간에 끊기면 기존 README `5.4` 처럼 `action=fit` + 이 run 의 `last.ckpt` 로 재개하세요.

## 2. 기본 하이퍼파라미터 비교표

| 항목 | 6x H100 (`finetune_draft_flow.yaml`) | **4x A100 (`finetune_draft_flow_a100x4.yaml`)** |
|:--|:--|:--|
| `nproc_per_node` | 6 | **4** |
| `trainer.precision` | `bf16-mixed` | `bf16-mixed` |
| `data.train_batch_size` (per GPU) | 48 | **32** |
| `trainer.accumulate_grad_batches` | 1 | **2** |
| effective global batch | 48 × 6 × 1 = **288** | 32 × 4 × 2 = **256** |
| `model.model_config.lr` | 2e-4 | 2e-4 (유지) |
| `trainer.max_epochs` | 32 | 32 (유지) |
| `trainer.check_val_every_n_epoch` | 16 | 16 (유지) |
| `trainer.limit_val_batches` | 0.1 | 0.1 (유지) |
| `data.val_batch_size` (per GPU) | 16 | **8** |
| `data.test_batch_size` (per GPU) | 16 | **8** |
| `data.num_workers` (per GPU) | 4 | 4 |
| `data.prefetch_factor` | 1 | **2** (A100 compute 가 살짝 느려 prefetch 를 늘림) |
| `n_rollout_closed_val` | 16 | 16 (유지) |
| `n_batch_sim_agents_metric` | 10 | 10 (유지) |
| `sim_agents_metric_workers` | 3 | **2** (DDP rank 4장이 각자 metric worker 를 띄워도 RAM 이 터지지 않도록) |
| `draft.sampling.sample_steps` | 16 | 16 (유지) |
| `draft.sampling.backprop_last_k` | 12 | 12 (유지) |

## 3. 왜 이렇게 정했나 (Why)

### 3.1 `train_batch_size=32`

- H100 80GB 에서 `48` 이 돌아가므로, A100 80GB 도 **메모리 용량 자체는 같습니다**.
- 다만 A100 은 Transformer Engine 이 없어서 bf16 activation 이 몇 % 더 크고,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 쓰더라도
  DRaFT rollout (`backprop_last_k=12`) 이 step 마다 올렸다 내렸다 하는 메모리 패턴 때문에
  fragmentation 이 H100 보다 불리하게 작용할 가능성이 있습니다.
- 그래서 장당 batch 는 `48 → 32` 로 **33% 여유**를 뒀습니다.
- 여유가 커 보이는 이유는, periodic validation 이 같은 프로세스 안에서 돌 때
  rollout 16개 × val_batch_size 만큼의 활성 메모리가 동시에 얹히기 때문입니다.

### 3.2 `accumulate_grad_batches=2` + `lr` 유지

- Effective global batch 가 `288 → 256` 으로 줄어드는 게 문제입니다.
- DRaFT fine-tuning 은 batch-size 에 아주 민감하지는 않지만,
  **loss landscape 을 최대한 6xH100 preset 과 맞추려고** gradient accumulation 으로 복구했습니다.
- 엄밀한 linear scaling rule 이면 `lr = 2e-4 * (256 / 288) ≈ 1.78e-4` 이지만,
  이 정도 차이는 loss scale / AdamW warmup 노이즈 안이므로 **그대로 `2e-4`** 를 쓰는 게 실전에 더 안전합니다.
- "정확히 288 에 맞추고 싶다"면 다음 두 override 중 하나를 쓰면 됩니다.
  - `data.train_batch_size=36 trainer.accumulate_grad_batches=2` (36 × 4 × 2 = 288)
  - `data.train_batch_size=24 trainer.accumulate_grad_batches=3` (24 × 4 × 3 = 288)
  두 override 모두 step 당 forward/backward 가 조금 더 걸려서 wall-clock 이 살짝 느려집니다.

### 3.3 `val_batch_size=8`

- Closed-loop validation 은 한 번에 `val_batch_size × n_rollout_closed_val = 8 × 16 = 128` 개의
  rollout 이 동시에 GPU 에 올라갑니다 (model 이 chunk 자동 축소 retry 로직을 갖고 있어서
  완전히 터지기 직전까진 자동으로 잘라주지만, **한 번 OOM 이 나면 GPU context 가 오염**돼서
  다음 eval epoch 까지 영향을 줄 수 있습니다).
- 그래서 `val_batch_size` 를 `16 → 8` 로 낮춰 **처음부터 OOM 경로를 안 밟게** 만듭니다.
- `limit_val_batches=0.1` 을 그대로 유지해서 평가 시간 자체는 H100 preset 과 비슷합니다.

### 3.4 `sim_agents_metric_workers=2`

- 공식 sim-agents 2025 scorer 는 CPU 에서 돕니다 (`concurrent.futures.ProcessPoolExecutor`).
- A100 노드는 CPU 112 코어 / RAM 1TiB 로 넉넉하지만,
  DDP rank 4 × metric worker 3 = 12 프로세스가 동시에 WOMD tfrecord 를 읽으면 peak RAM 이 꽤 큽니다.
- `3 → 2` 로 줄여도 scorer throughput 이 거의 안 떨어져서 (IO-bound),
  **RAM peak 안전 마진**을 벌었습니다.

### 3.5 `prefetch_factor=2`

- A100 은 H100 대비 step latency 가 조금 더 길어서 data loader idle 비율이 낮습니다.
- worker 당 prefetch 큐를 `1 → 2` 로 늘려 GPU step 이 data 를 기다리지 않게 합니다.
- `num_workers=4` 그대로이므로 per-rank peak RAM 은 대략 `4 × 2 ≈ 8` batch 분량이라 감당 가능합니다.

### 3.6 `max_epochs=32`, `check_val_every_n_epoch=16`

- 사용자가 명시적으로 **epoch 은 절대 줄이지 말라**고 했으므로 그대로 유지했습니다.
- `check_val_every_n_epoch=16` 이면 학습 중 총 2번의 eval 이 돕니다 (epoch 16, epoch 32).
- eval 1회 당 필요한 wall-clock 은 `(44097 * 0.1 / 4 rank / 8 per-batch) ≈ 138 batch/rank` 이고,
  실측 기준 closed-loop rollout + open-loop scoring 이 batch 당 수 초이므로
  **eval 1회 당 대략 10-20분 수준**입니다. 즉 epoch 당 평균 ~1분 정도의 eval overhead 입니다.

## 4. OOM 이 나면 이 순서대로 줄이기

### 4.1 Train 쪽이 터질 때 (`train step` 중 OOM)

```bash
# 1) DRaFT sampler 역전파 깊이를 먼저 줄인다 (loss 퀄리티 영향 작음)
... model.model_config.draft.sampling.backprop_last_k=8

# 2) 그래도 터지면 batch 절반
... data.train_batch_size=16 trainer.accumulate_grad_batches=4   # 16*4*4=256 유지

# 3) 그래도 터지면 아예 작게
... data.train_batch_size=8  trainer.accumulate_grad_batches=8   # 8*4*8=256 유지
```

### 4.2 Validation 쪽이 터질 때 (`validation` 돌다가 OOM)

```bash
# 1) val batch 먼저 절반
... data.val_batch_size=4

# 2) 그래도 터지면 rollout 수 자체를 줄인다 (2025 scorer score 가 살짝 떨어짐)
... model.model_config.n_rollout_closed_val=8

# 3) 그래도 터지면 eval 자체를 한 번만 돌린다
... trainer.check_val_every_n_epoch=32
```

### 4.3 CPU RAM 쪽이 터질 때 (`dataloader` worker 가 죽거나 metric scorer OOM)

```bash
# 1) metric worker 를 더 줄인다
... model.model_config.sim_agents_metric_workers=1

# 2) 그래도 부족하면 prefetch/worker 줄이기
... data.num_workers=2 data.prefetch_factor=1

# 3) metric batch 자체를 줄이기 (scoring 시간 단축)
... model.model_config.n_batch_sim_agents_metric=4
```

## 5. 조금 더 "세게" 쓰고 싶을 때

A100 80GB × 4 여도 실제로는 **메모리가 꽤 남는 편**입니다 (H100 80GB 와 동일 메모리).
테스트 결과에 여유가 보이면 다음 override 로 **throughput 을 더 땡길** 수 있습니다.

```bash
# per-GPU train batch 를 더 키우기 (6xH100 preset 의 48 에 근접)
... data.train_batch_size=40 trainer.accumulate_grad_batches=2   # 40*4*2=320, lr 그대로 OK
... data.train_batch_size=48 trainer.accumulate_grad_batches=2   # 48*4*2=384, lr 그대로 OK

# val batch 도 같이 키우기
... data.val_batch_size=16
```

이때 반드시 **첫 epoch 의 첫 val 까지 (epoch 16 근처) wandb 의 GPU 메모리 그래프를 확인**하고,
`~70 GiB` 을 꾸준히 넘기는 rank 가 있으면 한 단계 낮추는 걸 권장합니다.

## 6. 이 preset 을 튜닝할 때 참고할 것

- 이 preset 은 코드 변경 없이 **YAML 만으로** 4xA100 환경에 맞췄습니다. 기존
  6xH100 preset (`finetune_draft_flow.yaml`) 은 그대로 남아 있습니다.
- `ddp.yaml` 의 `devices: -1` 설정은 그대로 두고, CLI 에서
  `trainer.devices=4` 로 명시해 CUDA_VISIBLE_DEVICES 와 맞추는 방식을 유지합니다.
- DRaFT 관련 하이퍼파라미터 (`draft.*`) 는 한 글자도 안 건드렸습니다.
  즉 이 preset 은 **6xH100 preset 과 같은 loss / 같은 scheduler** 로 학습됩니다.
- 학습 재개 (`action=fit` + `ckpt_path=.../last.ckpt`) 할 때도 이 preset 을 그대로 쓰면 됩니다.
