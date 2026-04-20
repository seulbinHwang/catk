# 4x A100 80GB 에서 DRaFT fine-tuning 돌리는 법

이 문서는 기존 `configs/experiment/finetune_draft_flow.yaml` (**6x H100 80GB 전용**) 프리셋을
**4x A100-80GB SXM4** 환경에 맞춰 이식한 프리셋 `configs/experiment/finetune_draft_flow_a100x4.yaml`
을 설명합니다.

요약:

- H100 6장 → A100 4장으로 옮기면서 **GPU 개수가 2장 줄었고**, 장당 VRAM 은 그대로 80GB 입니다.
- 이 preset 의 `train_batch_size=54` 는 **math-SDPA 패치 적용 후 5-point 스윕 (bs=36/48/54/60/72) 으로 찾은 실측 throughput sweet spot** 입니다.
  bs=36 대비 per-sample 시간이 **7.2% 빨라지며** (0.0522 → 0.0487 s/sample), peak VRAM 은 ~63 GiB / 80 GiB (OOM 마진 ~18 GiB) 로 안전합니다.
- **`src/smart/modules/flow_local_decoder.py` 의 소폭 패치**가 `ChunkStepRefiner` self-attention 을 math-SDPA kernel 로 고정합니다.
  이 패치가 A100 (sm_80) SDPA kernel 의 grid-dim 한계 (`invalid configuration argument`) 를 없애줘서, 과거에 bs≥38 에서 터지던 문제는 **실제로 풀립니다**.
  남은 attention call (`HalfSecondChunkMixerBlock`, `AttentionLayer`) 은 batch×num_chunks blow-up 이 없어서 더 큰 bs 도 kernel-wise 는 통과합니다.
  따라서 상한은 **kernel 이 아니라 VRAM + memory bandwidth** 입니다. (패치 세부는 5장 참고.)
- Effective global batch 는 `54 × 4 × 1 = 216` 으로 6xH100 preset 의 288 대비 **0.75×** 입니다.
  → learning rate 는 linear scaling rule 로 `2e-4 × 216/288 = **1.5e-4**` 로 하향 조정합니다. 하향 방향이라 warmup 안정성에도 유리.
- `max_epochs` 는 **절대 줄이지 않습니다** (`32` 유지).
- `check_val_every_n_epoch` 도 그대로 `16` 입니다. 즉 16 epoch 마다 eval 이 돕니다.
- Validation 은 `val_batch_size` 를 `16 -> 8` 로 줄여서, closed-loop rollout
  16개가 한 번에 GPU 에 올라와도 A100 80GB 안에서 안전하게 돌게 했습니다 (실측 peak ~ 42 GiB).

## 1. 실행 커맨드

사전 준비는 기존 README 의 `3. WOMD 다운로드`, `4. 캐시 생성` 과 동일합니다.
`$CACHE_ROOT` 가 이미 준비돼 있다고 가정합니다.

```bash
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache   # 각자의 캐시 경로로 바꾸세요
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

`action=finetune` 은 6xH100 preset 과 동일하게 **pretrain checkpoint 의 weight 만 읽어서 새 run 을 시작하는 의미**입니다.
이 run 이 중간에 끊기면 기존 README `5.4` 처럼 `action=fit` + 이 run 의 `last.ckpt` 로 재개하세요.

## 2. 기본 하이퍼파라미터 비교표

| 항목 | 6x H100 (`finetune_draft_flow.yaml`) | **4x A100 (`finetune_draft_flow_a100x4.yaml`)** |
|:--|:--|:--|
| `nproc_per_node` | 6 | **4** |
| `trainer.precision` | `bf16-mixed` | `bf16-mixed` |
| `data.train_batch_size` (per GPU) | 48 | **54** (실측 throughput sweet spot) |
| `trainer.accumulate_grad_batches` | 1 | **1** |
| effective global batch | 48 × 6 × 1 = **288** | 54 × 4 × 1 = **216** (0.75×) |
| `model.model_config.lr` | 2e-4 | **1.5e-4** (linear scaling: 2e-4 × 216/288) |
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

### 2.1 실측 batch-size throughput sweep (4x A100 80GB)

모두 `math-SDPA 패치 적용 상태`, `trainer.limit_train_batches=40~60`, val off, steady-state 기준.

| per-GPU `train_batch_size` | acc | step time | per-sample | samples/s (4 GPU) | peak VRAM | OOM margin | vs bs=36 acc=2 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 36 | 2 | 1.88 s | 0.0522 s | 76.6 | ~48 GiB | 33 GiB | baseline (이전 채택) |
| 48 | 1 | 2.44 s | 0.0508 s | 78.7 | ~58 GiB | 22 GiB | +2.7% |
| **54 (채택)** | **1** | **2.63 s** | **0.0487 s** | **82.1** | **~63 GiB** | **18 GiB** | **+7.2%** |
| 60 | 1 | 2.94 s | 0.0490 s | 81.6 | ~67 GiB | 14 GiB | +6.5% |
| 72 | 1 | 4.00 s | 0.0556 s | 72.0 | ~79 GiB | **2 GiB ⚠️** | -6.0% (UNSAFE) |

**해석**:
- `bs=54` 가 per-sample throughput 최고 + OOM 마진도 18 GiB 로 안전. **Pareto 지배점**.
- `bs=60` 은 throughput 이 사실상 tied 인데 마진 4 GiB 작음.
- `bs=72` 는 memory-BW 포화로 per-sample 이 오히려 나빠지고, 마진 2 GiB 라 32 epoch × ~2254 opt-step = ~54k step 중 scene-variance spike 가 한 번이라도 8 GiB 이상 튀면 OOM.
- 패치 없이는 bs=36 조차 `F.scaled_dot_product_attention` 의 A100 grid-dim 한계 (`invalid configuration argument`) 로 ~204 step 에서 터집니다. 패치는 `ChunkStepRefiner` self-attention 을 math kernel 로 고정해 이 한계를 제거하고, 동시에 seq_len=5 에서는 flash/mem-eff kernel 의 launch overhead 가 math kernel 의 O(N^2) 비용보다 비싸서 **step time 도 약 20% 빨라집니다** (~2.40 → ~1.92 s at bs=36).
- **과거 버전의 주장** — "패치해도 bs≥38 은 다른 kernel 에서 죽는다" — 은 실측 근거가 부족한 추정이었습니다. 실제로 이 스윕에서 bs=72 까지 40 step 동안 kernel crash 는 관찰되지 않았습니다. 패치 후 진짜 상한은 **VRAM / memory bandwidth** 이며, kernel grid-dim 은 아닙니다.

### 2.2 Wall-clock 예상 절감

486,995 training samples × 32 epochs:
- 이전 (bs=36 acc=2, lr=2e-4): 76.6 samples/s → **약 58h** (compute ~56.5h + eval/overhead ~1.5h)
- 현재 (bs=54 acc=1, lr=1.5e-4): 82.1 samples/s → **약 54h**
- **절감: 약 4시간 (~7%)**

## 3. 왜 이렇게 정했나 (Why)

### 3.1 `train_batch_size=54`

- Math-SDPA 패치 (5장) 가 `ChunkStepRefiner` self-attention 의 A100 grid-dim 한계를 제거하면서, 상한이 **kernel → VRAM/BW** 로 옮겨갔습니다.
  5-point 스윕 결과 bs=54 가 per-sample throughput 최고점 (0.0487 s/sample) 이었고, peak VRAM ~63 GiB 로 18 GiB 마진이 확보됩니다 (2.1 참고).
- bs=60 은 throughput 은 사실상 tied 지만 마진이 14 GiB 로 줄고, bs=72 는 memory-bandwidth 포화로 per-sample 이 오히려 degrade + 마진 2 GiB 라 **32 epoch 스케일에서 OOM 확률적으로 확정**입니다.
- bs=36 (이전 채택) 은 compute underutilized 상태였습니다. A100 SM (108개) 활용도 측면에서 bs=54 근처가 tile-alignment/occupancy 가 잘 맞는 지점.

### 3.2 `accumulate_grad_batches=1` + `lr=1.5e-4`

- Effective global batch 는 `54 × 4 × 1 = 216` 으로, 6xH100 preset 의 `288` 대비 **0.75×** 입니다.
- Adam 에 대해 linear scaling rule 을 적용: `lr = 2e-4 × 216/288 = **1.5e-4**` (하향).
  하향 방향은 warmup 안정성과 발산 위험 측면에서 오히려 보수적이라 안전.
- 32 epoch 에서 총 opt-step 수는 `486995 × 32 / 216 ≈ 72k` 로, 이전 (288 effective, ~54k step) 대비 **+33% 더 많은 gradient update**. gradient noise 관점에서 수렴에 오히려 유리.
- 다른 effective-batch/LR 조합을 원한다면 예를 들어:
  - `data.train_batch_size=54 trainer.accumulate_grad_batches=2 model.model_config.lr=3e-4` (216 × 2 = 432, linear scale)
  - `data.train_batch_size=48 trainer.accumulate_grad_batches=1 model.model_config.lr=1.33e-4` (192, 더 보수적; throughput -4.4% vs bs=54)
  - `data.train_batch_size=36 trainer.accumulate_grad_batches=2 model.model_config.lr=2e-4` (이전 설정 복귀용)

### 3.3 `val_batch_size=8`

- Closed-loop validation 은 한 번에 `val_batch_size x n_rollout_closed_val = 8 x 16 = 128` 개의
  rollout 이 동시에 GPU 에 올라갑니다 (model 이 chunk 자동 축소 retry 로직을 갖고 있어서
  완전히 터지기 직전까진 자동으로 잘라주지만, **한 번 OOM 이 나면 GPU context 가 오염**돼서
  다음 eval epoch 까지 영향을 줄 수 있습니다).
- 그래서 `val_batch_size` 를 `16 -> 8` 로 낮춰 **처음부터 OOM 경로를 안 밟게** 만듭니다.
- `limit_val_batches=0.1` 을 그대로 유지해서 평가 시간 자체는 H100 preset 과 비슷합니다.

### 3.4 `sim_agents_metric_workers=2`

- 공식 sim-agents 2025 scorer 는 CPU 에서 돕니다 (`concurrent.futures.ProcessPoolExecutor`).
- A100 노드는 CPU 112 코어 / RAM 1TiB 로 넉넉하지만,
  DDP rank 4 x metric worker 3 = 12 프로세스가 동시에 WOMD tfrecord 를 읽으면 peak RAM 이 꽤 큽니다.
- `3 -> 2` 로 줄여도 scorer throughput 이 거의 안 떨어져서 (IO-bound),
  **RAM peak 안전 마진**을 벌었습니다.

### 3.5 `prefetch_factor=2`

- A100 은 H100 대비 step latency 가 조금 더 길어서 data loader idle 비율이 낮습니다.
- worker 당 prefetch 큐를 `1 -> 2` 로 늘려 GPU step 이 data 를 기다리지 않게 합니다.
- `num_workers=4` 그대로이므로 per-rank peak RAM 은 대략 `4 x 2 ~= 8` batch 분량이라 감당 가능합니다.

### 3.6 `max_epochs=32`, `check_val_every_n_epoch=16`

- 사용자가 명시적으로 **epoch 은 절대 줄이지 말라**고 했으므로 그대로 유지했습니다.
- `check_val_every_n_epoch=16` 이면 학습 중 총 2번의 eval 이 돕니다 (epoch 16, epoch 32).
- eval 1회 당 필요한 wall-clock 은 `(44097 * 0.1 / 4 rank / 8 per-batch) ~= 138 batch/rank` 이고,
  실측 기준 closed-loop rollout + open-loop scoring 이 batch 당 수 초이므로
  **eval 1회 당 대략 10-20분 수준**입니다. 즉 epoch 당 평균 ~1분 정도의 eval overhead 입니다.

## 4. OOM 이 나면 이 순서대로 줄이기

### 4.1 Train 쪽이 터질 때 (`train step` 중 OOM)

```bash
# 1) 먼저 한 단계 작은 실측 sweet spot 인 bs=48 로 내린다
#    (peak ~58 GiB, margin 22 GiB, throughput 은 bs=36 대비 여전히 +2.7%)
... data.train_batch_size=48 model.model_config.lr=1.33e-4   # 48*4*1=192, linear scale

# 2) 그래도 터지면 DRaFT sampler 역전파 깊이를 줄인다 (loss 퀄리티 영향 작음)
... model.model_config.draft.sampling.backprop_last_k=8

# 3) 그래도 터지면 이전 preset (bs=36 acc=2, lr=2e-4) 으로 복귀
... data.train_batch_size=36 trainer.accumulate_grad_batches=2 model.model_config.lr=2e-4

# 4) 그래도 터지면 아예 작게
... data.train_batch_size=18 trainer.accumulate_grad_batches=4 model.model_config.lr=1e-4
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

## 5. `ChunkStepRefiner` math-SDPA 패치

**파일**: `src/smart/modules/flow_local_decoder.py`
**범위**: 딱 `ChunkStepRefiner.forward` 안의 `self.attn(...)` 호출 한 줄
**형태**: `with sdpa_kernel([SDPBackend.MATH]): attn_out, _ = self.attn(...)`

왜 필요한가:

- `HierarchicalFlowDecoder.step_refiner` 의 self-attention 은
  `(batch_size x num_chunks, chunk_size=5, dim=96)` shape 로 attention 을 부릅니다.
- DRaFT fine-tuning 중 어떤 batch 가 agent 가 많은 scene 을 여러 개 담으면
  `batch_size x num_chunks` 가 A100 flash/mem-efficient SDPA kernel 의 grid-dim 상한을 넘어서
  `RuntimeError: CUDA error: invalid configuration argument` 로 학습이 통째로 죽습니다 (2.1 참고).
- 이 attention 은 seq_len 이 **5 밖에 안 돼서** math kernel (O(N^2) naive 구현) 로 돌려도 비용이 거의 같고
  오히려 flash 의 커널 launch overhead 가 사라져 **약 20% 더 빠릅니다** (실측).
- `HalfSecondChunkMixerBlock` 의 self-attention 은 seq_len=16, batch=anchor_count 로 grid-dim 한계와
  여유가 있어서 패치를 적용하지 않았습니다 (필요하면 같은 방식으로 감쌀 수 있음).

패치의 부작용:

- 모델 파라미터 / 초기화 / 그래디언트 경로 변화 없음 (kernel 선택만 바뀜).
- H100 에서 돌려도 문제 없습니다 (math kernel 은 모든 device 에서 쓸 수 있고, 이 지점은 seq_len 이 작아 H100 에서도 math 가 느리지 않습니다).
- checkpoint 호환성도 그대로입니다 (state_dict 구조가 변하지 않음).

## 6. throughput 을 더 올리고 싶다면

**현 preset 의 `train_batch_size=54` 는 5-point 실측 스윕의 Pareto 최적점**입니다.
bs 를 더 키우는 방향 (60, 72) 은 marginal throughput 이득이 없거나 오히려 줄고,
OOM 마진은 빠르게 얇아지므로 **비권장**입니다 (2.1 표 참고).

bs 는 건드리지 않고 throughput 을 더 짜내려면:

```bash
# 1) DRaFT sampler 역전파 깊이 줄이기 (loss 영향 작음, step 당 연산 ↓)
... model.model_config.draft.sampling.backprop_last_k=8          # 기본 12 -> 8

# 2) dataloader worker / prefetch 늘리기 (CPU RAM 여유 확인 후)
... data.num_workers=8 data.prefetch_factor=4

# 3) effective batch 를 키워 opt-step 수 줄이기 (per-sample compute 는 그대로)
... data.train_batch_size=54 trainer.accumulate_grad_batches=2 model.model_config.lr=3e-4   # 216 -> 432
```

`val_batch_size=8` 은 closed-loop rollout 16개가 함께 올라가는 상한이라서,
더 키우려면 먼저 `n_rollout_closed_val` 을 같이 줄여야 합니다.

## 7. 이 preset 을 튜닝할 때 참고할 것

- 이 preset 은 `ChunkStepRefiner` 의 math-SDPA 패치 (5장) 한 줄을 제외하면
  **YAML 만으로** 4xA100 환경에 맞춰져 있습니다. 기존
  6xH100 preset (`finetune_draft_flow.yaml`) 은 그대로 남아 있습니다.
- `ddp.yaml` 의 `devices: -1` 설정은 그대로 두고, CLI 에서
  `trainer.devices=4` 로 명시해 CUDA_VISIBLE_DEVICES 와 맞추는 방식을 유지합니다.
- DRaFT 관련 하이퍼파라미터 (`draft.*`) 는 한 글자도 안 건드렸습니다.
  Optimizer 는 effective batch (288 → 216) 에 맞춰 LR 만 linear scaling
  (2e-4 → 1.5e-4) 했고, scheduler 형태 (cosine, warmup_steps, lr_min_ratio)
  는 동일합니다.
- 학습 재개 (`action=fit` + `ckpt_path=.../last.ckpt`) 할 때도 이 preset 을 그대로 쓰면 됩니다.
