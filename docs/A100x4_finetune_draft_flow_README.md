# 4x A100 80GB 에서 DRaFT fine-tuning 돌리는 법

이 문서는 기존 `configs/experiment/finetune_draft_flow.yaml` (**6x H100 80GB 전용**) 프리셋을
**4x A100-80GB SXM4** 환경에 맞춰 이식한 프리셋 `configs/experiment/finetune_draft_flow_a100x4.yaml`
을 설명합니다.

요약:

- H100 6장 -> A100 4장으로 옮기면서 **GPU 개수가 2장 줄었고**, 장당 VRAM 은 그대로 80GB 입니다.
- 이 preset 의 `train_batch_size=36` 은 **실제 장비에서 돌려보며 찾은 실측 최대값**입니다.
  bs=38 / 40 / 48 은 `F.scaled_dot_product_attention` 의 A100 (sm_80) kernel 이
  `invalid configuration argument` 로 무작위 step 에서 터집니다 (OOM 아님, kernel grid-dim 한계).
- 또한 **`src/smart/modules/flow_local_decoder.py` 에 소폭 패치**를 적용해
  `ChunkStepRefiner` 의 self-attention 만 math-SDPA kernel 로 고정합니다.
  패치 없이 bs=36 을 쓰면 ~204 step 근처에서 위와 같은 CUDA 에러로 학습이 통째로 죽습니다 (실측).
  패치 적용 후에는 같은 bs=36 에서 500 step 이상 안정적으로 돕니다 (실측). sweep table 과 패치 세부 설명은 4장 / 5장 참고.
- Effective global batch 는 `accumulate_grad_batches=2` 로
  `36 x 4 x 2 = 288` 을 맞춰서 6xH100 preset 의 `48 x 6 x 1 = 288` 과 **정확히 동일**하게 유지합니다.
  -> learning rate 는 원래 값 `2e-4` 를 그대로 씁니다 (linear-scaling 보정 불필요).
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
| `data.train_batch_size` (per GPU) | 48 | **36** (실측 max) |
| `trainer.accumulate_grad_batches` | 1 | **2** |
| effective global batch | 48 x 6 x 1 = **288** | 36 x 4 x 2 = **288** (exact match) |
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

### 2.1 실측 batch-size sweep (4x A100 80GB)

`seed=817 / 42`, `trainer.limit_train_batches=N` (짧은 건 N=10~20, 긴 건 N=80~500), val off,
peak `nvidia-smi` 기준입니다.

| per-GPU `train_batch_size` | 코드 패치 없이 | 코드 패치 적용 후 | peak GPU memory | step time |
|:--|:--|:--|:--|:--|
| 32 | OK (4 step 확인) | 미측정 | ~45 GiB / 80 GiB | ~2.50 s/step |
| **36 (채택)** | **UNSAFE** - 80 step 에선 OK 지만 첫 epoch 의 204 step 근처에서 rank 1 에서 SDPA kernel error | **OK (500 step 확인, 크래시 0)** | ~48 GiB / 80 GiB | **~1.92 s/step** (패치 후) |
| 38 | **UNSAFE** - 20~60 step 안에 rank 1 SDPA kernel error | (미사용) | | |
| 40 | **UNSAFE** - 초반 2~4 step 안에 rank0/rank3 중 하나가 SDPA kernel error | (미사용) | | |
| 48 (H100 원본) | **UNSAFE** - step 1 안에 SDPA kernel error | (미사용) | | |

에러 메시지는 모두

```text
F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
RuntimeError: CUDA error: invalid configuration argument
```

이고, `HierarchicalFlowDecoder.step_refiner` 의 `nn.MultiheadAttention` 호출 시점입니다.
`torch.backends.cuda.enable_flash_sdp(False)` 로 flash kernel 을 꺼봐도 동일하게 터져서,
이 한계는 attention backend 선택 문제가 아니라 **A100 (sm_80) 의 flash / memory-efficient SDPA
커널들이 `batch x num_chunks` 차원이 커지면 grid-dim limit (~65535 블록) 을 못 맞추는 구조적
한계**로 보입니다. 문제 step 의 batch 에 agent 가 특별히 많은 scene 이 섞여 들어가면 발생하므로
**step 수만 채우면 언젠가는 반드시 맞닥뜨리는 확률적 크래시**입니다 (bs=36 기준 epoch 의
~6% 지점에서 한 번은 나옴).

그래서 이 preset 은 `ChunkStepRefiner` 의 self-attention 만
math kernel 로 강제하는 **작은 코드 패치**를 함께 적용합니다 (5장 참고). 이 패치 이후 bs=36 에서
500 step 을 이어 돌려도 크래시가 없고, step time 도 오히려 `~2.40 s -> ~1.92 s` 로 약 20% 빨라집니다
(seq_len 이 5 뿐이라 math kernel 의 O(N^2) 비용이 flash 의 커널 launch overhead 보다 싸서 그렇습니다).

## 3. 왜 이렇게 정했나 (Why)

### 3.1 `train_batch_size=36`

- H100 원본은 `48` 인데, A100 에서는 위 sweep 대로 bs=38 이상이
  `HierarchicalFlowDecoder.step_refiner` 안의 SDPA kernel 에서 CUDA kernel 한계에 걸립니다.
- 메모리 기준으로는 bs=36 에서 peak 48 GiB / 80 GiB 이므로 VRAM 은 꽤 남습니다.
- 문제는 VRAM 이 아니라 **A100 (sm_80) SDPA kernel 의 grid-dim 한계**라는 점이 특이합니다.
- 따라서 "메모리가 많이 남으니 조금 더 키워도 된다" 라고 섣불리 bs 를 올리면,
  학습 중 무작위 스텝에서 CUDA 에러로 학습이 통째로 죽을 수 있습니다.
- `36` 이 실측 기준 가장 공격적인 안전값입니다.

### 3.2 `accumulate_grad_batches=2` + `lr` 유지

- Effective global batch 가 `288 -> 288` 로 **정확히 같습니다** (`36 x 4 x 2 = 48 x 6 x 1`).
- DRaFT fine-tuning 의 loss landscape 이 6xH100 preset 과 동일하므로
  linear scaling rule 으로 lr 을 건드릴 필요가 없습니다. 그대로 `2e-4`.
- 다른 effective-batch 를 원한다면 예를 들어 아래처럼 override 할 수 있습니다.
  - `data.train_batch_size=32 trainer.accumulate_grad_batches=2` (32 x 4 x 2 = 256, lr 을 약간 낮춰도 됨: 1.78e-4)
  - `data.train_batch_size=24 trainer.accumulate_grad_batches=3` (24 x 4 x 3 = 288)
  - `data.train_batch_size=18 trainer.accumulate_grad_batches=4` (18 x 4 x 4 = 288, bs 가 반이라 fragmentation 은 더 안전)

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

**주의**: 이 preset 에서 `train_batch_size=36` 은 임의 고른 값이 아니라 **실측 상한**입니다.
A100 80GB 에는 메모리가 아직 ~30 GiB 남아 있지만,
그 여유로 bs 를 더 키우면 SDPA kernel 이 무작위로 터집니다 (위 2.1 참고).
따라서 **bs 를 키워서 더 땡기려는 시도는 비권장**입니다.

throughput 을 키우고 싶으면 오히려 아래 방향이 안전합니다.

```bash
# 1) gradient accumulation 을 늘려 effective batch 만 키우기 (wall-clock 은 더 걸림)
... data.train_batch_size=36 trainer.accumulate_grad_batches=4   # 36*4*4=576 (큰 batch)

# 2) DRaFT sampler 역전파 깊이를 줄여 step 당 연산 줄이기
... model.model_config.draft.sampling.backprop_last_k=8          # 기본 12 -> 8

# 3) dataloader worker / prefetch 늘리기 (CPU RAM 여유 확인 후)
... data.num_workers=8 data.prefetch_factor=4
```

`val_batch_size=8` 은 closed-loop rollout 16개가 함께 올라가는 상한이라서,
더 키우려면 먼저 `n_rollout_closed_val` 을 같이 줄여야 합니다.

## 7. 이 preset 을 튜닝할 때 참고할 것

- 이 preset 은 **A100x4 YAML preset + ChunkStepRefiner math-SDPA 패치**를 함께 전제로 합니다. 기존
  6xH100 preset (`finetune_draft_flow.yaml`) 은 그대로 남아 있습니다.
- `ddp.yaml` 의 `devices: -1` 설정은 그대로 두고, CLI 에서
  `trainer.devices=4` 로 명시해 CUDA_VISIBLE_DEVICES 와 맞추는 방식을 유지합니다.
- DRaFT 관련 하이퍼파라미터 (`draft.*`) 는 한 글자도 안 건드렸습니다.
  즉 이 preset 은 **6xH100 preset 과 같은 loss / 같은 scheduler** 로 학습됩니다.
- 학습 재개 (`action=fit` + `ckpt_path=.../last.ckpt`) 할 때도 이 preset 을 그대로 쓰면 됩니다.
