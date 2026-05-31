# UniMM Anchor-Based-4s

이 브랜치는 [Revisit Mixture Models for Multi-Agent Simulation: Experimental Study within a Unified Framework](https://arxiv.org/abs/2501.17015)의 **UniMM Anchor-Based-4s** 설정만 구현한다.

기존 CAT-K, SMART next-token prediction, RoaD 기반 fine-tuning을 기본 학습 경로에서 제외하고, 기존 WOMD cache와 WOSAC 평가/submission 유틸리티만 재사용한다.

## 구현 범위

| 항목 | 값 |
| --- | ---: |
| 모델 | anchor-based continuous mixture model |
| anchor 수 | 2048 |
| anchor 생성 | agent category별 training data k-means |
| prediction horizon `Tpred` | 4초, 40 step |
| simulation update interval `tau` | 0.5초, 5 step |
| closed-loop sample | approximate posterior policy |
| posterior planning horizon `Tpost` | 0.5초, 5 step |
| positive matching horizon `Tz*` | 0.5초, 5 step |
| optimizer | AdamW |
| weight decay | 0.0001 |
| base initial lr | 0.0005 |
| H100 x3x2 initial lr | 0.001224744871 |
| schedule | cosine annealing to 0 |
| final lr | 0 |
| warmup | 0 epoch |
| total decay epochs | 64 |
| epochs | 64 |
| base batch size | 32 scenes global |
| H100 x3x2 effective batch size | 192 scenes global |
| 모델 크기 | K=2048 기준 약 4.1M parameters |

논문에서 명시한 핵심은 `2048 anchors + continuous regression + Tpred=4s + closed-loop samples + approximate posterior policy + Tpost=Tz*=0.5s`다. 현재 구현도 이 조합만 대상으로 한다.

## 코드 구조

| 파일 | 역할 |
| --- | --- |
| `src/unimm/anchors.py` | anchor bank 로딩, category별 gather/matching, local/global 변환 |
| `src/unimm/processor.py` | 기존 WOMD cache를 UniMM 학습/rollout 입력으로 변환 |
| `src/unimm/modules.py` | continuous map encoder, factorized agent encoder, anchor-based decoder |
| `src/unimm/losses.py` | classification CE와 Laplace/von Mises NLL |
| `src/unimm/model/anchor_based_4s.py` | Lightning 학습, validation, closed-loop rollout, submission |
| `scripts/build_unimm_anchors.py` | training cache에서 category별 8초 anchor 생성 |
| `configs/model/unimm_anchor_based_4s.yaml` | 모델/하이퍼파라미터 |
| `configs/experiment/unimm_anchor_based_4s.yaml` | 64 epoch 학습 recipe |

## 환경 준비

기존 CAT-K 환경을 그대로 쓴다.

```bash
conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

WOMD는 기존 cache schema를 사용한다. 기본 split 구조는 아래와 같다.

```text
${CACHE_ROOT}/training/*.pkl
${CACHE_ROOT}/validation/*.pkl
${CACHE_ROOT}/testing/*.pkl
```

로컬 helper script는 `CACHE_ROOT`가 없으면 `/media/user/E/dataset/womd_v1_3/SMART_cache` 또는 `/scratch/cache/SMART`를 찾는다. 다른 위치를 쓰면 실행 시 직접 지정한다.

## Anchor 생성

UniMM Anchor-Based-4s는 학습 전에 category별 2048개 8초 anchor bank가 필요하다. 4초 모델은 이 8초 anchor의 앞 4초만 decoder 조건으로 사용하고, positive/posterior matching은 앞 0.5초만 사용한다.
기본 anchor 생성은 논문 설정에 맞게 training cache의 모든 valid agent trajectory를 category별 k-means 입력으로 사용한다. 빠른 실험용으로만 `--max-per-type`에 양수를 지정해 type별 입력 수를 제한할 수 있다.

```bash
CACHE_ROOT=/path/to/SMART_cache \
  bash scripts/build_unimm_anchors.sh
```

기본 출력:

```text
src/unimm/anchors/unimm_anchors_8s_k2048.pkl
```

출력 위치를 바꾸려면:

```bash
CACHE_ROOT=/path/to/SMART_cache \
OUTPUT=/path/to/unimm_anchors_8s_k2048.pkl \
  bash scripts/build_unimm_anchors.sh
```

GPU에서 k-means를 돌리려면:

```bash
CACHE_ROOT=/path/to/SMART_cache \
OUTPUT=/path/to/unimm_anchors_8s_k2048.pkl \
  bash scripts/build_unimm_anchors.sh --device cuda
```

anchor 파일은 pickle dict 형식이다.

```text
anchors:
  veh: [2048, 80, 3]
  ped: [2048, 80, 3]
  cyc: [2048, 80, 3]
posterior_error_threshold:
  veh/ped/cyc별 threshold
```

논문은 posterior plan error threshold의 정확한 값을 공개하지 않는다. 이 구현은 training trajectory의 nearest-anchor 0.5초 error 95% quantile을 category별 threshold로 저장한다.

## 학습

단일 노드 실행:

```bash
CACHE_ROOT=/path/to/SMART_cache \
  bash scripts/train_unimm_anchor_based_4s.sh
```

anchor 파일이 기본 위치가 아니면:

```bash
CACHE_ROOT=/path/to/SMART_cache \
  bash scripts/train_unimm_anchor_based_4s.sh \
  model.model_config.anchor_file=/path/to/unimm_anchors_8s_k2048.pkl
```

8 GPU 학습에서 논문 batch size 32 scenes를 맞추기 위해 기본 experiment는 per-rank batch size 4로 둔다. `svvvv-2-{1..4}`처럼 4 node x 2 GPU이면 global batch가 32가 된다.

## V100 4 Pod 실행

대상 pod:

```text
svvvv-2-1, svvvv-2-2, svvvv-2-3, svvvv-2-4
```

각 pod의 dirty checkout을 건드리지 않도록 launcher는 `/tmp/catk_unimm_v100x4x2`에 clean checkout을 만들고, 공유 anchor는 기본적으로 아래 경로를 쓴다.

```text
/mnt/nuplan/projects/catk/artifacts/unimm/unimm_anchors_8s_k2048.pkl
```

먼저 anchor를 만든다.

```bash
python scripts/launch_unimm_v100x4x2.py --build-anchors --replace
```

anchor build 진행 확인:

```bash
kubectl exec -it -n p-pnc svvvv-2-1 -c main -- tmux attach -t unimm-v100x4x2
```

분산 학습 smoke run:

```bash
python scripts/launch_unimm_v100x4x2.py --smoke --replace
```

전체 학습:

```bash
python scripts/launch_unimm_v100x4x2.py --replace
```

중단:

```bash
python scripts/launch_unimm_v100x4x2.py --stop
```

## H100 2 Pod 실행

대상 pod:

```text
hsb-npc-training-3-1, hsb-npc-training-3-2
```

각 pod는 H100 3장을 쓰므로 전체 world size는 `2 nodes x 3 GPUs = 6`이다. launcher는 기존 shared checkout을 건드리지 않고 각 pod의 `/tmp/catk_unimm_h100x3x2`에 clean checkout을 만든다. 기본 anchor는 `UniMM` 브랜치에 커밋된 파일을 사용한다.

분산 학습 smoke run:

```bash
python scripts/launch_unimm_h100x3x2.py \
  --smoke \
  --smoke-batches 2 \
  --train-batch-size 32 \
  --replace
```

batch size OOM 탐색은 validation을 끈 smoke run으로 작은 값에서 큰 값 순서로 확인한다.

```bash
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 4 --train-batch-size 16 --replace
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 4 --train-batch-size 24 --replace
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 40 --train-batch-size 32 --replace
```

2026-05-31 기준 `hsb-npc-training-3-1`, `hsb-npc-training-3-2`에서 `train_batch_size=32`가 40 batch smoke run을 CUDA OOM 없이 통과했다. 이 값은 per-GPU batch size이고 global batch size는 `32 x 6 = 192`이다.

LR은 기존 기준인 effective batch size 32, lr 0.0005에서 sqrt scaling으로 조정한다.

```text
scaled_lr = 0.0005 * sqrt(192 / 32) = 0.001224744871
```

`scripts/launch_unimm_h100x3x2.py`의 기본 `--learning-rate`는 이 값이며, 실행 시 `model.model_config.lr=0.001224744871` Hydra override로 전달된다.

스케줄은 논문과 같은 cosine annealing to zero를 유지하되, 64 epoch 학습에 맞춰 decay 끝점을 64 epoch 끝으로 둔다. warmup은 논문에 명시되지 않았고 LR도 linear scaling보다 보수적인 sqrt scaling이므로 `lr_warmup_steps=0`을 유지한다. 구현상 `lr_total_steps=${trainer.max_epochs}`라서 H100 x3x2 기본 실행은 `0.001224744871 -> 0`을 64 epoch 동안 no-restart cosine으로 감소시킨다.

전체 학습:

```bash
python scripts/launch_unimm_h100x3x2.py \
  --train-batch-size 32 \
  --replace
```

중단:

```bash
python scripts/launch_unimm_h100x3x2.py --stop
```

## Validation / Test / Submission

validation은 기본적으로 open-loop loss와 closed-loop WOSAC metric을 모두 계산한다.

```bash
CACHE_ROOT=/path/to/SMART_cache \
  bash scripts/train_unimm_anchor_based_4s.sh \
  action=validate \
  ckpt_path=/path/to/checkpoint.ckpt
```

submission 파일을 만들 때는 `sim_agents_submission.is_active=true`로 켜고, Waymo 요구 rollout 수와 metadata를 맞춘다.

```bash
CACHE_ROOT=/path/to/SMART_cache \
  bash scripts/train_unimm_anchor_based_4s.sh \
  action=test \
  ckpt_path=/path/to/checkpoint.ckpt \
  model.model_config.sim_agents_submission.is_active=true \
  model.model_config.sim_agents_submission.authors='[Your Name]' \
  model.model_config.sim_agents_submission.affiliation='Your Affiliation' \
  model.model_config.sim_agents_submission.account_name='Your Account'
```

## 구현상 해석

논문 본문에 없는 세부값은 다음처럼 정했다.

| 항목 | 현재 선택 |
| --- | --- |
| hidden dimension | 176, K=2048 기준 약 4.1M parameters |
| attention heads | 4 heads, head dim 16 |
| activation | ReLU |
| dropout | 0.1 |
| map self-attention | 1 layer |
| factorized attention | 2 layers |
| distance `d(.,.)` | position squared error + heading squared error |
| posterior threshold | category별 nearest-anchor 0.5초 error 95% quantile |
| output distribution | position Laplace, heading von Mises, timestep/coordinate independent |

이 값들은 논문이 공개한 `K=2048`, `Tpred=4s`, `tau=Tpost=Tz*=0.5s`, AdamW, weight decay와 충돌하지 않는 선에서 재현 가능성과 기존 codebase 적합성을 우선해 선택한 값이다. 현재 학습 recipe는 64 epochs이며, LR은 H100 x3x2 effective batch size 192에 맞춰 sqrt scaling을 적용한다. Scheduler는 epoch index 0에서 multiplier 1.0으로 시작하고 epoch index 64에서 multiplier 0.0이 되도록 계산한다.

## 빠른 검증

```bash
python -m compileall -q src/unimm scripts/build_unimm_anchors.py scripts/launch_unimm_v100x4x2.py
python -m pytest tests/test_unimm_anchor_based_4s.py -q
```

실제 cache smoke test는 tiny anchor 파일을 만들어 `trainer.limit_train_batches=1`로 돌리면 된다. 전체 성능 재현은 full training cache로 2048 anchor를 만든 뒤 64 epoch 학습해야 한다.
