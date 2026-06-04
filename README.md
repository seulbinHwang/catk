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
| H100 x3x2 initial lr | 0.001185854123 |
| schedule | cosine annealing to 0 |
| final lr | 0 |
| warmup | 4 epochs |
| total decay epochs | 64 |
| epochs | 64 |
| base batch size | 32 scenes global |
| H100 x3x2 effective batch size | 180 scenes global |
| H100 x3x2 validation batch size | 72 scenes global |
| validation interval | every 16 epochs |
| fit-time scorer scenes | 1680 |
| 모델 크기 | K=2048 기준 약 7.1M parameters |

논문에서 명시한 핵심은 `2048 anchors + continuous regression + Tpred=4s + closed-loop samples + approximate posterior policy + Tpost=Tz*=0.5s`다. 현재 구현도 이 조합만 대상으로 한다.
학습 objective는 hard `z*` 하나에 대한 CE/regression이 아니라, GT와 가까운 top-M anchor 후보의 `scorer probability x trajectory likelihood`를 합산하는 mixture NLL이다. Auxiliary CE는 후보 set 전체에 확률 mass를 주도록 돕는 작은 보조항으로만 사용한다.
학습 context는 1초 history 끝인 raw step 10부터 8초 rollout 마지막 decision point인 raw step 85까지 `tau=0.5s` 간격으로 만든다. `Tpred=4s` 전체 GT가 남지 않는 late context는 `[40, 3]` target shape을 유지하되 남은 valid future만 supervision에 쓰고 나머지 timestep은 invalid mask로 제외한다.

## 코드 구조

| 파일 | 역할 |
| --- | --- |
| `src/unimm/anchors.py` | anchor bank 로딩, category별 gather/matching, local/global 변환 |
| `src/unimm/processor.py` | 기존 WOMD cache를 UniMM 학습/rollout 입력으로 변환 |
| `src/unimm/modules.py` | continuous map encoder, factorized agent encoder, anchor-based decoder |
| `src/unimm/losses.py` | top-M mixture NLL, candidate-set auxiliary CE, Laplace/von Mises NLL |
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
기본 anchor 생성은 논문 설정에 맞게 training cache의 모든 valid agent trajectory를 category별 k-means 입력으로 사용한다. 빠른 mini-batch k-means로 초기화한 뒤, 전체 training trajectory를 chunk 단위로 빠짐없이 assignment/update하는 full-data Lloyd refinement를 수행한다. 빠른 실험용으로만 `--max-per-type`에 양수를 지정해 type별 입력 수를 제한할 수 있다.

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

기본 anchor build는 아래 설정을 쓴다.

```text
initialization: mini-batch k-means, 200 iterations, batch size 8192
refinement: full-training-data Lloyd, max 20 sweeps, tol 1e-4
assignment chunks: 2048 trajectory rows x 256 anchors
distance: mean(pos_sq + heading_weight * wrap_angle(heading_diff)^2)
heading_weight: 1.0
empty cluster policy: high-error trajectory로 재초기화
posterior threshold calibration: raw context step 10,15,...,85의 nearest-anchor 0.5초 error 95% quantile
```

이 distance는 학습/closed-loop에서 positive/posterior anchor를 고르는 기본 기준과 동일하게 맞춘 것이다. 다만 full-Lloyd anchor에서는 앞 0.5초가 거의 같은 anchor들이 4초 tail에서 크게 갈라질 수 있다. UniMM Anchor-Based-4s decoder는 선택된 4초 anchor를 continuous regression 조건으로 쓰므로, 학습용 positive `z*`는 `d0.5`가 best와 `0.0001` 이내인 near-tie anchor들 안에서만 `d4s`로 tail tie-break를 적용한다. posterior rollout matching은 실제로 앞 0.5초만 실행하므로 논문 설정대로 순수 `d0.5`를 유지한다. `--lloyd-iters 0`을 주면 refinement를 끌 수 있지만, 논문 재현 목적의 기본 anchor에는 쓰지 않는다.

anchor 파일은 pickle dict 형식이다.

```text
anchors:
  veh: [2048, 80, 3]
  ped: [2048, 80, 3]
  cyc: [2048, 80, 3]
posterior_error_threshold:
  veh/ped/cyc별 threshold
```

논문은 posterior plan error threshold의 정확한 값을 공개하지 않는다. 이 구현은 8초 rollout training 분포와 맞추기 위해 raw context step `10,15,...,85`에서 training trajectory의 nearest-anchor 0.5초 error 95% quantile을 category별 threshold로 저장한다. 기존 anchor 좌표는 유지하고 threshold만 다시 맞추려면 아래 명령을 쓴다.

```bash
python scripts/recompute_unimm_anchor_thresholds.py \
  --train-cache-dir /workspace/womd_v1_3/SMART_cache/training \
  --anchor-file src/unimm/anchors/unimm_anchors_8s_k2048.pkl \
  --output src/unimm/anchors/unimm_anchors_8s_k2048.pkl \
  --device cuda \
  --threshold-start-step 10 \
  --threshold-end-step 85 \
  --threshold-step 5
```

현재 커밋된 anchor file의 posterior threshold는 full training cache 기준 `veh=0.0027350509`, `ped=0.0440671258`, `cyc=0.0091966363`이다. threshold 산정 window 수는 각각 `veh=184,926,339`, `ped=13,986,957`, `cyc=1,376,585`이다.
새 anchor bank를 만들면 component index의 의미가 바뀌므로 기존 checkpoint에서 resume하지 말고 scratch 학습을 시작한다.

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
  --train-batch-size 30 \
  --replace
```

batch size OOM 탐색은 validation을 끈 smoke run으로 작은 값에서 큰 값 순서로 확인한다.

```bash
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 4 --train-batch-size 16 --replace
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 4 --train-batch-size 24 --replace
python scripts/launch_unimm_h100x3x2.py --smoke --smoke-batches 4 --train-batch-size 30 --replace
```

2026-06-03 최신 `UniMM@f64502e` 기준으로 `hsb-npc-training-3-1`, `hsb-npc-training-3-2`에서 memory-balanced train sampler와 positive anchor near-tie tail tie-break를 켠 상태로 batch size를 다시 탐색했다. `train_batch_size=32`는 200-batch train-only smoke에서 CUDA OOM이 재현됐고, `train_batch_size=30`은 200-batch train-only smoke를 exit status 0으로 통과했다. 안정 기본값은 per-GPU `train_batch_size=30`이고 global batch size는 `30 x 6 = 180`이다. Train dataloader는 기본적으로 memory-balanced distributed batch sampler를 사용해 agent/map이 많은 dense scenario가 한 rank-local batch에 몰리는 것을 줄인다. 이 sampler가 이미 rank별 batch를 직접 나누므로 UniMM trainer는 `use_distributed_sampler=false`로 둔다.

H100 x3x2 validation은 per-GPU batch size 12, global batch size `12 x 6 = 72`로 실행한다. 학습 중 validation은 `check_val_every_n_epoch=16` 주기로 돌며, Fast WOSAC `sim_agents_2025` scorer는 `scorer_scene_num=1680` 기준으로 GPU 수와 validation batch size에 맞춰 batch 수를 계산한다. H100 x3x2 기본값에서는 `ceil(ceil(1680 / 6) / 12) = 24`이므로 `n_batch_sim_agents_metric=24`로 자동 조정된다.

LR은 기존 기준인 effective batch size 32, lr 0.0005에서 sqrt scaling으로 조정한다.

```text
scaled_lr = 0.0005 * sqrt(180 / 32) = 0.001185854123
```

`scripts/launch_unimm_h100x3x2.py`의 기본 `--learning-rate`는 이 값이며, 실행 시 `model.model_config.lr=0.001185854123` Hydra override로 전달된다.

스케줄은 cosine annealing to zero를 유지하되, 64 epoch 학습에 맞춰 decay 끝점을 64 epoch 끝으로 둔다. H100 x3x2 기본 실행은 SMART 비교 실험과 맞춰 `lr_warmup_steps=4`를 사용하고, warmup 이후 `0.001185854123 -> 0`을 no-restart cosine으로 감소시킨다.

전체 학습:

```bash
python scripts/launch_unimm_h100x3x2.py \
  --train-batch-size 30 \
  --val-batch-size 12 \
  --replace
```

OOM retry/resume 전체 학습:

```bash
TASK_NAME=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_oom_retry \
INITIAL_BS=30 \
OOM_STEP=2 \
MIN_BS=16 \
  bash scripts/launch_unimm_h100x3x2_with_oom_retry.sh
```

이 래퍼는 첫 시도를 per-GPU `train_batch_size=30`으로 시작한다. 두 pod 중 하나라도 CUDA OOM marker를 로그에 남기면 양쪽 `unimm-h100x3x2` tmux 학습 세션을 종료하고, 같은 `TASK_NAME`의 최신 `epoch_last.ckpt` 또는 `last.ckpt`를 찾아 `train_batch_size -= OOM_STEP`으로 재시작한다. 기본 `OOM_STEP=2`라서 `30 -> 28 -> 26 -> ...` 순서로 낮춘다. 기본적으로 각 attempt의 LR은 현재 batch size에 맞춰 `0.0005 * sqrt((batch_size * 6) / 32)`로 다시 계산한다. `LEARNING_RATE`를 직접 지정하면 모든 attempt에서 그 값을 고정 사용한다. `OOM_STEP=0`이면 batch size를 낮추지 않고 `MAX_SAME_BS_OOM_RETRIES` 횟수만큼 같은 batch size로 재시도한다.

공유 pod에서 다른 사용자의 GPU 작업을 방해하지 않으려면 guarded launcher를 사용한다. 이 스크립트는 양쪽 pod의 GPU compute process, GPU memory, GPU utilization, 동일 tmux session 존재 여부를 먼저 확인하고, 하나라도 바쁘면 학습을 시작하지 않는다.

상태 확인만 할 때:

```bash
bash scripts/start_unimm_h100x3x2_pretrain_if_idle.sh --status
```

파드가 idle일 때만 full pretrain 시작:

```bash
TASK_NAME=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_guarded_$(date +%Y%m%d_%H%M%S) \
SESSION=unimm-h100x3x2 \
MASTER_PORT=29578 \
INITIAL_BS=30 \
OOM_STEP=2 \
MIN_BS=16 \
WANDB_MODE=online \
  bash scripts/start_unimm_h100x3x2_pretrain_if_idle.sh
```

dry-run으로 실제 실행 명령만 확인:

```bash
DRY_RUN=1 bash scripts/start_unimm_h100x3x2_pretrain_if_idle.sh --dry-run
```

`--force`는 현재 GPU 사용자와 확인한 뒤에만 사용한다. 기본 idle 기준은 GPU memory `<=1024MiB`, utilization `<=10%`, compute process 없음이다. 필요하면 `IDLE_MAX_MEMORY_MIB`, `IDLE_MAX_UTIL_PCT`로 조정할 수 있다.

이번 H100 x3x2 guarded full pretrain recipe는 다음 실험이다.

| 항목 | 값 |
| --- | --- |
| 모델 | UniMM Anchor-Based-4s |
| 브랜치 | `UniMM` |
| 실행 action | `fit` |
| 대상 pod | `hsb-npc-training-3-1`, `hsb-npc-training-3-2` |
| GPU 구성 | 2 nodes x 3 H100 = 6 GPUs |
| cache root | `/workspace/womd_v1_3/SMART_cache` |
| anchor file | `src/unimm/anchors/unimm_anchors_8s_k2048.pkl` |
| per-GPU train batch | 30 |
| effective train batch | 30 x 6 = 180 |
| OOM retry | `30 -> 28 -> 26 -> ...`, `OOM_STEP=2`, `MIN_BS=16` |
| LR | `0.001185854123` 기본, 또는 retry wrapper의 sqrt scaling |
| optimizer | AdamW |
| weight decay | 0.0001 |
| LR schedule | warmup 4 epoch + cosine annealing to epoch 64 |
| epochs | 64 |
| precision | `bf16-mixed` |
| gradient clip | 0.5 |
| validation 주기 | `check_val_every_n_epoch=16` |
| val batch | per-GPU 12 |
| Fast WOSAC scorer | `scorer_scene_num=1680`, H100 x3x2 기본 `n_batch_sim_agents_metric=24` |
| closed-loop training | true |
| prediction horizon | 40 steps = 4초 |
| commit interval | 5 steps = 0.5초 |
| positive candidates | top-M=8, 5 steps = 0.5초 matching + near-tie tail tie-break |
| loss | top-M mixture NLL weight 1.0 + candidate-set auxiliary CE weight 0.2 |
| inference temperature | 1.0 |
| inference top-k | 0 |
| inference top-p | 1.0 |
| train sampler | memory-balanced distributed batch sampler |

2026-06-04 21:57 KST에는 최신 `UniMM@a39fcdf` 기준으로 top-M mixture objective full pretrain을 새로 시작했다. 시작 전 `hsb-npc-training-3-1`, `hsb-npc-training-3-2` 모두 UniMM 학습 process가 없었고, GPU memory는 4MiB, utilization은 0%로 idle 상태였다. 두 pod의 `/tmp/catk_unimm_h100x3x2` checkout은 `origin/UniMM@a39fcdf`로 맞춘 뒤 `unimm-h100x3x2` tmux session에서 DDP 6-rank 학습을 시작했다.

```text
running full pretrain:
  task_name=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_topm8_membal_temp1_20260604_215533
  wandb_run_id=t2w2q5e1
  wandb_url=https://wandb.ai/jksg01019-naver-labs/SMART-FLOW/runs/t2w2q5e1
  commit=a39fcdf
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=30, val_batch_size=12
  learning_rate=0.001185854123
  inference_temperature=1.0
  inference_top_k=0
  inference_top_p=1.0
  scorer_scene_num=1680
  n_batch_sim_agents_metric=24
  result=running
```

실행 검증은 launcher 성공 여부만 보지 않고 실제 학습 batch까지 확인했다. DDP 6개 rank가 모두 등록됐고, epoch 0에서 `220` step 이상 OOM/Traceback/NaN 없이 진행됐으며, 같은 시점 GPU memory는 rank-local H100별 약 `76~81GiB`, utilization은 `93~100%`였다. 로그에는 `scorer_scene_num=1680`이 H100 x3x2의 world size와 val batch에 맞춰 `n_batch_sim_agents_metric=24`로 조정됐고, trainable parameter 수는 `7.1M`으로 출력됐다. W&B 기준 `trainer/global_step=19 -> 219` 사이 `train/loss=7.08834 -> 1.86461`, `train/loss_mixture=6.48800 -> 1.54931`, `train/loss_aux_ce=3.00168 -> 1.57647`로 finite하게 감소했다. warm-up 이후 속도는 약 `0.81s/batch`이며, 64 epoch train-only 예상 소요는 약 39~45시간이다. 16 epoch마다 실행되는 closed-loop validation/checkpoint overhead를 포함한 전체 예상 소요는 약 43~52시간이다.

아래 2026-06-04 00:28 KST 실행 기록은 top-M mixture 학습 변경 이전에 시작한 기존 run이라 hard `z*` 기반 loss와 `inference_top_k=256`을 사용했다. 최신 기본 inference setting은 위 표의 `temperature=1.0 / top_k=0 / top_p=1.0`이다.

2026-06-04 00:28 KST에 최신 `UniMM@f3cd9fc` 기준 guarded launcher로 full pretrain을 실제 시작했다. 시작 전 두 pod 모두 compute process 없음, GPU memory 4MiB, utilization 0%로 idle 상태였다. launcher는 `/tmp/catk_unimm_launcher`에서 실행했고, 각 pod의 `/tmp/catk_unimm_h100x3x2` clean checkout을 `origin/UniMM@f3cd9fc`로 맞춘 뒤 `unimm-h100x3x2` tmux session을 생성했다.

```text
running full pretrain:
  task_name=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_tiebreak_membal_temp1_20260604_002845
  wandb_run_id=srjtmkii
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=30, val_batch_size=12
  learning_rate=0.001185854123
  inference_temperature=1.0
  inference_top_k=256
  inference_top_p=1.0
  scorer_scene_num=1680
  result=running
```

실행 검증은 단순 프로세스 확인이 아니라 실제 학습 step까지 확인했다. DDP 6개 rank가 모두 등록됐고, epoch 0에서 `288/2706` batch까지 OOM/Traceback/NaN 없이 진행됐다. 같은 시점 GPU memory는 rank-local H100별 약 `78~81GiB`, utilization은 `93~100%`였다. W&B summary 기준 `trainer/global_step=239`, `train/loss=4.4607`, `train/loss_cls=2.8750`, `train/loss_reg=1.5857`, `train/posterior_accept_rate=0.9077`로 finite loss와 posterior 통계가 정상 로깅됐다. epoch 0 진행 속도 기준 train-only는 약 40.6시간, 16 epoch마다 실행되는 validation/checkpoint overhead를 포함한 전체 예상 소요는 약 43~45시간이다.

### Inference-only Sampling Tuning

UniMM closed-loop inference는 매 0.5초마다 scorer logits에서 anchor component를 sampling한다. 기본값은 `inference_temperature=1.0`, `inference_top_k=0`, `inference_top_p=1.0`이다. `inference_top_k=0`은 top-k truncation을 끄고 2048개 전체 anchor에서 확률 sampling한다는 뜻이다.

2026-06-04에 `hsb-npc-training-1`의 6개 H100에서 `unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_tiebreak_membal_temp1_20260604_002845`의 `Epoch_last.ckpt`를 사용해 inference-only 후보를 검증했다. checkpoint는 `/tmp/unimm_infer_tune_ckpts/epoch_last_h100x3x2_20260604.ckpt`, `epoch=30`, `global_step=89676`이었고, 검증 cache는 `/workspace/womd_v1_3/SMART_cache`다. `scorer_scene_num=1680`, `world_size=6`, `val_batch_size=12`라 실제 scorer batch는 24개로 자동 조정되며, scorer에는 1728 scenarios가 들어간다.

| 후보 | scenes | rollouts | RMM | Kinematic | Interactive | Map-based | minADE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `temp=1.0, top_k=0, top_p=1.0` | 1728 | 32 | 0.76953 | 0.46793 | 0.79907 | 0.90389 | 0.21799 |
| `temp=1.0, top_k=256, top_p=1.0` | 1728 | 32 | 0.76968 | 0.46676 | 0.79989 | 0.90395 | 0.21935 |
| `temp=0.75, top_k=0, top_p=1.0` | 1728 | 32 | **0.77004** | 0.46546 | 0.80092 | **0.90437** | 0.23674 |
| `temp=0.5, top_k=0, top_p=1.0` | 1728 | 32 | 0.76728 | 0.45927 | 0.80012 | 0.90107 | 0.27323 |
| `temp=0.75, top_k=256, top_p=1.0` | 1728 | 32 | 0.76930 | 0.46534 | 0.80065 | 0.90268 | 0.23727 |
| `temp=0.75, top_k=512, top_p=1.0` | 1728 | 32 | 0.76998 | 0.46560 | **0.80120** | 0.90376 | 0.23634 |
| `temp=0.75, top_k=128, top_p=1.0` | 1728 | 32 | 0.76945 | 0.46445 | 0.80085 | 0.90337 | 0.24018 |

위 표는 inference-only sweep 기록이다. 운영 기본값은 논문 의도와 가장 가까운 `temp=1.0, top_k=0, top_p=1.0`로 둔다. Sweep에서는 `temp=0.75, top_k=0`이 가장 높은 RMM을 보였지만, 이 값은 실험 후보로 README에 남겨두고 기본값으로 강제하지 않는다.

2026-06-03 23:35 KST에 guarded launcher와 H100 x3x2 파이프라인을 실제 pod/cache/GPU로 재검증했다. 검증 전후 `--status`에서 두 pod 모두 compute process 없음, GPU memory 4MiB, utilization 0%로 idle 상태였고, 검증용 tmux session은 종료 후 정리했다.

```text
fit smoke:
  task_name=unimm_h100x3x2_script_pipeline_verify_20260603_233529
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=30, val_batch_size=2
  trainer.max_epochs=1
  trainer.limit_train_batches=2
  trainer.limit_val_batches=1
  trainer.check_val_every_n_epoch=1
  model.model_config.n_rollout_closed_val=2
  model.model_config.n_batch_sim_agents_metric=1
  result=exit status 0

checkpoint-sync test smoke:
  task_name=unimm_h100x3x2_script_pipeline_verify_20260603_233529_test
  ckpt_path=/mnt/nuplan/projects/catk/logs/unimm_h100x3x2_script_pipeline_verify_20260603_233529/runs/2026-06-03_23-35-35/checkpoints/epoch_last.ckpt
  synced_ckpt_path=/tmp/unimm_h100x3x2_synced_ckpts/unimm_h100x3x2_script_pipeline_verify_20260603_233529_test_5fb8787bcf68/epoch_last.ckpt
  trainer.limit_test_batches=1
  model.model_config.n_rollout_closed_val=2
  result=exit status 0
```

2026-06-04 21:39 KST에는 top-M mixture objective 변경 후 같은 H100 x3x2 pod와 실제 cache로 다시 smoke를 돌렸다. 로컬 단위 테스트는 `23 passed`였고, 원격 conda에는 `pytest`가 없어 원격에서는 compile과 DDP runtime으로 검증했다.

```text
top-M mixture DDP smoke:
  task_name=unimm_topm8_ddp_smoke_20260604_213934
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=8, val_batch_size=2
  trainer.max_epochs=1
  trainer.limit_train_batches=3
  trainer.limit_val_batches=1
  trainer.check_val_every_n_epoch=1
  positive_top_m=8
  loss=top-M mixture NLL + candidate-set auxiliary CE
  inference_temperature=1.0
  inference_top_k=0
  model.model_config.n_rollout_closed_val=2
  model.model_config.scorer_scene_num=12
  result=exit status 0 on both pods
```

이 smoke에서 6개 DDP rank가 모두 등록됐고, training 3 batch와 closed-loop validation 1 batch가 완료됐다. W&B offline summary에는 `val_open/loss=17.38987`, `val_open/loss_mixture=16.29450`, `val_open/loss_aux_ce=5.47683`, `val_open/aux_mixture_ratio=0.34602`, `val_closed/sim_agents_2025_mean/metametric=0.16552`가 finite로 기록됐다. 검증 종료 후 두 pod 모두 `unimm-topm-smoke` session 없이 GPU memory `4MiB`, utilization `0%`로 복귀했다.

2026-06-04 21:47~21:52 KST에는 같은 commit `66eb5af`를 clean pull한 뒤 실제 H100 x3x2 pod와 실제 cache로 top-M objective를 다시 검증했다. 먼저 train+closed-loop validation 경로를 확인했고, 이후 운영 배치 크기인 per-GPU batch 30에서도 loss가 finite하게 내려가는지 확인했다.

```text
top-M train+validation smoke:
  task_name=unimm_topm8_train5_val1_verify_20260604_214715
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=8, val_batch_size=2
  trainer.limit_train_batches=5
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=2
  model.model_config.scorer_scene_num=12
  result=exit status 0 on both pods
  train/loss=14.57285, train/loss_mixture=13.52017, train/loss_aux_ce=5.26342
  train/loss trend=down, train/loss_mixture trend=down
  train/posterior_accept_rate=0.91106
  val_open/loss=14.10991
  val_closed/sim_agents_2025_mean/metametric=0.19329  # untrained tiny smoke, metric value is not a quality estimate

top-M operating-batch loss smoke:
  task_name=unimm_topm8_bs30_train3_loss_verify_20260604_215137
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=30, effective_train_batch_size=180
  lr=0.001185854123
  trainer.limit_train_batches=3
  result=exit status 0 on both pods
  train/loss trend=8->4->1 normalized sparkline, W&B: █▄▁
  train/loss_mixture trend=8->4->1 normalized sparkline, W&B: █▄▁
  train/loss=17.26965, train/loss_mixture=16.19053, train/loss_aux_ce=5.39556
  train/aux_mixture_ratio=0.33325
  train/posterior_accept_rate=0.89787
```

예전 hard `z*` run `unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_tiebreak_membal_temp1_20260604_002845`와 비교하면, objective가 달라져 loss 절댓값은 직접 비교하면 안 된다. 다만 예전 run도 초반 `train/loss`가 `8.99189 -> 7.73801 -> 6.91717`로 내려갔고 NaN 없이 진행됐으며, posterior accept rate는 `0.90479~0.92092` 범위였다. 새 top-M smoke도 `train/loss`와 `train/loss_mixture`가 내려가고 posterior accept rate가 `0.89787~0.91106`으로 같은 범위에 있어, 짧은 실제-cache/GPU 검증 기준으로는 학습 수렴을 막는 런타임/수치 오류가 보이지 않는다. `loss_aux_ce`는 보조항이라 batch별로 흔들릴 수 있지만 weight가 `0.2`이고 `aux_mixture_ratio`가 약 `0.33~0.39`로 주 loss를 압도하지 않았다.

2026-06-03 23:46 KST에 최신 `UniMM@1dbe124` 기준으로 학습-추론 파이프라인을 코드 레벨과 실제 cache/GPU로 다시 감사했다. 검토 대상은 `src/unimm/model/anchor_based_4s.py`, `src/unimm/processor.py`, `src/unimm/anchors.py`, `src/unimm/losses.py`, `src/unimm/modules.py`, datamodule, memory-balanced sampler, H100 launcher였다.

```text
actual-cache source QA:
  pod=hsb-npc-training-3-1
  cache_root=/workspace/womd_v1_3/SMART_cache
  context_indices=[1,2,...,16]  # raw step 10..85
  target_local_shape=(59,16,40,3)
  target_valid_shape=(59,16,40)
  context85_invalid_suffix_count=0
  z_star_range=0..2039
  posterior_accept_rate=0.93153
  posterior_error_p95=0.00540
  loss=38.36959, loss_cls=7.76818, loss_reg=30.60141
  reg_cls_ratio=3.93933
  bad_grads=[]
  rollout_shapes=(24,2,80,2)/(24,2,80)/(24,2,80)
  result=UNIMM_PIPELINE_QA_OK

DDP train-only optimization smoke:
  task_name=unimm_h100x3x2_code_audit_train20_20260603_234402
  pods=hsb-npc-training-3-1, hsb-npc-training-3-2
  world_size=6, train_batch_size=30
  trainer.max_epochs=1
  trainer.limit_train_batches=20
  trainer.limit_val_batches=0
  result=exit status 0
  final_train_loss=8.96504
  final_train_loss_cls=4.85498
  final_train_loss_reg=4.11006
  final_reg_cls_ratio=0.84657
  final_posterior_accept_rate=0.91873
```

이 감사에서 context 10..85 생성, late-context partial-valid padding, 0.5초 positive matching과 near-tie tail tie-break, posterior fallback, NLL loss finite check, gradient finite check, closed-loop rollout tensor shape, 2-node DDP 초기 최적화 감소 추세를 확인했다. 원격 pod에는 `pytest`가 설치되어 있지 않아 pytest 대신 실제 cache batch를 직접 통과시키는 runtime QA와 20-batch DDP smoke로 검증했다.

짧은 검증 실행은 아래처럼 batch/epoch limit을 걸어 사용한다.

```bash
TASK_NAME=unimm_h100x3x2_oom_retry_smoke \
MAX_EPOCHS=1 \
LIMIT_TRAIN_BATCHES=1 \
LIMIT_VAL_BATCHES=0 \
WANDB_MODE=offline \
EXTRA_HYDRA_OVERRIDES='model.model_config.val_open_loop=false model.model_config.val_closed_loop=false logger.wandb.offline=true logger.wandb.log_model=false' \
  bash scripts/launch_unimm_h100x3x2_with_oom_retry.sh
```

retry wrapper의 로컬 로그는 `logs/_unimm_h100x3x2_oom_retry/<TASK_NAME>/attempt_*.log`에 저장되고, 원격 tmux 로그는 `/mnt/nuplan/projects/catk/logs/tmux_unimm_h100x3x2/<TASK_NAME>/`에 저장된다.

H100 x3x2 launcher는 `--ckpt-path`가 지정되면 기본적으로 master pod의 checkpoint를 `/tmp/unimm_h100x3x2_synced_ckpts/<TASK_NAME>/` 아래로 복사하고, 같은 파일을 모든 pod의 동일 경로에 동기화한 뒤 그 경로를 Hydra `ckpt_path`로 넘긴다. 따라서 `validate`, `test`, OOM retry resume처럼 모든 rank가 checkpoint를 직접 읽어야 하는 실행에서도 rank0 pod에만 checkpoint가 있어서 worker rank가 깨지는 문제를 피한다. 대용량 checkpoint 복사는 `kubectl cp --retries=3`으로 재시도한다. 정말 모든 pod가 같은 shared filesystem path를 보고 있다는 것이 확실할 때만 `--no-sync-ckpt`를 사용한다.

2026-06-01에 `hsb-npc-training-3-1`, `hsb-npc-training-3-2`에서 최신 `UniMM` 코드 기준으로 아래 경로를 검증했다. 검증은 실제 `/workspace/womd_v1_3/SMART_cache`와 committed full-Lloyd anchor file을 사용했고, 모든 실행은 2 nodes x 3 H100 DDP에서 exit status 0으로 끝났다.

```text
fit smoke:
  task_name=unimm_verify_8250_312_20260601
  trainer.max_epochs=1
  trainer.limit_train_batches=2
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=2

explicit validate:
  task_name=unimm_verify_8250_312_validate_20260601
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=2

explicit test:
  task_name=unimm_verify_8250_312_test_20260601
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_test_batches=1
  model.model_config.n_rollout_closed_val=2
```

최신 `c9e6c3e` 기준 전체 학습-추론 smoke 검증은 아래 조건으로 다시 통과했다. 같은 코드에서 `train_batch_size=32`는 backward 중 CUDA OOM이 재현되어 H100 x3x2 기본값에서 제외했다.

```text
fit + fit-time validation:
  task_name=unimm_pipeline_fullcheck_c9e6c3e_bs24_20260601_192513
  2 nodes x 3 H100, train_batch_size=24 per GPU, val_batch_size=12 per GPU
  trainer.max_epochs=1
  trainer.limit_train_batches=2
  trainer.limit_val_batches=1
  trainer.check_val_every_n_epoch=1
  model.model_config.n_rollout_closed_val=4
  result=exit 0, epoch_last.ckpt saved

explicit validate:
  task_name=unimm_pipeline_fullcheck_c9e6c3e_bs24_20260601_192513_validate
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=4
  result=exit 0

explicit test:
  task_name=unimm_pipeline_fullcheck_c9e6c3e_bs24_20260601_192513_test
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_test_batches=1
  result=exit 0

pushed launcher default smoke before batch retune:
  task_name=unimm_default_bs24_lr_verify_20260601_193311
  remote checkout HEAD=9674c66
  default train_batch_size=24 per GPU
  default learning_rate=0.001060660172
  trainer.max_epochs=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=0
  result=exit 0
```

2026-06-01에 memory-balanced sampler를 켜기 전 H100 x3x2 batch size를 탐색했다. `train_batch_size=32`는 2-batch backward에서 CUDA OOM이 재현됐고, `train_batch_size=31`은 2-batch smoke를 통과했지만 full-run 안정 여유가 얇았다. `train_batch_size=30`과 `28`은 full pretrain 시도 중 실제 CUDA OOM이 재현됐고, OOM retry가 2씩 낮춰 `train_batch_size=26`으로 재시작하는 것을 확인했다. 이 기록은 memory-balanced sampler 적용 전 결과이며, 현재 기본값은 위 H100 섹션의 `train_batch_size=30`이다.

```text
OOM boundary probes:
  bs=30, task_name=unimm_bs30_oom_probe_20260601_194225
    trainer.max_epochs=1, trainer.limit_train_batches=2
    result=exit 0 on both pods
  bs=31, task_name=unimm_bs31_oom_probe_20260601_194535
    trainer.max_epochs=1, trainer.limit_train_batches=2
    result=exit 0 on both pods
  bs=32, task_name=unimm_bs32_oom_probe_20260601_194330
    trainer.max_epochs=1, trainer.limit_train_batches=2
    result=CUDA OOM on rank1 during backward
  full-run retry, task_name=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_lloyd_trajsum_posteriorcalib_20260601_195332
    bs=30 -> CUDA OOM during real full-run training
    bs=28 -> CUDA OOM during real full-run training
    bs=26 -> RUNNING beyond 200 optimizer steps on both pods

bs=30 full pipeline smoke before full-run OOM retune:
  task_name=unimm_pipeline_fullcheck_bs30_20260601_194705
  2 nodes x 3 H100, train_batch_size=30 per GPU, val_batch_size=12 per GPU
  learning_rate=0.001185854123
  trainer.max_epochs=1
  trainer.limit_train_batches=20
  trainer.limit_val_batches=1
  trainer.check_val_every_n_epoch=1
  model.model_config.n_rollout_closed_val=4
  result=exit 0, epoch_last.ckpt saved

explicit validate:
  task_name=unimm_pipeline_fullcheck_bs30_20260601_194705_validate
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=4
  result=exit 0

explicit test:
  task_name=unimm_pipeline_fullcheck_bs30_20260601_194705_test
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_test_batches=1
  result=exit 0
```

2026-06-01에 `last_train_context_step=85`와 late-context partial valid future padding 변경도 같은 H100 x3x2 pod에서 검증했다.

```text
direct processor/model checks on both pods:
  context_indices = [1, ..., 16]  # raw step 10,15,...,85
  target_local shape = [3, 16, 40, 3]
  target_valid shape = [3, 16, 40]
  final context raw step 85 has only first 5 target steps valid
  synthetic training_step and 80-step closed-loop rollout both finite

DDP fit smoke:
  task_name=unimm_context85_ddp_smoke_20260601
  2 nodes x 3 H100, train_batch_size=32 per GPU  # historical one-batch smoke
  trainer.max_epochs=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=0
  result=exit 0, no CUDA OOM

DDP fit + validation smoke:
  task_name=unimm_context85_val_smoke_20260601
  2 nodes x 3 H100, train_batch_size=8 per GPU, val_batch_size=4 per GPU
  trainer.max_epochs=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=1
  model.model_config.n_rollout_closed_val=2
  result=exit 0, val_open and val_closed/sim_agents_2025 metrics produced
```

2026-06-01에 posterior threshold 계측과 context-distribution threshold calibration도 같은 H100 x3x2 pod에서 검증했다. 검증 checkout은 `/tmp/catk_unimm_posterior_verify`를 사용했고, 실제 `/workspace/womd_v1_3/SMART_cache/training`과 committed anchor file을 사용했다.

```text
threshold recompute:
  context steps = [10, 15, ..., 85]
  quantile = 0.95
  distance = mean(pos_sq + heading_weight * wrap_angle(heading_diff)^2)
  result thresholds:
    veh = 0.0027350509 from 184,926,339 windows
    ped = 0.0440671258 from 13,986,957 windows
    cyc = 0.0091966363 from 1,376,585 windows

direct processor/model checks on both pods:
  posterior_stats contains accept_rate, error_mean, error_p50/p90/p95,
  error_over_threshold, type accept rates, and context accept rates
  context accept-rate diagnostic steps = raw step 10,15,...,80
  result=all direct checks passed

DDP fit smoke:
  task_name=unimm_posterior_threshold_ddp_smoke_20260601
  2 nodes x 3 H100, train_batch_size=8 per GPU
  trainer.max_epochs=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=0
  result=exit 0, no CUDA OOM
  logged keys include train/posterior_accept_rate,
  train/posterior_error_mean, train/posterior_error_p50/p90/p95,
  train/posterior_error_over_threshold, type rates,
  and train/posterior_accept_rate_ctx_10...ctx_80
```

2026-06-02에 regression objective를 per-valid-timestep NLL mean으로 되돌린 뒤 H100 x3x2에서 다시 검증했다. 검증은 `origin/UniMM@103df9e` clean checkout, 실제 `/workspace/womd_v1_3/SMART_cache`, committed full-Lloyd anchor file, `train_batch_size=26`, `val_batch_size=12`, `learning_rate=0.001103970108` 조건으로 실행했다.

```text
fit + fit-time validation:
  task_name=unimm_perstep_objective_verify2_20260602
  2 nodes x 3 H100, train_batch_size=26 per GPU, val_batch_size=12 per GPU
  trainer.max_epochs=1
  trainer.limit_train_batches=20
  trainer.limit_val_batches=1
  trainer.check_val_every_n_epoch=1
  model.model_config.n_rollout_closed_val=4
  result=exit 0
  train/loss_cls=3.71625
  train/loss_reg=32.08650
  train/reg_cls_ratio=8.63410
  val_open/reg_cls_ratio=6.39625
```

2026-06-03에 full-Lloyd anchor의 positive matching을 H100 pod에서 실제 training cache로 진단했다. `Tz*=0.5s` primary distance는 유지하되 4초 anchor tail tie-break를 넣으면, 같은 6804개 agent/context row에서 선택 anchor의 4초 tail ADE가 `26.06m -> 0.385m`로 내려갔다. 차량 row만 보면 `27.51m -> 0.39m`이다. 이 검증은 WOSAC metric이 아니라 학습 label 생성 QA이며, posterior rollout matching은 여전히 순수 0.5초 기준이다.

from-scratch 학습-추론 파이프라인을 짧게 재검증하려면 아래 순서로 돌린다. 이 검증은 안정 기본값인 `train_batch_size=30`, `val_batch_size=12`, H100 x3x2 DDP를 사용하되 batch/epoch 수만 줄여서 학습, 학습 중 validation, checkpoint 저장, explicit validation, test inference를 모두 통과시키는 용도다.

```bash
TASK_NAME=unimm_pipeline_fullcheck_fromscratch_$(date +%Y%m%d_%H%M%S)

python scripts/launch_unimm_h100x3x2.py \
  --action fit \
  --task-name "$TASK_NAME" \
  --session unimm-pipeline-fullcheck \
  --master-port 29575 \
  --train-batch-size 30 \
  --learning-rate 0.001185854123 \
  --val-batch-size 12 \
  --wandb-mode offline \
  --max-epochs 1 \
  --limit-train-batches 2 \
  --limit-val-batches 1 \
  --extra-hydra-overrides 'trainer.check_val_every_n_epoch=1 model.model_config.n_rollout_closed_val=4 logger.wandb.offline=true logger.wandb.log_model=false' \
  --replace

CKPT=$(
  kubectl exec hsb-npc-training-3-1 -- bash -lc \
    "ls -t /mnt/nuplan/projects/catk/logs/${TASK_NAME}/runs/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1" |
  tail -n 1 | tr -d '\r'
)

python scripts/launch_unimm_h100x3x2.py \
  --action validate \
  --task-name "${TASK_NAME}_validate" \
  --session unimm-pipeline-fullcheck-validate \
  --master-port 29576 \
  --ckpt-path "$CKPT" \
  --val-batch-size 12 \
  --wandb-mode offline \
  --limit-val-batches 1 \
  --extra-hydra-overrides 'model.model_config.n_rollout_closed_val=4 logger.wandb.offline=true logger.wandb.log_model=false' \
  --replace

python scripts/launch_unimm_h100x3x2.py \
  --action test \
  --task-name "${TASK_NAME}_test" \
  --session unimm-pipeline-fullcheck-test \
  --master-port 29577 \
  --ckpt-path "$CKPT" \
  --test-batch-size 4 \
  --wandb-mode offline \
  --extra-hydra-overrides 'trainer.limit_test_batches=1 model.model_config.n_rollout_closed_val=4 logger.wandb.offline=true logger.wandb.log_model=false' \
  --replace
```

2026-06-02에 `origin/UniMM@4155d12` clean checkout, 실제 `/workspace/womd_v1_3/SMART_cache`, committed full-Lloyd anchor file 조건으로 bs30 memory-balanced fullcheck를 통과했다.

```text
fit + fit-time validation:
  task_name=unimm_pipeline_fullcheck_bs30_membal_clean_20260602_130001
  train_batch_size=30 per GPU, global batch=180
  learning_rate=0.001185854123
  trainer.limit_train_batches=20
  trainer.limit_val_batches=1
  result=exit 0

explicit validate:
  task_name=unimm_pipeline_fullcheck_bs30_membal_clean_20260602_130001_validate_retry
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_val_batches=1
  result=exit 0

explicit test:
  task_name=unimm_pipeline_fullcheck_bs30_membal_clean_20260602_130001_test_retry
  ckpt_path=<fit smoke epoch_last.ckpt, synced to both pods>
  trainer.limit_test_batches=1
  result=exit 0
```

실전 full pretrain은 아래처럼 OOM retry wrapper로 시작한다. 처음부터 학습할 때는 `CKPT_PATH`를 지정하지 않는다. wrapper는 CUDA OOM 또는 재시도 가능한 종료가 발생했을 때만 같은 `TASK_NAME`의 최신 checkpoint를 찾아 resume한다.

```bash
TASK_NAME=unimm_anchor_based_4s_h100x3x2_pretrain_globalbs180_topm8_temp1_$(date +%Y%m%d_%H%M%S) \
SESSION=unimm-h100x3x2 \
MASTER_PORT=29578 \
INITIAL_BS=30 \
OOM_STEP=2 \
MIN_BS=16 \
WANDB_MODE=online \
EXTRA_HYDRA_OVERRIDES="model.model_config.inference_temperature=1.0 model.model_config.inference_top_k=0 model.model_config.scorer_scene_num=1680" \
  bash scripts/launch_unimm_h100x3x2_with_oom_retry.sh
```

중단:

```bash
python scripts/launch_unimm_h100x3x2.py \
  --stop \
  --task-name "$TASK_NAME" \
  --session unimm-h100x3x2
```

## Validation / Test / Submission

validation은 기본적으로 open-loop loss와 closed-loop WOSAC metric을 모두 계산한다.

```bash
python scripts/launch_unimm_h100x3x2.py \
  --action validate \
  --ckpt-path /path/to/checkpoint.ckpt \
  --val-batch-size 12 \
  --replace
```

submission 파일을 만들 때는 `sim_agents_submission.is_active=true`로 켜고, Waymo 요구 rollout 수와 metadata를 맞춘다.

```bash
python scripts/launch_unimm_h100x3x2.py \
  --action test \
  --ckpt-path /path/to/checkpoint.ckpt \
  --extra-hydra-overrides "model.model_config.sim_agents_submission.is_active=true model.model_config.sim_agents_submission.authors='[Your Name]' model.model_config.sim_agents_submission.affiliation='Your Affiliation' model.model_config.sim_agents_submission.account_name='Your Account'" \
  --replace
```

## 구현상 해석

논문 본문에 없는 세부값은 다음처럼 정했다.

| 항목 | 현재 선택 |
| --- | --- |
| hidden dimension | 232, K=2048 기준 약 7.1M parameters |
| attention heads | 4 heads, head dim 20 |
| activation | ReLU |
| dropout | 0.1 |
| map self-attention | 1 layer |
| factorized attention | 2 layers |
| distance `d(.,.)` | `mean(pos_sq + heading_weight * wrap_angle(heading_diff)^2)` |
| positive candidates | `Tz*=0.5s` 기준 top-M=8 후보. 첫 후보는 `d0.5`가 best와 `0.0001` 이내인 near-tie anchor들 안에서만 `d4s`로 안정화 |
| posterior threshold | category별 raw context `10,15,...,85` nearest-anchor 0.5초 error 95% quantile |
| train context starts | raw step `10,15,...,85`; late context는 남은 valid future만 supervision |
| output distribution | position Laplace, heading von Mises, timestep/coordinate independent |
| heading concentration cap | `max_von_mises_concentration=100.0`; bf16/von Mises NLL의 과확신 NaN을 막는 수치 안정화 |
| training objective | top-M mixture NLL over candidate anchors + candidate-set auxiliary CE |
| trajectory NLL reduction | candidate별 valid timestep NLL mean |
| loss weights | `mixture=1.0`, `aux_ce=0.2` |
| inference sampling | `temperature=1.0`, `top_k=0`, `top_p=1.0`; 0.5초마다 2048개 전체 anchor에서 확률 sampling |

이 값들은 논문이 공개한 `K=2048`, `Tpred=4s`, `tau=Tpost=Tz*=0.5s`, AdamW, weight decay와 충돌하지 않는 선에서 재현 가능성과 기존 codebase 적합성을 우선해 선택한 값이다. 현재 학습 recipe는 64 epochs이며, LR은 H100 x3x2 effective batch size 180에 맞춰 sqrt scaling을 적용한다. Scheduler는 4 epoch linear warmup 뒤 epoch index 64에서 multiplier 0.0이 되도록 계산한다. 학습 중 validation은 비용을 줄이기 위해 16 epoch마다 실행하고, `scorer_scene_num=1680`으로 world size와 validation batch size가 바뀌어도 scorer 대상 scene 수가 같은 규모가 되도록 맞춘다.
Loss 로그에서 `loss_mixture`는 top-M 후보들의 scorer probability와 trajectory likelihood를 합친 주 loss이고, `loss_aux_ce`는 후보 set 전체에 scorer 확률 mass를 주는 보조 loss다. `top_m_error`와 `top_m_nll`은 후보 set 품질과 decoder 보정 품질을 확인하기 위한 진단값이다. objective가 바뀌었으므로 기존 hard `z*` checkpoint에서 resume하지 말고 scratch pretrain을 사용한다.
closed-loop posterior 품질 진단을 위해 학습 중 `train/posterior_accept_rate`, `train/posterior_error_mean`, `train/posterior_error_p50`, `train/posterior_error_p90`, `train/posterior_error_p95`, `train/posterior_error_over_threshold`, type별 accept rate, context-step별 accept rate를 로깅한다.

## 빠른 검증

```bash
python -m compileall -q src/unimm scripts/build_unimm_anchors.py scripts/recompute_unimm_anchor_thresholds.py scripts/launch_unimm_v100x4x2.py
python -m pytest tests/test_unimm_anchor_based_4s.py -q
```

실제 cache smoke test는 tiny anchor 파일을 만들어 `trainer.limit_train_batches=1`로 돌리면 된다. 전체 성능 재현은 full training cache로 2048 anchor를 만든 뒤 64 epoch 학습해야 한다.
