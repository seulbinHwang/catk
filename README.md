# MDG-WOSAC

이 브랜치는 WOMD를 사용해 Waymo Sim Agents / WOSAC closed-loop simulation 제출물을 만들기 위한 MDG 구현이다. SMART token prediction, CAT-K, RoaD fine-tuning, planning, guidance 모드는 기본 파이프라인에서 사용하지 않는다.

구현 목표는 [MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments](https://arxiv.org/abs/2511.17496)의 WOSAC closed-loop 부분이다. 논문에서 명확하지 않은 dynamics 세부는 현재 repo의 WOMD cache와 WOSAC 제출 파이프라인에 맞춰 합리적으로 고정했다.

## 구현 범위

- 입력: WOMD agent history 1초(11 step), map polyline, traffic light.
- 출력: 최대 128개 agent의 8초 미래 궤적 32개.
- 학습 target: future trajectory를 acceleration / yaw-rate action으로 바꾼 continuous action tensor.
- denoising: agent-time별 Gaussian mask, `K=5`, alpha schedule `0.99 -> 0.01`.
- inference: full Gaussian noise에서 시작하며, validation/test/submission 기본값은 5-step denoising이다.
- closed-loop: Waymax 없이 repo 내부 rollout으로 1Hz replanning을 수행한다. 매 1초 구간만 history에 반영하고 다시 MDG를 호출해 80 step을 채운다.
- 평가/제출: 기존 Fast WOSAC metric, RMM, WOSAC submission archive 생성 코드를 재사용한다.

## 주요 설정

| 항목 | 값 |
| --- | ---: |
| history steps | 11 |
| future steps | 80 |
| action chunk | 2 |
| reduced action steps | 40 |
| train agents | 64 |
| eval/submission agents | all current-valid cached agents |
| map polylines | 320 |
| waypoints per polyline | 16 |
| traffic lights | 16 |
| hidden dim | 192 |
| modality encoder MLP-Mixer layers | 2 |
| scene encoder layers | 6 |
| denoiser blocks | 2 |
| attention heads | 8 |
| FFN dim | 704 |
| dropout | 0.1 |
| auxiliary modes | 6 |
| relation Fourier bands | 4 |
| auxiliary loss weight | 5 |
| model parameters | 7.11M |
| optimizer | AdamW |
| learning rate | 0.00052915 |
| weight decay | 0.01 |
| LR warmup | 457 steps |
| LR decay | 914 steps마다 0.98 |
| precision | 16-mixed on V100, bf16-mixed on BF16 지원 GPU |
| epochs | 64 |
| grad clip | 1.0 |

MDG의 `[N, Ta, 5]` physical state는 `x, y, cos(heading), sin(heading), speed`로 둔다. raw heading 대신 `cos/sin`을 쓰면 angle wrapping 불연속이 줄어든다. `Ta=40 -> T=80` 복원은 action 하나를 0.1초 step 두 번에 걸쳐 적분한다.

기본 모델 파라미터 수는 `7,111,168`개다. 모듈별로는 scene encoder `4,017,374`, denoiser `2,778,434`, auxiliary predictor `315,360`개다. encoder/denoiser/mixer depth와 attention head 수는 유지하고, `D=192`, `FFN=704`로 폭만 줄인 설정이다.

learning rate는 기존 effective batch `32`에서 쓰던 `0.0002`를 기준으로, testas A100 7장 기본 effective batch `32 * 7 = 224`에 맞춰 sqrt scaling을 적용했다. 계산식은 `0.0002 * sqrt(224 / 32) = 0.00052915`다. LR schedule은 논문 설정인 global batch `32`, `20 epochs`, warmup `1000 steps`, decay interval `2000 steps`를 기준으로 전체 학습 진행률을 보존하도록 조정했다. 우리 설정은 global batch가 7배 크고 epoch이 `20 -> 64`로 `3.2`배 길어졌으므로 step scale은 `(64 / 20) / 7 = 0.457142...`이다. 따라서 warmup은 `1000 * 0.457142 ~= 457 steps`, decay interval은 `2000 * 0.457142 ~= 914 steps`로 둔다. decay factor `0.98`은 유지한다.

## 설치

```bash
conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

WandB를 사용하지 않는 로컬 스모크는 다음처럼 실행한다.

```bash
WANDB_MODE=offline python -m src.run experiment=mdg_pretrain logger=[] callbacks=[]
```

## 데이터 생성

WOMD 원본 TFRecord를 받은 뒤 cache를 만든다. 기존 SMART cache도 fallback으로 읽을 수 있지만, 새로 cache를 만들면 `mdg_map`, `mdg_traffic_signal` 필드가 추가되어 MDG 입력과 더 잘 맞는다. `mdg_traffic_signal`은 WOMD `dynamic_map_states`의 현재 phase와 stop point를 저장한다.

기본 단일 split cache script는 다음처럼 실행한다.

```bash
DATA_SPLIT=validation \
WOMD_INPUT_DIR=/path/to/womd_v1_3/scenario \
CACHE_ROOT=/path/to/MDG_cache \
bash scripts/cache_womd.sh
```

`ssh user@10.60.188.78`의 `/media/user/E/dataset/womd_v1_3/scenario` 원본 TFRecord에서
`/media/user/F/dataset/womd_v1_3/MDG_cache`를 빠르게 만들 때는 전용 병렬 script를 사용한다.
이 script는 `training`, `validation`, `testing`을 동시에 처리하고, validation용
`validation_tfrecords_splitted`도 함께 만든다.

```bash
ssh user@10.60.188.78
cd /media/user/E/projects/catk
git checkout MDG
git pull --ff-only
tmux new-window -t hsb-rl-train -n mdg-cache \
  'bash -lc "cd /media/user/E/projects/catk && bash scripts/cache_mdg_womd_10_60_188_78.sh"'
```

전용 script의 기본값:

| 항목 | 기본값 |
| --- | --- |
| raw WOMD root | `/media/user/E/dataset/womd_v1_3/scenario` |
| cache root | `/media/user/F/dataset/womd_v1_3/MDG_cache` |
| conda env | `catk` |
| log dir | `/media/user/F/dataset/womd_v1_3/MDG_cache/logs` |
| worker 배분 | CPU 개수 기준 training 83%, validation 10%, testing 나머지 |
| 128 logical CPU 기준 | training 106, validation 12, testing 10 |
| 최소 여유 공간 검사 | 300GB |
| 최소 inode 검사 | 1,000,000 |

진행 상황은 아래 로그에서 확인한다.

```bash
tail -f /media/user/F/dataset/womd_v1_3/MDG_cache/logs/supervisor.log
tail -f /media/user/F/dataset/womd_v1_3/MDG_cache/logs/training.log
tail -f /media/user/F/dataset/womd_v1_3/MDG_cache/logs/validation.log
tail -f /media/user/F/dataset/womd_v1_3/MDG_cache/logs/testing.log
```

완료 기준 count는 WOMD v1.3 Sim Agents split 기준으로 다음과 같다.

| 출력 디렉터리 | 기대 파일 수 |
| --- | ---: |
| `training` | 486,995 |
| `validation` | 44,097 |
| `validation_tfrecords_splitted` | 44,097 |
| `testing` | 44,920 |

`/media/user/F` 저장 공간은 script 시작 전에 `df -h`와 `df -ih`로 검사한다. 실제 계측에서는
기존 SMART cache 대비 MDG pickle 크기 비율이 약 1.38배였고, validation split TFRecord까지
포함한 전체 cache 예상 크기는 약 180GB 전후였다. 그래서 `/media/user/F`에 수백 GB 이상
여유가 있으면 충분하지만, 장기 실행 전에는 script의 storage check 결과를 확인한다.

생성된 MDG cache를 Nubes에 업로드하려면 아래 script를 사용한다. 기본 목적지는
`labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache`이고, 기존 실험에서 가장 빨랐던
`-j 96`을 기본값으로 둔다.

```bash
ssh user@10.60.188.78
cd /media/user/E/projects/catk
git checkout MDG
git pull --ff-only
tmux new-window -t hsb-rl-train -n mdg-upload \
  'bash -lc "cd /media/user/E/projects/catk && bash scripts/upload_mdg_cache_to_nubes.sh"'
```

업로드 script 기본값:

| 항목 | 기본값 |
| --- | --- |
| local cache | `/media/user/F/dataset/womd_v1_3/MDG_cache` |
| Nubes remote | `labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache` |
| nubes jobs | 96 |
| Nubes gateway | `c.nubes.sto.navercorp.com:8000` |
| expected files | 620,109 |

업로드 전 local file count를 검증하고, 업로드 후 `nubescli list -R`로 remote file count가
620,109개인지 확인한다. 이미 존재하는 파일은 `nubescli dir-upload -s` 옵션으로 skip한다.
cache root 아래 `logs/` 같은 부가 디렉터리가 있어도 Nubes에는 `training`, `validation`,
`testing`, `validation_tfrecords_splitted` 4개 data split만 업로드한다.

Nubes에 업로드된 MDG cache를 `testas` 파드의 `/workspace/womd_v1_3/MDG_cache`로
다운로드하려면 `ssh user@10.60.188.78`에서 아래 script를 실행한다.

```bash
cd /media/user/E/projects/catk
git checkout MDG
git pull --ff-only
bash scripts/download_mdg_cache_to_testas.sh
```

다운로드 script 기본값:

| 항목 | 기본값 |
| --- | --- |
| pod | `testas` |
| local cache | `/workspace/womd_v1_3/MDG_cache` |
| Nubes remote | `labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/MDG_cache` |
| nubes jobs | 96 |
| expected files | 620,109 |

`download_mdg_cache_to_testas.sh`는 `testas` 내부의 `/mnt/nuplan/projects/catk`를 `MDG`
브랜치로 맞춘 뒤, pod 내부 tmux session `mdg-cache-download`에서
`download_mdg_cache_from_nubes.sh`를 실행한다. 로그는
`/workspace/womd_v1_3/logs/download_mdg_cache_from_nubes_*.log`에 남는다.

기본 config는 cache root 아래 split을 다음처럼 기대한다.

```text
${paths.cache_root}/training
${paths.cache_root}/validation
${paths.cache_root}/validation_tfrecords_splitted
${paths.cache_root}/testing
```

주요 config:

- 데이터: `configs/data/mdg_waymo.yaml`
- 모델: `configs/model/mdg.yaml`
- 학습: `configs/experiment/mdg_pretrain.yaml`
- 제출: `configs/experiment/mdg_wosac_sub.yaml`

## 학습

단일 노드 기본 실행:

```bash
CACHE_ROOT=/workspace/womd_v1_3/MDG_cache bash scripts/train.sh
```

`CACHE_ROOT`를 명시하지 않는 경우에도 MDG 브랜치의 기본 학습/launcher script는
`/workspace/womd_v1_3/MDG_cache`를 기본 cache root로 사용한다.

직접 실행:

```bash
python -m src.run \
  experiment=mdg_pretrain \
  paths.cache_root=/workspace/womd_v1_3/MDG_cache \
  task_name=mdg_wosac_pretrain
```

4노드 x 노드당 V100 2장 실행 예시:

```bash
export NNODES=4
export NPROC_PER_NODE=2
export MASTER_ADDR=<rank0-pod-ip>
export MASTER_PORT=29531
export NODE_RANK=<0|1|2|3>
export CACHE_ROOT=/workspace/womd_v1_3/MDG_cache

torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  -m src.run \
  experiment=mdg_pretrain \
  trainer=ddp \
  trainer.devices="$NPROC_PER_NODE" \
  trainer.num_nodes="$NNODES" \
  paths.cache_root="$CACHE_ROOT" \
  task_name=mdg_wosac_v100x4x2
```

논문 설정은 L40S 8장 기준이고 precision은 bf16이다. V100은 bf16을 지원하지 않으므로 기본 학습/제출 config는 `trainer.precision=16-mixed`로 둔다. L40S/A100/H100처럼 bf16을 지원하는 GPU에서는 `trainer.precision=bf16-mixed`로 바꿔도 된다.
V100에서 메모리가 부족하면 `data.train_batch_size`만 먼저 낮춘다. 학습은 논문 설정처럼 SDC 포함 nearest 64 agents를 쓰지만, validation/test/submission은 Fast WOSAC이 요구하는 `sim_agent_ids`가 누락되지 않도록 cache의 current-valid agents를 모두 유지하고 batch 안에서 agent 축을 동적 padding한다. 모델 구조, `n_rollout_closed_val=32`, `closed_loop_denoising_steps=5`는 제출 검증에서 유지해야 한다.
학습 중 validation은 closed-loop 32 rollout과 5-step denoising을 포함한다. `mdg_pretrain` 기본값은 `trainer.check_val_every_n_epoch=16`, `trainer.limit_val_batches=0.1`, `data.val_batch_size=12`이며, `model.model_config.scorer_scene_num=1680`이 켜져 있으면 GPU 수와 validation batch size에 맞춰 Fast WOSAC scorer batch 수를 자동으로 맞춘다. A100 7장, per-GPU validation batch 12에서는 per-rank `n_batch_sim_agents_metric=20`으로 보정되어 총 1,680개 scenario가 scorer에 들어간다. fit 중에는 checkpoint 점수 계산 시간을 제한하기 위해 validation loop cap도 이 scorer batch 수로 줄인다. 전체 validation/submission은 `mdg_wosac_sub` 또는 별도 validate/test 실행에서 수행한다.

### testas A100 7장 pretrain

`testas` 파드에 cache가 `/workspace/womd_v1_3/MDG_cache`로 준비되어 있으면, `ssh user@10.60.188.78`에서 아래처럼 학습을 시작한다.

```bash
cd /media/user/E/projects/catk
git checkout MDG
git pull --ff-only

TRAIN_BATCH_SIZE=32 \
VAL_BATCH_SIZE=12 \
DATA_NUM_WORKERS=4 \
bash scripts/start_mdg_pretrain_testas_a100x7.sh
```

기본값은 A100 7장 단일 pod, `bf16-mixed`, per-GPU `train_batch_size=32`, global batch `224`, `max_epochs=64`이다. checkpoint monitor는 기존 pretrain과 동일하게 closed-loop metric `val_closed/sim_agents_2025/realism_meta_metric`을 `max` 기준으로 사용한다. 학습 중 validation 기본값은 `VAL_BATCH_SIZE=12`, `LIMIT_VAL_BATCHES=0.1`, `SCORER_SCENE_NUM=1680`, `N_BATCH_SIM_AGENTS_METRIC=10`이다. `SCORER_SCENE_NUM`이 양수이면 코드가 `N_BATCH_SIM_AGENTS_METRIC`을 런타임에 덮어쓴다. A100 7장에서는 `ceil(ceil(1680 / 7) / 12) = 20`이므로 rank마다 20 validation batch까지 Fast WOSAC scorer를 업데이트한다.

CUDA OOM이 나면 자동으로 batch size를 낮춰 같은 `TASK_NAME`의 최신 `epoch_last.ckpt`에서 resume하려면 OOM retry wrapper를 쓴다. 기본값은 `INITIAL_BS=32`, `OOM_STEP=2`, `MIN_BS=24`이며, OOM이 아닌 외부 종료 코드 `134,143`은 batch size를 유지하고 최대 2회 재시도한다.

```bash
cd /media/user/E/projects/catk
git checkout MDG
git pull --ff-only

INITIAL_BS=32 \
OOM_STEP=2 \
MIN_BS=24 \
TASK_NAME=mdg_wosac_pretrain_testas_a100x7_oom_retry_bs32 \
bash scripts/start_mdg_pretrain_testas_a100x7_with_oom_retry.sh
```

이 wrapper는 testas pod를 새로 만들거나 재시작하지 않는다. 각 attempt마다 testas 내부 tmux session `mdg-pretrain-a100x7-oom-retry`를 교체하고, 로그에서 `OutOfMemoryError`, `CUDA out of memory`, `torch.OutOfMemoryError` 등을 감지하면 세션을 멈춘 뒤 `train_batch_size -= OOM_STEP`로 다시 시작한다. retry 로그는 로컬 repo의 `logs/_mdg_testas_a100x7_oom_retry/<TASK_NAME>/attempt_*.log`와 testas 내부 `${PROJECT_ROOT}/logs/testas_mdg_pretrain_a100x7_oom_retry/`에 남는다.

MDG의 multi-step closed-loop denoising은 full-noise mask에서 시작해 clean mask로 내려가는 schedule을 쓴다. 현재 구현은 5-level discrete mask embedding을 사용하므로 `closed_loop_denoising_steps`를 `1..5` 범위로 제한한다. 기본값 `closed_loop_denoising_steps=5`, `num_noise_levels=5`에서는 `[4, 3, 2, 1, 0]` schedule로 모든 discrete noise level을 한 번씩 사용한다. 각 intermediate step은 모델의 clean action estimate를 다음 mask level로 다시 noising한 뒤 다음 denoising call에 넣는다. 논문 WOSAC leaderboard 설정은 one-step closed-loop denoising이므로 strict 재현을 원하면 `closed_loop_denoising_steps=1`로 실행한다.

이 launcher는 기본적으로 `TRAIN_MEMORY_BALANCED_BATCHING=true`를 켠다. 이 기능은 `semi_control_stable`의 memory-balanced batching 개념을 MDG dataloader에 맞춘 것으로, training cache의 agent 수, 현재 valid agent 수, valid agent step 수, map 수 metadata를 만든 뒤 무거운 scene이 한 rank-local batch에 몰리지 않도록 batch sampler가 sample 순서를 재배치한다. metadata cache 기본 위치는 `${CACHE_ROOT}/.catk_metadata/training_mdg_memory_balance_v1.pt`이고, 첫 실행 때만 생성된다.

denoiser는 inter-agent / agent-scene relation이 action timestep `Ta=40` 동안 변하지 않는 점을 이용해 relation embedding을 time-shared로 계산한다. 즉 기존처럼 `[B, N*Ta, S, D]` relation activation을 물리적으로 만들지 않고 `[B, N, S, D]`를 timestep 전체에서 공유한다. temporal attention은 relation bias가 없으므로 PyTorch SDPA 경로를 사용한다. 이 변경은 attention 수식, 학습 objective, 모델 파라미터 수를 바꾸지 않는 실행 최적화다.

testas A100 7장 튜닝 결과는 다음과 같다.

| per-GPU train batch | 결과 |
| ---: | --- |
| 36 | 32 step 통과, peak `78,493 MiB`, 약 `0.86 it/s`; 메모리 여유가 작고 sample/sec도 낮아 운영 기본값에서 제외 |
| 34 | 96 step 통과, peak `74,229 MiB`, 약 `0.95 it/s` |
| 32 | 96 step 통과, peak `70,647 MiB`, 약 `1.01 it/s`; closed-loop checkpoint monitor smoke도 통과 |
| 28 | 48 step 통과, peak `61,321 MiB`, 약 `1.11 it/s` |
| 24 | 48 step 통과, peak `52,963 MiB`, 약 `1.27 it/s` |
| 20 | 48 step 통과, peak `44,417 MiB`, 약 `1.49 it/s` |
| 10 | 48 step 통과, peak `22,955 MiB`, 약 `2.59 it/s` |

MDG pretrain은 agent/map tensor를 각각 `[B, 64, ...]`, `[B, 320, ...]`로 고정 pad하므로, SMART/PyG 계열처럼 scene별 graph 크기 차이가 CUDA activation shape을 크게 바꾸지는 않는다. memory-balanced batching은 무거운 scene/rank 편중을 줄이는 안정화 용도로 유지한다. `bs=34`도 통과하지만 `bs=32`와 예상 epoch time이 거의 같고 메모리 여유가 더 작으므로, 운영 기본값은 안정성과 처리량을 함께 고려해 `TRAIN_BATCH_SIZE=32`로 둔다.

training split `486,995`개 기준 step 수는 `ceil(486995 / 224) = 2,175` step/epoch이다. 최적화 후 memory-balanced `bs=32` 96-step probe에서 약 `1.01 it/s`가 관측되었고, 보수적으로 0.95-1.01 it/s를 잡으면 train step만 기준으로 1 epoch는 약 36-38분이다. 여기에 closed-loop validation 10 batch가 추가되므로 실제 wall-clock은 validation 수행 시간만큼 더 길어진다.

진행 확인:

```bash
kubectl exec -it -n p-pnc testas -c main -- tmux attach -t mdg-pretrain-a100x7
kubectl exec -n p-pnc testas -c main -- bash -lc \
  'tail -f /mnt/nuplan/projects/catk/logs/testas_mdg_pretrain_a100x7/*.log'
```

OOM retry wrapper로 띄운 run은 session/log path가 다르다.

```bash
kubectl exec -it -n p-pnc testas -c main -- tmux attach -t mdg-pretrain-a100x7-oom-retry
tail -f logs/_mdg_testas_a100x7_oom_retry/<TASK_NAME>/attempt_*.log
```

임시 throughput 디버깅 용도로만 closed-loop checkpoint를 끄려면 아래처럼 실행할 수 있다. 정식 pretrain에서는 사용하지 않는다.

```bash
VAL_CLOSED_LOOP=false \
N_BATCH_SIM_AGENTS_METRIC=0 \
CHECKPOINT_MONITOR=val/loss \
CHECKPOINT_MODE=min \
bash scripts/start_mdg_pretrain_testas_a100x7.sh
```

memory-balanced batching을 끄고 비교하려면 아래처럼 실행한다.

```bash
TRAIN_MEMORY_BALANCED_BATCHING=false \
bash scripts/start_mdg_pretrain_testas_a100x7.sh
```

## 검증

checkpoint 검증:

```bash
CKPT_PATH=/path/to/model.ckpt CACHE_ROOT=/workspace/womd_v1_3/MDG_cache bash scripts/local_val.sh
```

Fast WOSAC metric을 직접 켜는 핵심 조건:

```bash
python -m src.run \
  experiment=mdg_pretrain \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  paths.cache_root=/workspace/womd_v1_3/MDG_cache \
  model.model_config.n_rollout_closed_val=32 \
  model.model_config.closed_loop_denoising_steps=5 \
  model.model_config.scorer_scene_num=1680
```

MDG eval/test/submission dataloader는 더 이상 `eval_max_agents` hard cap을 쓰지 않는다. 각 scene의 current-valid cached agents를 모두 유지한 뒤 batch-local max agent 수로 padding하므로, validation TFRecord의 `sim_agent_ids`가 128명을 넘는 scene도 Fast WOSAC metric에 필요한 prediction agent를 누락하지 않는다. 기존 command에 `data.eval_max_agents=...` override가 남아 있어도 deprecated no-op으로 무시된다.

## 제출물 생성

validation split 제출물:

```bash
ACTION=validate \
CKPT_PATH=/path/to/model.ckpt \
CACHE_ROOT=/workspace/womd_v1_3/MDG_cache \
bash scripts/wosac_sub.sh
```

test split 제출물:

```bash
ACTION=test \
CKPT_PATH=/path/to/model.ckpt \
CACHE_ROOT=/workspace/womd_v1_3/MDG_cache \
bash scripts/wosac_sub.sh
```

결과 archive는 run directory 아래 `sim_agents_2025_submission.tar.gz`로 저장된다. multi-node 실행에서는 각 노드의 shard를 rank 0으로 자동 전송해 하나의 archive로 묶는다. 포트 충돌이 있으면 `CATK_SUBMISSION_SHARD_STREAM_PORT`를 다른 값으로 지정한다. `configs/experiment/mdg_wosac_sub.yaml`의 `model.model_config.sim_agents_submission.*` 메타데이터는 실제 제출 계정에 맞게 바꿔야 한다.

## 로컬 스모크 테스트

GPU 없이 pipeline shape과 loss/backward만 확인하려면:

```bash
python -m compileall -q src/mdg src/data_preprocess.py
pytest -q tests/test_mdg_pipeline.py
```

작은 CPU 학습 1 batch:

```bash
python src/run.py action=fit \
  paths.cache_root=/workspace/womd_v1_3/MDG_cache \
  logger=[] callbacks=[] \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  trainer.limit_train_batches=1 trainer.limit_val_batches=0 trainer.max_epochs=1 \
  data.num_workers=0 data.train_batch_size=1 data.val_batch_size=1 \
  model.model_config.val_closed_loop=false \
  model.model_config.backbone.hidden_dim=32 \
  model.model_config.backbone.num_encoder_layers=1 \
  model.model_config.backbone.num_denoiser_blocks=1 \
  model.model_config.backbone.num_heads=4 \
  model.model_config.backbone.ffn_dim=64 \
  model.model_config.backbone.num_mixer_layers=1 \
  model.model_config.backbone.predictor_modes=2
```

Fast WOSAC 1-batch 스모크:

```bash
CATK_TF_INTRA_OP_THREADS=1 CATK_TF_INTER_OP_THREADS=1 python src/run.py action=validate \
  paths.cache_root=/workspace/womd_v1_3/MDG_cache \
  logger=[] callbacks=[] \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  trainer.limit_val_batches=1 \
  data.num_workers=0 data.val_batch_size=1 \
  data.max_map_polylines=16 data.max_traffic_lights=4 \
  model.model_config.n_rollout_closed_val=32 \
  model.model_config.closed_loop_denoising_steps=2 \
  model.model_config.rollout_chunk_size=4 \
  model.model_config.n_batch_sim_agents_metric=1 \
  model.model_config.backbone.hidden_dim=16 \
  model.model_config.backbone.num_encoder_layers=1 \
  model.model_config.backbone.num_denoiser_blocks=1 \
  model.model_config.backbone.num_heads=4 \
  model.model_config.backbone.ffn_dim=32 \
  model.model_config.backbone.num_mixer_layers=1 \
  model.model_config.backbone.predictor_modes=2
```

## 코드 구조

- `src/mdg/data.py`: WOMD cache loader, train nearest-64 구성, eval/test all-agent dynamic padding, DDP exact eval sampler.
- `src/mdg/modules.py`: scene encoder, differentiable kinematic dynamics, MDG denoiser, auxiliary predictor.
- `src/mdg/model.py`: LightningModule, mask/noise objective, 1~5-step closed-loop rollout, Fast WOSAC metric/submission 연결.
- `src/mdg/geometry.py`: 좌표 변환, angle wrapping, relation feature.
- `src/data_preprocess.py`: 새 cache 생성 시 MDG map/signal field 저장.

## 주의사항

- Waymax는 사용하지 않는다. closed-loop 효과는 제출 궤적 내부에서 1Hz replanning으로 근사한다.
- validation/test/submission은 기본적으로 full noise에서 시작해 `closed_loop_denoising_steps=5`로 denoising한다. 기본 schedule은 `[4, 3, 2, 1, 0]`이며, 같은 replanning segment 안에서는 scene encoder를 한 번만 실행하고 auxiliary predictor는 호출하지 않는다.
- WOSAC 제출은 반드시 32 rollout이어야 한다. submission mode에서 다른 값이면 모델 초기화 시 실패한다.
- evaluation/test DDP는 padding 없는 exact sampler를 사용한다. 제출 archive에서 scenario 중복이 생기지 않도록 하기 위함이다.
- 기존 SMART cache는 fallback으로 읽히지만, 논문 설정에 더 가까운 입력을 쓰려면 MDG field가 포함된 cache를 새로 만드는 편이 낫다.
