# MDG-WOSAC

이 브랜치는 WOMD를 사용해 Waymo Sim Agents / WOSAC closed-loop simulation 제출물을 만들기 위한 MDG 구현이다. SMART token prediction, CAT-K, RoaD fine-tuning, planning, guidance 모드는 기본 파이프라인에서 사용하지 않는다.

구현 목표는 [MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments](https://arxiv.org/abs/2511.17496)의 WOSAC closed-loop 부분이다. 논문에서 명확하지 않은 dynamics 세부는 현재 repo의 WOMD cache와 WOSAC 제출 파이프라인에 맞춰 합리적으로 고정했다.

## 구현 범위

- 입력: WOMD agent history 1초(11 step), map polyline, traffic light.
- 출력: 최대 128개 agent의 8초 미래 궤적 32개.
- 학습 target: future trajectory를 acceleration / yaw-rate action으로 바꾼 continuous action tensor.
- denoising: agent-time별 Gaussian mask, `K=5`, alpha schedule `0.99 -> 0.01`.
- inference: full Gaussian noise에서 1-step denoising.
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
| eval/submission agents | 128 |
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
| learning rate | 0.0002 |
| weight decay | 0.01 |
| LR warmup | 1000 steps |
| LR decay | 2000 steps마다 0.98 |
| precision | 16-mixed on V100, bf16-mixed on BF16 지원 GPU |
| epochs | 20 |
| grad clip | 1.0 |

MDG의 `[N, Ta, 5]` physical state는 `x, y, cos(heading), sin(heading), speed`로 둔다. raw heading 대신 `cos/sin`을 쓰면 angle wrapping 불연속이 줄어든다. `Ta=40 -> T=80` 복원은 action 하나를 0.1초 step 두 번에 걸쳐 적분한다.

기본 모델 파라미터 수는 `7,111,168`개다. 모듈별로는 scene encoder `4,017,374`, denoiser `2,778,434`, auxiliary predictor `315,360`개다. encoder/denoiser/mixer depth와 attention head 수는 유지하고, `D=192`, `FFN=704`로 폭만 줄인 설정이다.

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

```bash
bash scripts/cache_womd.sh
```

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
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache bash scripts/train.sh
```

직접 실행:

```bash
python -m src.run \
  experiment=mdg_pretrain \
  paths.cache_root=/workspace/womd_v1_3/SMART_cache \
  task_name=mdg_wosac_pretrain
```

4노드 x 노드당 V100 2장 실행 예시:

```bash
export NNODES=4
export NPROC_PER_NODE=2
export MASTER_ADDR=<rank0-pod-ip>
export MASTER_PORT=29531
export NODE_RANK=<0|1|2|3>
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache

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
V100에서 메모리가 부족하면 `data.train_batch_size`만 먼저 낮춘다. 모델 구조나 `eval_max_agents=128`, `n_rollout_closed_val=32`, `closed_loop_denoising_steps=16`은 제출 검증에서 유지해야 한다.
학습 중 validation은 closed-loop 32 rollout과 16-step denoising을 포함하므로 `mdg_pretrain` 기본값은 `trainer.limit_val_batches=10`으로 제한한다. 전체 validation/submission은 `mdg_wosac_sub` 또는 별도 validate/test 실행에서 수행한다.

## 검증

checkpoint 검증:

```bash
CKPT_PATH=/path/to/model.ckpt CACHE_ROOT=/workspace/womd_v1_3/SMART_cache bash scripts/local_val.sh
```

Fast WOSAC metric을 직접 켜는 핵심 조건:

```bash
python -m src.run \
  experiment=mdg_pretrain \
  action=validate \
  ckpt_path=/path/to/model.ckpt \
  paths.cache_root=/workspace/womd_v1_3/SMART_cache \
  model.model_config.n_rollout_closed_val=32 \
  model.model_config.closed_loop_denoising_steps=16 \
  model.model_config.n_batch_sim_agents_metric=10 \
  data.eval_max_agents=128
```

`data.eval_max_agents`를 128보다 작게 줄이면 Waymo GT scenario의 agent 수와 prediction agent 수가 달라져 Fast WOSAC metric이 실패할 수 있다. 빠른 개발 스모크에서만 줄이고, metric/submission 검증은 128로 둔다.

## 제출물 생성

validation split 제출물:

```bash
ACTION=validate \
CKPT_PATH=/path/to/model.ckpt \
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache \
bash scripts/wosac_sub.sh
```

test split 제출물:

```bash
ACTION=test \
CKPT_PATH=/path/to/model.ckpt \
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache \
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
  paths.cache_root=womd_v1_3/cache/SMART \
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
  paths.cache_root=womd_v1_3/cache/SMART \
  logger=[] callbacks=[] \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  trainer.limit_val_batches=1 \
  data.num_workers=0 data.val_batch_size=1 data.eval_max_agents=128 \
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

- `src/mdg/data.py`: WOMD cache loader, fixed-size MDG tensor 구성, DDP exact eval sampler.
- `src/mdg/modules.py`: scene encoder, differentiable kinematic dynamics, MDG denoiser, auxiliary predictor.
- `src/mdg/model.py`: LightningModule, mask/noise objective, 16-step closed-loop rollout, Fast WOSAC metric/submission 연결.
- `src/mdg/geometry.py`: 좌표 변환, angle wrapping, relation feature.
- `src/data_preprocess.py`: 새 cache 생성 시 MDG map/signal field 저장.

## 주의사항

- Waymax는 사용하지 않는다. closed-loop 효과는 제출 궤적 내부에서 1Hz replanning으로 근사한다.
- validation/test/submission은 기본적으로 full noise에서 시작해 16번 denoising한다. 같은 replanning segment 안에서는 scene encoder를 한 번만 실행하고 auxiliary predictor는 호출하지 않는다.
- WOSAC 제출은 반드시 32 rollout이어야 한다. submission mode에서 다른 값이면 모델 초기화 시 실패한다.
- evaluation/test DDP는 padding 없는 exact sampler를 사용한다. 제출 archive에서 scenario 중복이 생기지 않도록 하기 위함이다.
- 기존 SMART cache는 fallback으로 읽히지만, 논문 설정에 더 가까운 입력을 쓰려면 MDG field가 포함된 cache를 새로 만드는 편이 낫다.
