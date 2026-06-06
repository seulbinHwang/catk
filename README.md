# CAT-K semi_mdg

이 브랜치는 **2초 Control-State MDG Pretrain** 전용입니다. 기존 SMART map/context trunk와 18개 context slot, 16개 train anchor, 2초 미래 window, 0.5초 closed-loop commit, raw 10Hz WOSAC rollout 경로는 유지합니다.

Flow Matching velocity target, ODE sampling, self-forced fine tuning, LQR bridge, matched-token 외부 출력 경로는 실험 대상에서 제외했습니다.
active MDG config에는 Flow ODE solver 설정이 없고, 이전 path-flow velocity API를 호출하면 즉시 `RuntimeError`가 나도록 막아 둡니다.

## 핵심 구조

학습 흐름은 다음과 같습니다.

```text
SMART cache scene
-> 18 context slot + 16 train anchor 구성
-> 2초 GT pose를 0.1초 단위 3D control로 변환
-> clean control을 5D state로 복원
-> MDG noise level mask 생성
-> clean control에 Gaussian noise 적용
-> noisy control을 5D state로 복원
-> context-conditioned denoiser가 clean control 예측
-> 예측 control을 5D state로 복원
-> valid future step에 대해서만 state MSE
-> 같은 anchor context에서 20-step best-mode auxiliary trajectory loss
```

기본 tensor 의미는 다음입니다.

| 항목 | 값 |
| --- | --- |
| future horizon | 20 step = 2초 |
| commit interval | 5 step = 0.5초 |
| train context/anchor | raw step 10, 15, ..., 85 |
| train anchor 수 | 16 |
| control dim | `[delta_s, delta_n, delta_yaw]` |
| MDG state dim | `[x/20, y/20, cos(yaw), sin(yaw), speed/10]` |
| noise level | 1..5, alpha `[0.99, 0.745, 0.5, 0.255, 0.01]` |
| auxiliary prediction | 6 modes x 20 step x `[local_x, local_y, delta_yaw]` |
| auxiliary head | `128 -> 896 -> 360`, 438,760 params |
| auxiliary loss weight | 5.0 |
| inference default | `sample_steps=1`, `action_reuse=true` |

## MDG Mask Sampling

학습 noise corruption은 MDG 논문 설명에 맞춰 batch 안의 masking rate가 낮은 값부터 높은 값까지 균등하게 들어가도록 만듭니다. 각 active scene-anchor pair는 한 training update 안에서 stratified rate를 하나 받습니다.

```text
r = [0, 1 / (B - 1), 2 / (B - 1), ..., 1]
```

DDP 학습에서는 rank별 local batch가 아니라 가능하면 전체 active scene-anchor 수를 모아 global batch 기준 rate grid를 만들고, 각 rank가 자기 slice를 받습니다. time-axis masking과 agent-axis masking도 active scene-anchor 안에서 거의 1:1이 되도록 배치합니다. Gaussian noising 수식, alpha schedule, time-axis/agent-axis mask 구조는 그대로 유지합니다.

## Auxiliary Trajectory Loss

`semi_mdg`는 MDG 논문의 scene-context regularization 의도를 2초 horizon에 맞춰 적용합니다. 보조 head는 denoiser 출력 뒤가 아니라, noise를 넣기 전 anchor별 context vector에서 바로 20-step future pose를 예측합니다.

```text
anchor context [P, 128]
-> auxiliary head [128 -> 896 -> 360]
-> [P, 6, 20, 3]
```

보조 예측 길이, target, best-mode loss 정의는 그대로 유지하고 MLP 중간 폭만 896으로 둡니다. 이 head의 파라미터 수는 438,760개이며, 20-step 고정 조건 안에서 MDG 논문 Waymo 설정의 auxiliary predictor 규모에 가깝게 맞추기 위한 설정입니다.

보조 target은 anchor 현재 위치/방향 기준 local pose입니다.

```text
[local_x_m, local_y_m, delta_heading_rad]
```

6개 mode 중 best mode는 20-step valid prefix의 local xy L2로만 고릅니다. 선택된 mode에는 local x, local y, wrapped delta heading 전체에 Smooth L1 loss를 적용하고, 유효 future step 수로 평균냅니다. 최종 pretrain loss는 다음입니다.

```text
train/loss = train/loss_mdg + 5.0 * train/loss_aux
```

이 auxiliary head는 학습 전용입니다. open-loop/closed-loop validation, WOSAC metric, submission rollout에는 사용하지 않습니다.

## Control Dynamics

이 브랜치에서는 **사람, 자전거, 자동차를 모두 non-holonomic control-state dynamics로 처리**합니다. 보행자도 별도 holonomic 예외 경로를 쓰지 않습니다.

제어값은 여전히 3차원입니다.

```text
[delta_s, delta_n, delta_yaw]
```

하지만 semi_mdg 기본 학습/추론에서는 모든 agent type에 대해 `delta_n=0`인 non-holonomic 경로만 의미 있게 사용합니다. vehicle/cyclist는 기존 no-slip midpoint arc 식을 유지하고, pedestrian은 raw heading 대신 위치 변화에서 얻은 보행 진행방향으로 먼저 회전한 뒤 전진합니다. 따라서 보행자 lateral GT motion은 `delta_n` 직접 이동으로 맞추지 않고, `delta_yaw`로 진행방향을 돌린 후 `delta_s`로 흡수합니다. 이 방식은 보행자를 holonomic으로 만들지 않으면서 보행자 position round-trip error를 크게 줄이기 위한 target 생성 방식입니다.

`use_holonomic_model_only` 옵션은 더 이상 노출하지 않습니다. 호환성 때문에 일부 함수 인자로 남아 있더라도 `true`가 들어오면 즉시 에러를 내서 잘못된 실험이 조용히 학습되지 않게 막습니다.

출력은 기존 평가 경로와 동일하게 raw 10Hz position/heading rollout으로 변환됩니다.

2026-06-05 기준 testas A100x7 검증:

- `tests/test_kinematic_control.py`: 21 passed
- control metric / prefix-valid loss / type-aware yaw scale 관련 테스트: 23 passed
- SMART cache training 512 scene 샘플에서 pedestrian action->state round-trip position error가 mean `0.03298m -> 0.00197m`, p95 `0.11366m -> 0.01457m`, max `1.47296m -> 0.02000m`로 감소
- real cache 기반 7-GPU smoke: 1 train batch + open-loop validation + closed-loop Fast WOSAC validation 정상 종료

## 데이터

두 pod에서 같은 cache 경로가 보여야 합니다.

```text
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

학습에는 `training/*.pkl`과 `validation/*.pkl`이 필요하고, closed-loop WOSAC validation에는 validation TFRecord split도 필요합니다.

### Semi-MDG token/flow sidecar

학습 속도 병목인 deterministic token/flow target은 sidecar로 미리 계산할 수 있습니다. 이 sidecar는 학습 전용입니다. validation, closed-loop rollout, WOSAC metric, submission 경로는 기존 cache를 그대로 사용합니다.

생성되는 sidecar는 기존 cache를 수정하지 않고 별도 폴더에 저장됩니다.

```text
${CACHE_ROOT}/semi_mdg_sidecar/training/*.pkl
```

testas A100 x7 pod 안에서 전체 training split을 7개 shard로 나누어 생성하려면:

```bash
cd /mnt/nuplan/projects/catk
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache \
SIDECAR_ROOT=/workspace/womd_v1_3/SMART_cache/semi_mdg_sidecar \
bash scripts/precompute_semi_mdg_sidecar_a100x7.sh
```

작은 smoke는 `LIMIT`으로 줄일 수 있습니다.

```bash
LIMIT=128 bash scripts/precompute_semi_mdg_sidecar_a100x7.sh
```

testas launcher는 기본적으로 이 sidecar를 사용합니다. 다른 위치를 쓰는 경우에만 `TRAIN_SIDECAR_DIR`를 바꾸면 됩니다.

```bash
TRAIN_SIDECAR_DIR=/workspace/womd_v1_3/SMART_cache/semi_mdg_sidecar/training \
bash scripts/start_semi_mdg_testas_a100x7_pretrain.sh
```

sidecar가 켜졌는데 특정 scenario의 sidecar file이 없으면 학습은 즉시 실패합니다. 조용히 기존 on-the-fly path로 fallback하지 않습니다. 이는 sidecar target과 on-the-fly target이 섞이는 실험을 막기 위한 fail-fast 정책입니다.

2026-06-06 testas 검증:

| check | result |
| --- | --- |
| sidecar format | `semi_mdg_token_flow_sidecar_v1` |
| batch 20 equality | map token, context token, flow target, loss mask, agent type/length all matched |
| token/flow target time | on-the-fly `51.47ms` -> sidecar `0.54ms` |
| 1-GPU train smoke | 2 train batches passed |
| 7-GPU DDP smoke | 1 train batch passed |

## H100 x3x2 학습

대상 pod:

```text
hsb-npc-training-3-1
hsb-npc-training-3-2
```

기본 실행:

```bash
python scripts/launch_mdg_h100x3x2_static_pods.py \
  --replace \
  --task-name semi_mdg_pretrain_h100x3x2_$(date +%Y%m%d_%H%M%S) \
  --pod-cache-root hsb-npc-training-3-1=/workspace/womd_v1_3/SMART_cache \
  --pod-cache-root hsb-npc-training-3-2=/workspace/womd_v1_3/SMART_cache
```

OOM retry는 `train_batch_size`를 2씩 낮추며 재시작합니다. `1254aea` 검증 중 H100x3x2에서 per-GPU 28/26/24는 실제 train batch OOM이 났고, per-GPU 22는 통과했지만 peak reserved memory가 약 98%였습니다. 기본값은 장기 학습 안정성을 위해 per-GPU 20으로 둡니다.

```text
initial_bs: 20
oom_step: 2
min_bs: 2
nproc_per_node: 3
effective batch = final_train_batch_size x 6
train_memory_balanced_batches: true
trainer.use_distributed_sampler: false
```

중지:

```bash
python scripts/launch_mdg_h100x3x2_static_pods.py \
  --stop \
  --task-name <task_name>
```

짧은 smoke 검증 예시:

```bash
python scripts/launch_mdg_h100x3x2_static_pods.py \
  --replace \
  --task-name semi_mdg_smoke \
  --initial-bs 4 \
  --limit-train-batches 2 \
  --limit-val-batches 1 \
  --extra-hydra-overrides 'trainer.max_epochs=1 logger.wandb.offline=true' \
  --pod-cache-root hsb-npc-training-3-1=/workspace/womd_v1_3/SMART_cache \
  --pod-cache-root hsb-npc-training-3-2=/workspace/womd_v1_3/SMART_cache
```

## Testas A100 x7 학습

`testas` 단일 pod에 붙은 A100 80GB 7장을 모두 사용하려면 아래 스크립트를 씁니다.
스크립트는 기존 `testas` pod를 재시작하지 않고, pod 안의 repo를 `semi_mdg`
최신 브랜치로 동기화한 뒤 tmux 세션에서 7-GPU DDP 학습을 시작합니다.

```bash
bash scripts/start_semi_mdg_testas_a100x7_pretrain.sh
```

기본 실행 조건은 다음입니다.

```text
pod: testas
cache_root: /workspace/womd_v1_3/SMART_cache
train_sidecar_dir: /workspace/womd_v1_3/SMART_cache/semi_mdg_sidecar/training
experiment: mdg_pretrain_h100x3x2
session: catk-semi-mdg-testas-a100x7
nproc_per_node: 7
initial_bs: 20 per GPU
min_bs: 16 per GPU
oom_step: 2
effective batch: 140
val_batch_size: 12 per GPU
train_memory_balanced_batches: true
trainer.use_distributed_sampler: false
max_epochs: 64
precision: bf16-mixed
validation: check_val_every_n_epoch=16, limit_val_batches=0.1
closed-loop validation: n_rollout_closed_val=32, scorer_scene_num=1680
validation sampling: sample_steps=1, action_reuse=true, antithetic_pairs=true
optimizer LR: 0.00068313
OOM retry: train_batch_size를 2씩 낮추고 latest epoch_last.ckpt에서 재시작
LR retry policy: retry된 train_batch_size 기준으로 sqrt scaling LR 재계산
```

Testas 기본 LR은 flow-control baseline의 global batch `108`, LR `6e-4`를
기준으로 global batch `140`에 sqrt scaling을 적용한 값입니다:
`6e-4 * sqrt(140 / 108) = 0.00068313`.

OOM retry가 batch를 낮추면 LR도 같은 규칙으로 다시 계산합니다. 예를 들어
per-GPU batch가 `18`로 내려가면 global batch는 `126`이고 LR은
`6e-4 * sqrt(126 / 108) = 0.00064807`입니다.

학습 중지:

```bash
bash scripts/start_semi_mdg_testas_a100x7_pretrain.sh \
  --stop \
  --task-name <task_name>
```

짧은 train-only smoke:

```bash
bash scripts/start_semi_mdg_testas_a100x7_pretrain.sh \
  --replace \
  --task-name semi_mdg_testas_train_smoke \
  --wandb-mode offline \
  --initial-bs 20 \
  --min-bs 20 \
  --max-epochs 1 \
  --limit-train-batches 2 \
  --limit-val-batches 0 \
  --extra-hydra-overrides 'logger.wandb.log_model=false model.model_config.val_open_loop=false model.model_config.val_closed_loop=false'
```

짧은 train + open/closed-loop validation smoke:

```bash
bash scripts/start_semi_mdg_testas_a100x7_pretrain.sh \
  --replace \
  --task-name semi_mdg_testas_val_smoke \
  --wandb-mode offline \
  --initial-bs 20 \
  --min-bs 20 \
  --val-batch-size 12 \
  --max-epochs 1 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --extra-hydra-overrides 'trainer.check_val_every_n_epoch=1 model.model_config.n_rollout_closed_val=32 model.model_config.scorer_scene_num=84 logger.wandb.log_model=false'
```

원격 로그는 다음 위치에 저장됩니다.

```text
/mnt/nuplan/projects/catk/logs/tmux_testas_a100x7_semi_mdg/<task_name>/tmux.log
/mnt/nuplan/projects/catk/logs/<task_name>/runs/<run_id>/
```

2026-06-06 KST에 `semi_mdg@6f9ecb7`, `testas` A100 80GB x7, 실제
SMART cache와 precomputed semi_mdg sidecar로 최신 학습/추론/평가 경로를
다시 검증했습니다.

| check | result |
| --- | --- |
| per-GPU bs28, 5 train batches | CUDA OOM on first train step |
| per-GPU bs24, 5 train batches | CUDA OOM after first train step |
| per-GPU bs22, 10 train batches | CUDA OOM on first train step |
| per-GPU bs20, 20 train batches | passed, global batch 140, peak reserved memory 91.16% |
| per-GPU bs20, train + validation smoke | passed, val batch 12, 32 rollouts, 84 Fast WOSAC scenes, closed-loop metrics logged |

`bs22` 이상은 최신 code + sidecar path에서도 OOM이므로, default long-run
start batch는 보수 안정값 `bs20`입니다. OOM retry는 계속 켜져 있으며, 이후
더 큰 scene batch에서 memory spike가 나면 batch를 `2`씩 낮추고 해당 batch에
맞춰 LR도 다시 sqrt scaling합니다.

## 주요 설정

기본 preset은 `configs/experiment/mdg_pretrain_h100x3x2.yaml`입니다.

| 항목 | 값 |
| --- | --- |
| model | `smart_flow` 기반 MDG control-state denoiser |
| action | `fit` |
| precision | `bf16-mixed` |
| max epochs | 64 |
| train batch | per-GPU 20, effective 120 on H100x3x2 |
| optimizer LR | `0.00068313` |
| warmup | 4 epoch |
| validation 주기 | 16 epoch |
| validation rollout | 32 |
| auxiliary prediction | 6 modes x 20 step x 3, weight 5.0 |
| `validation_rollout_sampling.sample_steps` | 1 |
| `validation_rollout_sampling.action_reuse` | true |
| `decoder.closed_loop_rollout_mode` | `raw_mdg` |
| `decoder.use_lqr` | false |
| DDP `find_unused_parameters` | true |
| `finetune.enabled` | false |

## 추론 설정

기본 추론은 매 0.5초마다 새 Gaussian control noise에서 시작해 2초 control을 한 번 denoise하고, 앞 0.5초만 commit합니다.

```text
sample_steps=1
action_reuse=true
```

`sample_steps > 1`로 MDG multi-step denoising을 실험할 때도 denoiser 호출 mask는 학습된 noise level만 사용합니다. 예를 들어 5-step은 `[5, 4, 3, 2, 1]` 순서로 denoiser를 호출하고, Algorithm 2의 마지막 clean transition은 `m0=0`, `alpha(m0)=1`인 identity로 처리합니다. 따라서 마지막에 별도 `m=0` denoiser 호출을 추가하지 않고, 마지막 clean estimate를 그대로 최종 control로 사용합니다.

action reuse는 기본으로 켜져 있습니다. 끄는 ablation이 필요할 때만 다음처럼 override합니다.

```bash
model.model_config.validation_rollout_sampling.action_reuse=false
```

action reuse는 이전 2초 predicted control을 0.5초 앞으로 shift한 뒤, 새 noise와 섞어 다음 block의 초기 action으로 사용합니다. 기본값은 true입니다.

## 검증 기준

구현 변경 후 최소 확인 항목:

```text
1. local compile/import 통과
2. 실제 SMART cache batch로 train step loss finite
3. H100 x3x2 DDP smoke fit exit 0
4. validation open-loop + closed-loop smoke exit 0
5. action_reuse=true closed-loop smoke exit 0
6. pred_traj_10hz / pred_head_10hz shape 유지
7. train/loss, train/loss_mdg, train/loss_aux가 NaN 없이 감소
```

이 브랜치의 목표는 Flow Matching과 fine tuning을 섞지 않고, 2초 control-state MDG pretrain 단일 방법론을 빠르고 안정적으로 검증하는 것입니다.

## 최근 검증 기록

2026-06-05 KST에 `testas` A100 80GB x7과 실제 SMART cache로 auxiliary trajectory head를 `128 -> 896 -> 360`으로 확대한 뒤 검증했습니다.

```text
auxiliary head parameter check:
  aux_trajectory.hidden_dim: 896
  aux_head_params: 438,760
  aux_head_output_shape: [2, 360]
  model summary total params: 7.5M

unit / regression tests:
  command: PYTHONPATH=/mnt/nuplan/projects/catk pytest tests/test_aux_trajectory_loss.py tests/test_open_loop_empty_target_loss.py -q
  result: 9 passed

pipeline compatibility tests:
  command: PYTHONPATH=/mnt/nuplan/projects/catk pytest tests/test_control_metric_conversion.py tests/test_aux_trajectory_loss.py tests/test_open_loop_empty_target_loss.py tests/test_mask_aware_prefix_valid_decoder.py -q
  result: 23 passed

train + open/closed-loop validation smoke:
  task_name: semi_mdg_aux896_train_val_smoke_20260605
  train_batch_size: 1 per GPU
  val_batch_size: 1 per GPU
  n_rollout_closed_val: 2
  scorer_scene_num: 7
  result: exit status 0
  train/loss: 34.44249
  train/loss_mdg: 0.24784
  train/loss_aux: 6.83893
  val_open/ADE2s: 4.28147
  val_closed/sim_agents_2025/realism_meta_metric: 0.56220
  val_closed/sim_agents_2025/scenario_counter: 7
```

2026-06-05 KST에 `testas` A100 80GB x7과 실제 SMART cache로 batch-stratified MDG mask sampling 적용 후 다시 검증했습니다. 이 재검증에서 일부 DDP rank의 active scene-anchor pair가 0개인 경우에도 모든 rank가 동일한 collective 경로를 지나도록 보강했습니다.

```text
unit / regression tests:
  command: PYTHONPATH=/mnt/nuplan/projects/catk pytest tests/test_control_metric_conversion.py tests/test_aux_trajectory_loss.py tests/test_open_loop_empty_target_loss.py tests/test_mask_aware_prefix_valid_decoder.py -q
  result: 22 passed

preprocessing / sampler compatibility tests:
  command: PYTHONPATH=/mnt/nuplan/projects/catk pytest tests/test_prefix_valid_future_loss_mask.py tests/test_memory_balanced_batch_sampler.py -q
  result: 14 passed

DDP mask-plan zero-active-rank contract:
  command: torchrun --standalone --nnodes=1 --nproc_per_node=7 /tmp/test_mdg_stratified_mask_ddp_zero.py
  result: no collective deadlock with ranks 0 and 3 holding zero active pairs
  global delta grid matched linspace(0, 1) across 23 active scene-anchor samples
  time-axis / agent-axis counts: 13 / 10

train-only smoke:
  task_name: semi_mdg_aux_strat_mask_train_smoke_20260605
  train_batch_size: 1 per GPU
  global_batch_size: 7
  limit_train_batches: 2
  result: exit status 0
  train/loss: 19.74106
  train/loss_mdg: 0.13753
  train/loss_aux: 3.92071

train + open/closed-loop validation smoke:
  task_name: semi_mdg_aux_strat_mask_val_smoke_20260605
  train_batch_size: 1 per GPU
  val_batch_size: 1 per GPU
  n_rollout_closed_val: 2
  scorer_scene_num: 7
  result: exit status 0
  train/loss: 34.40338
  train/loss_mdg: 0.24784
  train/loss_aux: 6.83111
  val_open/ADE2s: 4.33034
  val_closed/sim_agents_2025/realism_meta_metric: 0.58547
  val_closed/sim_agents_2025/scenario_counter: 7
```

2026-06-05 KST에 `testas` A100 80GB x7과 실제 SMART cache로 20-step auxiliary trajectory loss 적용 후 검증했습니다.

```text
unit tests:
  command: pytest tests/test_aux_trajectory_loss.py tests/test_open_loop_empty_target_loss.py -q
  result: 8 passed

train-only smoke:
  task_name: semi_mdg_aux_train_smoke_20260605
  train_batch_size: 2 per GPU
  global_batch_size: 14
  limit_train_batches: 2
  result: exit status 0
  train/loss: 11.85204
  train/loss_mdg: 0.06043
  train/loss_aux: 2.35832

train + open/closed-loop validation smoke:
  task_name: semi_mdg_aux_val_smoke_20260605
  train_batch_size: 1 per GPU
  val_batch_size: 1 per GPU
  n_rollout_closed_val: 2
  scorer_scene_num: 7
  result: exit status 0
  train/loss: 34.40209
  train/loss_mdg: 0.24735
  train/loss_aux: 6.83095
  val_closed/sim_agents_2025/realism_meta_metric: 0.58544
  val_closed/sim_agents_2025/scenario_counter: 7
```

2026-06-04 KST에 Flow ODE solver 잔여 config 제거와 path-flow velocity API guard를 적용한 뒤, 같은 H100x3x2 pod와 실제 SMART cache로 다시 검증했습니다.

```text
local compile:
  files:
    src/smart/modules/smart_flow_decoder.py
    src/smart/modules/flow_agent_decoder.py
    src/smart/model/smart_flow.py
  result: py_compile exit 0

removed Flow Matching API guard:
  pod: hsb-npc-training-3-1
  checked:
    SMARTFlowDecoder.path_flow_velocity_for_anchor0 -> RuntimeError
    SMARTFlowAgentDecoder.path_flow_velocity_for_anchor0 -> RuntimeError
  result: guard exit 0

DDP train-only smoke:
  task_name: semi_mdg_intent_guard_ddp_smoke_20260604
  pods: hsb-npc-training-3-1, hsb-npc-training-3-2
  cache_root: /workspace/womd_v1_3/SMART_cache
  world_size: 2 nodes x 3 H100 = 6
  train_batch_size: 2 per GPU
  limit_train_batches: 2
  validation: disabled for speed
  result: exit status 0
  final train/loss_mdg: 0.27901
  train_setup/global_batch_size: 12
  peak reserved memory: 13.13%

DDP fit + open/closed-loop validation smoke:
  task_name: semi_mdg_intent_guard_val2_smoke_20260604
  pods: hsb-npc-training-3-1, hsb-npc-training-3-2
  cache_root: /workspace/womd_v1_3/SMART_cache
  world_size: 2 nodes x 3 H100 = 6
  train_batch_size: 2 per GPU
  val_batch_size: 2 per GPU
  n_rollout_closed_val: 2
  scorer_scene_num: 1680
  result: exit status 0
  train/loss_mdg: 0.54827
  val_open/ADE2s: 3.65683
  val_closed/sim_agents_2025/realism_meta_metric: 0.48892
  val_closed/sim_agents_2025/scenario_counter: 1680

DDP action reuse closed-loop smoke:
  task_name: semi_mdg_action_reuse_val_smoke_20260604
  pods: hsb-npc-training-3-1, hsb-npc-training-3-2
  cache_root: /workspace/womd_v1_3/SMART_cache
  world_size: 2 nodes x 3 H100 = 6
  validation_rollout_sampling.action_reuse: true
  n_rollout_closed_val: 2
  scorer_scene_num: 12
  result: exit status 0
  train/loss_mdg: 0.54829
  val_closed/sim_agents_2025/realism_meta_metric: 0.53048
  val_closed/sim_agents_2025/scenario_counter: 12

DDP find_unused_parameters check:
  tried: find_unused_parameters=false
  result: first train step did not finish within normal smoke time and was stopped
  decision: keep find_unused_parameters=true for the H100x3x2 preset
```

2026-06-03 KST에 최신 `semi_mdg` 코드에 all non-holonomic dynamics 변경을 적용한 뒤, `hsb-npc-training-3-1`, `hsb-npc-training-3-2`에서 실제 cache와 H100 GPU로 검증했습니다.

```text
unit/control QA:
  pod: hsb-npc-training-3-1
  checkout: /tmp/catk_semi_mdg_validate
  cache-independent control check:
    pedestrian lateral GT motion -> delta_n=0
    pedestrian round-trip error: 6.0m
    use_holonomic_model_only=true -> ValueError
  result: ALL_NONHOL_UNIT_QA_OK

DDP fit + closed-loop validation smoke:
  task_name: semi_mdg_all_nonhol_ddp_smoke_20260603_235642
  pods: hsb-npc-training-3-1, hsb-npc-training-3-2
  cache_root: /workspace/womd_v1_3/SMART_cache
  world_size: 2 nodes x 3 H100 = 6
  train_batch_size: 2 per GPU
  val_batch_size: 2 per GPU
  limit_train_batches: 2
  limit_val_batches: 1
  n_rollout_closed_val: 2
  result: exit status 0
  final train/loss_mdg: 0.31906
  closed-loop WOSAC path: completed

DDP train-only convergence smoke:
  task_name: semi_mdg_all_nonhol_train20_20260603_235750
  pods: hsb-npc-training-3-1, hsb-npc-training-3-2
  cache_root: /workspace/womd_v1_3/SMART_cache
  world_size: 2 nodes x 3 H100 = 6
  train_batch_size: 4 per GPU
  limit_train_batches: 20
  validation: disabled for speed
  result: exit status 0
  final train/loss_mdg: 0.04677
  peak reserved memory: 33.13%
```

검증 후 두 pod의 GPU compute process가 없는 idle 상태를 확인했습니다.
