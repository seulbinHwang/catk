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
| inference default | `sample_steps=1`, `action_reuse=false` |

## Control Dynamics

이 브랜치에서는 **사람, 자전거, 자동차를 모두 같은 non-holonomic control-state dynamics로 처리**합니다. 보행자도 별도 holonomic 예외 경로를 쓰지 않습니다.

제어값은 여전히 3차원입니다.

```text
[delta_s, delta_n, delta_yaw]
```

하지만 semi_mdg 기본 학습/추론에서는 모든 agent type에 대해 `delta_n=0`인 non-holonomic 경로만 의미 있게 사용합니다. 즉 lateral 이동을 한 step에서 순간이동처럼 직접 맞추는 대신, `delta_s`와 `delta_yaw`가 만드는 곡선 진행으로 상태를 복원합니다. 이 제약으로 복원할 수 없는 lateral GT motion은 round-trip error로 잡히며, `control_round_trip_max_position_error_m` 필터 대상이 됩니다.

`use_holonomic_model_only` 옵션은 더 이상 노출하지 않습니다. 호환성 때문에 일부 함수 인자로 남아 있더라도 `true`가 들어오면 즉시 에러를 내서 잘못된 실험이 조용히 학습되지 않게 막습니다.

출력은 기존 평가 경로와 동일하게 raw 10Hz position/heading rollout으로 변환됩니다.

## 데이터

두 pod에서 같은 cache 경로가 보여야 합니다.

```text
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
```

학습에는 `training/*.pkl`과 `validation/*.pkl`이 필요하고, closed-loop WOSAC validation에는 validation TFRecord split도 필요합니다.

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

`testas` 단일 pod에 붙은 A100 80GB 7장을 모두 사용하려면 아래 launcher를 씁니다.

```bash
python scripts/launch_mdg_testas_a100x7.py \
  --replace \
  --task-name semi_mdg_pretrain_testas_a100x7_$(date +%Y%m%d_%H%M%S)
```

기본 실행 조건은 다음입니다.

```text
pod: testas
cache_root: /workspace/womd_v1_3/SMART_cache
experiment: mdg_pretrain_h100x3x2
session: catk-semi-mdg-testas-a100x7
nproc_per_node: 7
initial_bs: 20 per GPU
effective batch: 140
val_batch_size: 12 per GPU
train_memory_balanced_batches: true
trainer.use_distributed_sampler: false
max_epochs: 64
precision: bf16-mixed
validation: check_val_every_n_epoch=16, limit_val_batches=0.1
closed-loop validation: n_rollout_closed_val=32, scorer_scene_num=1680
validation sampling: sample_steps=1, action_reuse=false, antithetic_pairs=true
optimizer LR: 0.00068313
OOM retry: train_batch_size를 2씩 낮추고 latest epoch_last.ckpt에서 재시작
```

Testas 기본 LR은 flow-control baseline의 global batch `108`, LR `6e-4`를
기준으로 global batch `140`에 sqrt scaling을 적용한 값입니다:
`6e-4 * sqrt(140 / 108) = 0.00068313`.

`31dd89a` 이후 testas A100x7에서 memory-balanced batching을 켠 뒤 확인한
결과, per-GPU `30/28/26/24`는 실제 train batch에서 CUDA OOM이 났고
per-GPU `22`는 2-batch smoke를 통과했지만 peak reserved memory가 약
`97.37%`로 장기 학습에는 타이트했습니다. 기본값은 8-batch train smoke와
1680-scene closed-loop validation smoke를 통과한 보수적 안전값
per-GPU `20`입니다.

학습 중지:

```bash
python scripts/launch_mdg_testas_a100x7.py \
  --stop \
  --task-name <task_name>
```

짧은 train-only smoke:

```bash
python scripts/launch_mdg_testas_a100x7.py \
  --replace \
  --task-name semi_mdg_testas_train_smoke \
  --wandb-mode offline \
  --initial-bs 16 \
  --max-epochs 1 \
  --limit-train-batches 2 \
  --limit-val-batches 0 \
  --extra-hydra-overrides 'logger.wandb.log_model=false model.model_config.val_open_loop=false model.model_config.val_closed_loop=false'
```

짧은 train + open/closed-loop validation smoke:

```bash
python scripts/launch_mdg_testas_a100x7.py \
  --replace \
  --task-name semi_mdg_testas_val_smoke \
  --wandb-mode offline \
  --initial-bs 2 \
  --val-batch-size 2 \
  --max-epochs 1 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --extra-hydra-overrides 'trainer.check_val_every_n_epoch=1 model.model_config.n_rollout_closed_val=2 model.model_config.scorer_scene_num=14 logger.wandb.log_model=false'
```

원격 로그는 다음 위치에 저장됩니다.

```text
/mnt/nuplan/projects/catk/logs/tmux_testas_a100x7_semi_mdg/<task_name>/tmux.log
/mnt/nuplan/projects/catk/logs/<task_name>/runs/<run_id>/
```

2026-06-04 KST에 `testas` A100 80GB x7과 실제 SMART cache로 검증한 결과:

```text
train-only smoke:
  task_name: semi_mdg_testas_a100x7_smoke_20260604_122557
  train_batch_size: 4 per GPU
  global_batch_size: 28
  limit_train_batches: 2
  result: exit status 0
  final train/loss_mdg: 0.15257
  peak reserved memory: 26.92%

train + open/closed-loop validation smoke:
  task_name: semi_mdg_testas_a100x7_val_smoke_20260604_122701
  train_batch_size: 2 per GPU
  val_batch_size: 2 per GPU
  n_rollout_closed_val: 2
  scorer_scene_num: 14
  result: exit status 0
  val_closed/sim_agents_2025/realism_meta_metric: 0.56800
  val_closed/sim_agents_2025/scenario_counter: 14

batch-size safety check:
  bs20: 20-batch train smoke hit CUDA OOM
  bs18: 20-batch train smoke passed, but peak reserved memory was 97.90%
  bs16: 20-batch train smoke passed with peak reserved memory 85.74%
  decision: use bs16 as the default long-run start batch for testas
```

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
| `validation_rollout_sampling.sample_steps` | 1 |
| `validation_rollout_sampling.action_reuse` | false |
| `decoder.closed_loop_rollout_mode` | `raw_mdg` |
| `decoder.use_lqr` | false |
| DDP `find_unused_parameters` | true |
| `finetune.enabled` | false |

## 추론 설정

기본 추론은 매 0.5초마다 새 Gaussian control noise에서 시작해 2초 control을 한 번 denoise하고, 앞 0.5초만 commit합니다.

```text
sample_steps=1
action_reuse=false
```

`sample_steps > 1`로 MDG multi-step denoising을 실험할 때도 denoiser 호출 mask는 학습된 noise level만 사용합니다. 예를 들어 5-step은 `[5, 4, 3, 2, 1]` 순서로 denoiser를 호출하고, Algorithm 2의 마지막 clean transition은 `m0=0`, `alpha(m0)=1`인 identity로 처리합니다. 따라서 마지막에 별도 `m=0` denoiser 호출을 추가하지 않고, 마지막 clean estimate를 그대로 최종 control로 사용합니다.

action reuse ablation이 필요할 때만 다음처럼 켭니다.

```bash
model.model_config.validation_rollout_sampling.action_reuse=true
```

action reuse는 이전 2초 predicted control을 0.5초 앞으로 shift한 뒤, 새 noise와 섞어 다음 block의 초기 action으로 사용합니다. 기본값은 false입니다.

## 검증 기준

구현 변경 후 최소 확인 항목:

```text
1. local compile/import 통과
2. 실제 SMART cache batch로 train step loss finite
3. H100 x3x2 DDP smoke fit exit 0
4. validation open-loop + closed-loop smoke exit 0
5. action_reuse=true closed-loop smoke exit 0
6. pred_traj_10hz / pred_head_10hz shape 유지
7. train/loss, train/loss_mdg가 NaN 없이 감소
```

이 브랜치의 목표는 Flow Matching과 fine tuning을 섞지 않고, 2초 control-state MDG pretrain 단일 방법론을 빠르고 안정적으로 검증하는 것입니다.

## 최근 검증 기록

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
