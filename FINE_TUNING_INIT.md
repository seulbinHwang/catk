# Fine-tuning Init Guide

작성일: 2026-05-29
현재 branch: `pareto_dmd_baseline`

이 문서는 새 작업 세션에서 이 repo를 빠르게 fine-tuning 중심으로 이해하기 위한 초기화 문서입니다. 상세 배경과 긴 실행 예시는 `README.md`와 `CLAUDE.md`를 기준으로 확인합니다.

## 1. 현재 작업 인식

이 저장소는 **CAT-K / SMART 기반 Flow Matching 모델의 fine-tuning 작업장**입니다. 기본 pretrain 경로도 남아 있지만, 현재 이 branch에서 주로 볼 것은 다음입니다.

- Flow Matching pretrained checkpoint에서 출발하는 fine-tuning.
- WOSAC 2025 closed-loop metric, 특히 `realism_meta_metric`(RMM)과 CPD 추적.
- Self-Forced NPFM 기반 **DMD / SiD** fine-tuning.
- Self-forced와 분리된 **OCSC(Open-Closed Self-Consistency)** fine-tuning.

즉, 새 작업을 시작할 때 `pre_bc_flow` pretrain 자체보다 `finetune_flow_*`, `self_forced_npfm*`, `ocsc_ft` 설정을 먼저 확인합니다.

## 2. Repo 한 줄 요약

`README.md` 기준으로 이 repo는 CrossEntropy next-token 경로를 제거하고 `smart_flow` 계열만 사용하는 Flow Matching 학습/추론/평가 전용 버전입니다.

- 기존 SMART map/context trunk를 재사용하고 agent 쪽을 flow decoder로 바꿉니다.
- `FlowTokenProcessor`는 18-token context와 16개 NTP-aligned anchor를 만듭니다.
- prefix-valid future loss mask가 기본 방향입니다. tail anchor는 존재하는 future prefix에만 loss를 줍니다.
- closed-loop inference는 0.5초씩 commit하며 WOSAC 2025 Sim Agents 포맷과 직접 연결됩니다.
- stop-motion gate는 전 경로에서 실질적으로 꺼져 있습니다.
- validation closed-loop metric은 vendored TrajTok Fast WOSAC 2025 evaluator를 사용합니다.

## 3. Fine-tuning 모드 구분

| 모드 | 주요 config/script | 핵심 |
|---|---|---|
| Pure FM range/prefix fine-tune | `configs/experiment/finetune_flow_range.yaml`, `finetune_flow_prefix_valid_*.yaml` | pretrained weight만 로드하고 Flow Matching loss로 학습 범위 또는 target mask를 조정 |
| Self-Forced DMD | `configs/experiment/self_forced_npfm.yaml`, `self_forced_npfm_pareto.yaml`, `self_forced_npfm_h100_*.yaml` | closed-loop self-rollout을 DMD-style distribution matching으로 보정 |
| Self-Forced SiD | `configs/experiment/self_forced_npfm_sid*.yaml` | DMD 대신 SiD-lite objective 사용 |
| OCSC | `configs/experiment/ocsc_ft.yaml`, `scripts/train_ocsc_ft.sh`, `scripts/train_ocsc_pose2hz_baseline.sh` | self-forced framework와 격리된 open-loop/closed-loop self-consistency fine-tune |
| RoaD | `configs/experiment/road_flow.yaml`, `scripts/road_flow_finetune.sh` | epoch마다 closed-loop RoaD cache 생성 후 학습 |

## 4. DMD / Self-Forced NPFM

DMD는 self-forced fine-tuning의 기본 distribution matching objective입니다.

핵심 구성:

- `F_rho`: pretrained `SMARTFlowDecoder`를 복사한 frozen target teacher.
- `F_psi`: generated path-flow estimator. 현재 generator의 detached self-rollout 위에서 online update됩니다.
- Generator EMA: validation / checkpoint 선택 / submission 시 EMA가 준비되면 EMA generator를 사용합니다.
- Generator update와 estimator update는 엄격히 분리되어야 합니다.
- 기본 objective는 `model.model_config.self_forced.distribution_matching_objective=dmd`입니다.
- SiD-lite로 바꾸면 `distribution_matching_objective=sid`를 사용합니다.

대표 실행:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
torchrun --standalone --nproc_per_node=2 -m src.run \
  experiment=self_forced_npfm_pareto \
  action=finetune \
  ckpt_path=logs/pretrained/pretrained.ckpt \
  paths.cache_root="$CACHE_ROOT" \
  task_name=pareto_dmd_ft
```

중요 규칙:

- pretrained checkpoint에서 새 self-forced run을 시작할 때는 `action=finetune`.
- self-forced run을 이어서 학습할 때만 `action=fit ckpt_path=<self_forced_run>/last.ckpt`.
- self-forced checkpoint와 pretrained checkpoint를 `action` 반대로 섞으면 보조 state(`F_rho`, `F_psi`, EMA) 때문에 의도적으로 막히는 구조입니다.
- `use_anchor_flow_matching_loss=false`가 주요 DMD/pareto 실험의 기본 방향입니다.

자주 보는 knob:

```text
model.model_config.self_forced.dmd_beta
model.model_config.self_forced.estimator_warmup_steps
model.model_config.self_forced.estimator_lr
model.model_config.self_forced.n_rollouts
model.model_config.self_forced.n_anchors
model.model_config.self_forced.anchor_stride
model.model_config.self_forced.sampling.noise_scale
model.model_config.self_forced.sampling.random_terminal_step.policy
model.model_config.self_forced.sampling.backprop_last_k
model.model_config.self_forced.unfrozen_range
```

## 5. OCSC Fine-tuning

OCSC는 self-forced와 다른 별도 fine-tune mode입니다.

활성 조건:

```yaml
model:
  model_config:
    self_forced:
      enabled: false
    finetune:
      enabled: true
      mode: ocsc_ft
```

코드 진입:

- `src/smart/model/smart_flow.py:_is_ocsc_ft_enabled`
- `src/smart/model/smart_flow.py:_run_flow_ocsc_ft_step`
- `training_step()`에서 OCSC가 켜져 있으면 self-forced보다 먼저 OCSC branch로 들어갑니다.

알고리즘 요약:

1. anchor 0(history end) 기준으로 현재 scene context를 만듭니다.
2. frozen reference flow decoder로 `M`개 open-loop sample을 생성합니다.
3. student로 `G`개 closed-loop rollout을 생성합니다.
4. 각 closed-loop rollout에 대해 가장 가까운 open-loop sample을 찾습니다.
5. 10 Hz 전체가 아니라 2 Hz coarse point, 즉 0.5초 block 끝점 기준으로 paired pose-norm L2를 학습합니다.
6. 기본 loss weight는 `pos_w=1.0`, `head_w=0.01`입니다.

현재 안정 baseline wrapper:

```bash
bash scripts/train_ocsc_pose2hz_baseline.sh
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 \
MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=2 LIMIT_VAL_BATCHES=0 \
TRAIN_B=2 VAL_B=2 OCSC_N_ROLLOUTS=2 OCSC_N_OL_ROLLOUTS=4 \
bash scripts/train_ocsc_pose2hz_baseline.sh
```

OCSC 핵심 knob:

```text
OCSC_N_ROLLOUTS        # G: student closed-loop rollout 수
OCSC_N_OL_ROLLOUTS     # M: frozen ref open-loop sample 수
OCSC_OL_NEAREST_MATCH  # per-CL nearest OL matching
OCSC_GT_TARGET         # OL 대신 GT target 사용 여부
OCSC_USE_PRETRAINED_REF
OCSC_POSITION_WEIGHT
OCSC_HEADING_WEIGHT
OCSC_VELOCITY_HEAD_ONLY
OCSC_FULL_FLOW_DECODER
```

OCSC와 DMD의 차이:

- DMD: closed-loop generator를 frozen teacher / generated estimator 차이로 보정하는 distribution matching.
- OCSC: critic 없이 frozen reference decoder의 open-loop distribution을 closed-loop rollout target pool로 사용합니다.
- OCSC는 `finetune.mode=ocsc_ft`, DMD는 `self_forced.enabled=true` 경로입니다. 두 경로를 같은 run에서 섞지 않습니다.

## 6. Metric / Checkpoint 기준

주요 validation metric:

```text
val_closed/sim_agents_2025/realism_meta_metric
val_closed/WOSAC-CPD/value
val_closed/WOSAC-CES/value
```

기본 checkpoint monitor는 RMM입니다.

```yaml
callbacks:
  model_checkpoint:
    monitor: val_closed/sim_agents_2025/realism_meta_metric
    mode: max
```

CPD baseline 보존율을 보고 싶으면 `model.model_config.wosac_cpd_reference`에 비교 기준 CPD 값을 넣습니다.

## 7. 실행 전 체크리스트

1. `ckpt_path`가 현재 `flow_window_steps`와 같은 horizon으로 pretrain된 checkpoint인지 확인합니다.
2. 새 fine-tuning 시작은 보통 `action=finetune`, 같은 run resume는 `action=fit`입니다.
3. DMD/SiD는 self-forced checkpoint state가 있으므로 resume 규칙을 특히 지킵니다.
4. OCSC는 `self_forced.enabled=false`이고 `finetune.mode=ocsc_ft`여야 합니다.
5. `paths.cache_root` 아래에 `training`, `validation`, `testing` cache가 있는지 확인합니다.
6. 이 repo의 로컬 작업 규칙상 학습/평가 GPU는 기본적으로 `CUDA_VISIBLE_DEVICES=2,3`만 사용합니다.
7. Hydra resolved config는 실행 로그의 `config_tree.log`와 `.hydra/config.yaml`에서 확인합니다.

## 8. 자주 확인할 파일

```text
README.md                                      # 전체 설명과 긴 실행 예시
CLAUDE.md                                     # repo 작업 규칙과 빠른 가이드
configs/run.yaml                              # Hydra top-level entry
configs/model/smart_flow.yaml                 # self_forced / finetune / decoder 기본값
configs/experiment/self_forced_npfm_pareto.yaml
configs/experiment/ocsc_ft.yaml
scripts/train_ocsc_pose2hz_baseline.sh
scripts/train_ocsc_ft.sh
src/smart/model/smart_flow.py                 # training_step, DMD, OCSC 구현
src/smart/modules/self_forced_*.py            # DMD/SiD helper
tests/test_self_forced_*.py                   # self-forced 정합성 테스트
```

## 9. 검증 명령

가벼운 import sanity:

```bash
conda run -n catk python -c "from src.smart.model.smart_flow import SMARTFlow; print('ok')"
```

self-forced 관련 단위 테스트:

```bash
conda run -n catk pytest \
  tests/test_self_forced_update_separation.py \
  tests/test_self_forced_dmd_direction_beta.py \
  tests/test_self_forced_sid_loss.py \
  tests/test_self_forced_trainable_range.py -q
```

OCSC smoke는 별도 pytest보다 wrapper의 작은 batch 실행으로 확인하는 편이 현재 구조와 맞습니다.

```bash
CUDA_VISIBLE_DEVICES=3 NPROC_PER_NODE=1 \
MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=0 \
TRAIN_B=1 VAL_B=1 OCSC_N_ROLLOUTS=1 OCSC_N_OL_ROLLOUTS=2 \
bash scripts/train_ocsc_pose2hz_baseline.sh
```
