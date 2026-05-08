# Repo notes

CATK / SMART-Flow 기반 motion forecasting 코드베이스.  **현재 브랜치
`OCSC_clean_v2` 는 `8b2f9b7dc6bb1506e921f55ef8d76e892e11b1b4` (self-forcing
라인의 마지막 commit) 에서 분기**해서, self-forcing / DRaFT / RoaD
fine-tuning 잔재를 모두 제거한 뒤 `origin/OCSC_clean` 의 OCSC fine-tuning
알고리즘만 surgical 이식한 라인입니다.  즉:

- **모델 본체** (pretraining / inference / rollout) = `8b2f9b7d` 의 새 모델
  (`flow_agent_decoder.py` 1883줄, `flow_local_decoder.py`,
  `dynamic_light_time`, `draft_physics.DEFAULT_LIMITS` 등 그대로 유지)
- **Fine-tuning 알고리즘** = `origin/OCSC_clean` 의 OCSC 본체를 복붙 위주로
  이식 (mmd_consistency_loss, HardSimAgentsMetrics, FinetuneConfig,
  `_run_flow_ocsc_ft_step` 832 줄 등)
- **새 추가 surgical 부품** = `ResidualFlowVelocityHead`, FlowODE 의
  `use_adjoint_for_bptt` / `last_n_grad_solver_steps` 토글

## 작업 규칙 (반드시 준수)

1. **코드 변경 = commit + push 페어**.  변경 전 상태 commit 하고, 변경 후
   commit 해서 push 한 뒤 다음 단계로 넘어갈 것.  중간 상태로 멈추지 말 것.
2. **GPU 할당**: GPU `0`, `1` 은 사용자 할당이 아니므로 절대 사용 금지.  학습 /
   추론 / parity check 모두 `CUDA_VISIBLE_DEVICES=2` 또는 `3` 에서만 진행할 것
   (멀티 GPU 가 필요하면 `2,3`).  런처 스크립트 default 는 이 정책을 따라야 함.
3. **시간 표기는 KST (Asia/Seoul)**.  로그 파일명, commit 메시지, 사용자 보고
   문, 스케줄링 모두 KST 기준.
4. **사용자 응답은 한국어**로 설명할 것.  코드 symbol / 변수명 / log message /
   commit 메시지는 영어, docstring 은 한국어 (기존 코드베이스 컨벤션 유지).

## Branches

- `main` — baseline.
- `origin/fix-hard-rmm` — 옛 OCSC reference (검증된 production 라인).
  **모델 본체가 다름** (`flow_agent_decoder.py` 1138줄짜리 옛 모델).
- `origin/OCSC_clean` — fix-hard-rmm 위에 사용자가 OCSC 알고리즘을 추가
  발전시킨 라인 (split MMD pos/heading 등).  **모델 본체는 fix-hard-rmm 와
  byte-identical** 이라 project_3 새 모델과 호환 안 됨.  OCSC 알고리즘 추출
  reference 로만 사용.
- `8b2f9b7d` — self-forcing 라인의 마지막 commit.  project_3 새 모델 본체.
- **`OCSC_clean_v2` — 현재 브랜치**.  `8b2f9b7d` 위에 self-forcing/DRaFT/RoaD
  fine-tuning 잔재를 모두 제거하고 `origin/OCSC_clean` 의 OCSC 본체를
  복붙 위주로 이식한 라인.

## 호환 ckpt

```
/home2/pnc2/repos_python/project_3/logs/pretrained/epoch_last.ckpt   # 87MB
```

`OCSC_clean_v2` base 와 strict load 시 `missing=0 / unexpected=0` 으로 검증
됨 (Phase A.3).  `ResidualFlowVelocityHead` 추가 후에도 strict=False load 시
missing=6 (residual_velocity_head 새 weight 만) 이고 OCSC default 는
`flow_velocity_head_only=True` 라 residual head 가 frozen 처리되므로
`run.py finetune` action 의 `_validate_finetune_loaded_trainable_params`
체크는 통과함.

`/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt`
(85MB, fix-hard-rmm 호환) 은 OCSC_clean_v2 와 모델 본체가 달라 호환 안 됨.
launcher 의 default `CKPT_PATH` 는 85MB 를 가리키므로 OCSC_clean_v2 학습
시에는 반드시 87MB 로 override.

## Key modules

```
src/smart/model/smart_flow.py                       (~2475 줄)
   - SMARTFlow LightningModule 본체.  base flow matching training step 은
     project_3 측 그대로, OCSC 분기는 _is_ocsc_ft_enabled / training_step 분기로
     붙어 있음.
   - OCSC 메서드 그룹 (training_step 직전, ~line 1112-2222):
     · _is_ocsc_ft_enabled
     · _world_traj_to_flow_norm / _compute_soft_rmm / _compute_rmm_group
     · _compute_rmm_bptt_gt_fm_loss   (anchor 후 batch-level GT FM regularization)
     · _compute_ocsc_train_hard_rmm   (HardSimAgentsMetrics 기반 monitoring)
     · _run_flow_ocsc_ft_step         (832 줄 OCSC 본체 — anchor sequential 루프
                                       + per-anchor 즉시 backward + MMD/paired L2 분기
                                       + GT vs OL target 분기 + BPTT 토글 처리)
   - on_train_start: OCSC mode 일 때 frozen ref_flow_decoder 생성 + NaN guard hook
   - __init__ 에 OCSC 전용 attribute (eval_sampling_noise, ref_flow_decoder,
     _ocsc_train_hard_rmm, _ocsc_train_hard_rmm_ref) 추가.
src/smart/modules/flow_local_decoder.py
   - FlowODE.__init__ 의 BPTT 토글: use_adjoint_for_bptt, last_n_grad_solver_steps.
     OCSC step 본체에서 매 step 직전 set 후 끝나면 reset.
   - FlowODE.generate: use_adjoint_for_bptt=True 면 model_fn 을 torch.utils.checkpoint
     으로 감싸 BPTT 메모리 ↓.  last_n_grad_solver_steps>0 이면 backprop_last_k 로 매핑.
   - HierarchicalFlowDecoder 에 ResidualFlowVelocityHead 추가
     (flow_dim → bottleneck → 4, last layer zero-init).  forward_components 로
     base/residual 분리 접근, forward 는 base+residual 합산 (zero-init 이라
     pretraining inference path 영향 없음).
src/smart/modules/flow_agent_decoder.py
   - SMARTFlowAgentDecoder (project_3 새 모델 본체).
     prepare_inference_cache, training_rollout_from_cache, sliding-window history.
   - self_forced_epoch / detach_block_transition / random terminal step 분기 모두 제거.
src/smart/modules/smart_flow_decoder.py
   - SMARTFlowDecoder wrapper (encoder.agent_encoder 계층).
src/smart/metrics/mmd_consistency_loss.py
   - mmd_from_stacked / mmd_per_rollout_proxy / mmd_precompute_sigma_sq.
   - per-channel-group split: pos (ch[0:2]) 와 heading (ch[2:4]) 각각 별도 sigma + RBF.
src/smart/metrics/hard_sim_agents_metrics.py
   - HardSimAgentsMetrics + _hard_load_and_log_feat_worker + _LIKELIHOOD_NAMES.
     OCSC_clean 측 monolithic metrics/__init__.py 에서 추출해 별도 파일로 분리.
src/smart/metrics/wosac_metametric_pytorch.py            # PyTorch hard RMM
src/smart/metrics/wosac_metametric_pytorch_differentiable.py
src/smart/metrics/wosac_metric_features_torch/          # PyTorch feature 12 파일
src/smart/utils/finetune.py
   - FinetuneConfig dataclass (67 fields, OCSC + dormant 다른 finetune mode 들).
   - parse_finetune_config / set_model_for_finetuning.  OCSC default 는
     flow_velocity_head_only=True → residual_velocity_head 만 frozen, 나머지
     velocity_head 만 unfreeze.
configs/experiment/flow_consistency_bptt.yaml          # OCSC 실험 config
scripts/train_flow_consistency_bptt_single.sh          # OCSC single-GPU 런처
scripts/launch_ocsc_clean_current.sh                   # 도는 GT-target 셋팅 핀
```

## Run

```bash
# OCSC fine-tuning (single GPU, OCSC_clean_v2 default)
# launcher default CKPT_PATH 가 85MB 를 가리키므로 87MB 로 반드시 override.
CKPT_PATH=/home2/pnc2/repos_python/project_3/logs/pretrained/epoch_last.ckpt \
  sh scripts/train_flow_consistency_bptt_single.sh

# 자주 쓰는 toggle (env override):
CUDA_VISIBLE_DEVICES=2 OCSC_GT_TARGET=false \
  CKPT_PATH=/home2/pnc2/repos_python/project_3/logs/pretrained/epoch_last.ckpt \
  sh scripts/train_flow_consistency_bptt_single.sh

OCSC_USE_MMD=true sh ...                  # MMD² (rel_disp/pos/heading_weight 무시)
OCSC_USE_MMD=false OCSC_POSITION_WEIGHT=1.0 sh ...   # paired L2 with channel weights
BPTT_LAST_COARSE_ONLY=false sh ...        # 모든 coarse step grad
BPTT_SEQUENTIAL_ROLLOUTS=true sh ...      # G rollout 순차 backward (메모리 ↓)
VAL_CHECK_INTERVAL=200 sh ...
TRAIN_B=8 VAL_B=8 sh ...                  # OOM 회피
MY_TASK_NAME=ocsc-... sh ...              # wandb run name
WANDB_PROJECT=My-Run sh ...

# Hard-RMM parity check (vs 공식 TF)
WOSAC_PARITY_TFRECORD_DIR=<dir> python scripts/verify_wosac_metametric_pytorch_parity.py --n 50
```

## OCSC 핵심 토글 (launcher default)

| key | default | 의미 |
|---|---|---|
| `OCSC_GT_TARGET` | true | GT 궤적 vs (false) ref decoder 의 OL sample |
| `OCSC_N_ROLLOUTS` (G) | 4 | scenario 당 closed-loop rollout 수 |
| `OCSC_ANCHOR_STRIDE` | 1 | 매 N번째 2Hz step 만 anchor |
| `OCSC_PRED_MAX_STEPS` | 2 | closed-loop coarse step (×0.5s) |
| `OCSC_USE_MMD` | false | true=split MMD² (pos/heading group 별 sigma), false=paired L2 |
| `OCSC_USE_PRETRAINED_REF` | true | frozen ref decoder 로 OL 생성 |
| `OCSC_POSITION_WEIGHT` | 0.0 | pos channel ([x/20,y/20]) — paired L2 / split-MMD 모두 active |
| `OCSC_REL_DISP_WEIGHT` | 1.0 | paired L2 분기 전용 (MMD 분기에선 무시) |
| `OCSC_HEADING_WEIGHT` | 0.0 | heading channel ([cos_h,sin_h]) — paired L2 / split-MMD 모두 active |
| `OCSC_FM_REG_LAMBDA` | 0.1 | GT FM regularization (batch-level 1회) |
| `BPTT_USE_ADJOINT` | true | flow_ode model_fn ckpt |
| `BPTT_LAST_COARSE_ONLY` | true | 마지막 1 coarse step 만 grad (warm = pred_steps - 1) |
| `BPTT_SEQUENTIAL_ROLLOUTS` | false | true 면 G rollout 순차 backward + 2-pass MMD |
| `FLOW_VELOCITY_HEAD_ONLY` | true | velocity_head 만 학습 (residual_velocity_head zero+frozen) |

**중요 (split MMD)**: `OCSC_USE_MMD=true` 분기는 channel group 별
(pos = `[x/20, y/20]`, heading = `[cos_h, sin_h]`) MMD² 를 따로 계산해
`OCSC_POSITION_WEIGHT`, `OCSC_HEADING_WEIGHT` 로 가중합.  group 별로
median-heuristic sigma 가 따로 잡혀서 position scale 이 heading channel 을
saturate 시키는 문제가 사라짐.  단 두 weight 가 모두 0 이면 loss=0 (no-op)
이므로 launcher 에서 적어도 하나는 명시적으로 양수로 set.
`OCSC_REL_DISP_WEIGHT` 는 paired L2 분기 전용 (MMD 에선 무시).

## 알려진 차이 (OCSC_clean 대비)

| 항목 | OCSC_clean | OCSC_clean_v2 |
|---|---|---|
| 모델 본체 | fix-hard-rmm 옛 모델 | project_3 새 모델 |
| 호환 ckpt | 85MB (`/project/logs/pretrained/`) | 87MB (`/project_3/logs/pretrained/`) |
| `last_n_grad_solver_steps` semantic | velocity 만 detach (x_t chain 유지) | x_t no_grad+detach (project_3 측 backprop_last_k 와 동일) |
| 다른 fine-tuning mode (adjoint_matching, kinematic_*, rmm_bptt_ft, ref_nll_ft, dice_ft, flow_epg_ft, flow_rwr_ft) | 활성 | dormant (FinetuneConfig field 만 존재, 분기 없음) |
| `wosac_metametric_api.py` | 존재 (dead code) | 미이식 (이름 mismatch dead code) |
| Train HardRMM monitoring (`_compute_ocsc_train_hard_rmm`) | 활성 (config 분기) | 통째 제거 (사용자 요청) |
| `data["tfrecord_path"]` / `data["scenario_id"]` 검증 | 강제 raise | 제거 (HardRMM 미사용) |

## Phase D smoke test 알려진 limitation

- **`BPTT_USE_ADJOINT=true` 가 `torch.utils.checkpoint` shape recompute
  mismatch 로 깨짐**. project_3 측 forward path 가 BPTT backward 시점의
  재연산과 텐서 shape 이 다른 corner-case 가 있음 (e.g. `forward_components`
  의 base+residual 결합 path). 회피: launcher 에 `BPTT_USE_ADJOINT=false`
  로 override 후 OCSC 학습 가능. 메모리 사용량은 약간 증가하지만 OCSC
  consistency loss 는 그대로 산출됨 (Phase D smoke test 5-step 검증:
  `train/consistency_loss=0.00014`).
- 추후 model_fn 의 deterministic recompute 보장 디버깅 필요.

## Validation backend

`model.model_config.validation_metric: "real" | "hard"`:
- `real`: 공식 TF SimAgentsMetrics (subprocess + TF, 정확하지만 비용 큼).
- `hard` (launcher default): PyTorch in-process RMM (`HardSimAgentsMetrics`).
  Parity 검증됨.

## Env vars (런처가 export)

| var | 소비처 |
|---|---|
| `OMP_NUM_THREADS` 등 BLAS thread | numpy/torch/MKL/OpenBLAS auto |
| `WOSAC_HARD_POOL_WORKERS` | hard RMM forkserver pool |
| `WOSAC_REAL_POOL_WORKERS` | official TF metric forkserver pool |
| `WOSAC_VERIFY` | hard vs real cross-check (디버그) |
| `CKPT_PATH` | **반드시 87MB 로 override** (default 가 85MB → 모델 mismatch) |

## Conventions

- 한 클래스 한 파일.
- Korean docstring, English code 주석/symbol.
- Hydra 진입점은 `src/run.py`, primary config `configs/run.yaml`.
- 새 finetune 모드 추가 시: `finetune_config.mode == "<name>"` 분기 패턴
  (`_is_ocsc_ft_enabled` 처럼).
- ckpt 가 OCSC_clean_v2 와 호환 안 되는 경우 (shape mismatch) 가 가장 흔한
  init 실패.  먼저 `CKPT_PATH` 가 87MB 인지 확인.
