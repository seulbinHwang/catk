# Repo notes

CATK / SMART-Flow 기반 motion forecasting 코드베이스. **현재 브랜치
`OCSC_clean` 는 `origin/fix-hard-rmm` 의 src/configs/scripts 트리를
verbatim 복사**한 상태이며, OCSC (Open-Closed Self-Consistency)
fine-tuning 을 fix-hard-rmm reference 그대로 검증/실험하는 라인입니다.

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
- `origin/fix-hard-rmm` — OCSC reference (검증된 production 라인).
- **`OCSC_clean` — 현재 브랜치**.  `self_forcing_w_track_loss` 에서 분기했지만
  src/configs/scripts 가 fix-hard-rmm 와 byte-identical 하게 동기화됐고
  self-forcing 잔재는 모두 제거됨.  ckpt 호환성도 fix-hard-rmm 와 매칭.
- `OL_consistency_training` (별도 라인) — 이전 additive 통합 시도.  RMM
  regression 이 명확해서 사용자가 "fix-hard-rmm 복붙 재구현" 을 지시,
  현재는 사용 안 함.  reset 하지 말 것 (이전 commit history 보존).

## 호환 ckpt

```
/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt   # 85MB, fix-hard-rmm 호환 (launcher default)
```

`logs/pretrained/epoch_last.ckpt` (87MB, project_3 안) 은 OL_consistency_training
시점에 학습된 다른 모델로, fix-hard-rmm 의 agent_encoder shape 와 호환 안 됨
(token_emb / freqs.weight 차원 차이).  OCSC_clean 학습 / 평가에는 반드시
`/project/logs/pretrained/epoch_last.ckpt` 를 사용 (launcher default).

## Key modules

```
src/smart/model/smart_flow.py
   - _is_ocsc_ft_enabled, _run_flow_ocsc_ft_step (1786~)
     · OCSC training step 본체
     · anchor sequential 루프 + per-anchor 즉시 backward (peak mem O(1 graph))
     · MMD² (use_mmd=true) / paired L2 (use_mmd=false) 분기
     · GT vs OL target 분기 (ocsc_gt_target)
     · BPTT 토글: bptt_use_adjoint, bptt_last_coarse_only, bptt_warm_coarse_steps,
       bptt_last_n_solver_steps, bptt_last_n_coarse_steps, bptt_grad_clip_traj
   - GT FM regularization 은 anchor 루프 이후 batch-level 1회
     (_compute_rmm_bptt_gt_fm_loss)
src/smart/modules/flow_local_decoder.py
   - FlowODE (use_adjoint_for_bptt, last_n_grad_solver_steps 토글),
     ResidualFlowVelocityHead, HierarchicalFlowDecoder.forward_components
src/smart/modules/flow_agent_decoder.py
   - SMARTFlowAgentDecoder: rollout_from_cache, training_rollout_from_cache,
     prepare_inference_cache, _build_rollout_noise_tape, sliding-window history
src/smart/modules/smart_flow_decoder.py
   - SMARTFlowDecoder wrapper
src/smart/metrics/mmd_consistency_loss.py
   - mmd_from_stacked, mmd_per_rollout_proxy, mmd_precompute_sigma_sq
src/smart/metrics/wosac_metametric_pytorch.py
   - PyTorch hard RMM 본 계산기 (validation_metric=hard 일 때)
src/smart/metrics/wosac_metric_features_torch/
   - PyTorch feature 계산 (interaction / map / traffic light / trajectory)
configs/experiment/flow_consistency_bptt.yaml   # OCSC 실험 config
scripts/train_flow_consistency_bptt_single.sh   # OCSC single-GPU 런처
scripts/launch_ocsc_clean_current.sh            # 현재 도는 GT-target 셋팅 핀
```

## Run

```bash
# OCSC fine-tuning (single GPU, fix-hard-rmm production defaults)
sh scripts/train_flow_consistency_bptt_single.sh

# 자주 쓰는 toggle (env override):
CUDA_VISIBLE_DEVICES=2 OCSC_GT_TARGET=false sh scripts/train_flow_consistency_bptt_single.sh
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
| `OCSC_USE_MMD` | false | true=proper MMD², false=paired L2 |
| `OCSC_USE_PRETRAINED_REF` | true | frozen ref decoder 로 OL 생성 |
| `OCSC_POSITION_WEIGHT` | 0.0 | paired L2 분기에서만 효과 |
| `OCSC_REL_DISP_WEIGHT` | 1.0 | 동일 |
| `OCSC_HEADING_WEIGHT` | 0.0 | 동일 |
| `OCSC_FM_REG_LAMBDA` | 0.1 | GT FM regularization (batch-level 1회) |
| `BPTT_USE_ADJOINT` | true | flow_ode model_fn ckpt |
| `BPTT_LAST_COARSE_ONLY` | true | 마지막 1 coarse step 만 grad (warm = pred_steps - 1) |
| `BPTT_SEQUENTIAL_ROLLOUTS` | false | true 면 G rollout 순차 backward + 2-pass MMD |
| `FLOW_VELOCITY_HEAD_ONLY` | true | velocity_head 만 학습 (residual_velocity_head zero+frozen) |

**중요**: `OCSC_USE_MMD=true` 분기는 `pos/rel_disp/heading_weight` 토글을
**무시**하고 4ch norm tensor 전체를 RBF 거리에 넣음.  채널 가중치 효과를 보려면
`OCSC_USE_MMD=false` 로 paired L2 분기를 써야 한다.

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

## Conventions

- 한 클래스 한 파일.
- Korean docstring, English code 주석/symbol.
- Hydra 진입점은 `src/run.py`, primary config `configs/run.yaml`.
- 새 finetune 모드 추가 시: `finetune_config.mode == "<name>"` 분기 패턴
  (`_is_ocsc_ft_enabled` 처럼).
- ckpt 가 OCSC_clean 와 호환 안 되는 경우 (shape mismatch) 가 가장 흔한
  init 실패.  먼저 `CKPT_PATH` 가 fix-hard-rmm 호환인지 확인.
