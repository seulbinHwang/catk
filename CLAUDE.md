# Repo notes

CATK / SMART-Flow 기반 motion forecasting 코드베이스.  **현재 브랜치
`OCSC_clean`** 은 `origin/fix-hard-rmm` 의 src/configs/scripts 를 verbatim 복사한
뒤 두 개의 patch 가 추가된 상태:

1. **`d945af2` (5/8)**: OCSC nearest-match (M>G OL samples + per-anchor argmin paired L2)
2. **`642d91d` (5/10)**: `bptt_last_coarse_only` field 를 `FinetuneConfig` schema 에 추가
   — 이전엔 schema 누락으로 launcher 의 `BPTT_LAST_COARSE_ONLY=true` 가 silently False
   였음 (fix-hard-rmm 도 동일 버그).  fix 후 토글이 진짜 동작 → 마지막 1 coarse step
   만 grad 흘림.  ⚠ LR 민감 (lr=5e-6 + last_coarse_only=true 가 RMM 폭락 사례 있음).

## 작업 규칙 (반드시 준수)

1. **코드 변경 = commit + push 페어**.  변경 전 상태 commit, 변경 후 commit + push.
   중간 상태로 멈추지 말 것.
2. **GPU 할당** (5/10 갱신): GPU 2/3 우선.  GPU 0/1 은 사용자가 명시 허용한 경우만
   사용; 동료 사용 통보 받으면 즉시 종료.
3. **시간 표기는 KST (Asia/Seoul)**.  로그 파일명, commit 메시지, 보고문, 스케줄링 모두 KST.
4. **사용자 응답은 한국어**.  코드 symbol / 변수명 / log message / commit 메시지는 영어.
   docstring 은 한국어 (기존 컨벤션).

## Branches

- `main` — baseline.
- `origin/fix-hard-rmm` — OCSC reference (검증된 production 라인).  85MB ckpt 호환.
- **`OCSC_clean` — 현재 브랜치**.  fix-hard-rmm + 위 두 patch.  85MB ckpt 호환.
- `OCSC_clean_v2` (sister) — `8b2f9b7d` (self-forcing 라인) 위에 OCSC 알고리즘 surgical
  이식.  87MB ckpt 호환 (`/project_3/logs/pretrained/`).  추가 patch (verbose tracing,
  GT padding fix, `ocsc_loss_global_frame` toggle).  실험 결과 RMM 회귀 — 87MB 모델 +
  OCSC 가 잘 안 맞는 듯.  코드 구현이 잘못 되었을 수 있음.
- `OL_consistency_training` — 폐기 보존, reset 금지.

## 호환 ckpt

```
/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt   # 85MB, fix-hard-rmm/OCSC_clean 호환 (launcher default)
/home2/pnc2/repos_python/project_3/logs/pretrained/epoch_last.ckpt # 87MB, OCSC_clean_v2 전용 (다른 모델 arch)
```

OCSC_clean 은 반드시 85MB (default).  OCSC_clean_v2 는 87MB 로 override.

## 좌표계 / L2 semantics (사용자 자주 묻는 부분)

CL / GT / OL **세 trajectory 모두 같은 anchor-local normalized frame** 에서 L2:

- **CL**: rollout → world XY (commit_bridge 통해) → `_cl_to_norm` → anchor-local normalized
- **GT**: `tokenized_agent["gt_pos"]` 는 **raw continuous** (token vocab snap 없음),
  단 시간 spacing 만 2Hz (= every shift=5 of 10Hz).  → `_cl_to_norm` → anchor-local normalized
- **OL**: `flow_decoder.generate(x_init, _model_fn=lambda x_t,tau: flow_decoder(anchor_hidden, ...))`
  의 출력 자체가 anchor-frame normalized [x/20, y/20, cos, sin] (native).  변환 불필요.

`_cl_to_norm = _world_traj_to_flow_norm` = `transform_to_local(world_xy, world_head, current_pos, current_head)` + `pos/20` + `cos/sin(head_local)`.  per-agent rotation, 모든 timestep 이 같은 anchor reference.

**Time alignment**:
- CL `pred_traj_10hz[..., 4::5][:T_target]` = 2Hz at +0.5s, +1.0s (with pred_max_steps=2)
- GT `gt_pos[..., anchor+1:anchor+3]` = 2Hz at +0.5s, +1.0s
- 직접 매칭 (downsample 필요한 건 model 의 의사결정이 2Hz coarse step 이라서)

## 핵심 모듈

```
src/smart/model/smart_flow.py
   - _is_ocsc_ft_enabled, _run_flow_ocsc_ft_step (1815~)
     · anchor sequential 루프 + per-anchor 즉시 backward
     · MMD² (use_mmd=true) / paired L2 (use_mmd=false) 분기
     · GT vs OL target 분기 (ocsc_gt_target)
     · M_ol > G + nearest_match: argmin paired L2 target
     · _cl_to_norm closure: anchor-frame normalize
   - _world_traj_to_flow_norm (1266): transform_to_local + /20 + cos/sin
   - GT FM regularization 은 anchor 루프 후 batch-level 1회 (_compute_rmm_bptt_gt_fm_loss)
src/smart/modules/flow_local_decoder.py
   - FlowODE (use_adjoint_for_bptt, last_n_grad_solver_steps),
     ContinuousCommitBridge.commit (normalized → world XY)
src/smart/modules/flow_agent_decoder.py
   - SMARTFlowAgentDecoder: rollout_from_cache, _sample_open_loop_future_from_hidden,
     _build_rollout_noise_tape
src/smart/utils/finetune.py
   - FinetuneConfig dataclass + parse_finetune_config
   - 5/10 추가 field: bptt_last_coarse_only, ocsc_ol_nearest_match (둘 다 OCSC_clean 만)
src/smart/utils/rollout.py
   - transform_to_local (per-agent SE(2) inverse + rotation)
src/smart/tokens/token_processor.py / flow_token_processor.py
   - tokenize_agent: tokenized_agent["gt_pos"] = pos[:, ::shift] (raw continuous, 2Hz)
src/smart/metrics/mmd_consistency_loss.py / wosac_metametric_pytorch.py / wosac_metric_features_torch/
configs/experiment/flow_consistency_bptt.yaml   # OCSC 실험 config
scripts/train_flow_consistency_bptt_single.sh   # OCSC single-GPU 런처
configs/experiment/flow_dmd.yaml                # Self-Forcing DMD 실험 config
scripts/train_flow_dmd_single.sh                # Self-Forcing DMD single-GPU 런처
```

`_run_flow_dmd_ft_step` (smart_flow.py 신규): Self-Forcing DMD.  Anchor sequential
루프 + 2 manual_backward (gen / fake_score).  ref_flow_decoder 를 real_score 로 재사용,
__init__ 에서 deepcopy 한 self.fake_score_decoder 를 critic 으로 학습.  configure_optimizers
가 [opt_gen, opt_fake] tuple 반환 → training_step 의 DMD branch 가 두 opt.step() 분리.

## OCSC 핵심 토글 (launcher default)

| key | default | 의미 |
|---|---|---|
| `OCSC_GT_TARGET` | **false** | true=GT 궤적 / false=OL sample (default 5/11: OL setting) |
| `OCSC_GT_RESOLUTION` | 2hz | GT target 의 시간 해상도 ("2hz" 또는 "10hz" raw fine) |
| `OCSC_N_ROLLOUTS` (G) | 4 | 시나리오당 closed-loop rollout 수 |
| `OCSC_N_OL_ROLLOUTS` (M) | -1 (=G) | OL sample 수.  M>G 면 nearest match |
| `OCSC_OL_NEAREST_MATCH` | false | true=각 CL g 가 M 개 OL 중 argmin 으로 paired L2 |
| `OCSC_NEAREST_INCLUDE_GT` | false | nearest_match pool 에 raw 10Hz GT 1 개 추가 |
| `OCSC_STRICT_ACTIVE_MASK` | **true** | future fine step 모두 valid 인 agent 만 OCSC 학습 (main training 과 일관) |
| `OCSC_ANCHOR_STRIDE` | 1 | 매 N번째 2Hz step 만 anchor |
| `OCSC_PRED_MAX_STEPS` | 2 | CL coarse step (×0.5s) |
| `OCSC_USE_MMD` | false | true=split MMD² (pos/heading 따로 sigma) |
| `OCSC_USE_PRETRAINED_REF` | true | frozen ref decoder 로 OL 생성 |
| `OCSC_POSITION_WEIGHT` | 1.0 | pos channel L2 가중치 |
| `OCSC_HEADING_WEIGHT` | 0.1 | heading channel L2 가중치 |
| `OCSC_REL_DISP_WEIGHT` | 0.0 | rel_disp (delta-pos) L2; paired L2 분기 전용 |
| `OCSC_FM_REG_LAMBDA` | 0.1 | GT FM regularization |
| `BPTT_USE_ADJOINT` | true | flow_ode model_fn checkpoint |
| `BPTT_LAST_COARSE_ONLY` | false | 마지막 1 coarse step 만 grad — ⚠ schema fix 후 진짜 동작 |
| `FLOW_VELOCITY_HEAD_ONLY` | true | velocity_head 만 학습 |

## Self-Forcing DMD 핵심 토글 (mode `self_forcing_dmd`, launcher default)

알고리즘 요약:  generator (main flow_decoder) 가 CL rollout → x_gen;  τ~U sample 후
score networks (ref=real, fake=critic) 로 synthetic gradient `g=(1/β)·v_fake−v_real` 계산
→ MSE trick 으로 opt_gen update.  같은 anchor 에서 fake_score 를 x_gen.detach() 에 대한
FM loss 로 opt_fake update.  ref_flow_decoder 는 frozen, fake_score_decoder 는 __init__
에서 main 으로부터 deepcopy 후 on_train_start 에서 in-place state_dict sync.

| key | default | 의미 |
|---|---|---|
| `DMD_BETA` | **1.0** | entropy knob. <1 diversity↑ (smoothing), >1 sharpening |
| `DMD_N_ROLLOUTS` (G) | 1 | 시나리오당 closed-loop rollout 수 (variance reduction) |
| `DMD_PRED_MAX_STEPS` | 2 | CL coarse step (×0.5s) |
| `DMD_USE_REAL_SCORE` | true | frozen ref_flow_decoder 를 real_score teacher 로 사용 |
| `DMD_FAKE_LR_SCALE` | 1.0 | lr_fake = lr_gen × scale |
| `DMD_NORMALIZE` | true | Self-Forcing abs-mean normalizer |
| `DMD_ANCHOR_STRIDE` | 1 | 매 N번째 2Hz step 만 anchor |
| `DMD_STRICT_ACTIVE_MASK` | true | future fine step 모두 valid 인 agent 만 anchor |
| `DMD_WARMUP_FAKE_ONLY_STEPS` | 0 | 초기 N step fake_score-only (cold-start 안전) |
| `DMD_GEN_GRAD_CLIP` | 0.0 | gen 별도 clip (0 = BPTT_GRAD_CLIP_TRAJ 따름) |
| `DMD_EVAL_HARD_RMM` / `_INTERVAL` | true / 5 | HardRMM 모니터링 |
| `FLOW_FT_TARGET` | **full** | DMD generator 측은 decoder 전체 학습이 표준 |
| `BPTT_USE_ADJOINT` | true | 3× decoder 메모리 (gen+ref+fake) 부담 완화 |

**Logged keys (wandb)**:  `train/dmd/{gen_loss, fake_loss, score_diff_norm, v_real_norm,
v_fake_norm, normalizer_mean, beta, n_valid_anchors, skip_gen, hard_rmm}` + `train/loss` alias.

**β=1 sanity 기대값**:  pretrained 85MB 근처 RMM ≈ 0.7669 ±0.005 stable.  `score_diff_norm`
은 0 에서 시작해 학습 중 0.01~0.1 범위로 증가 (fake_score 가 generator 따라잡으며 분리되는 신호).

## 5/10 검증된 best 셋팅 (85MB OCSC_clean, M=24 OL, b32)

| task | LR | GT? | best RMM | step |
|---|---|---|---|---|
| `GT-lastF-lr5e6` | 5e-6 | true | **0.77340** ★ | 599 |
| `GT-lastF-lr1e6` | 1e-6 | true | 0.77287 (slow but stable) | 2599 |
| `M24-lastF-lr1e6` | 1e-6 | false | 0.77025 | 2399 |
| `M24-lastF-lr5e6` | 5e-6 | false | 0.77064 | 1199 |

baseline (pretrained 85MB) ≈ **0.7669**.  공통: TRAIN_B=32, VAL_B=16, last_coarse_only=false,
pos=1.0/h=0.01/rd=0, M=24/-1, fm_reg=0, anchor_stride=1, pred_max=2.

**불안정 사례** (실패): `lastT-lr5e6` (last_coarse_only=true + LR=5e-6) → RMM 0.7669→0.27 폭락
(schema fix 후 last_coarse 가 진짜 작동, gradient 발산).  `GT-lastF-lr5e5` (LR=5e-5) → RMM
초기 dip 후 0.762 정체.

## 알려진 fix / 패치 (OCSC_clean_v2, 비교 참고)

- GT padding fix: tokenizer 가 invalid GT 를 raw 0.0 으로 채움 → `_cl_to_norm` 후 50+ huge value
  → mask 적용 path (paired L2) 는 영향 없음, 하지만 MMD/aux logging 오염.  v2 commit `5db83d2`
  에서 `torch.where(_gt_valid, _gt_pos, current_pos_active.unsqueeze(1))` 로 수정.
  OCSC_clean 에는 미적용 (필요시 cherry-pick).
- `ocsc_loss_global_frame` toggle (v2 commit `bed4b64`): L2 만 world-frame 에서 측정.
  실험에서 결정적 효과 X (Adam 이 grad scale 불변), 폐기.

## Validation backend

`model.model_config.validation_metric: "real" | "hard"`:
- `real`: 공식 TF SimAgentsMetrics (느림, 정확)
- `hard` (launcher default): PyTorch in-process HardSimAgentsMetrics.  Parity 검증됨.

## Env vars (런처 export)

| var | 소비처 |
|---|---|
| `OMP_NUM_THREADS` 등 BLAS thread | numpy/torch/MKL/OpenBLAS |
| `WOSAC_HARD_POOL_WORKERS` | hard RMM forkserver pool (default 16, 학습 중엔 4 권장) |
| `WOSAC_REAL_POOL_WORKERS` | official TF metric forkserver |
| `WOSAC_VERIFY` | hard vs real cross-check (디버그) |
| `OCSC_VERBOSE` | (v2 only) 좌표 frame raw dump + per-channel loss decomp |
| `WANDB_RUN_ID` + `WANDB_RESUME=allow` | wandb 같은 run id 로 이어붙이기 |

## Conventions

- 한 클래스 한 파일.
- Korean docstring, English code 주석/symbol.
- Hydra 진입점 `src/run.py`, primary config `configs/run.yaml`.
- 새 finetune 모드 추가 시: `finetune_config.mode == "<name>"` 분기 패턴
  (`_is_ocsc_ft_enabled` 처럼).
- ckpt 가 OCSC_clean 와 호환 안 되는 경우 (shape mismatch) 가 가장 흔한 init 실패.
  OCSC_clean → 85MB, OCSC_clean_v2 → 87MB 확인.
