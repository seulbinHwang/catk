# Repo notes

CATK / SMART-Flow 기반 motion forecasting 코드베이스. 현재 브랜치는 OCSC
(Open-Closed Self-Consistency) 파인튜닝을 self-forcing + track-loss 라인 위에
얹은 상태입니다.

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
- `self_forcing` 계열 (`self_forcing`, `self_forcing_bugfix`, `self_forcing_7m`,
  `self_forcing_w_road`, `self_forcing_w_track_loss`) — self-forced + track-loss
  학습 라인. 현재 작업 브랜치 `OL_consistency_training` 의 부모는
  `self_forcing_w_track_loss`.
- `OL_consistency_training` — **현재 브랜치**. self-forcing 코드를 보존한 채로
  OCSC 인프라를 *additive* 방식으로 얹었음. 절대 self-forcing 경로를 부수지 말 것.
- `origin/fix-hard-rmm` — OCSC 의 reference 구현. 디코더 API 가 더 발전된
  대신 self-forcing 코드는 없음. 가져올 때 verbatim copy 가 아니라 **현재
  브랜치 primitives 위에 어댑터로 통합**한 형태가 정답.

## Key modules

```
src/smart/model/smart_flow.py          # SMARTFlow LightningModule
                                       # OCSC 진입점: _is_ocsc_ft_enabled,
                                       # _training_step_ocsc_ft, _run_flow_ocsc_ft_step
src/smart/modules/flow_agent_decoder.py
   - rollout_from_cache (no-grad, validation/inference)
   - training_rollout_from_cache (gradient 유지, OCSC + self-forcing 공용)
   - _rollout_from_cache_impl (둘의 공용 backbone, 539 lines)
   - _build_rollout_noise_tape (share_noise_across_time 토글)
src/smart/modules/flow_local_decoder.py
   - FlowODE.generate (use_adjoint_for_bptt, last_n_grad_solver_steps,
     기존 backprop_last_k 와 별도 동작)
src/smart/modules/self_forced_*.py     # 기존 self-forcing 학습 경로 (보존)
src/smart/metrics/
   - sim_agents_metrics.py  # 공식 TF (subprocess + forkserver)
   - hard_sim_agents_metrics.py  # PyTorch in-process. SimAgentsMetrics 와 동일 인터페이스
   - wosac_metametric_pytorch{,_differentiable}.py  # 본 RMM 계산기 (TF 없이)
   - wosac_metric_features_torch/  # PyTorch feature 계산
   - mmd_consistency_loss.py  # OCSC consistency MMD²
   - wosac_distribution_metrics.py  # CPD / CES / DPR
configs/experiment/flow_consistency_bptt.yaml  # OCSC 실험 config
scripts/train_flow_consistency_bptt_single.sh  # OCSC single-GPU 런처
```

## OCSC 통합 원칙

1. **기존 self-forcing/track-loss 코드 절대 수정 금지**. OCSC 는 추가-only.
2. 디코더 신규 파라미터는 default no-op 로 추가 (`warm_coarse_steps=0`,
   `noise_tape_override=None`, `share_noise_across_time=False`,
   `bptt_grad_clip_traj=0.0`).
3. FlowODE BPTT 속성 (`use_adjoint_for_bptt`, `last_n_grad_solver_steps`)
   은 OCSC step 진입 시 활성, finally 절에서 원복.
4. `automatic_optimization=False` 는 `bptt_sequential_rollouts=true` 일 때만.
5. `_run_flow_ocsc_ft_step` 은 단일 함수 (~400 lines) 로 유지. 분리 금지.

## Validation backend

`model.model_config.validation_metric: "real" | "hard"`:
- `real` (default): 공식 TF SimAgentsMetrics. 정확하지만 subprocess + TF 비용.
- `hard`: HardSimAgentsMetrics (PyTorch in-process). 동일 인터페이스 (drop-in
  replace). Parity 검증됨 (max delta ≈ 6e-8 vs official, see Phase 1-3).

## Diversity metrics

`WOSACDistributionMetrics` 가 `__init__` 에서 자동 wiring. CPD / CES 는 항상
계산, DPR 은 `model_config.wosac_cpd_reference: <float>` 설정 시에만.
RMM backend 와 무관하게 closed-loop rollout 끝나면 `pred_traj` 만으로 갱신.

## Run

```bash
# OCSC fine-tuning (single GPU, fix-hard-rmm production defaults)
sh scripts/train_flow_consistency_bptt_single.sh

# 자주 쓰는 토글:
OCSC_GT_TARGET=false sh ...           # GT 대신 open-loop sample target
OCSC_USE_MMD=true sh ...              # MMD² loss
BPTT_SEQUENTIAL_ROLLOUTS=true sh ...  # G rollout 순차 backward + manual opt
VALIDATION_METRIC=real sh ...         # 공식 TF 검증
WANDB_PROJECT=My-Run sh ...

# Hard-RMM parity check (vs 공식 TF)
WOSAC_PARITY_TFRECORD_DIR=<dir> python scripts/verify_wosac_metametric_pytorch_parity.py --n 50
```

## Pretrained checkpoint

`logs/pretrained/epoch_last.ckpt` (about 87 MB). 런처가
`SCRIPT_DIR + ../logs/pretrained/epoch_last.ckpt` 로 자동 해결.

## Env vars (런처가 export)

| var | 소비처 |
|---|---|
| `OMP_NUM_THREADS` 등 BLAS thread | numpy/torch/MKL/OpenBLAS auto |
| `WOSAC_HARD_POOL_WORKERS` | `hard_sim_agents_metrics.py` |
| `WOSAC_REAL_POOL_WORKERS` | `sim_agents_metrics.py:_resolve_sim_agents_metric_workers` (env > yaml) |
| `WOSAC_VERIFY` | hard vs real cross-check (디버그) |
| `WOSAC_HARD_LOG_CACHE_DIR` | hard-RMM log feature 디스크 캐시 |

## Conventions

- 한 클래스 한 파일 (e.g. `HardSimAgentsMetrics` 는 `__init__.py` 가 아니라
  `hard_sim_agents_metrics.py`).
- Korean docstring, English code 주석/symbol.
- Hydra 진입점은 `src/run.py`, primary config `configs/run.yaml`.
- 새 finetune 모드 추가 시: `finetune_config.mode == "<name>"` 분기 패턴 따라가기
  (`_is_ocsc_ft_enabled` 처럼).
