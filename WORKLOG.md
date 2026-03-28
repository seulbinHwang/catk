# Work Log

---

## 2026-03-27 — Final Step Terminal Cost Fine-tuning 검수 & 실행 시도

### 목표
`flow_terminal_cost_final_step.py` (DRAFT 구현체)를 검수하고, 버그 수정 후 fine-tuning 실행.

---

### 구현 개요

**방식**: ODE 샘플링 후 마지막 diffusion step에서만 reward gradient를 역전파.
- `residual_velocity_head` (5K params) 만 학습, 나머지 7.1M frozen
- Adjoint Matching(AM)과 달리 전체 SDE trajectory를 저장하지 않고, 마지막 step에서만 gradient 유지
- Open-loop (batch_idx % 2 == 0): GT context에서 anchor 독립 샘플링 → terminal cost minimize
- Closed-loop (batch_idx % 2 == 1): receding horizon 16 steps × 16-step SDE rollout → 각 step terminal cost 누적

**SDE rollout 핵심 로직** (`_rollout_memoryless_sde_last_step_grad`):
```python
for step_idx in range(rollout_steps):
    velocity_dict = flow_decoder.forward_components(...)
    drift = flow_ode.drift_from_velocity(x_t, velocity, tau)
    next_state = x_t + dt * drift + sqrt(dt) * sigma * noise

    if step_idx < rollout_steps - 1:
        current_state = next_state.detach()  # 중간 step: gradient 끊음
    else:
        current_state = next_state           # 마지막 step: gradient 유지
```

---

### 발견한 버그 & 수정 내역

#### 버그 1: `_build_step_times` range 오류 (수정 완료)
- **위치**: `flow_terminal_cost_final_step.py:60`
- **문제**: `range(self.rollout_steps + 1)` → 마지막 t=1.0 텐서가 생성되지만 loop에서 미사용
- **수정**: `range(self.rollout_steps)` 로 변경

#### 버그 2: NaN/Inf guard 없음 (수정 완료)
- **위치**: `flow_terminal_cost_final_step.py`
- **문제**: AM(`flow_adjoint_matching.py`)에는 `_assert_finite_tensor` 가 있으나 이 파일엔 없어 gradient가 NaN으로 폭발해도 silent fail
- **수정**: `_assert_finite_tensor` static method 추가, 3곳에서 호출
  - `_rollout_memoryless_sde_last_step_grad` 종료 후 → `final_state`
  - `forward_open_loop` → `terminal_cost`, `projection_gap`
  - `forward_closed_loop` → 각 horizon step의 `final_state`

#### 설계상 허용 가능 (수정 불필요)
- OT path에서 `drift_from_velocity` / `memoryless_sigma`의 beta 미적용: `sigma_min=1e-3` → 오차 < 0.1%, 무시 가능
- 중간 SDE 노이즈 unscaled: AM과 동일한 의도 (`rollout_noise_scale`은 초기 조건만 제어)
- `_zero_loss_with_trainable_dependency` 첫 param만 연결: gradient graph 유지에 충분

---

### Fine-tuning 실행 현황

**설정 파일**: `configs/experiment/am_finetune_flow_reward_final_step.yaml`
**실행 스크립트**: `scripts/train_flow_finalstep_reward.sh`
**체크포인트**: `logs/pretrained/epoch_last.ckpt`
**WandB**: `https://wandb.ai/se99an/clsft-catk`

**GPU 환경**: A100 × 2 (GPU 2, 3), 80GB each

#### 시도 1 — batch_size=8 (기본값)
- GPU 메모리: 4GB/80GB (5%), GPU util: ~50%
- 속도: ~0.4 it/s → epoch당 약 21시간 (30,438 steps)
- **중단**: 속도 문제로 kill

#### 시도 2 — batch_size=64 (진행 중 중단)
- 30 step benchmark 실행 중 user가 중단
- 첫 배치: ~78s (초기화 포함), 이후 steady-state 미측정

---

### 미해결 과제

#### GPU 활용률 최적화 (batch_size 증가)
- 현재 4GB/80GB → batch_size ≈ 64-128 per GPU 가능 (20x 증가 여지)
- 30K steps → batch=64 시 ~3,800 steps

#### Epoch 길이 단축
- `limit_train_batches=1.0` (전체) → 정수값(예: 200-400)으로 고정 필요
- 목표: 1시간/epoch
- 주 병목: closed-loop = 16 horizon steps × 16 rollout steps = 256 flow forward calls per step

#### 예상 설정 (미검증)
```sh
TRAIN_B=64 LIMIT_TRAIN_BATCHES=200 PREFETCH_FACTOR=4 MAX_EPOCHS=8 \
bash scripts/train_flow_finalstep_reward.sh
```
- 200 batches × (open ~8s + closed ~25s 평균 ~16.5s) ≈ 3,300s ≈ 55분 (추정)

---

### 관련 파일 목록

| 파일 | 역할 |
|------|------|
| `src/smart/modules/flow_terminal_cost_final_step.py` | 핵심 구현체 (이번 작업의 주 대상) |
| `src/smart/model/smart_flow.py` | training_step에서 분기 처리 |
| `src/smart/utils/finetune.py` | FinetuneConfig, set_model_for_finetuning |
| `src/smart/modules/flow_adjoint_matching.py` | AM 구현체 (참조용) |
| `src/smart/modules/flow_local_decoder.py` | FlowODE (drift, sigma 공식) |
| `configs/experiment/am_finetune_flow_reward_final_step.yaml` | 실험 설정 |
| `scripts/train_flow_finalstep_reward.sh` | 실행 스크립트 |
