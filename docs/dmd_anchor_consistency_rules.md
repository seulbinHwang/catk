# DMD GT-grounded per-anchor rollout — 정합성 검증 규칙

OCSC `self_forcing_dmd` 의 multi time-anchor 를 현재 kinematic 모델/inference 위에
GT-grounded(🅐) 로 이식할 때 만족해야 하는 규칙. 구현 후 이 규칙들로 검증한다.

## 배경
- 구 동작(🅑): 단일 closed-loop rollout 에서 `anchor_idx*stride` 만큼 들어간 window 를
  잘라 anchor 로 사용 → `anchor_idx>0` 은 generator 자기 rollout 의 drift 위에서 출발.
- OCSC(🅐): anchor 마다 **GT history 로 새로 출발**하는 독립 rollout.
- 가능성 근거(코드): `gt_pos = pos[:, coarse_end_steps]` 는 full-episode coarse GT.
  `coarse_end_steps = arange(shift, n_step, shift)` → coarse step `c` = 10Hz step `(c+1)*shift`.
  anchor k current(10Hz) = `raw_current_steps[k] = shift*(k+2)` = coarse step `k+1`.
  `step_current_2hz = (num_historical_steps-1)//shift = 2`.

## 규칙 (VR)

- **VR1 (GT-grounded 출발)**: anchor k rollout cache 의 current 상태(pos_window[:, -1],
  head_window[:, -1])는 GT 의 anchor k current(coarse step `k+1`) 상태와 일치.
  (= gt_pos[:, k+1], gt_heading[:, k+1]). self-rollout drift 미사용.
- **VR2 (anchor 독립성)**: anchor 마다 독립 cache·독립 rollout. anchor k 의 x_gen 은
  다른 anchor 의 rollout 에 의존하지 않는다.
- **VR3 (coarse window 정합)**: anchor k 의 cache coarse history window =
  `gt_pos[:, k : k+step_current_2hz]` (full-episode gt_pos 에서 anchor offset k 슬라이스).
- **VR4 (fine history 정합)**: anchor k 의 `rollout_init_fine_*_history` 는 10Hz raw 에서
  current_raw_step=`shift*(k+2)` 기준 최근 `shift+1`(=6) 스텝, 마지막이 anchor k current.
- **VR5 (frame 정합)**: `_pack_self_forced_committed_rollout(rollout_k, anchor_idx=0)` 의
  정규화 기준 frame(anchor k current pos/head) == score 평가
  `path_flow_velocity_for_anchor_k(anchor_idx=k)` 의 conditioning anchor frame.
  (둘 다 anchor k 의 GT current 좌표계여야 x_gen 과 score 가 같은 frame.)
- **VR6 (n_anchors=1 무회귀)**: `n_anchors=1` 이면 per-anchor 경로의 anchor 0 결과가
  기존 base rollout(anchor 0) 과 bit-identical.
- **VR7 (active mask)**: anchor k 는 anchor k current 가 valid 인 agent 만 (strict 면
  future window 전부 valid 인 agent 만) 포함.
- **VR8 (DMD math 불변)**: per-anchor 전환이 DMD synthetic-gradient 수식
  (pred_x0=β·x_t+σ·v, g=(1/β)x0_f − x0_r, normalizer=|x0_r|.mean, target=(x_gen−g_n).detach,
  L=0.5·MSE) 을 바꾸지 않는다.
- **VR9 (conditioning 공유)**: anchor 내 real/fake/gen 의 인코더가 frozen-identical
  (`unfrozen_range=full_flow_decoder`) → 동일 conditioning. (기존 audit [A] 유지.)
- **VR10 (finite/shape)**: 모든 anchor 의 x_gen/loss 가 finite, shape 일관.

## 검증 방법
- 단위(합성 텐서, GPU 불필요): VR3/VR4 — per-anchor 토큰 빌더 헬퍼의 슬라이스·fine-history
  인덱스 검증. VR6 — anchor 0 경로 일치. VR1 — cache current == gt 일치(작은 stub).
- 통합(smoke, GPU): VR1/VR5/VR7/VR10 — 실제 model+data 로 anchor k rollout 의 current
  상태와 frame, mask, finite 확인.
- 구조(by construction): VR2/VR8/VR9.
