# Overnight DMD 자율 탐색 로그 (2026-06-04 밤 → 06-05 09:00 KST)

## ★★★ 11:20 알고리즘 FIX (사용자 커밋 321ec50 "train DMD on raw control actions") ★★★
- **밤새 모든 결과(천장 0.78068)는 구버전(81d6ee2) 코드 — DMD/critic이 pose→control 역변환(round-trip, 0.5m 오차)된 control을 봤음.**
- **FIX**: rollout이 실제 실행한 raw control(`pred_control_10hz`)을 직접 DMD에 사용(`build_anchor_k_normalized_committed_control`, return_committed_control=True). kinematic은 action 뒤 state transition으로만(pose/head rollout 유지). round-trip 오염 제거.
- 새 테스트 5 passed. 재런치: GPU2 raw-ctrl except_map(5wrtua4v), GPU3 raw-ctrl flow-head(7v1ceetl). base g1e-6/f5e-4/cad5/s16/anchor1/fp32.
- ★핵심 질문: raw-control DMD가 천장 0.78068을 깨고 baseline 초과하는지. (밤새 천장은 이제 "구버전 결과"로 재해석 필요.)

- 11:45 사용자 지시: raw-control + except_map에서 critic(fake) lr 1e-6/1e-7. GPU2 f1e-6(n1iv2n6g), GPU3 f1e-7(n9nisecv). (raw-ctrl f5e-4는 ep149 max0.77996에서 중단.) 관전: 구버전 critic1e-7=0.76 급붕괴가 raw-control서 바뀌는지.

## ★★★ 12:35 핵심 발견: 밤새 탐색이 잘못된 regime이었음! ★★★
- 커밋 8c88d17이 **검증된 STABLE_RISING peak** 문서화: **gen lr 5e-5 · critic lr 1e-4 · updates4 · anchor4×stride4(cadence 16:1) · scope velocity_head_only · s16, β1.** RMM Δ=+0.019 상승, 크래시 없음. (단일-scene overfit sweep 전역 peak; 좌우 lr 4e-5 FLAT/1e-4 폭주, cadence 12·20 FLAT/8·48 crash로 둘러싼 좁은 peak.)
- **내 밤새 탐색(gen 1e-6/cadence5/anchor1/except_map)은 전혀 다른 저-lr regime → 그래서 천장 0.781에 막힌 것.** 상승 regime = **gen 5e-5 × cadence16 × velocity_head_only**. (밤새 velocity_head_only가 flat이던 것도 gen 1e-6이라 그랬던 것 — gen 5e-5면 상승.)
- 12:36 raw-control로 재현+탐색: GPU2 PEAK gen5e-5(6cbhzc0j), GPU3 gen6e-5(fs97ki0t). 둘 다 critic1e-4/updates4/anchor4s4/velhead/s16, floor 0.770. ★RMM 상승(0.785+) 여부 집중.

## ★★★ 13:44 새 알고리즘: OCSC GT-target global L2 (DMD 아님!) ★★★
- DMD 천장 우회 — student closed-loop rollout(2초)을 **GT future에 직접 L2 매칭**(teacher distill 아님 → 구조적으로 baseline 초과 가능).
- 구현: `_ocsc_world_traj_to_anchor0_pose_norm`에 `frame` 추가(global=raw world meter, 정규화X). config `ocsc_match_frame`. test 3 passed. 런처 `scripts/_ocsc_tv16.sh`.
- 세팅: flow head only(velocity_head_only)·lr 1e-6·G=4·gt_target=true·2초·tv16(Train=Val). 
- 13:44 GPU2 global L2(2i0xhjbm), GPU3 local L2 대조(md7qc8ru). ★RMM 상승 여부 핵심. floor 0.700.
- 13:55 사용자: lr 1e-4도 같이. GPU2 global lr1e-4(dy2mf23o), GPU3 local lr1e-4(ft86kdo3). [14:0x 사용자 지시로 중단]
- 14:05 사용자: lr1e-4 둘 종료, lr 1e-7 + except_map_encoder로 재런치. SCOPE knob 추가(_ocsc_tv16.sh). GPU2 global lr1e-7 except_map(dyzonhx6), GPU3 local lr1e-7 except_map(3lbp8d5p). 기존 lr1e-6 velhead 페어는 유지. scope 정상 확인(train_except_map_encoder 우선).
  현재: GPU2={lr1e-6 global velhead 2i0xhjbm, lr1e-7 global except_map dyzonhx6}, GPU3={lr1e-6 local velhead md7qc8ru, lr1e-7 local except_map 3lbp8d5p}. [lr1e-6 14:2x 중단]
- 14:22 사용자: lr1e-6 둘 종료, lr1e-7 except_map 같은 세팅에 batch(시나리오)=32. BATCH knob 추가. GPU2 global b32(rf3ek34o), GPU3 local b32(z7t89kj7). 기존 b16 유지 → batch16 vs 32 비교.
  현재: GPU2={global except_map b16 dyzonhx6, b32 rf3ek34o}, GPU3={local except_map b16 3lbp8d5p, b32 z7t89kj7}. 메모리 ~53GB/80GB, OOM 주시.

## 12:55 검증 peak가 tv16서 안 오른 원인 + warmup 재테스트
- 검증 peak(8c88d17)와 tv16 차이: **① estimator_warmup_steps 200 vs 0(tv16) ② 정상학습 vs 16-scene degenerate overfit(limit_batches=1).** random_terminal off는 양쪽 동일(full BPTT).
- 첫 시도(warmup0): peak gen5e-5 max0.77807 하락(ep112), gen6e-5 max0.77758 하락(ep99 트리거). → **상승 재현 실패.**
- 12:55 ESTIMATOR_WARMUP knob 추가, warmup=200 재테스트: GPU2 gen5e-5+w200(m4l7tsl6), GPU3 gen6e-5+w200(67nhv7ai), MAX_EPOCHS=600. tv16 1step/epoch라 ep200까지 critic warmup(RMM~baseline), ep200+ generator 학습 시 상승하는지 관찰. floor 0.770.
- ★ 만약 warmup으로도 tv16서 안 오르면 → degenerate overfit 환경 문제 → 검증 launcher train_kinematic_dmd.sh(정상학습, FAKE_WARMUP=200 기본)로 전환 필요.

## 13:13 새 커밋 4b18969 "project control actions before commit"
- 변경: commit 전 control을 `project_control_norm_to_kinematic_manifold`로 사영 — vehicle/cyclist lateral(delta_n) 채널=0 강제(non-holonomic), pedestrian holonomic 유지. raw-control DMD가 매칭하는 control을 물리적 feasible로 정제. (test 23 passed)
- 새 코드로 warmup200 peak 재런치: GPU2 gen5e-5(3j70evvp), GPU3 gen6e-5(fsh1vsad). [13:16 사용자 지시로 중단]
- 13:16 사용자 "warmup 빼고": GPU2 gen5e-5 w0(bmw2c6pf), GPU3 gen6e-5 w0(5kt2ownm). 새 코드(project+raw-ctrl), warmup0, MAX375. 첫 val(~ep10)부터 거동. project 코드가 이전 하락(구 raw-ctrl warmup0=하락) 개선하는지 확인.

## raw-control critic lr 결과 (12:0x, except_map, gen1e-6/cad5/s16/anchor1)
| critic lr | 트리거 ep / max RMM | 구버전 대비 |
|---|---|---|
| 1e-7 | ~ep119 / max 0.78076(전체 신기록, noise) | 구버전 0.76 급붕괴 → raw-ctrl 안 무너짐(개선!) |
| 1e-6 | ep109 / max 0.78057 | ~동급 |
| 5e-4 | ep149중단 / max 0.77996 | — |
→ **통찰17: raw-control fix가 낮은 critic lr(1e-7) 급붕괴를 제거(거동 개선). 단 천장 ~0.781 그대로, 0.785+ 없음.** 0.78076이 전체 최고지만 noise 범위.

## raw-control 첫 결과 (11:4x, base g1e-6/f5e-4/cad5/s16/anchor1)
| run | 트리거 ep / max RMM | 구버전 대비 |
|---|---|---|
| raw-ctrl flow-head | ep159 / max 0.77905 | 구 0.77889 ≈ 동급 |
| raw-ctrl except_map | 진행 ep104 / max 0.77996 | 구 0.78068보다 살짝↓ |
→ **통찰16: raw-control fix가 이 하이퍼파라미터에선 천장 못 깸(여전히 ~0.779-0.780 erode). 알고리즘은 더 정합적이나 RMM 거동 유사.** 단, 손실 지형이 바뀌었으니 **최적 하이퍼파라미터가 이동했을 수 있음** → raw-control 하에서 critic lr/cadence 재스윕 필요할 수 있음.

## ★★ 아침 요약 (FINAL, 04:1x — 구버전 코드 기준; 11:20 알고리즘 FIX로 재검증 중) ★★
**핵심 결론: 허용된 4축(train ODE step·precision·lr비율·cadence)만으로는 RMM이 baseline(0.779)을 명확히(0.785+) 넘지 못했다.**
**전체 run 통틀어 최고 RMM = 0.78068 (baseline +0.0017, noise 범위). 0.785+ 단 한 번도 없음. 총 ~20개 run 검증.**

전체 최고 RMM top5 (모든 run): 0.78068 / 0.78057 / 0.78054 / 0.78050 / 0.78044 — 전부 ~0.780, baseline와 사실상 동급.

확정된 최적값 (각 축):
- **critic(fake) lr = 5e-4** (∩형 최적: 1e-7 급락 / 1e-4 ep149 / **5e-4 ep289** / 1e-3 ep139). critic lr > gen lr 필수.
- **cadence = 5** (10·20은 critic 과학습으로 오히려 더 빨리 erode).
- **train ODE step**: 16=가장 높은 RMM이나 erode(ep289) / **4=가장 안정(750ep 완주 무붕괴, ~0.778)** / 8·32 더 나쁨. (val=16 고정)
- **precision**: fp32 ≥ fp16-mixed(미세 우세). fp16 NaN 없음.
- **gen_lr↓**(1e-6→5e-7→2e-7): 낮을수록 hold↑(더 안정), 천장은 baseline 그대로. 극한=frozen=baseline.

**가장 안정적(추천) config = g1e-6/f5e-4/cad5/s4 또는 gen_lr↓ 변형** — 붕괴 없이 baseline 유지. 하지만 **개선(상승)은 아님.**

**왜 baseline 초과 못 하나(구조적 천장):** DMD가 frozen teacher(=pretrained≈baseline)로 distill → 천장이 teacher. β=1 고정 하에선 generator가 teacher 분포를 재현할 뿐 초과 불가. erode는 critic이 stale/과격할 때 generator가 잘못 표류해 발생.

**RMM↑ 위한 추천(허용 축 밖, 사용자 판단 필요):**
1. **β>1 (sharpening)** — realism↑(diversity↓) 직접 lever. (이번엔 금지돼 미검증; DMD_BETA knob 준비됨.)
2. teacher를 baseline보다 나은 것으로 교체하거나, distill이 아닌 reward-기반(RMM 직접 최적화) objective.
3. scope·anchor 등 고정축 해제.

---

## 목표
**RMM(realism_meta_metric)이 명백히 baseline(~0.779) 위로 오르는 세팅을 찾는다.**
(pretrained baseline RMM ≈ 0.779. 지금까지 best case는 "baseline 유지", 초과 못 함.)

## 고정/공통
- setup: tv16 overfit (train=val=validation 16 scene, shuffle off, limit_batches=1), no-ckpt.
- gen_lr = 1e-6 (지금까지 고정). scope = except_map_encoder (별도 명시 없으면).
- 트리거: `RmmFloorStop` floor=0.775 → val(10ep마다)에서 RMM<0.775면 자동 종료(exit0) → 다음 런.
- script: `scripts/_dmd_tv16.sh`, knobs: GPU GEN_LR CRITIC_LR UPDATES(=cadence, n_anchors×UPDATES) N_ANCHORS ANCHOR_STRIDE SCOPE MAX_EPOCHS RMM_FLOOR DMD_BETA DMD_NORMALIZE.
- step 환산: epoch≈step, wandb _step≈epoch×5.35. MAX_EPOCHS=375≈2000 step. **개선 중이면 자유롭게 늘려도 됨(사용자 허락).**

## 확정된 통찰
1. **critic(fake) lr > gen lr 이어야 함.** critic이 stale하면 stale/biased gradient로 generator가 무방비 표류 → 붕괴. (작은 신호여도 방향이 틀려 망가짐; nonorm gen_loss 1e-5인데도 0.759로 붕괴가 증거.)
2. **critic이 더 빠르고 더 수렴할수록 RMM hold가 길어짐** (아래 표). f1e-7 급락 → f1e-4 ep149까지 → f5e-4 ep289까지.
3. 하지만 **vanilla β=1로는 baseline 초과 못 함** — DMD는 frozen teacher(=pretrained≈baseline)로 distill하므로 천장이 teacher. → **β>1 sharpening이 천장을 깰 유일한 후보(미검증).**
4. scope: except_map은 결국 erode(critic 빠를수록 느리게). head_only는 flat(무학습, 0.780 유지).
5. 진단은 `score_diff_norm`(raw fake−real)·`v_fake−v_real`로 봐야. normalize on의 `direction_abs_mean`/`gen_loss`는 per-agent normalizer(clamp1e-7) 폭발로 오염.

## 결과 표 (RMM = 최종/트리거 시점)
| run | gen/critic | cadence | scope | β | 결과 | wandb |
|---|---|---|---|---|---|---|
| head_only | 1e-6/1e-7 | 3 | velhead | 1 | 0.7802 flat(무학습) | o63e7d31 |
| except_map | 1e-6/1e-7 | 5 | except_map | 1 | 0.7614 ↓붕괴 | mnpuvp2f |
| nonorm | 1e-6/1e-7 | 5 | except_map | 1 | 0.7590 ↓(gradient≈0인데도 붕괴) | 6os7whh7 |
| f1e-4 | 1e-6/1e-4 | 5 | except_map | 1 | ep149 trig 0.7745 (ep110까지 0.779 유지) | 62swxmmw |
| f5e-4 | 1e-6/5e-4 | 5 | except_map | 1 | ep289 trig 0.7749 (ep240까지 0.780 유지) | jhbbzsi1 |
| cad200 | 1e-6/5e-4 | 200 | except_map | 1 | 진행중 | l26iy2iy |
| cad300 | 1e-6/5e-4 | 300 | except_map | 1 | 진행중 | mbcpk6ba |

## ★ 사용자 제약 업데이트 (22:55)
- **β 고정(=1, 건들지 말 것).** scope=except_map 고정. anchor1 고정. normalize 고정.
- 허용 탐색 축 **4개만**: ① **training ODE step**(`self_forced.sampling.sample_steps`, 기본16) ② **precision**(32-true/16-mixed/bf16-mixed) ③ **lr 비율**(gen:critic) ④ **cadence**(UPDATES).
- **validation ODE step = 16 고정** (`validation_rollout_sampling.sample_steps`, 스크립트가 안 건드려 자동 유지). train만 바꿀 것.
- 2000step 부족하면 MAX_EPOCHS 자유롭게 ↑ 허용.
- cad200/300은 6~9분/epoch로 너무 느려 폐기(22:56 kill). extreme cadence는 빠른 config서 유망하면 재방문.

## 계획 (웨이브) — 축: train ODE step × precision × lr비율 × cadence
base = g1e-6/f5e-4 cad5 (지금까지 best hold: ep289). 여기서 새 축 변동.
- **W1 (진행중, 22:57)**: GPU2 sample_steps=**32**/fp32 (`j9nbqjgp`), GPU3 sample_steps=16/**fp16-mixed** (`nb2s6sa4`).
- **W2**: 결과 보고 best ODE step에서 sample_steps=**8**(coarse) vs **64**, precision은 살아남은 것.
- **W3**: best(step,prec) × lr비율 강화(critic 1e-3) / cadence10.
- **W4**: best 조합 + MAX_EPOCHS↑ 수렴까지. RMM 0.785+ 뜨면 연장·재현.
- 천장 문제(β=1은 teacher 천장) 여전 — 상승 안 보이면 ODE step/precision이 closed-loop fidelity를 바꿔 teacher 초과하는지가 관건.

## 운영 규칙(자율)
- GPU 2,3만. 한 run 끝나면(트리거/자연) 결과를 이 파일에 추가하고 다음 웨이브 런치.
- 둘 다 바쁘면 ~30분마다 모니터링(RMM 추세/GPU/디스크). 명백히 hold-flat이라 목표(상승)에 무익하면 피벗.
- 디스크: no-ckpt라 안전하지만 wandb 미디어 누적 주의. (현재 여유 ~4T)
- RMM이 **0.785+ 명확히** 뜨면 = 성공 후보 → epoch 늘려 재현/연장하고 강조 기록.

## 결과 표 2 (새 축: train step/precision; base g1e-6/f5e-4 cad5 except_map β1)
| run | train ODE step | precision | 결과 | wandb |
|---|---|---|---|---|
| baseline | 16 | 32-true | ep289 trig 0.7749 (best hold) | jhbbzsi1 |
| s32 | **32** | 32-true | **ep139 trig 0.7749 — 훨씬 빨리 erode(나쁨)** | j9nbqjgp |
| fp16 | 16 | **16-mixed** | ep209 trig 0.7739 (fp32 ep289보다 약간 빠른 erode) | nb2s6sa4 |

## 추가 통찰
6. **train ODE step ↑(32) = 더 빨리 붕괴.** → step↓(8,4)이 hold 길어지는지가 다음 검증. (val은 16 고정이라 train<16이면 train/val mismatch — 방향 효과 확인 필요.)
7. **fp16-mixed 안전**(NaN 없음, graph attention fp32 강제 덕). 성능은 fp32와 큰 차이 없음.

## 결과 표 3 (train ODE step 스윕, base g1e-6/f5e-4 cad5 fp32)
| sample_steps | 트리거 ep / RMM | 비교 |
|---|---|---|
| 4 | 진행중(ep124 0.779 유지) | ≈baseline | do4kh4tg |
| 8 | ep139 / 0.7746 | 16보다 나쁨 | x633ykq6 |
| **16 (base)** | **ep289 / 0.7749** | **best hold** | jhbbzsi1 |
| 32 | ep139 / 0.7749 | 훨씬 나쁨 | j9nbqjgp |
→ **통찰8: train ODE step은 16(=val과 일치)이 best. 8·32 deviation은 더 빨리 erode. 4는 baseline급(진행중 확인).** 어느 것도 baseline 초과 못 함.

## 결과 표 4 (critic lr 스윕, base g1e-6 cad5 s16 fp32) — sweet spot 확정
| critic lr | 트리거 ep / 결과 | wandb |
|---|---|---|
| 1e-7 | ~ep180 0.76 급락 | mnpuvp2f |
| 1e-4 | ep149 / 0.7745 | 62swxmmw |
| **5e-4** | **ep289 / 0.7749 — best hold** | jhbbzsi1 |
| 1e-3 | ep139 / 0.7748 (너무 높음) | 7ix68g8x |
→ **통찰9: critic lr 최적 = 5e-4 (∩형, 너무 낮아도 높아도 빨리 erode).**
→ **통찰10(정정): sample_steps=4는 ep375 완주(0.777 안정), 16보다 안정적. 16은 높지만(0.780) ep289 erode. "stable vs high" trade.** 어느 것도 baseline(0.779) 명확 초과 못 함(천장=teacher, β1 고정).

## 진행 메모 (시간순 append)
- 22:37 cad200/cad300 런치 → 22:56 너무 느려 폐기.
- 22:57 W1: s32/fp16 런치. 23:25 s32 ep139 트리거 → GPU2 s8.
- 23:30 fp16 ep209 트리거(fp32 우세) → GPU3 s4.
- 23:38 s8 ep139 트리거 → GPU2 f1e-3.
- 00:00 f1e-3 ep139 트리거(5e-4가 최적 확정). s4 ep374 완주(안정). → cadence 축 시작: GPU2 cad10, GPU3 cad20 (둘 다 f5e-4 s16).
- 00:44 cad10 ep119 트리거(cad5 ep289보다 빨라 cadence↑ 나쁨). cad20도 하락중.
  → **통찰11: cadence 최적=5. critic lr 5e-4가 이미 freshness sweet → cadence↑는 critic 과학습→DMD 과격→더 빨리 erode.**
  → 미탐색 축 gen_lr↓로: GPU2 g5e-7/f5e-4 cad5 s16 런치.

- 01:00 cad20 ep119 트리거(cad10과 동일) → cadence 축 종결, **cad5 최적**. GPU3 free → s4 연장(750ep) 런치, GPU2 g5e-7 진행중.
- 01:14 점검: g5e-7 ep209 RMM~0.778(안정, gen lr↓로 더 안정), s4-750 ep109 RMM~0.778. 둘 다 baseline 초과 없음, 피크 ~0.780. 가동 중.
- 01:20 g5e-7 ep309 트리거(0.7742) — g1e-6 ep289보다 더 오래 버팀. **gen_lr↓ = 더 안정(hold↑)이나 결국 erode, 상승 없음.**
  → **★ 전체 run 최고 RMM = 0.7807 (baseline 0.779의 noise 범위). 0.785+ 단 한 번도 없음.** GPU2 free → gen_lr 축 마무리 g2e-7 런치.

- 01:58 점검: g2e-7 ep242 RMM~0.780(가장 안정), s4-750 ep599/750 RMM~0.777. 0.785+ 없음.
- 02:0x s4-750 **749ep 완주 무트리거**(max 0.7805, final 0.778) → **s4 = 안정성 winner(장기 무붕괴), 단 상승 없음.** GPU3 free → 미검증 조합 g5e-7×s4 750ep 런치.

- 03:28 안정-레퍼런스 점검: g2e-7×s4 ep281 max0.7804, g5e-7×s4 ep719/750 max0.7803. **0.785+ 없음, 결론 유지.**
- 03:5x g5e-7×s4 749ep 완주 무트리거(final 0.7766, max 0.7803). s4 조합 장기 안정 확인. **GPU3 idle** (탐색 완료, 과도 런치 자제). GPU2 g2e-7×s4만 유지(ep353/750).
- **최종 stable winner 후보: s4 계열(g1e-6 or gen_lr↓ /f5e-4/cad5/s4) — 750ep 무붕괴 ~0.777-0.780. RMM↑은 허용 축 내 불가.**
- 04:1x g2e-7×s4 749ep 완주(final 0.7776, max 0.7804, CPD 0.2051). **모든 run 종료, 양 GPU idle. 탐색 완전 종료.** 전체 최고 RMM 0.78068. 아침요약 FINAL 확정.

## 09:29 사용자 복귀 — anchor 축 재개 (제약 해제)
- 사용자 지시: best 세팅에 anchor4. anchor1 고정 해제됨.
- GPU2 s16+anchor4(z8gd9063), GPU3 s4+anchor4(wdod66sq). 둘 다 g1e-6/f5e-4/cad5/fp32, n_anchors=4 stride=4(token 0/4/8/12), val16, floor0.775.
- 관전: 다중 anchor 평균이 DMD direction variance↓로 baseline 초과 가능한지.
- 09:57 점검: s16+anchor4 ep67 ~0.777(살짝 erode), s4+anchor4 ep75 ~0.780 안정(max 0.7806).
- 10:2x s16+anchor4 **ep119 트리거**(max 0.7793) — anchor1 s16(ep289)보다 빨리 erode(effective cadence20 효과, cad20과 동일). s4+anchor4 ep148 max0.7806 ~baseline 유지.
  → **통찰13: anchor4(다중평균)도 천장 0.7807 못 깸. s16엔 해롭고 s4엔 중립.** anchor 축도 baseline 초과 불가.
- 추가 논의(10:1x): self-forced rollout BPTT가 block(0.5초) 단위 truncated(detach_training_rollout_state) — backward credit assignment 부재가 천장 원인 가능성. kinematic 변환은 block 내에선 미분되며 탐(control→pose→control 라운드트립). 구조 변경(멀티-block BPTT) 후보.

## 10:1x 정정 + backprop_last_k 축 (새 발견)
- **정정: 밤새/현재 run은 이미 full BPTT였음.** detach_block_transition=false(게이트 off), backprop_last_k=None(미설정+random_terminal off) → block·ODE step 전체 미분. 내가 truncation 활성이라 한 건 오류(함수 존재만 보고 단정). 천장 0.78068은 이미 full-grad 결과.
- **backprop_last_k 구현**(flow_local_decoder.py:339-366): forward 16 step 다 돎, 마지막 K step만 grad(앞 16-K는 no_grad+detach). knob=self_forced.sampling.backprop_last_k. 현재 None=16 전체.
- 10:26 BACKPROP_K knob 추가. last-4(GPU2 v4sbsrwy)/last-8(GPU3 kiswqi6q) 런치. base g1e-6/f5e-4/cad5/s16/anchor1/fp32. full(16)=0.78068 대비 후반부만 미세조정 효과 보는 중.

## 결과 표 6 (backprop_last_k, base g1e-6/f5e-4/cad5/s16/anchor1)
| backprop_last_k | 트리거 ep / max RMM | wandb |
|---|---|---|
| 4 | ep109 / max 0.78004 | v4sbsrwy |
| 8 | ~ep142 진행 / max 0.77971 | kiswqi6q |
| 16(full) | ep289 / max 0.78068 | jhbbzsi1 |
→ **통찰14: backprop_last_k↓ = 약간 빨리 erode·max 약간↓. full(16)이 근소 우세. 천장 못 깸.**

## 10:5x scope=full_flow_decoder (flow-head only) 테스트
- 사용자 지시: bk4/bk8을 scope만 flow-head(full_flow_decoder)로. encoder 전체 freeze, flow decoder만 학습.
- GPU2 flowhead+bk4(q8p3a038), GPU3 flowhead+bk8(l97w0u7p). base g1e-6/f5e-4/cad5/s16/anchor1/fp32.
- 관전: except_map(인코더 학습→erode)과 달리 인코더 안 건드려 더 안정/개선 가능한지 vs head만으론 학습력 부족(밤새 velocity_head_only는 flat).

## 결과 표 7 (scope=full_flow_decoder, flow-head only, base g1e-6/f5e-4/cad5/s16)
| scope+bk | 트리거 ep / max RMM | wandb |
|---|---|---|
| flowhead bk8 | ep99 / max 0.77972 | l97w0u7p |
| flowhead bk4 | ~ep137 진행 / max 0.77889 | q8p3a038 |
→ **통찰15: flow-head only도 erode, max RMM이 except_map(0.7807)보다 더 낮음. 인코더 freeze해도 flow decoder 단독으로 DMD 신호 하에 하락 → 붕괴는 인코더 오염만의 문제 아님.** scope 3종(except_map/velocity_head/full_flow_decoder) 모두 baseline 초과 못 함.

## 아침에 사용자에게 보고할 한 줄
허용 4축(train ODE step·precision·lr비율·cadence, β/scope/anchor 고정) 풀 sweep 완료 → **RMM은 baseline 0.779 유지가 한계(최고 0.7807), 0.785+ 불가**. critic lr 5e-4·cadence 5·sample16(높음)/4(안정)·fp32·gen_lr↓(안정)이 최적이지만 전부 "유지"지 "상승" 아님. **상승하려면 금지했던 β>1(sharpening) 또는 distill이 아닌 RMM-direct reward objective 필요.** s4 계열이 가장 안정(750ep 무붕괴).

## 결과 표 5 (gen_lr 스윕, f5e-4 cad5 s16)
| gen_lr | 트리거 ep | 비교 |
|---|---|---|
| 1e-6 | ep289 | base |
| 5e-7 | ep309 | 더 안정 |
| 2e-7 | ep599 (max 0.7804) | 가장 안정 |
| 2e-6 | **ep69 (max 0.7789)** | gen_lr↑ = 급속 붕괴 |
→ **통찰12(완결): gen_lr 축 = 단조. ↓일수록 안정(2e-6:69 < 1e-6:289 < 5e-7:309 < 2e-7:599 trigger ep). 어느 쪽도 baseline 초과 없음. 최적 안정점 = gen_lr 최소(+s4).**
→ **★ 단일축 탐색 완전 소진. 허용 4축 모두 baseline 유지가 최선, 0.785+ 불가 확정.**

## ★ 중간 결론 (00:45, 01:00 갱신)
모든 축 sweep 결과 **best config = g1e-6/f5e-4/cad5/s16(또는 s4)/fp32**, 전부 **baseline 0.779 근처 plateau(~0.777~0.780), 명확한 초과(0.785+) 없음.** 구조적 천장=frozen teacher(β=1 distill). 허용 축(ODE step/precision/lr/cadence)으로는 teacher 초과가 구조적으로 어려움 — **β>1(sharpening, 현재 금지)이 천장 깰 유력 lever**. 남은 시도: gen_lr↓(가장 안정점 찾기), best config MAX_EPOCHS 대폭 연장(혹시 매우 느린 상승), s4×다른조합.
