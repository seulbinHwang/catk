# SMART 속도 최적화 후보 정리

기준:

- 비교 기준: `origin/SMART@77802ce` vs `origin/main@f2c206a`
- 작성 기준 SMART HEAD: `9d141db SMART 학습 속도 최적화 반영`
- 목표: 학습 성능은 거의 유지하면서, SMART pretrain 학습 속도를 줄일 수 있는 main 쪽 구현 후보를 정리한다.

## 후보 요약

| 후보 | main에는 있음 | SMART@77802ce에는 있음 | 현재 SMART 상태 | 학습 성능 영향 | 속도 기대 | SMART 적용 적합성 |
|---|---|---|---|---|---|---|
| unused attention_weight 저장 제거 | 있음 | 없음 | `d0ab85d` 묶음에 포함됐지만 `b5f1bcc`에서 revert됨 | 없음에 가까움 | 작음 | 단독 적용은 여전히 가능 |
| FourierEmbedding compile | 있음 | 없음 | `d0ab85d`에서 적용 후 `b5f1bcc`에서 revert됨 | 거의 없음, fp 오차 수준 | 중간 후보였으나 SMART A100 probe에서는 유의미 개선 미확인 | 재적용은 보류 |
| static map 유지 방식 | 있음 | 없음 | `9d141db`에서 적용됨 | 의도상 거의 동일 | 큼 | 적용 완료 |
| token processor local contour/direct vectorized 처리 | 있음 | 없음 | `9d141db`에서 적용됨 | 기본 `num_k=1`에서는 거의 동일 | 작음~중간 | 적용 완료 |
| bf16 + graph attention fp32 경계 | 있음 | 없음 | 미적용 | precision recipe 변화라 영향 가능 | A100에서 클 수 있음 | 현재 SMART `32-true` 학습에는 비추천 |

## 직접 실험해본 것

### 1. static map 유지 방식 + token processor local contour/direct vectorized 처리

적용 커밋:

```text
9d141db SMART 학습 속도 최적화 반영
```

개념:

- map token을 시간 step마다 복제하지 않고 static map token으로 유지한다.
- 여러 시간의 agent token이 같은 static map token을 참조하도록 map-agent edge를 구성한다.
- 기본 `num_k=1` agent token matching에서는 global contour를 만들었다가 다시 local frame으로 변환하는 중간 계산을 줄이고, previous-token local frame에서 직접 matching한다.

동등성 검증:

| 검증 항목 | 결과 |
|---|---:|
| token index mismatch | `0` |
| token position max diff | `3.814697265625e-6` |
| token heading max diff | `1.3113021850585938e-6` |
| static map old/new edge count | `6423 / 6423` |
| static map edge missing/extra | `0 / 0` |
| static map relation embedding max diff | `3.5762786865234375e-7` |

파이프라인 검증:

| 검증 | 조건 | 결과 |
|---|---|---|
| train smoke | `testa + testaa`, A100 8GPU, 3 train batches | 정상 완료 |
| validation smoke | `testa + testaa`, A100 8GPU, closed-loop validation 1 batch | 정상 완료 |
| 오류 확인 | OOM / RuntimeError / NCCL fatal error | 없음 |

속도 측정:

| 조건 | 속도 |
|---|---:|
| 변경 전 SMART speed probe | 약 `0.98~0.99 it/s` |
| 변경 후 80 train step probe | 약 `1.15~1.17 it/s` |

판단:

- train step 기준 약 `15~18%` 시간 단축을 기대할 수 있다.
- output-close 검증과 train/validation smoke를 통과했으므로 현재 SMART에 유지하는 것이 타당하다.

### 2. unused attention_weight 저장 제거 + FourierEmbedding compile 묶음

적용/제거 커밋:

```text
d0ab85d SMART 학습 속도 최적화 경로 추가
b5f1bcc Revert "SMART 학습 속도 최적화 경로 추가"
```

개념:

- `AttentionLayer`에서 학습 loss에 쓰지 않는 attention weight 저장을 제거한다.
- `FourierEmbedding`의 continuous embedding 계산을 `torch.compile` 경로로 묶는다.
- compile 이슈가 생기면 eager fallback으로 돌아가도록 했다.

실험 결과:

| 조건 | 결과 |
|---|---:|
| compile off speed probe | 약 `0.98 it/s` |
| compile on speed probe | 약 `0.98 it/s` |

판단:

- SMART A100x4x2 probe에서는 유의미한 속도 개선을 확인하지 못했다.
- 사용자 요청에 따라 `b5f1bcc`에서 revert했다.
- attention weight 제거 단독 효과는 작을 가능성이 있으나, 단독 실험은 따로 하지 않았다.

## 아직 실험하지 않았거나 현재 적용하지 않은 것

### 1. unused attention_weight 저장 제거 단독 적용

상태:

- main에는 있음.
- SMART에는 현재 없음.
- `d0ab85d` 묶음에는 포함됐지만, 단독으로 분리해서 속도/메모리 효과를 측정하지 않았다.

판단:

- 학습 성능 영향은 거의 없을 가능성이 높다.
- 다만 기대 속도 개선은 작다.
- 필요하면 단독 적용 후 짧은 A100x4x2 speed probe로 판단하는 것이 맞다.

### 2. FourierEmbedding compile 재적용

상태:

- main에는 있음.
- SMART에는 현재 없음.
- `d0ab85d`에서 적용했지만 A100x4x2 probe에서 유의미한 개선이 없었고, 이후 revert됐다.

판단:

- 이론적으로는 relation embedding 병목에 유효할 수 있다.
- 하지만 SMART 현재 recipe에서는 이미 A100 probe상 개선이 확인되지 않았다.
- 재적용하려면 `CATK_COMPILE_FOURIER_EMBEDDING` fallback, cudagraph 비활성화, 장기 학습 안정성 검증이 필요하다.

### 3. bf16 + graph attention fp32 경계

상태:

- main에는 있음.
- SMART에는 현재 없음.
- SMART 현재 A100 pretrain preset은 `precision=32-true`라 이 기능의 직접 적용 대상이 아니다.

판단:

- bf16 mixed precision 학습으로 recipe를 바꾸는 경우에는 A100에서 sparse graph attention penalty를 줄이는 데 의미가 있을 수 있다.
- 하지만 SMART 현재 recipe는 fp32 학습이므로, 이 기능을 넣어도 이득이 거의 없다.
- precision recipe 자체가 바뀌면 학습 성능과 수치 경로가 달라질 수 있으므로 현재 SMART에는 비추천이다.

## 현재 결론

| 분류 | 항목 |
|---|---|
| 유지 추천 | static map 유지 방식, token processor local contour/direct vectorized 처리 |
| 보류 | FourierEmbedding compile |
| 단독 실험 후보 | unused attention_weight 저장 제거 |
| 현재 비추천 | bf16 + graph attention fp32 경계 |

현재 SMART에서 실측으로 유의미한 속도 개선이 확인된 것은 `9d141db`의 static map 유지 방식과 token processor local-contour 최적화 조합이다.
