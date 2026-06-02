# TrajTok 논문과 현재 구현의 vocab 생성 차이

이 문서는 `trajtok` 브랜치의 최신 코드 기준으로, TrajTok 논문
([arXiv:2506.21618](https://arxiv.org/pdf/2506.21618))의 설명과 현재
vocabulary 생성 구현이 어디에서 다르게 해석될 수 있는지 쉽게 정리한 문서다.

핵심은 모델 구조가 아니라 **궤적 token vocabulary를 어떻게 만드는가**이다.
TrajTok에서는 모델이 직접 연속 궤적을 새로 그리는 것이 아니라, 미리 만들어 둔
궤적 token 후보 중 하나를 고른다. 따라서 vocab을 어떻게 만들었는지가 학습과
closed-loop rollout 품질에 영향을 줄 수 있다.

## 전체 결론

현재 구현과 논문은 큰 방향에서는 같다.

- WOMD 로그에서 0.5초 길이의 agent trajectory를 모은다.
- trajectory endpoint를 기준으로 x-y grid에 넣는다.
- 충분히 의미 있는 grid를 고른다.
- 각 grid마다 대표 trajectory token을 만든다.

다만 논문 문장을 아주 엄격하게 읽으면, 현재 코드에는 수정 후보가 세 가지 있다.

| 항목 | 쉽게 말한 차이 | 논문 문장 그대로라면 |
|---|---|---|
| 데이터 사용량 | vocab 생성에 쓸 로그 파일/trajectory 수를 제한한다 | 제한 없이 WOMD trajectory를 쓰는 쪽이 더 자연스럽다 |
| empty grid endpoint | grid 칸의 시작점에 가까운 좌표를 endpoint로 쓴다 | grid 칸의 중심점을 endpoint로 쓰는 쪽이 더 자연스럽다 |
| non-empty token 생성 | 평균 trajectory를 만든 뒤 endpoint/yaw를 후처리한다 | 해당 grid의 실제 trajectory 평균을 그대로 쓰는 쪽이 더 직접적이다 |

아래에서 각 항목을 개념적으로 설명한다.

## 1. 데이터 사용량

논문은 WOMD에서 0.5초 trajectory를 추출한다고 설명한다. 즉 기본 아이디어는
실제 주행 로그에서 짧은 행동 조각을 많이 모아 vocab 후보를 만드는 것이다.

현재 코드는 같은 방식으로 0.5초 trajectory를 모으지만, vocab 생성 시 기본적으로
사용량 제한을 둔다.

- 최대 파일 수: `50,000`
- agent type별 최대 trajectory 수: `12,000,000`
- 좌우 flip을 추가하면 type별 후보는 최대 `24,000,000`개까지 늘 수 있다.

쉽게 말하면, 논문은 "WOMD trajectory를 모은다"고만 말하고, 현재 코드는
"너무 많으면 일정 개수까지만 모은다"는 안전장치를 둔 것이다.

이 제한은 계산 시간과 메모리를 줄이는 데 도움이 된다. 하지만 제한 때문에
아주 드문 행동이 vocab 생성에 덜 반영될 가능성도 있다. 예를 들어 드문 회전,
특이한 차선 변경, 드문 보행자 움직임 같은 행동이 충분히 들어오지 않을 수 있다.

논문 본문에는 `50,000`이나 `12,000,000` 같은 cap 값이 명시되어 있지 않다.
따라서 논문 문장 그대로만 따른다면 cap을 끄거나, 적어도 실험 옵션으로 두는
쪽이 더 보수적인 재현이다. 다만 실제 제출용 내부 코드에 비공개 cap이 있었는지는
논문만으로 알 수 없다.

## 2. token endpoint 위치

TrajTok vocab은 trajectory의 마지막 위치, 즉 endpoint를 기준으로 grid를 만든다.
여기서 중요한 질문은 이것이다.

> grid 칸 하나를 대표하는 endpoint 좌표를 어디로 둘 것인가?

논문은 빈 grid에서 endpoint가 grid cell center라고 설명한다. 말 그대로 해석하면
각 grid 칸의 가운데 점을 endpoint로 써야 한다.

현재 코드는 `x_min + idx * interval` 형태의 좌표를 쓴다. 일반적인 grid 해석으로는
이 값이 cell center라기보다 cell 시작점에 더 가깝다.

차이는 크지 않다.

| agent | 현재 코드 endpoint | cell center 해석 | 좌표 차이 |
|---|---:|---:|---:|
| vehicle x | `x_min + x * 0.1` | `x_min + (x + 0.5) * 0.1` | `+0.05m` |
| vehicle y | `y_min + y * 0.05` | `y_min + (y + 0.5) * 0.05` | `+0.025m` |
| pedestrian/cyclist x,y | `min + idx * 0.05` | `min + (idx + 0.5) * 0.05` | `+0.025m` |

즉 이 차이는 센티미터 단위다. WOSAC RMM을 크게 바꿀 정도의 차이라고 단정하기는
어렵다. 다만 논문의 "cell center"라는 문장을 문자 그대로 구현하려면
`idx + 0.5` 방식이 더 자연스럽다.

한 가지 주의할 점은 현재 코드가 endpoint bin을 `round()` 기반으로 정한다는 것이다.
그래서 작성자는 `idx * interval + min`을 round 기반 bin의 대표점으로 의도했을
가능성이 있다. 따라서 이 항목은 논문 표현과 코드 해석의 차이라고 보는 것이 맞다.

## 3. non-empty token 생성

이 항목이 세 가지 중 가장 직접적인 차이다.

non-empty grid란 해당 grid 안에 실제 로그 trajectory가 들어온 경우를 말한다.
논문은 이 경우 token을 단순하게 정의한다.

> 그 grid에 들어온 실제 trajectory들의 평균을 token으로 쓴다.

쉽게 말하면 다음과 같다.

```text
어떤 grid 안에 실제 trajectory 100개가 있다.
그러면 그 100개 trajectory의 평균 모양을 대표 token으로 쓴다.
```

현재 코드도 먼저 평균 trajectory를 만든다. 여기까지는 논문과 같다. 하지만 그 뒤에
두 가지 후처리가 들어간다.

첫째, 평균 trajectory의 마지막 `x,y`를 grid endpoint 좌표로 덮어쓴다. 즉 실제
평균이 도착한 마지막 위치를 그대로 쓰지 않고, grid가 대표하는 endpoint 위치로
맞춘다.

둘째, 평균 trajectory의 yaw 변화가 너무 크다고 판단되면 평균 token을 버리고,
가까운 grid의 trajectory yaw를 이용해 보간 token을 만든다.

따라서 논문식과 현재 코드식의 차이는 이렇게 볼 수 있다.

| 방식 | 의미 |
|---|---|
| 논문 문장 그대로 | 실제 로그 trajectory 평균을 그대로 token으로 사용 |
| 현재 코드 | 평균을 만들되, endpoint와 yaw가 불안정하면 grid에 맞게 보정 |

현재 코드의 후처리는 나쁜 의도라기보다 안정화 장치로 볼 수 있다. 평균을 그대로 쓰면
여러 움직임이 섞인 grid에서 yaw가 이상해질 수 있고, endpoint가 grid 대표 위치와
어긋날 수 있다. 반대로 논문 문장에 가장 직접적으로 맞추려면 이 후처리를 제거해야 한다.

## 구현 우선순위

논문 문장 그대로 맞추는 것이 목표라면 우선순위는 다음과 같다.

1. **non-empty token은 평균값 그대로 사용**

   논문 수식과 가장 직접적으로 연결된 부분이다. 현재 코드의 endpoint overwrite와
   yaw-jump interpolation fallback은 논문에 명시되어 있지 않다.

2. **empty grid endpoint를 cell center로 변경**

   논문의 "cell center"라는 표현을 문자 그대로 따르는 수정이다. 다만 실제 좌표 차이는
   작다.

3. **trajectory/file cap 제거 또는 옵션화**

   논문에는 cap이 명시되어 있지 않다. 완전한 paper-literal 재현을 원하면 cap을 끄는
   실험을 해볼 수 있다. 하지만 계산 비용이 커지고, 실제 제출 코드에 cap이 있었는지는
   논문만으로 확정할 수 없다.

## RMM 관점의 주의점

위 차이들이 논문 문장과의 차이를 설명해 주는 것은 맞지만, 수정한다고 해서 반드시
WOSAC RMM이 오른다고 보장할 수는 없다. 특히 현재 구현의 일부 후처리는 noisy token을
줄이기 위한 안정화일 수 있다.

따라서 이 문서의 결론은 다음과 같다.

- 논문 문장을 엄격하게 따르는 수정 후보는 위 세 가지다.
- 그중 non-empty token 생성 방식은 vocab 모양을 가장 크게 바꿀 수 있다.
- 하지만 최종 RMM 개선 여부는 새 vocab 생성, 새 pretrain, fast-RMM/정식 validation
  ablation으로 확인해야 한다.
