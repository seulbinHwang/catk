# Flow RMM-BPTT Fine-Tuning 가이드

이 문서는 본 저장소의 **`rmm_bptt_ft`** 모드(Closed-loop BPTT + 미분 가능한 **Soft RMM**)와, 검증 시 사용하는 **Official / Hard RMM**의 관계, 그리고 **Soft histogram**의 정의를 정리합니다.

**수식 표기**: GitHub·VS Code 등에서 `$$...$$` 블록이 렌더링됩니다. 로컬 미리보기에서 안 보이면 [HTML 버전](flow_bptt_finetuning_guide.html)(MathJax)을 사용하세요.

---

## 0. 기호·수학 정식화 (요약)

### 0.1 Metametric (공통 골격)

Waymo Sim Agents 2025 설정에서 각 항목 $i$에 가중치 $w_i \ge 0$ (proto의 `metametric_weight`)가 붙습니다. 항목별 **가징 likelihood** $\ell_i \in [0,1]$ 를 만든 뒤, **metametric**은 가중합으로 정의됩니다.

$$
M \;=\; \sum_{i} w_i \, \ell_i \,.
$$

항목 $i$는 예를 들어 `linear_speed`, `linear_acceleration`, `collision_indication` 등입니다. Official·Hard·Soft 모두 **같은 $w_i$**를 쓰지만, **$\ell_i$를 만드는 연산**(이산 vs 미분 가능 근사)이 다릅니다.

### 0.2 학습 목적함수 (`rmm_bptt_ft`)

시나리오 $s$마다 Soft metametric $M_s^{\mathrm{soft}}$를 구하고, 배치에서 유효한 시나리오만 평균 $\overline{M}^{\mathrm{soft}}$를 냅니다. 기본 학습은 **$M$을 크게** 하는 방향입니다.

$$
\mathcal{L}_{\mathrm{RMM}} \;=\; - \,\overline{M}^{\mathrm{soft}}
\;=\; - \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} M_s^{\mathrm{soft}} \,.
$$

선택적 **GT flow-matching** 항(계수 $\lambda_{\mathrm{fm}} =$ `flow_reg_lambda`):

$$
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{RMM}} + \lambda_{\mathrm{fm}} \cdot \mathcal{L}_{\mathrm{FM}} \,.
$$

---

## 1. Fine-tuning 프레임워크 개요

### 1.1 모델과 진입점

- **모델**: `SMARTFlow` (`src/smart/model/smart_flow.py`)
- **실험 프리셋**: `configs/experiment/flow_bptt_ft.yaml` (`experiment=flow_bptt_ft`, `action=finetune`)
- **Fine-tune 모드**: `model.model_config.finetune.mode: rmm_bptt_ft`

`FinetuneConfig` (`src/smart/utils/finetune.py`)에 정의된 플래그들이 Hydra로 주입되며, `set_model_for_finetuning()`이 **트렁크 동결 + 학습 대상 헤드만 활성화** 등을 담당합니다.

### 1.2 `rmm_bptt_ft`가 하는 일 (한 줄 요약)

1. **인코더/맵**는 동결(`no_grad`)로 캐시를 만들고,
2. **Flow ODE**로 **closed-loop coarse rollout**을 수행해 시뮬 궤적을 얻고,
3. 위 절의 $M_s^{\mathrm{soft}}$를 계산한 뒤,
4. $\mathcal{L}_{\mathrm{RMM}} = -\overline{M}^{\mathrm{soft}}$ (필요 시 $\mathcal{L}_{\mathrm{FM}}$ 추가)로 역전파합니다.

구현: `_run_flow_bptt_ft_step()` → `compute_wosac_metametric_soft_batched()` (`src/smart/model/smart_flow.py`, `src/smart/metrics/wosac_metametric_pytorch_differentiable.py`).

### 1.3 학습에 필요한 데이터

- 배치에 **`tfrecord_path`**, **`scenario_id`**, 에이전트 **`object_id`**가 있어야 합니다 (rollout과 log feature 정합).
- Log 쪽 metric feature는 TFRecord 시나리오에서 **한 번 계산·슬라이스**되어 캐시됩니다 (`_LOG_FEAT_DICT_CACHE`).
- 상세 경로/스플릿 설명은 `flow_bptt_ft.yaml`의 `data:` 주석을 참고하세요.

---

## 2. BPTT 알고리즘과 주요 하이퍼파라미터

### 2.1 시간 축: coarse step vs Flow solver step

- **Coarse step**: rollout **외부** 인덱스 $s = 1,\ldots,S$ (대략 `shift`×$S$ 만큼 10Hz 스텝).
- **`bptt_max_coarse_steps`**: $S$의 상한. `null`/0 이하면 보통 **16** (전체 지평).
- **Flow ODE `flow_solver_steps`**: 각 coarse step **안**에서 ODE를 $N_{\mathrm{ode}}$번 이산 적분 (예: Euler, midpoint).

### 2.2 Rollout 개수 `bptt_n_rollouts` ($G$)

시나리오당 **$G$개** 독립 rollout $\{r=1,\ldots,G\}$. 병렬 모드에서는 rollout 축에 대해 $M$을 평균합니다.

### 2.3 메모리·그래프 절감 옵션

| 설정 | 의미 |
|------|------|
| `bptt_use_adjoint` | Flow `generate()` 내부를 **gradient checkpoint**로 감싸 활성화 메모리 절감 (Neural ODE adjoint의 이산 대응). |
| `bptt_sequential_rollouts` | $G>1$일 때 rollout을 **순차** 실행하고 매번 `backward` → 피크 메모리는 줄지만 **DDP 다중 GPU에서는 비권장** (bucket reducer 이슈). |
| `bptt_warm_coarse_steps` | 앞쪽 coarse step을 `no_grad`+`detach`로 **슬라이딩 윈도 BPTT**. |
| `bptt_last_n_coarse_steps` | 마지막 $N$ coarse step에만 gradient (warm을 자동으로 키움). |
| `bptt_last_n_solver_steps` | ODE solver의 **마지막 $N$ step**에만 velocity→파라미터 gradient. |
| `bptt_grad_clip_traj` | `pred_traj` / `pred_head`에 **gradient L2 norm clip** 훅 (0이면 끔). |

### 2.4 보조 손실

- **`flow_reg_lambda`**: GT 궤적에 대한 **velocity flow-matching MSE** $\mathcal{L}_{\mathrm{FM}}$ (`_compute_rmm_bptt_gt_fm_loss`).

### 2.5 참조 모델 로깅

- **`rmm_bptt_ref_train` / `rmm_bptt_ref_val`**: 동결된 **pretrained** `flow_decoder`로 동일 조건 rollout 후 Soft RMM을 구해 **`train/rmm_ref`, `val_ref/rmm`** 및 **`*/rmm_delta` ($M^{\mathrm{ft}} - M^{\mathrm{ref}}$)** 로깅.

### 2.6 파라미터 학습 범위

- **`flow_velocity_head_only: true`** (기본): `HierarchicalFlowDecoder.velocity_head`만 학습, 트렁크·`residual_velocity_head` 동결.

---

## 3. 검증 메트릭: `validation_metric`

`model.model_config.validation_metric`:

| 값 | 구현 클래스 | 설명 |
|----|-------------|------|
| **`real`** (기본 `smart_flow.yaml`) | `SimAgentsMetrics` | Waymo 공식 **`wdl_limited.sim_agents_metrics.metrics`** 를 **서브프로세스**에서 TF로 실행. |
| **`hard`** (`flow_bptt_ft.yaml` 권장) | `HardSimAgentsMetrics` | **PyTorch in-process**, **`compute_wosac_metametric_from_features_torch`** — **$M^{\mathrm{hard}}$**, 공식과 parity 목표. |

**중요**: **Soft RMM** ($M^{\mathrm{soft}}$)은 **학습 목적 함수**용이며, **`validation_metric`** 과는 별개입니다. 검증은 **Official(`real`) 또는 Hard**를 쓰고, Soft는 **train/rmm_soft** 등으로 로깅됩니다.

---

## 4. Official / Hard vs Soft: 동일 $w_i$, 다른 $\ell_i$

### 4.1 공통

- **동일** `challenge_2025_sim_agents_config.textproto` → $w_i$, 히스토그램 구간 $[a_i,b_i]$, 빈 개수 $K_i$, 의사카운트 $\alpha_i$.
- **동일** feature 키 (텐서 형상은 `MetricFeatures` 규약).

### 4.2 시계열 항목: log-likelihood 그리드 → 마스크 평균 → likelihood

한 객체·한 항목에 대해, 시간 $t$마다 (로그·시뮬 샘플로부터) **log-likelihood** $\mathrm{LL}_t$를 얻습니다. 유효 마스크 $m_t \in \{0,1\}$ (Hard) 또는 $m_t \in [0,1]$ (Soft)로 가중 평균한 뒤 지수를 취해 $\ell \in [0,1]$에 넣습니다.

**Hard** (`wosac_metametric_pytorch.py`):

$$
\ell \;=\; \exp\!\left( \frac{\sum_t m_t \, \mathrm{LL}_t}{\sum_t m_t} \right), \qquad m_t \in \{0,1\}\,.
$$

**Soft**: 같은 겉구조이나, $m_t$가 곱으로 정의된 **연속** 마스크이고, $\mathrm{LL}_t$ 자체가 아래 **soft histogram**으로 계산됩니다.

### 4.3 Kinematic validity: speed / acceleration 마스크

시각 $t$에서 유효 플래그를 $v_t \in \{0,1\}$라 하면 (로그 궤적의 `valid`에서 옴).

**Hard** (`compute_kinematic_validity`): 중앙 시각 $t$에서 속도 유효는 대략

$$
m^{\mathrm{spd}}_t \;=\; v_{t-1} \;\wedge\; v_t \;\wedge\; v_{t+1} \quad (\text{끝은 패딩})\,.
$$

가속도 유효는 $m^{\mathrm{spd}}$에 **같은 AND 연산**을 한 번 더 적용한 결과입니다.

**Soft** (`compute_kinematic_validity_soft`): 논리곱을 **곱**으로 바꿉니다 ($v_t \in [0,1]$로 해석).

$$
m^{\mathrm{spd}}_t \;\approx\; v_{t-1} \cdot v_t \cdot v_{t+1}\,, \qquad
m^{\mathrm{acc}}_t \;\approx\; m^{\mathrm{spd}}_{t-1} \cdot m^{\mathrm{spd}}_t \cdot m^{\mathrm{spd}}_{t+1}\,.
$$

그래서 **linear_acceleration** 항목은 Hard에서는 **이산 AND**로 잘린 시점만 쓰고, Soft에서는 **연속 가중**으로 평균해 gradient가 흐릅니다.

### 4.4 히스토그램 추정기 (Hard vs Soft)

구간 $[a,b]$를 $K$개의 균일 빈으로 나누고, 경계를 $a = e_0 < e_1 < \cdots < e_K = b$, 빈 **중심** $c_k = (e_k + e_{k+1})/2$.

#### Hard (`histogram_estimate_torch`)

시뮬 샘플 $x^{(s)}_1,\ldots,x^{(s)}_M$ 각각에 **하나의** 빈 인덱스 $b(x) \in \{0,\ldots,K-1\}$ (반개구간 규칙, 비유한 값은 상단 빈 등 Waymo 규칙과 동일). 카운트

$$
\hat{n}_k \;=\; \sum_{m=1}^{M} \mathbf{1}[b(x^{(s)}_m)=k] + \alpha \,.
$$

정규화 확률 $p_k = \hat{n}_k / \sum_j \hat{n}_j$. 로그 궤적 샘플 $x^{(\log)}$에 대해

$$
\mathrm{LL}\bigl(x^{(\log)}\bigr) \;=\; \log p_{b(x^{(\log)})} \,.
$$

#### Soft (`histogram_estimate_soft_torch`, 온도 $\tau > 0$)

빈 가중은 **거리 기반 softmax**:

$$
w_k(x) \;=\; \frac{\exp\!\bigl(-|x - c_k|/\tau\bigr)}{\sum_{j=0}^{K-1} \exp\!\bigl(-|x - c_j|/\tau\bigr)} \,.
$$

시뮬 샘플로 **소프트 카운트** $\tilde{n}_k = \sum_m w_k(x^{(s)}_m) + \alpha$, $p_k = \tilde{n}_k / \sum_j \tilde{n}_j$. 로그 샘플 쪽은 같은 $w_k(x^{(\log)})$로

$$
\mathrm{LL}_{\mathrm{soft}}\bigl(x^{(\log)}\bigr) \;=\; \sum_{k=0}^{K-1} w_k\bigl(x^{(\log)}\bigr)\, \log p_k \,.
$$

기본 $\tau =$ `default_tau` (보통 $0.1$). **angular_speed** 등은 필요 시 `custom_taus`로 항목별 조정.

### 4.5 시나리오 항 (collision, offroad, …): `any` vs `soft_any`

시간 $t=1,\ldots,T$에 스칼라 $z_t$가 있을 때 (예: 충돌 지표).

**Hard**: $\mathrm{any}_t z_t = \mathbf{1}[\exists t: z_t \neq 0]$ 처럼 이산화된 뒤 Bernoulli/histogram 경로.

**Soft** (`soft_any`): $\beta > 0$ (기본 10)에 대해

$$
\widetilde{\mathrm{any}}(z) \;=\; \sum_{t=1}^{T} z_t \cdot \frac{\exp(\beta z_t)}{\sum_{s=1}^{T} \exp(\beta z_s)} \,.
$$

(코드는 비유한 값을 0으로 치환한 뒤 위 가중합을 씁니다.) 이 스칼라들이 다시 **soft histogram** likelihood로 들어갑니다.

### 4.6 Likelihood → $[0,1]$ 스칼라 (`_likelihood_from_log_ll`)

Soft 경로에서는 그리드 $\mathrm{LL}_t$를 마스크로 평균한 **스칼라** $\overline{\mathrm{LL}}$에 대해

$$
\ell \;=\; \exp\!\bigl( \mathrm{clip}(\overline{\mathrm{LL}}, -80, 0) \bigr)
$$

로 항목 likelihood를 만듭니다 (오버플로·역전파 안정화).

---

## 5. 항목별 메모 (kinematic, linear acceleration, …)

| 항목 | Hard | Soft |
|------|------|------|
| **linear_speed** | $m^{\mathrm{spd}}_t$ (이산 AND) + hard histogram | 연속 $m^{\mathrm{spd}}_t$ + soft histogram |
| **linear_acceleration** | $m^{\mathrm{acc}}_t$ (이산 AND) + hard histogram | 연속 $m^{\mathrm{acc}}_t$ + soft histogram |
| **angular_speed** | 동일 구조 | 동일 구조, $\tau$는 기본값 통일(과거 0.01은 gradient 스케일 이슈) |
| **collision / offroad / …** | 시간 `any` (불리언) | `soft_any` + soft histogram |

**Surrogate**: `SurrogateConfig`의 온도로 연속화된 위험 feature를 쓰므로, 공식 TF의 이산 처리와 **완전 동일하지 않을 수** 있습니다.

---

## 6. 요약 표

| 구분 | 용도 | $\ell_i$ 내부 | Gradient |
|------|------|----------------|----------|
| **Official `real`** | 검증·리더보드 | TF/TFP, 이산 | 없음 |
| **Hard** | 빠른 검증 | PyTorch hard bin + hard mask | 없음 |
| **Soft RMM** | `rmm_bptt_ft` 학습 | soft bin ($\tau$) + soft mask + soft_any | 있음 |

---

## 7. 참고 파일

| 파일 | 내용 |
|------|------|
| `src/smart/model/smart_flow.py` | BPTT rollout, Soft RMM loss, 검증 루프 |
| `src/smart/utils/finetune.py` | `FinetuneConfig`, `set_model_for_finetuning` |
| `src/smart/metrics/wosac_metametric_pytorch.py` | Hard PyTorch metametric (parity 목표) |
| `src/smart/metrics/wosac_metametric_pytorch_differentiable.py` | Soft RMM |
| `src/smart/metrics/__init__.py` | `SimAgentsMetrics` vs `HardSimAgentsMetrics` |
| `configs/experiment/flow_bptt_ft.yaml` | 실험 기본값 |

---

*세부 수치·가중치 $w_i$는 Waymo `challenge_2025_sim_agents_config.textproto`에 따릅니다.*
