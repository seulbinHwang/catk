# CLAUDE.md — kinematic_flow 가이드

> 이 문서는 Claude Code가 이 repo를 빠르게 이해하고 작업하기 위한 짧은 가이드입니다.
> 상세 설명은 `README.md`(약 3000줄)을 참고하세요.

## 1. 한 줄 요약

이미 잘 학습된 **flow-matching 기반 SMART backbone**을 **Waymo Open Sim Agents Challenge (WOSAC) 2025** 지표로 fine-tuning하는 프로젝트.

## 2. 현재 작업 목표 (이 branch)

- **Branch**: `pareto_rmm_cpd` (base: `semi_control_stable`)
- **목표**: 두 개의 핵심 metric을 **동시에 pareto optimal**로 만드는 fine-tuning 알고리즘 개발
  - **WOSAC RMM** (= `realism_meta_metric`) — 현실성 점수 (↑ 클수록 좋음)
  - **WOSAC CPD** — rollout 다양성 점수 (↑ 클수록 좋음)
- 두 지표는 trade-off 관계가 알려져 있음 (현실성 높이면 모드 collapse로 다양성 감소).

## 3. 핵심 metric 위치

| Metric | 정의/계산 위치 | 로그 키 |
|---|---|---|
| **RMM** | `src/smart/metrics/wosac_fast_eval_tool/fast_sim_agents_metrics/metrics.py:_compute_metametric()` (weighted sum of likelihoods) | `val_closed/sim_agents_2025/realism_meta_metric` |
| RMM 집계기 | `src/smart/metrics/sim_agents_metrics.py` (`SimAgentsMetrics`) | `val_closed/sim_agents_2025/*` |
| **CPD** | `src/smart/metrics/wosac_distribution_metrics.py` (`WOSACDistributionMetrics.compute()`) | `val_closed/WOSAC-CPD/value` |
| CES (보조, GT 있을 때만) | 같은 파일 | `val_closed/WOSAC-CES/value` |
| CPD 보존율 (DPR) | `wosac_cpd_reference` 설정 시에만 기록 | `*/WOSAC-CPD/DPR` |

- LightningModule(`src/smart/model/smart_flow.py`) 안에서 `self.sim_agents_metrics`(RMM), `self.wosac_distribution_metrics`(CPD), `self.test_wosac_distribution_metrics`(test) 로 등록됨.
- 기본 checkpoint monitor는 **RMM** (`callbacks/model_checkpoint`: `monitor=val_closed/sim_agents_2025/realism_meta_metric`, `mode=max`).

## 4. Repo 구조 (한 눈에)

```
src/
  run.py                    # Hydra main, action=fit/finetune/road_finetune/validate/test
  smart/
    model/smart_flow.py     # SMARTFlow LightningModule (제일 중요)
    modules/                # Decoder, self-forced, kinematic control 등
      smart_flow_decoder.py, flow_agent_decoder.py, flow_local_decoder.py
      self_forced_*         # Self-forced fine-tuning 관련
      kinematic_control.py, dynamic_limits.py
    tokens/                 # FlowTokenProcessor, agent vocab
    metrics/                # ★ RMM(sim_agents_metrics) + CPD(wosac_distribution_metrics)
      wosac_fast_eval_tool/ # TrajTok Fast WOSAC 2025 (vendored)
    datamodules/            # MultiDataModule, samplers
    datasets/scalable_dataset.py
    layers/                 # attention, fourier embedding, MLP
    road/                   # RoaD fine-tuning
    utils/                  # 기하, finetune helper, rollout
  utils/                    # Lightning instantiators, W&B, Waymo submission
  data_preprocess.py        # WOMD cache 만들 때 사용

configs/                    # Hydra
  run.yaml                  # 최상위
  model/smart_flow.yaml     # 모델 + 모든 model_config (CPD/RMM 설정 포함)
  data/waymo.yaml
  experiment/               # 실제 학습 recipe (★ 여기를 보고 시작)
    pre_bc_flow*.yaml       #   - flow pretrain (BC)
    finetune_flow_*.yaml    #   - flow fine-tuning (현재 작업 대상)
    self_forced_npfm*.yaml  #   - Self-forced N-second path flow fine-tuning
    road_flow.yaml          #   - RoaD fine-tuning
    local_val_flow.yaml     #   - 로컬 validation only
  trainer/, callbacks/, logger/, paths/, hydra/

scripts/                    # 학습/평가/스윕 launcher (대부분 멀티노드/스윕용)
  train_flow.sh, local_val_flow.sh, road_flow_finetune.sh
  launch_fast_rmm_*sweep*.py  # RMM 스윕 launcher (현재 branch의 최근 작업)
  launch_finetune_flow_*.py, launch_self_forced_*.py
tools/                      # 분석/검증 스크립트 (예: compare_fast_wosac_metric.py)
tests/                      # pytest (★ 정합성 검증용)
install/                    # Dockerfile, requirements.txt
```

## 5. 실행 방법

모든 학습/평가는 Hydra entry point `python -m src.run`를 통해 돌아갑니다.

```bash
# 환경
conda activate catk

# 1) Flow pretrain (BC)
python -m src.run experiment=pre_bc_flow task_name=<name>

# 2) Flow fine-tuning (이 작업의 baseline)
python -m src.run experiment=finetune_flow_prefix_valid_a100_4x2 \
  action=finetune ckpt_path=<pretrain_ckpt> task_name=<name>

# 3) Self-forced fine-tuning
python -m src.run experiment=self_forced_npfm \
  action=finetune ckpt_path=<pretrain_ckpt> task_name=<name>
#   주의: 이미 self-forced 진행 중인 체크포인트는 action=fit 으로 resume.

# 4) Validation only (RMM + CPD 로컬 계산)
python -m src.run experiment=local_val_flow \
  action=validate ckpt_path=<ckpt> task_name=<name>

# 5) 빠른 sanity (스크립트 wrapper)
bash scripts/train_flow.sh
bash scripts/local_val_flow.sh
```

`action`의 의미:
- `fit`: Lightning 전체 resume (optimizer/scheduler/EMA 포함). Self-forced 재개에 사용.
- `finetune`: weight-only 로딩(`strict=False`) + 새 optimizer. Pretrain → fine-tune 시 사용.
- `road_finetune`: RoaD 알고리즘 사용 (`src/smart/road/`).
- `validate`/`test`: trainer.validate/test 호출.

## 6. 작업 시 룰 (사용자 지정)

1. **한국어로 답변할 것.** 코드/명령어를 제외한 텍스트는 항상 한글.
2. **주기적 pull / commit**: 코드 수정-실행 사이에 `git pull --rebase`와 의미 단위 `git commit`을 반복.
3. **수정 완료 후 정합성 검증**: 변경 후에는 짧은 테스트(pytest 일부 + `python -c "from src.smart..."` 등)로 import/실행 오류가 없는지 확인.
4. **새 branch에서 작업**: 큰 실험/알고리즘 변경은 새 branch로 분리.
5. **TaskCreate / TaskUpdate로 진행상황 추적**.

## 7. 자주 쓰는 검증 명령

```bash
# 가장 가벼운 sanity (import만)
conda run -n catk python -c "from src.smart.model.smart_flow import SMARTFlow; \
  from src.smart.metrics import SimAgentsMetrics, WOSACDistributionMetrics; print('ok')"

# CPD 단위 테스트 (빠름, GPU 불필요할 수 있음)
conda run -n catk pytest tests/test_wosac_distribution_metrics.py -q

# Fast WOSAC metric 회귀 테스트
conda run -n catk pytest tests/test_fast_wosac_metric.py -q

# Fast WOSAC vs 공식 scorer 수치 비교 (3 시나리오, 1e-6 기준)
conda run -n catk python tools/compare_fast_wosac_metric.py \
  --num-scenarios 3 --threshold 1e-6 --device cpu \
  --json-output artifacts/fast_wosac_compare_3scenarios.json
```

## 8. Pareto (RMM × CPD) 작업 시 참고 포인트

- **CPD 정규화 scale**: `model.model_config.wosac_distribution_type_scale=[vehicle, ped, cyc]` (기본값은 training cache에서 offline 계산됨). 비교 가능성을 위해 실험 간 같은 scale 유지 권장.
- **CPD reference**: 비교 baseline의 CPD를 `model.model_config.wosac_cpd_reference`에 넣으면 DPR(보존율)이 로그됨 — pareto front 추적에 유용.
- **Diversity ↑ knob**:
  - `model.model_config.self_forced.sampling.noise_scale` (>1.0 이면 다양성↑, 현실성↓ 경향)
  - `model.model_config.self_forced.sampling.sample_steps`
  - `random_terminal_step.policy` (`paper_uniform` vs `all`)
- **Realism ↑ knob**:
  - 더 많은 fine-tuning step, prefix-valid loss mask, kinematic control flow.
- **Closed-loop rollout 모드**: `model.model_config.decoder.closed_loop_rollout_mode`
  - `raw_fm`(기본) — 네트워크 raw 출력
  - `matched_token_chunk` — 외부 export에만 token chunk 반영
  - `use_lqr=true` — vehicle/bicycle만 LQR + kinematic bridge

## 9. 환경 / 의존성 요약

- Python 3.11.9, PyTorch 2.4.x, CUDA 12.1, PyG (`torch_geometric`, `torch_scatter`, `torch_cluster`)
- `waymo-open-dataset-tf-2-12-0==1.6.7` (Sim Agents 2025 proto 검증)
- TensorFlow (proto/scorer용), `ffmpeg`(시각화)
- W&B (`WANDB_PROJECT=SMART-FLOW`)
- 자세한 설치는 `README.md` §2 참조.

---

추가로 알아야 할 세부 사항(`closed_loop_rollout_mode`, `self_forced`, `prefix_valid_future_loss_mask`, traffic light static feature, LQR bridge, motion missingness, FP32 graph attention 등)은 `README.md` 본문에 한글로 잘 정리되어 있으니 필요 시 해당 섹션만 골라 읽으세요.
