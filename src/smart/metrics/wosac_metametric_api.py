from __future__ import annotations

"""
Object-oriented facade for WOSAC metametric.

목표:
- 학습/평가 코드에서 하나의 객체만 선택하면
  - official TF 평가,
  - PyTorch hard RMM,
  - PyTorch fully soft RMM(+surrogate)
  를 같은 인터페이스로 호출할 수 있게 한다.

이 파일은 기존 구현들을 래핑만 하고, 내부 로직은 수정하지 않는다.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as tf_metrics

from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
    compute_scenario_rollouts_features,
)
from src.smart.metrics.wosac_metric_features_torch.surrogate import SurrogateConfig
from src.smart.metrics.wosac_metametric_pytorch import (
    compute_scenario_metrics_for_features_bundle as compute_rmm_hard_torch,
)
from src.smart.metrics.wosac_metametric_pytorch_differentiable import (
    compute_scenario_metrics_for_features_bundle_soft as compute_rmm_soft_torch,
)


class WosacEvaluator(Protocol):
    """공통 인터페이스: 시나리오+rollouts -> dict(str -> float) 메트릭."""

    def __call__(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> dict:
        ...


@dataclass(frozen=True)
class OfficialWosac:
    """공식 TF 파이프라인을 그대로 쓰는 evaluator (비미분, 가장 느림)."""

    def __call__(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> dict:
        # 공식 API는 feature bundle까지 내부에서 생성한다.
        return tf_metrics.compute_scenario_metrics_for_rollouts(
            scenario=scenario,
            scenario_rollouts=scenario_rollouts,
        )


@dataclass(frozen=True)
class HardTorchWosac:
    """PyTorch Stage① + PyTorch hard RMM.

    - 그래프는 끊겨 있어도 되는 evaluation/validation 용도.
    """

    def _features(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> Tuple[object, object]:
        log_feat, sim_feat = compute_scenario_rollouts_features(
            scenario,
            scenario_rollouts,
        )
        return log_feat, sim_feat

    def __call__(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> dict:
        log_feat, sim_feat = self._features(scenario, scenario_rollouts)
        return compute_rmm_hard_torch(
            logged_features=log_feat,
            simulated_features=sim_feat,
        )


@dataclass(frozen=True)
class SoftTorchWosac:
    """PyTorch Stage①(+optional surrogate) + fully soft differentiable RMM.

    - Flow/생성 모델 학습 시 loss로 사용하는 용도.
    - surrogate=True 이면 collision/offroad/tl 이벤트를 연속 확률 surrogate로 계산.
    """

    surrogate: bool = True

    def _features(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> Tuple[object, object]:
        surrogate_cfg = SurrogateConfig() if self.surrogate else None
        log_feat, sim_feat = compute_scenario_rollouts_features(
            scenario,
            scenario_rollouts,
            surrogate=surrogate_cfg,
        )
        return log_feat, sim_feat

    def __call__(
        self,
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> dict:
        log_feat, sim_feat = self._features(scenario, scenario_rollouts)
        # soft 버전은 gradient가 흘러야 하므로, 그대로 호출해 주면 된다.
        return compute_rmm_soft_torch(
            logged_features=log_feat,
            simulated_features=sim_feat,
        )


def make_wosac_evaluator(
    kind: str,
    *,
    surrogate: bool = True,
) -> WosacEvaluator:
    """kind에 따라 official/hard/soft evaluator를 생성하는 간단한 팩토리.

    Args:
      kind: \"official\" | \"hard\" | \"soft\"
      surrogate: kind=\"soft\"일 때 Stage① surrogate 이벤트를 쓸지 여부.
    """
    k = kind.lower()
    if k == "official":
        return OfficialWosac()
    if k == "hard":
        return HardTorchWosac()
    if k == "soft":
        return SoftTorchWosac(surrogate=surrogate)
    raise ValueError(f"Unknown WOSAC kind: {kind}")


__all__ = [
    "WosacEvaluator",
    "OfficialWosac",
    "HardTorchWosac",
    "SoftTorchWosac",
    "make_wosac_evaluator",
]

