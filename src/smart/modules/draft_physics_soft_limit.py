from __future__ import annotations

import torch
from torch import Tensor

from src.smart.modules.draft_physics_topk import TopKDraftPhysicsRegularizer


class SoftLimitTopKDraftPhysicsRegularizer(TopKDraftPhysicsRegularizer):
    """soft-limit 비율이 적용된 DRaFT 물리 손실을 계산합니다.

    기존 DRaFT hard 물리 손실은 물리량이 hard limit를 넘은 뒤부터만
    벌점을 줍니다. 이 클래스는 그 시작점을 ``soft_limit_ratio`` 만큼
    앞당깁니다. 기본값 ``1.0`` 에서는 기존 hard-only 방식과 같은 값이 됩니다.

    Args:
        *args: 기존 ``TopKDraftPhysicsRegularizer`` 에 그대로 넘길 위치 인자입니다.
        soft_limit_ratio: 물리 한계값 대비 벌점 시작 비율입니다.
            값은 ``0 < soft_limit_ratio <= 1`` 이어야 합니다.
            ``1.0``이면 기존 hard-only 방식과 같습니다.
        **kwargs: 기존 ``TopKDraftPhysicsRegularizer`` 에 그대로 넘길 이름 인자입니다.
    """

    def __init__(
        self,
        *args: object,
        soft_limit_ratio: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < float(soft_limit_ratio) <= 1.0:
            raise ValueError("soft_limit_ratio must satisfy 0 < soft_limit_ratio <= 1.")
        self.soft_limit_ratio = float(soft_limit_ratio)

    def _phi(self, value: Tensor) -> Tensor:
        """기존 hard-limit 벌점 시작점을 soft-limit 비율만큼 앞당깁니다.

        Args:
            value: 기존 코드가 넘기는 ``물리량 / hard_limit - 1`` 값입니다.
                shape은 임의입니다.

        Returns:
            Tensor: ``max(0, 물리량 / hard_limit - soft_limit_ratio)^2`` 값입니다.
                shape은 입력과 같습니다.
        """
        shifted_value = value + (1.0 - self.soft_limit_ratio)
        return torch.relu(shifted_value).square()
