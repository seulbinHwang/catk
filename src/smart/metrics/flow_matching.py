from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingLoss(nn.Module):
    """Flow-Planner 스타일의 2항 손실.

    이 손실은 아래 두 항만 사용한다.

    1. conditional flow matching 기본 손실
       - 네트워크가 예측한 velocity field와 목표 velocity field의 평균 제곱 오차
    2. 인접 future segment 경계 일치 손실
       - 겹치는 경계점이 서로 자연스럽게 이어지도록 하는 평균 제곱 오차

    주의:
        ``test_new`` 파이프라인과의 함수 호출 호환을 유지하기 위해,
        ``forward`` 시그니처에는 현재 사용하지 않는 인자도 그대로 둔다.
        하지만 실제 최적화에는 오직 두 손실만 반영한다.

    Args:
        flow_weight: 기본 flow 손실 가중치.
        consistency_weight: segment 경계 일치 손실 가중치.
    """

    def __init__(self, flow_weight: float = 1.0, consistency_weight: float = 1.0) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight

    @staticmethod
    def _masked_mean(square_error: Tensor, mask: Tensor) -> Tensor:
        """마스크된 평균을 계산한다.

        Args:
            square_error: 임의 shape의 제곱 오차 텐서.
            mask: ``square_error`` 와 broadcast 가능한 마스크.

        Returns:
            마스크가 적용된 스칼라 평균값.
        """
        weight = mask.to(square_error.dtype)
        denom = torch.clamp(weight.sum(), min=1.0)
        return (square_error * weight).sum() / denom

    def _flow_loss(
        self,
        flow_pred: Tensor,
        flow_target: Tensor,
        loss_mask: Tensor,
    ) -> Tensor:
        """기본 conditional flow matching 손실을 계산한다.

        Args:
            flow_pred: ``[N, 4, 6, 4]`` 예측 velocity field.
            flow_target: ``[N, 4, 6, 4]`` 목표 velocity field.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.

        Returns:
            스칼라 기본 손실.
        """
        seg_mask = loss_mask[:, None, None, None]  # [N, 1, 1, 1]
        return self._masked_mean((flow_pred - flow_target) ** 2, seg_mask)

    def _consistency_loss(self, pred_segments: Tensor, loss_mask: Tensor) -> Tensor:
        """겹치는 future segment 경계 일치 손실을 계산한다.

        Flow-Planner 공개 코드처럼, 각 경계 손실을 먼저 만든 뒤 평균을 낸다.
        즉, 경계 3개를 단순 합하지 않고 평균한다.

        Args:
            pred_segments: ``[N, 4, 6, 4]`` clean future segment 예측값.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.

        Returns:
            스칼라 경계 일치 손실.
        """
        boundary_mask = loss_mask[:, None]  # [N, 1]

        boundary_losses = [
            self._masked_mean((pred_segments[:, 0, -1] - pred_segments[:, 1, 0]) ** 2, boundary_mask),
            self._masked_mean((pred_segments[:, 1, -1] - pred_segments[:, 2, 0]) ** 2, boundary_mask),
            self._masked_mean((pred_segments[:, 2, -1] - pred_segments[:, 3, 0]) ** 2, boundary_mask),
        ]
        return torch.stack(boundary_losses).mean()

    def forward(
        self,
        flow_pred: Tensor,
        flow_target: Tensor,
        pred_segments: Tensor,
        gt_segments: Tensor,
        pred_future_local: Tensor,
        gt_future_local: Tensor,
        loss_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """손실을 계산한다.

        Args:
            flow_pred: ``[N, 4, 6, 4]`` 예측 velocity field.
            flow_target: ``[N, 4, 6, 4]`` 목표 velocity field.
            pred_segments: ``[N, 4, 6, 4]`` clean future segment 예측값.
            gt_segments: ``[N, 4, 6, 4]`` GT clean segment.
                현재는 API 호환을 위해 받기만 하고 쓰지 않는다.
            pred_future_local: ``[N, 21, 4]`` 조립된 clean future 예측값.
                현재는 API 호환을 위해 받기만 하고 쓰지 않는다.
            gt_future_local: ``[N, 21, 4]`` GT local future.
                현재는 API 호환을 위해 받기만 하고 쓰지 않는다.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.

        Returns:
            다음 두 값을 담은 tuple.

            - total loss scalar
            - logging dict
        """
        del gt_segments, pred_future_local, gt_future_local

        flow = self._flow_loss(flow_pred=flow_pred, flow_target=flow_target, loss_mask=loss_mask)
        consistency = self._consistency_loss(pred_segments=pred_segments, loss_mask=loss_mask)

        total = self.flow_weight * flow + self.consistency_weight * consistency
        zero = torch.zeros((), device=total.device, dtype=total.dtype)

        log_dict = {
            "flow": flow.detach(),
            "consistency": consistency.detach(),
            "total": total.detach(),
            # 아래 두 키는 기존 test_new 로그 경로와의 호환을 위한 alias다.
            "overlap": consistency.detach(),
            "recon": zero.detach(),
        }
        return total, log_dict
