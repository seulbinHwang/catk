from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        xy_scale_m: loss에서만 x, y 오차를 나눌 고정 길이 스케일.
        overlap_weight: ``consistency_weight`` 의 하위 호환 alias.
        recon_weight: 더 이상 쓰지 않지만 기존 config 호환을 위해 받는다.
    """

    def __init__(
        self,
        flow_weight: float = 1.0,
        consistency_weight: Optional[float] = None,
        xy_scale_m: float = 20.0,
        overlap_weight: Optional[float] = None,
        recon_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        if consistency_weight is None:
            consistency_weight = overlap_weight if overlap_weight is not None else 1.0
        if xy_scale_m <= 0.0:
            raise ValueError(f"xy_scale_m must be positive, got {xy_scale_m}.")

        del recon_weight

        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
        self.xy_scale_m = xy_scale_m

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
        _, weight = torch.broadcast_tensors(square_error, weight)
        denom = torch.clamp(weight.sum(), min=1.0)
        return (square_error * weight).sum() / denom

    def _channel_balanced_loss(self, diff: Tensor, mask: Tensor) -> Tensor:
        """xy와 heading 채널을 분리 평균해 채널 불균형을 줄인다.

        geometry에 해당하는 trajectory/segment 자체는 raw 공간에 두고,
        최종 loss residual에만 xy 스케일 정규화를 적용한다.
        """
        xy_loss = self._masked_mean((diff[..., :2] / self.xy_scale_m) ** 2, mask)
        heading_loss = self._masked_mean(diff[..., 2:] ** 2, mask)
        return 0.5 * (xy_loss + heading_loss)

    def _flow_loss(
        self,
        flow_pred: Tensor,
        flow_target: Tensor,
        loss_mask: Tensor,
    ) -> Tensor:
        """기본 conditional flow matching 손실을 계산한다.

        Args:
            flow_pred: ``[N, S, P, 4]`` 예측 velocity field.
            flow_target: ``[N, S, P, 4]`` 목표 velocity field.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.

        Returns:
            스칼라 기본 손실.
        """
        seg_mask = loss_mask[:, None, None, None]  # [N, 1, 1, 1]
        return self._channel_balanced_loss(flow_pred - flow_target, seg_mask)

    def _consistency_loss(self, pred_segments: Tensor, loss_mask: Tensor) -> Tensor:
        """겹치는 future segment 경계 일치 손실을 계산한다.

        Flow-Planner 공개 코드처럼, 각 경계 손실을 먼저 만든 뒤 평균을 낸다.
        즉, 경계 3개를 단순 합하지 않고 평균한다.

        Args:
            pred_segments: ``[N, S, P, 4]`` clean future segment 예측값.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.

        Returns:
            스칼라 경계 일치 손실.
        """
        boundary_mask = loss_mask[:, None]  # [N, 1]
        boundary_diffs = pred_segments[:, :-1, -1] - pred_segments[:, 1:, 0]  # [N, 3, 4]
        boundary_losses = [
            self._channel_balanced_loss(boundary_diffs[:, i], boundary_mask)
            for i in range(boundary_diffs.shape[1])
        ]
        return torch.stack(boundary_losses).mean()

    def forward(
        self,
        flow_pred: Tensor,
        flow_target: Tensor,
        pred_segments: Tensor,
        loss_mask: Tensor,
        gt_segments: Optional[Tensor] = None,
        pred_future_local: Optional[Tensor] = None,
        gt_future_local: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """손실을 계산한다.

        Args:
            flow_pred: ``[N, S, P, 4]`` 예측 velocity field.
            flow_target: ``[N, S, P, 4]`` 목표 velocity field.
            pred_segments: ``[N, S, P, 4]`` clean future segment 예측값.
            loss_mask: ``[N]``. 학습 대상 agent 마스크.
            gt_segments: ``[N, S, P, 4]`` GT clean segment.
                train slim path에서는 ``None`` 일 수 있고, 현재는 받기만 하고 쓰지 않는다.
            pred_future_local: ``[N, T, 4]`` 조립된 clean future 예측값.
                train slim path에서는 ``None`` 일 수 있고, 현재는 받기만 하고 쓰지 않는다.
            gt_future_local: ``[N, T, 4]`` GT local future.
                train slim path에서는 ``None`` 일 수 있고, 현재는 받기만 하고 쓰지 않는다.

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
