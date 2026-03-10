from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingLoss(nn.Module):
    """Sparse factorized flow matching loss.

    이 손실은 세 가지를 합친다.
    1. conditional flow matching velocity MSE
    2. 겹치는 segment 경계 일치 손실
    3. optional 2초 open-loop 재구성 MSE
    """

    def __init__(
        self,
        flow_weight: float = 1.0,
        overlap_weight: float = 0.25,
        recon_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.overlap_weight = overlap_weight
        self.recon_weight = recon_weight

    @staticmethod
    def _masked_mean(square_error: Tensor, mask: Tensor) -> Tensor:
        """마스크된 평균을 계산한다.

        Args:
            square_error: 임의 shape의 제곱 오차.
            mask: ``square_error`` 와 broadcast 가능한 bool/float mask.

        Returns:
            스칼라 평균값.
        """
        weight = mask.to(square_error.dtype)
        denom = torch.clamp(weight.sum(), min=1.0)
        return (square_error * weight).sum() / denom

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
            flow_pred: ``[N, 4, 6, 4]`` predicted velocity field.
            flow_target: ``[N, 4, 6, 4]`` target velocity field.
            pred_segments: ``[N, 4, 6, 4]`` clean segment prediction.
            gt_segments: ``[N, 4, 6, 4]`` clean GT segment.
            pred_future_local: ``[N, 21, 4]`` assembled clean prediction.
            gt_future_local: ``[N, 21, 4]`` assembled GT future.
            loss_mask: ``[N]``. 학습 대상 agent인지와 2초 GT 유효성까지 포함한 mask.

        Returns:
            tuple:
                - total loss scalar
                - logging dict
        """
        seg_mask = loss_mask[:, None, None, None]
        flow = self._masked_mean((flow_pred - flow_target) ** 2, seg_mask)
        overlap = self._masked_mean(
            (pred_segments[:, 0, -1] - pred_segments[:, 1, 0]) ** 2,
            loss_mask[:, None],
        )
        overlap = overlap + self._masked_mean(
            (pred_segments[:, 1, -1] - pred_segments[:, 2, 0]) ** 2,
            loss_mask[:, None],
        )
        overlap = overlap + self._masked_mean(
            (pred_segments[:, 2, -1] - pred_segments[:, 3, 0]) ** 2,
            loss_mask[:, None],
        )
        recon = self._masked_mean(
            (pred_future_local - gt_future_local) ** 2,
            loss_mask[:, None, None],
        )
        total = (
            self.flow_weight * flow
            + self.overlap_weight * overlap
            + self.recon_weight * recon
        )
        log_dict = {
            "flow": flow.detach(),
            "overlap": overlap.detach(),
            "recon": recon.detach(),
            "total": total.detach(),
        }
        return total, log_dict
