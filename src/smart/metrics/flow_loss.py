from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.smart.utils.flow_traj import (
    assemble_4x6_to_21,
    chunk_valid_21_to_4x6,
    overlap_consistency_residual,
)


@dataclass
class FlowLossOutput:
    """flow 손실 계산 결과를 담는 자료형입니다."""

    total_loss: Tensor
    flow_loss: Tensor
    overlap_loss: Tensor
    ade_2s: Tensor


class FlowMatchingLoss(nn.Module):
    """조각 기반 flow 학습 손실을 계산합니다.

    open-loop pretraining에서는 velocity field MSE를,
    short closed-loop fine-tuning에서는 조각 재구성 MSE를 씁니다.
    둘 다 overlap 경계 손실과 2초 ADE를 같이 계산합니다.

    Args:
        overlap_loss_weight: overlap 손실 가중치입니다.
    """

    def __init__(self, overlap_loss_weight: float) -> None:
        super().__init__()
        self.overlap_loss_weight = overlap_loss_weight

    def forward(
        self,
        pred_segments: Tensor,
        target_segments: Tensor,
        target_valid: Tensor,
        train_mask: Optional[Tensor] = None,
        pred_velocity: Optional[Tensor] = None,
        target_velocity: Optional[Tensor] = None,
    ) -> FlowLossOutput:
        """손실을 계산합니다.

        Args:
            pred_segments: 모델이 복원한 조각 미래입니다.
                shape: `[n_agent, 4, 6, 4]`
            target_segments: 정답 조각 미래입니다.
                shape: `[n_agent, 4, 6, 4]`
            target_valid: 점 단위 유효 마스크입니다.
                shape: `[n_agent, 21]`
            train_mask: 실제 학습 대상으로 쓸 agent 마스크입니다.
                shape: `[n_agent]`
            pred_velocity: 모델이 예측한 velocity field입니다.
                shape: `[n_agent, 4, 6, 4]`
            target_velocity: 정답 velocity field입니다.
                shape: `[n_agent, 4, 6, 4]`

        Returns:
            총 손실, 기본 손실, overlap 손실, 2초 ADE입니다.
        """
        point_mask = chunk_valid_21_to_4x6(target_valid)  # [n_agent, 4, 6]
        if train_mask is not None:
            point_mask = point_mask & train_mask[:, None, None]
        point_mask_f = point_mask.unsqueeze(-1).float()  # [n_agent, 4, 6, 1]
        denom = torch.clamp(point_mask_f.sum() * pred_segments.shape[-1], min=1.0)

        if pred_velocity is not None and target_velocity is not None:
            primary_err = (pred_velocity - target_velocity) ** 2
        else:
            primary_err = (pred_segments - target_segments) ** 2
        flow_loss = (primary_err * point_mask_f).sum() / denom

        overlap_mask = point_mask[:, :-1, -1] & point_mask[:, 1:, 0]  # [n_agent, 3]
        overlap_mask_f = overlap_mask.unsqueeze(-1).float()  # [n_agent, 3, 1]
        overlap_err = overlap_consistency_residual(pred_segments) ** 2  # [n_agent, 3, 4]
        overlap_denom = torch.clamp(overlap_mask_f.sum() * pred_segments.shape[-1], min=1.0)
        overlap_loss = (overlap_err * overlap_mask_f).sum() / overlap_denom

        pred_future = assemble_4x6_to_21(pred_segments)[:, 1:, :2]  # [n_agent, 20, 2]
        target_future = assemble_4x6_to_21(target_segments)[:, 1:, :2]  # [n_agent, 20, 2]
        ade_mask = target_valid[:, 1:].float()  # [n_agent, 20]
        if train_mask is not None:
            ade_mask = ade_mask * train_mask[:, None].float()
        ade_err = torch.norm(pred_future - target_future, dim=-1)  # [n_agent, 20]
        ade_denom = torch.clamp(ade_mask.sum(), min=1.0)
        ade_2s = (ade_err * ade_mask).sum() / ade_denom

        total_loss = flow_loss + self.overlap_loss_weight * overlap_loss
        return FlowLossOutput(
            total_loss=total_loss,
            flow_loss=flow_loss,
            overlap_loss=overlap_loss,
            ade_2s=ade_2s,
        )
