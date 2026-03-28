from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import tensor
from torchmetrics import Metric

from src.smart.utils import wrap_angle

class WeightedMeanMetric(Metric):
    """가중치가 있는 스칼라 평균을 DDP까지 포함해 안정적으로 누적합니다.

    각 batch에서 이미 평균으로 계산된 스칼라 값과,
    그 값이 대표하는 실제 표본 개수를 함께 받아 누적합니다.
    마지막에는 ``전체 합 / 전체 개수`` 로 계산하므로
    batch마다 표본 수가 달라도 진짜 표본 평균을 얻을 수 있습니다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor, weight: int | float | torch.Tensor) -> None:
        """한 batch의 평균값과 표본 개수를 누적합니다.

        Args:
            value: batch 안에서 미리 평균낸 스칼라 값입니다. shape은 ``[]`` 입니다.
            weight: 이 값이 대표하는 실제 표본 개수입니다. shape은 ``[]`` 또는 파이썬 숫자입니다.
        """
        weight_tensor = value.detach().new_tensor(float(weight))
        self.sum += value.detach() * weight_tensor
        self.count += weight_tensor

    def compute(self) -> torch.Tensor:
        """누적된 전체 표본 평균을 돌려줍니다.

        Returns:
            torch.Tensor: 전체 표본 기준 평균 스칼라입니다.
        """
        if self.count.item() == 0:
            return self.sum * 0.0
        return self.sum / self.count



def flow_matching_loss(flow_pred_norm: torch.Tensor, flow_target_norm: torch.Tensor) -> torch.Tensor:
    if flow_pred_norm.numel() == 0:
        return flow_pred_norm.sum() * 0.0
    return F.mse_loss(flow_pred_norm, flow_target_norm)


def ade_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    diff_xy = (pred_clean_norm[..., :2] - target_clean_norm[..., :2]) * 20.0
    return torch.norm(diff_xy, dim=-1).mean()


def fde_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    diff_xy = (pred_clean_norm[:, -1, :2] - target_clean_norm[:, -1, :2]) * 20.0
    return torch.norm(diff_xy, dim=-1).mean()



def yaw_ade_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    """2초 전체 구간의 평균 방향 오차를 degree 단위로 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화된 미래입니다.
            shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다.
            shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

    Returns:
        torch.Tensor: 평균 방향 오차입니다. degree 단위 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    pred_head = torch.atan2(pred_clean_norm[..., 3], pred_clean_norm[..., 2])
    target_head = torch.atan2(target_clean_norm[..., 3], target_clean_norm[..., 2])
    diff_head = wrap_angle(pred_head - target_head).abs()
    return diff_head.mul(180.0 / torch.pi).mean()


def yaw_fde_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    """2초 마지막 시점의 방향 오차를 degree 단위로 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화된 미래입니다.
            shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다.
            shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

    Returns:
        torch.Tensor: 마지막 시점 방향 오차입니다. degree 단위 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    pred_head = torch.atan2(pred_clean_norm[:, -1, 3], pred_clean_norm[:, -1, 2])
    target_head = torch.atan2(target_clean_norm[:, -1, 3], target_clean_norm[:, -1, 2])
    diff_head = wrap_angle(pred_head - target_head).abs()
    return diff_head.mul(180.0 / torch.pi).mean()
