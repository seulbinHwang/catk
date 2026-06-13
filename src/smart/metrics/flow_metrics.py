from __future__ import annotations

from typing import Sequence

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


def _validate_future_mask(value: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """미래 step mask의 shape을 확인하고 bool tensor로 돌려줍니다.

    Args:
        value: mask와 비교할 기준 tensor입니다. shape은 ``[n_anchor, n_future, ...]`` 입니다.
        valid_mask: loss 또는 metric에 포함할 미래 step입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 모든 step을 사용합니다.

    Returns:
        torch.Tensor | None: 기준 tensor와 같은 장치의 bool mask입니다.
    """
    if valid_mask is None:
        return None
    if tuple(valid_mask.shape) != tuple(value.shape[:2]):
        raise ValueError(
            "valid_mask shape must match the first two dimensions of value: "
            f"expected={tuple(value.shape[:2])}, actual={tuple(valid_mask.shape)}."
        )
    return valid_mask.to(device=value.device, dtype=torch.bool)


def _normalize_future_step_weights(
    value: torch.Tensor,
    future_step_weights: torch.Tensor | Sequence[float] | None,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """미래 step별 손실 가중치를 샘플별 평균 1로 정규화합니다.

    Args:
        value: loss 대상 tensor입니다. shape은 ``[n_anchor, n_future, ...]`` 입니다.
        future_step_weights: 미래 step별 원가중치입니다. shape은 ``[n_future]`` 입니다.
            값이 없으면 기존 loss와 같습니다.
        valid_mask: loss에 포함할 미래 step입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 전체 step을 사용합니다.

    Returns:
        torch.Tensor | None:
            가중치를 쓰지 않으면 ``None`` 입니다. 가중치를 쓰면 shape은
            ``[n_anchor, n_future]`` 이며, 각 샘플의 유효 step 평균이 1입니다.

    설명:
        commit-aligned Flow Matching은 open-loop 학습 objective에서만 명시적으로 넘겨 쓰는
        선택적 step 가중치입니다. 이 함수는 모든 샘플을 tensor broadcast로 한 번에 처리해
        Python loop 없이 유효 prefix별 평균 loss scale을 보존합니다.
    """
    if future_step_weights is None:
        return None
    n_anchor = int(value.shape[0])
    n_future = int(value.shape[1])
    step_weights = torch.as_tensor(
        future_step_weights,
        device=value.device,
        dtype=value.dtype,
    )
    if step_weights.numel() == 0:
        return None
    if step_weights.ndim != 1 or int(step_weights.shape[0]) != n_future:
        raise ValueError(
            "future_step_weights must have shape [n_future]: "
            f"expected={(n_future,)}, actual={tuple(step_weights.shape)}."
        )

    base_weight = step_weights.view(1, n_future).expand(n_anchor, n_future)
    if valid_mask is None:
        mean_weight = step_weights.mean().clamp_min(torch.finfo(value.dtype).tiny)
        return base_weight / mean_weight

    mask = _validate_future_mask(value=value, valid_mask=valid_mask)
    if mask is None:
        mean_weight = step_weights.mean().clamp_min(torch.finfo(value.dtype).tiny)
        return base_weight / mean_weight

    mask_float = mask.to(dtype=value.dtype)
    masked_weight = base_weight * mask_float
    valid_count = mask_float.sum(dim=1, keepdim=True)
    weight_sum = masked_weight.sum(dim=1, keepdim=True)
    return masked_weight * valid_count / weight_sum.clamp_min(torch.finfo(value.dtype).tiny)


def _masked_step_mean(step_value: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """step별 값을 mask가 켜진 위치만 평균냅니다.

    Args:
        step_value: step별 오차입니다. shape은 ``[n_anchor, n_future]`` 입니다.
        valid_mask: 평균에 포함할 step입니다. shape은 ``[n_anchor, n_future]`` 입니다.

    Returns:
        torch.Tensor: mask가 켜진 step만 평균낸 스칼라입니다.
    """
    if step_value.numel() == 0:
        return step_value.new_zeros(())
    if valid_mask is None:
        return step_value.mean()
    mask = _validate_future_mask(value=step_value, valid_mask=valid_mask)
    if mask is None:
        return step_value.mean()
    if not bool(mask.any().item()):
        return step_value.sum() * 0.0
    mask_float = mask.to(dtype=step_value.dtype)
    return (step_value * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def flow_matching_loss(
    flow_pred_norm: torch.Tensor,
    flow_target_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    future_step_weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Flow Matching loss를 유효한 미래 step 기준으로 계산합니다.

    Args:
        flow_pred_norm: 모델이 예측한 정규화 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 또는 ``[n_anchor, n_future, 3]`` 입니다.
        flow_target_norm: 정답 정규화 미래입니다.
            shape은 ``flow_pred_norm`` 과 같습니다.
        valid_mask: loss에 포함할 미래 step입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 전체 step을 사용합니다.
        future_step_weights: open-loop 학습에서만 쓰는 미래 step별 원가중치입니다.
            shape은 ``[n_future]`` 입니다. 값이 없으면 기존 평균 MSE와 같습니다.

    Returns:
        torch.Tensor: 평균 MSE loss 스칼라입니다.

    설명:
        ``future_step_weights`` 를 넘긴 경우에만 commit-aligned weighted MSE를 계산합니다.
        가중치는 각 샘플의 유효 future step 평균이 1이 되도록 정규화하므로 전체 loss
        scale을 유지합니다. 가중치가 없으면 기존 호출 경로와 같은 평균 MSE입니다.
    """
    if flow_pred_norm.numel() == 0:
        return flow_pred_norm.sum() * 0.0
    if valid_mask is None and future_step_weights is None:
        return F.mse_loss(flow_pred_norm, flow_target_norm)

    mask = _validate_future_mask(value=flow_pred_norm, valid_mask=valid_mask)
    if mask is not None and not bool(mask.any().item()):
        return flow_pred_norm.sum() * 0.0

    if future_step_weights is None:
        if mask is None:
            return F.mse_loss(flow_pred_norm, flow_target_norm)
        squared_error = (flow_pred_norm - flow_target_norm).square()
        mask_float = mask.to(dtype=squared_error.dtype)
        denom = mask_float.sum().clamp_min(1.0) * squared_error.shape[-1]
        return (squared_error * mask_float.unsqueeze(-1)).sum() / denom

    loss_pred_norm = flow_pred_norm.float()
    loss_target_norm = flow_target_norm.float()
    squared_error = (loss_pred_norm - loss_target_norm).square()
    future_weight = _normalize_future_step_weights(
        value=squared_error,
        future_step_weights=future_step_weights,
        valid_mask=mask,
    )
    if future_weight is None:
        if mask is None:
            return F.mse_loss(flow_pred_norm, flow_target_norm)
        mask_float = mask.to(dtype=squared_error.dtype)
        denom = mask_float.sum().clamp_min(1.0) * squared_error.shape[-1]
        return (squared_error * mask_float.unsqueeze(-1)).sum() / denom
    denom = future_weight.sum().clamp_min(1.0) * squared_error.shape[-1]
    return (squared_error * future_weight.unsqueeze(-1)).sum() / denom


def ade_future(
    pred_clean_norm: torch.Tensor,
    target_clean_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """미래 위치 평균 오차를 유효한 step만 사용해 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화 미래입니다. shape은 ``[n_anchor, n_future, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다. shape은 ``[n_anchor, n_future, 4]`` 입니다.
        valid_mask: 평균에 포함할 미래 step입니다. shape은 ``[n_anchor, n_future]`` 입니다.

    Returns:
        torch.Tensor: 위치 평균 오차 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    # diff_xy: [n_anchor, n_future, 2]
    diff_xy = (pred_clean_norm[..., :2] - target_clean_norm[..., :2]) * 20.0
    # step_error: [n_anchor, n_future]
    step_error = torch.norm(diff_xy, dim=-1)
    return _masked_step_mean(step_value=step_error, valid_mask=valid_mask)


def fde_future(
    pred_clean_norm: torch.Tensor,
    target_clean_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """각 anchor의 마지막 유효 미래 위치 오차를 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화 미래입니다. shape은 ``[n_anchor, n_future, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다. shape은 ``[n_anchor, n_future, 4]`` 입니다.
        valid_mask: 마지막 step을 고를 때 사용할 미래 step mask입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 전체 window의 마지막 step을 씁니다.

    Returns:
        torch.Tensor: 마지막 유효 미래 위치 오차 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    if valid_mask is None:
        diff_xy = (pred_clean_norm[:, -1, :2] - target_clean_norm[:, -1, :2]) * 20.0
        return torch.norm(diff_xy, dim=-1).mean()

    mask = _validate_future_mask(value=pred_clean_norm, valid_mask=valid_mask)
    if mask is None:
        diff_xy = (pred_clean_norm[:, -1, :2] - target_clean_norm[:, -1, :2]) * 20.0
        return torch.norm(diff_xy, dim=-1).mean()
    valid_step_count = mask.long().sum(dim=1)
    sample_mask = valid_step_count > 0
    if not bool(sample_mask.any().item()):
        return pred_clean_norm.sum() * 0.0

    last_index = valid_step_count[sample_mask] - 1
    gather_index = last_index.view(-1, 1, 1).expand(-1, 1, 2)
    # pred_last_xy: [n_valid_anchor, 2]
    pred_last_xy = pred_clean_norm[sample_mask, :, :2].gather(dim=1, index=gather_index).squeeze(1)
    # target_last_xy: [n_valid_anchor, 2]
    target_last_xy = target_clean_norm[sample_mask, :, :2].gather(dim=1, index=gather_index).squeeze(1)
    diff_xy = (pred_last_xy - target_last_xy) * 20.0
    return torch.norm(diff_xy, dim=-1).mean()


def yaw_ade_future(
    pred_clean_norm: torch.Tensor,
    target_clean_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """전체 미래 구간의 평균 방향 오차를 degree 단위로 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화된 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        valid_mask: 평균에 포함할 미래 step입니다. shape은 ``[n_anchor, n_future]`` 입니다.

    Returns:
        torch.Tensor: 평균 방향 오차입니다. degree 단위 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    pred_head = torch.atan2(pred_clean_norm[..., 3], pred_clean_norm[..., 2])
    target_head = torch.atan2(target_clean_norm[..., 3], target_clean_norm[..., 2])
    diff_head = wrap_angle(pred_head - target_head).abs()
    step_error = diff_head.mul(180.0 / torch.pi)
    return _masked_step_mean(step_value=step_error, valid_mask=valid_mask)


def yaw_fde_future(
    pred_clean_norm: torch.Tensor,
    target_clean_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """각 anchor의 마지막 유효 미래 방향 오차를 degree 단위로 계산합니다.

    Args:
        pred_clean_norm: 모델이 만든 정규화된 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        target_clean_norm: 정답 정규화 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        valid_mask: 마지막 step을 고를 때 사용할 미래 step mask입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 전체 window의 마지막 step을 씁니다.

    Returns:
        torch.Tensor: 마지막 유효 미래 방향 오차입니다. degree 단위 스칼라입니다.
    """
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    if valid_mask is None:
        pred_head = torch.atan2(pred_clean_norm[:, -1, 3], pred_clean_norm[:, -1, 2])
        target_head = torch.atan2(target_clean_norm[:, -1, 3], target_clean_norm[:, -1, 2])
        diff_head = wrap_angle(pred_head - target_head).abs()
        return diff_head.mul(180.0 / torch.pi).mean()

    mask = _validate_future_mask(value=pred_clean_norm, valid_mask=valid_mask)
    if mask is None:
        pred_head = torch.atan2(pred_clean_norm[:, -1, 3], pred_clean_norm[:, -1, 2])
        target_head = torch.atan2(target_clean_norm[:, -1, 3], target_clean_norm[:, -1, 2])
        diff_head = wrap_angle(pred_head - target_head).abs()
        return diff_head.mul(180.0 / torch.pi).mean()
    valid_step_count = mask.long().sum(dim=1)
    sample_mask = valid_step_count > 0
    if not bool(sample_mask.any().item()):
        return pred_clean_norm.sum() * 0.0

    last_index = valid_step_count[sample_mask] - 1
    gather_index = last_index.view(-1, 1, 1).expand(-1, 1, 2)
    # pred_last_heading_vec: [n_valid_anchor, 2]
    pred_last_heading_vec = pred_clean_norm[sample_mask, :, 2:4].gather(dim=1, index=gather_index).squeeze(1)
    # target_last_heading_vec: [n_valid_anchor, 2]
    target_last_heading_vec = target_clean_norm[sample_mask, :, 2:4].gather(dim=1, index=gather_index).squeeze(1)
    pred_head = torch.atan2(pred_last_heading_vec[:, 1], pred_last_heading_vec[:, 0])
    target_head = torch.atan2(target_last_heading_vec[:, 1], target_last_heading_vec[:, 0])
    diff_head = wrap_angle(pred_head - target_head).abs()
    return diff_head.mul(180.0 / torch.pi).mean()
