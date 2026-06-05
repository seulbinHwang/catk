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
) -> torch.Tensor:
    """Flow matching loss를 유효한 미래 step 기준으로 계산합니다.

    Args:
        flow_pred_norm: 모델이 예측한 정규화 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        flow_target_norm: 정답 정규화 미래입니다.
            shape은 ``[n_anchor, n_future, 4]`` 입니다.
        valid_mask: loss에 포함할 미래 step입니다.
            shape은 ``[n_anchor, n_future]`` 입니다. 값이 없으면 전체 step을 사용합니다.

    Returns:
        torch.Tensor: 평균 MSE loss 스칼라입니다.
    """
    if flow_pred_norm.numel() == 0:
        return flow_pred_norm.sum() * 0.0
    if valid_mask is None:
        return F.mse_loss(flow_pred_norm, flow_target_norm)

    mask = _validate_future_mask(value=flow_pred_norm, valid_mask=valid_mask)
    if mask is None:
        return F.mse_loss(flow_pred_norm, flow_target_norm)
    if not bool(mask.any().item()):
        return flow_pred_norm.sum() * 0.0

    # squared_error: [n_anchor, n_future, 4]
    squared_error = (flow_pred_norm - flow_target_norm).square()
    mask_float = mask.to(dtype=squared_error.dtype)
    denom = mask_float.sum().clamp_min(1.0) * squared_error.shape[-1]
    return (squared_error * mask_float.unsqueeze(-1)).sum() / denom


def mdg_state_loss(
    pred_state_norm: torch.Tensor,
    clean_state_norm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MDG control-state loss over valid 10Hz future steps."""
    return flow_matching_loss(
        flow_pred_norm=pred_state_norm,
        flow_target_norm=clean_state_norm,
        valid_mask=valid_mask,
    )


def auxiliary_best_mode_trajectory_loss(
    pred_local: torch.Tensor,
    target_local: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Best-mode 20-step auxiliary trajectory loss.

    Mode selection uses local xy L2 only. The selected mode receives Smooth L1
    supervision over local x, local y, and wrapped heading delta.
    """
    if pred_local.ndim != 4 or pred_local.shape[-1] != 3:
        raise ValueError(
            "pred_local must have shape [n_anchor, n_mode, n_future, 3], "
            f"got {tuple(pred_local.shape)}."
        )
    if target_local.ndim != 3 or target_local.shape[-1] != 3:
        raise ValueError(
            "target_local must have shape [n_anchor, n_future, 3], "
            f"got {tuple(target_local.shape)}."
        )
    if tuple(pred_local.shape[:1] + pred_local.shape[2:3]) != tuple(target_local.shape[:2]):
        raise ValueError(
            "pred_local and target_local anchor/future dimensions must match: "
            f"pred={tuple(pred_local.shape)}, target={tuple(target_local.shape)}."
        )
    mask = _validate_future_mask(value=target_local, valid_mask=valid_mask)
    if mask is None:
        raise ValueError("valid_mask is required for auxiliary trajectory loss.")
    if pred_local.numel() == 0:
        return pred_local.sum() * 0.0

    mask_float = mask.to(dtype=pred_local.dtype)
    valid_count = mask_float.sum(dim=-1).clamp_min(1.0)
    row_valid = mask.any(dim=-1).to(dtype=pred_local.dtype)
    xy_error_sq = (pred_local[..., :2] - target_local[:, None, :, :2]).square().sum(dim=-1)
    mode_distance = (xy_error_sq * mask_float[:, None, :]).sum(dim=-1)
    mode_distance = mode_distance / valid_count[:, None]
    best_mode = mode_distance.argmin(dim=1)

    row_index = torch.arange(pred_local.shape[0], device=pred_local.device)
    selected = pred_local[row_index, best_mode]
    error = selected - target_local
    error = error.clone()
    error[..., 2] = wrap_angle(error[..., 2])
    smooth_l1 = F.smooth_l1_loss(
        error,
        torch.zeros_like(error),
        reduction="none",
    ).sum(dim=-1)

    per_row_loss = (smooth_l1 * mask_float).sum(dim=-1) / valid_count
    return (per_row_loss * row_valid).sum() / row_valid.sum().clamp_min(1.0)


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


def ade_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    return ade_future(pred_clean_norm, target_clean_norm)


def fde_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    return fde_future(pred_clean_norm, target_clean_norm)


def yaw_ade_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    return yaw_ade_future(pred_clean_norm, target_clean_norm)


def yaw_fde_2s(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    return yaw_fde_future(pred_clean_norm, target_clean_norm)
