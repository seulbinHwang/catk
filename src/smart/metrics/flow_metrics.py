from __future__ import annotations

import torch
import torch.nn.functional as F


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


def _yaw_from_cos_sin(clean_norm: torch.Tensor) -> torch.Tensor:
    cos_sin = F.normalize(clean_norm[..., 2:4], dim=-1, eps=1e-6)
    return torch.atan2(cos_sin[..., 1], cos_sin[..., 0])


def _wrapped_angle_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return torch.atan2(diff.sin(), diff.cos())


def yaw_ade_2s_deg(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    pred_yaw = _yaw_from_cos_sin(pred_clean_norm)
    target_yaw = _yaw_from_cos_sin(target_clean_norm)
    yaw_diff = _wrapped_angle_diff(pred_yaw, target_yaw).abs()
    return torch.rad2deg(yaw_diff).mean()


def yaw_fde_2s_deg(pred_clean_norm: torch.Tensor, target_clean_norm: torch.Tensor) -> torch.Tensor:
    if pred_clean_norm.numel() == 0:
        return pred_clean_norm.new_zeros(())
    pred_yaw = _yaw_from_cos_sin(pred_clean_norm[:, -1])
    target_yaw = _yaw_from_cos_sin(target_clean_norm[:, -1])
    yaw_diff = _wrapped_angle_diff(pred_yaw, target_yaw).abs()
    return torch.rad2deg(yaw_diff).mean()
