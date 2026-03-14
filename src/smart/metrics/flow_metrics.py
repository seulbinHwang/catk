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
