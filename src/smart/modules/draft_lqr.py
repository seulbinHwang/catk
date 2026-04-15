from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.smart.modules.flow_local_decoder import ContinuousCommitBridge
from src.smart.utils import transform_to_local, wrap_angle


DRAFT_LQR_METRIC_KEYS = (
    "commit_mse",
    "commit_pos_ade_m",
    "commit_pos_fde_m",
    "commit_yaw_ade_deg",
    "commit_yaw_fde_deg",
    "active_anchor_count",
)


def _build_zero_output(reference: Tensor) -> Dict[str, Tensor]:
    """LQR penalty logging에 필요한 0 값 사전을 만듭니다.

    Args:
        reference: 장치와 자료형만 빌려올 기준 텐서입니다.
            shape은 임의입니다.

    Returns:
        Dict[str, Tensor]:
            스칼라 0 값들로 채운 결과 사전입니다.
    """
    zero = reference.new_zeros(())
    output = {
        "loss": zero,
        "raw_pred_loss": zero,
    }
    for key in DRAFT_LQR_METRIC_KEYS:
        output[key] = zero
    return output


class DraftLQRRegularizer(nn.Module):
    """실행된 첫 0.5초를 그대로 GT와 맞추는 LQR penalty 입니다.

    이 모듈은 raw 2초 미래를 그대로 비교하지 않습니다.
    먼저 현재 runtime과 같은 LQR 실행 경로로 다음 0.5초 5개 점을 만든 뒤,
    그 결과를 현재 flow target과 같은 local 정규화 표현으로 바꿔
    GT 첫 0.5초와 평균 제곱 오차로 비교합니다.

    Args:
        commit_bridge: 실제 runtime과 같은 LQR 실행 다리입니다.
        pos_scale_m: local ``x, y`` 를 meter로 바꿀 때 쓸 배율입니다.
    """

    def __init__(
        self,
        commit_bridge: ContinuousCommitBridge,
        pos_scale_m: float = 20.0,
    ) -> None:
        super().__init__()
        if not bool(getattr(commit_bridge, "use_lqr", False)):
            raise ValueError("DraftLQRRegularizer requires decoder.use_lqr=true.")
        if bool(getattr(commit_bridge, "use_stop_motion", False)):
            raise ValueError(
                "DraftLQRRegularizer requires decoder.use_stop_motion=false "
                "so the training penalty matches the requested runtime path."
            )
        if bool(getattr(commit_bridge.config, "clip_longitudinal_command", False)):
            raise ValueError(
                "DraftLQRRegularizer requires "
                "decoder.lqr_commit.clip_longitudinal_command=false."
            )
        if bool(getattr(commit_bridge.config, "clip_lateral_projection_and_final_curvature_state", False)):
            raise ValueError(
                "DraftLQRRegularizer requires "
                "decoder.lqr_commit.clip_lateral_projection_and_final_curvature_state=false."
            )
        self.commit_bridge = commit_bridge
        self.pos_scale_m = float(pos_scale_m)

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_current_pos: Tensor,
        packed_current_head: Tensor,
        packed_exec_pos_history: Tensor,
        packed_exec_head_history: Tensor,
        packed_exec_valid_history: Tensor,
        packed_agent_type: Tensor,
    ) -> Dict[str, Tensor]:
        """실행된 첫 0.5초와 GT 첫 0.5초의 차이를 계산합니다.

        Args:
            pred_future_norm: 샘플러가 만든 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            packed_current_pos: 각 anchor의 현재 위치입니다.
                shape은 ``[n_valid_anchor, 2]`` 입니다.
            packed_current_head: 각 anchor의 현재 방향입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_exec_pos_history: 최근 실제 fine 위치 6개입니다.
                shape은 ``[n_valid_anchor, 6, 2]`` 입니다.
            packed_exec_head_history: 최근 실제 fine 방향 6개입니다.
                shape은 ``[n_valid_anchor, 6]`` 입니다.
            packed_exec_valid_history: 최근 실제 fine 유효 여부입니다.
                shape은 ``[n_valid_anchor, 6]`` 입니다.
            packed_agent_type: anchor 순서대로 압축한 agent 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                LQR 실행 penalty와 보기 쉬운 요약 지표 사전입니다.
        """
        if pred_future_norm.numel() == 0:
            return _build_zero_output(pred_future_norm)

        lqr_mask = self._build_lqr_agent_mask(packed_agent_type)
        if not lqr_mask.any():
            return _build_zero_output(pred_future_norm)

        commit_pos, commit_head, _, _ = self.commit_bridge.execute_lqr_commit(
            y_hat_norm=pred_future_norm[lqr_mask],
            current_pos=packed_current_pos[lqr_mask],
            current_head=packed_current_head[lqr_mask],
            exec_pos_history=packed_exec_pos_history[lqr_mask],
            exec_head_history=packed_exec_head_history[lqr_mask],
            exec_valid_history=packed_exec_valid_history[lqr_mask],
            agent_type=packed_agent_type[lqr_mask],
        )
        pred_commit_norm, pred_commit_local_pos_m, pred_commit_local_head = self._build_commit_local_norm(
            commit_pos=commit_pos,
            commit_head=commit_head,
            current_pos=packed_current_pos[lqr_mask],
            current_head=packed_current_head[lqr_mask],
        )
        target_commit_norm = target_future_norm[lqr_mask, :5]
        target_commit_local_pos_m = target_commit_norm[..., :2] * self.pos_scale_m
        target_commit_local_head = torch.atan2(target_commit_norm[..., 3], target_commit_norm[..., 2])

        loss = F.mse_loss(pred_commit_norm, target_commit_norm, reduction="mean")
        pos_error_m = torch.norm(pred_commit_local_pos_m - target_commit_local_pos_m, dim=-1)
        yaw_error_deg = wrap_angle(pred_commit_local_head - target_commit_local_head).abs() * (180.0 / torch.pi)

        return {
            "loss": loss,
            "raw_pred_loss": loss,
            "commit_mse": loss,
            "commit_pos_ade_m": pos_error_m.mean(),
            "commit_pos_fde_m": pos_error_m[:, -1].mean(),
            "commit_yaw_ade_deg": yaw_error_deg.mean(),
            "commit_yaw_fde_deg": yaw_error_deg[:, -1].mean(),
            "active_anchor_count": lqr_mask.sum().to(dtype=pred_future_norm.dtype),
        }

    def _build_lqr_agent_mask(self, packed_agent_type: Tensor) -> Tensor:
        """LQR penalty를 걸 대상 anchor만 고릅니다.

        Args:
            packed_agent_type: anchor 순서대로 압축한 agent 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Tensor:
                vehicle 또는 bicycle 인지 나타내는 마스크입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
        """
        return (packed_agent_type.long() == 0) | (packed_agent_type.long() == 2)

    def _build_commit_local_norm(
        self,
        commit_pos: Tensor,
        commit_head: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """실행된 0.5초를 flow target과 같은 local 정규화 표현으로 바꿉니다.

        Args:
            commit_pos: LQR가 실행한 다음 0.5초 위치입니다.
                shape은 ``[n_anchor, 5, 2]`` 입니다.
            commit_head: LQR가 실행한 다음 0.5초 방향입니다.
                shape은 ``[n_anchor, 5]`` 입니다.
            current_pos: 각 anchor의 현재 위치입니다.
                shape은 ``[n_anchor, 2]`` 입니다.
            current_head: 각 anchor의 현재 방향입니다.
                shape은 ``[n_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - commit_norm: local 정규화 표현입니다. shape은 ``[n_anchor, 5, 4]`` 입니다.
                - commit_pos_local_m: local meter 위치입니다. shape은 ``[n_anchor, 5, 2]`` 입니다.
                - commit_head_local: local 방향입니다. shape은 ``[n_anchor, 5]`` 입니다.
        """
        commit_pos_local_m, commit_head_local = transform_to_local(
            pos_global=commit_pos,
            head_global=commit_head,
            pos_now=current_pos,
            head_now=current_head,
        )
        commit_norm = torch.stack(
            [
                commit_pos_local_m[..., 0] / self.pos_scale_m,
                commit_pos_local_m[..., 1] / self.pos_scale_m,
                commit_head_local.cos(),
                commit_head_local.sin(),
            ],
            dim=-1,
        )
        return commit_norm, commit_pos_local_m, commit_head_local
