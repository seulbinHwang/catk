from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.utils import wrap_angle
import torch.nn.functional as F


@dataclass(frozen=True)
class ConstraintHParams:
    """제약 projector가 공통으로 쓰는 고정 상수입니다.

    Attributes:
        dt: 연속 제어를 계산할 때 쓰는 시간 간격입니다. 단위는 초입니다.
        eps: 0으로 나누는 상황을 막기 위한 작은 값입니다.
        eta_slip: 외부 feasible 구현과 같은 slip 관련 밴드 값입니다.
        eta_speed: 외부 feasible 구현과 같은 속도 관련 밴드 값입니다.
        eta_inc: 외부 feasible 구현과 같은 가속도 관련 밴드 값입니다.
        eta_yaw: 외부 feasible 구현과 같은 yaw-rate 관련 밴드 값입니다.
        eta_fric: 외부 feasible 구현과 같은 마찰 관련 밴드 값입니다.
    """

    dt: float = 0.1
    eps: float = 1e-6
    eta_slip: float = 0.07
    eta_speed: float = 0.05
    eta_inc: float = 0.10
    eta_yaw: float = 0.05
    eta_fric: float = 0.05


@dataclass(frozen=True)
class DynamicLimits:
    """객체 종류별 동역학 제한값입니다.

    Attributes:
        v_max_mps: 최대 속도입니다. 단위는 m/s 입니다.
        a_max_mps2: 최대 종가속도 절대값입니다. 단위는 m/s^2 입니다.
        alpha_max_radps2: 최대 yaw 가속도 절대값입니다. 단위는 rad/s^2 입니다.
        a_lat_max_mps2: 최대 횡가속도 절대값입니다. 단위는 m/s^2 입니다.
        r_min_m: 최소 선회 반경입니다. 단위는 m 입니다.
        omega_max_abs_radps: 최대 yaw-rate 절대값입니다. 단위는 rad/s 입니다.
        v_b_y_max: body y 방향 속도 정규화에 쓰는 기준값입니다. 단위는 m/s 입니다.
        beta_max_rad: slip angle 상한입니다. 단위는 rad 입니다.
    """

    v_max_mps: float
    a_max_mps2: float
    alpha_max_radps2: float
    a_lat_max_mps2: float
    r_min_m: float
    omega_max_abs_radps: float
    v_b_y_max: float
    beta_max_rad: float


@dataclass
class AdjointMatchingResult:
    """Adjoint Matching 학습 한 번의 결과를 모아 둡니다.

    Attributes:
        loss: 실제로 역전파할 regression loss 입니다. shape은 ``[]`` 입니다.
        terminal_cost: 학습용 stochastic rollout 마지막 feasible cost 평균입니다. shape은 ``[]`` 입니다.
        projection_gap: stochastic rollout의 정규화된 projector gap 평균 절대값입니다. shape은 ``[]`` 입니다.
        projection_gap_vx_b_mps: stochastic rollout의 body x 속도 gap 평균 절대값입니다. 단위는 m/s 입니다.
        projection_gap_vy_b_mps: stochastic rollout의 body y 속도 gap 평균 절대값입니다. 단위는 m/s 입니다.
        projection_gap_yaw_rate_degps: stochastic rollout의 yaw-rate gap 평균 절대값입니다. 단위는 deg/s 입니다.
        delta_velocity_norm: student velocity와 frozen teacher velocity 차이의 평균 제곱 크기입니다.
            shape은 ``[]`` 입니다.
        final_sample: rollout 마지막 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        diagnostic_metrics: 추가 projector 진단 로그입니다.
            ``gt_*``, ``deterministic_*``, ``stochastic_*`` 접두사를 사용합니다.
    """

    loss: Tensor
    terminal_cost: Tensor
    projection_gap: Tensor
    projection_gap_vx_b_mps: Tensor
    projection_gap_vy_b_mps: Tensor
    projection_gap_yaw_rate_degps: Tensor
    delta_velocity_norm: Tensor
    final_sample: Tensor
    diagnostic_metrics: Dict[str, Tensor]


class SmoothControlProjector(nn.Module):
    """미분 가능한 feasible terminal cost를 만드는 projector 입니다.

    이 클래스는 외부 feasible 구현의 제한값과 제약 순서를 그대로 가져오되,
    reward 쪽에서는 det ach 기반 STE를 쓰지 않고 실제 미분이 흐르는 연산만 사용합니다.
    그래서 terminal cost는 batch 전체에 대해 바로 autograd로 미분할 수 있습니다.
    """

    def __init__(
        self,
        feasible_weight: float = 1.0,
        smooth_deadzone_epsilon: Sequence[float] = (0.01, 0.01, 0.01),
        smooth_deadzone_tau: float = 0.002,
        hparams: ConstraintHParams | None = None,
    ) -> None:
        super().__init__()
        self.feasible_weight = float(feasible_weight)
        self.smooth_deadzone_tau = float(smooth_deadzone_tau)
        self.hparams = ConstraintHParams() if hparams is None else hparams

        deadzone = torch.tensor(list(smooth_deadzone_epsilon), dtype=torch.float32)
        if deadzone.shape != (3,):
            raise ValueError(
                "smooth_deadzone_epsilon must contain 3 values for [vx, vy, omega]."
            )
        self.register_buffer("smooth_deadzone_epsilon", deadzone, persistent=False)

        self._limits: Dict[int, DynamicLimits] = {
            0: DynamicLimits(
                v_max_mps=35.0,
                a_max_mps2=8.0,
                alpha_max_radps2=1.75,
                a_lat_max_mps2=4.2,
                r_min_m=4.50,
                omega_max_abs_radps=0.9,
                v_b_y_max=1.0,
                beta_max_rad=0.27,
            ),
            1: DynamicLimits(
                v_max_mps=5.0,
                a_max_mps2=4.7,
                alpha_max_radps2=14.0,
                a_lat_max_mps2=3.2,
                r_min_m=0.00001,
                omega_max_abs_radps=3.3,
                v_b_y_max=1.3,
                beta_max_rad=10.0,
            ),
            2: DynamicLimits(
                v_max_mps=22.0,
                a_max_mps2=5.5,
                alpha_max_radps2=6.0,
                a_lat_max_mps2=4.4,
                r_min_m=0.5,
                omega_max_abs_radps=2.0,
                v_b_y_max=1.3,
                beta_max_rad=0.7,
            ),
        }

    def _build_per_anchor_limits(
        self,
        agent_type: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """객체 종류별 제한값을 anchor 축에 맞는 텐서로 만듭니다.

        Args:
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            device: 반환 텐서를 둘 장치입니다.
            dtype: 반환 텐서 자료형입니다.

        Returns:
            Dict[str, Tensor]: 제한값 사전입니다. 모든 값의 shape은 ``[n_valid_anchor]`` 입니다.
        """
        if agent_type.dim() != 1:
            raise ValueError(f"agent_type must be 1D, got shape={tuple(agent_type.shape)}")

        supported_mask = torch.zeros_like(agent_type, dtype=torch.bool)
        for class_id in self._limits:
            supported_mask |= agent_type == class_id
        if bool((~supported_mask).any()):
            unknown_types = agent_type[~supported_mask].detach().unique().cpu().tolist()
            raise ValueError(f"Unsupported agent_type values: {unknown_types}")

        def _gather(attr_name: str) -> Tensor:
            values = torch.empty(agent_type.shape[0], device=device, dtype=dtype)
            for class_id, limit in self._limits.items():
                class_mask = agent_type == class_id
                if bool(class_mask.any()):
                    values[class_mask] = getattr(limit, attr_name)
            return values

        return {
            "v_max": _gather("v_max_mps"),
            "a_max": _gather("a_max_mps2"),
            "alpha_max": _gather("alpha_max_radps2"),
            "a_lat_max": _gather("a_lat_max_mps2"),
            "r_min": _gather("r_min_m"),
            "omega_abs_max": _gather("omega_max_abs_radps"),
            "v_b_y_max": _gather("v_b_y_max"),
            "beta_max_rad": _gather("beta_max_rad"),
            "is_nonholonomic": agent_type != 1,
        }

    @staticmethod
    def _normalize_heading_vectors(pred_clean_norm: Tensor, eps: float) -> Tensor:
        """예측된 ``[cos, sin]`` 쌍을 단위 길이로 다시 맞춥니다.

        Args:
            pred_clean_norm: 정규화된 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            eps: 수치 안정용 작은 값입니다.

        Returns:
            Tensor: 단위 길이로 맞춘 ``[cos, sin]`` 입니다. shape은 ``[n_valid_anchor, 20, 2]`` 입니다.
        """
        heading_vec = pred_clean_norm[..., 2:4]
        denom = torch.sqrt((heading_vec * heading_vec).sum(dim=-1, keepdim=True) + float(eps))
        return heading_vec / denom

    def trajectory_to_controls(self, pred_clean_norm: Tensor) -> Dict[str, Tensor]:
        """정규화된 2초 미래를 body-control 시퀀스로 바꿉니다.

        Args:
            pred_clean_norm: 정규화된 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            Dict[str, Tensor]: 아래 텐서를 담은 사전입니다. ( 비 정규화)
                - ``vx_b``: body x 속도. shape은 ``[n_valid_anchor, 20]`` 입니다.
                - ``vy_b``: body y 속도. shape은 ``[n_valid_anchor, 20]`` 입니다.
                - ``omega``: yaw-rate. shape은 ``[n_valid_anchor, 20]`` 입니다.
        """
        if pred_clean_norm.dim() != 3 or int(pred_clean_norm.shape[-1]) != 4:
            raise ValueError(
                "pred_clean_norm must have shape [n_valid_anchor, 20, 4], "
                f"got {tuple(pred_clean_norm.shape)}"
            )


        dt = float(self.hparams.dt)
        eps = float(self.hparams.eps)
        positions = pred_clean_norm[..., :2] * 20.0
        heading_vec = self._normalize_heading_vectors(pred_clean_norm, eps)
        heading = torch.atan2(heading_vec[..., 1], heading_vec[..., 0])

        zeros_pos = torch.zeros(
            (pred_clean_norm.shape[0], 1, 2),
            device=pred_clean_norm.device,
            dtype=pred_clean_norm.dtype,
        )
        zeros_head = torch.zeros(
            (pred_clean_norm.shape[0], 1),
            device=pred_clean_norm.device,
            dtype=pred_clean_norm.dtype,
        )

        prev_positions = torch.cat([zeros_pos, positions[:, :-1]], dim=1)
        prev_heading = torch.cat([zeros_head, heading[:, :-1]], dim=1)

        delta_positions = (positions - prev_positions) / dt
        delta_heading = wrap_angle(heading - prev_heading)
        heading_mid = prev_heading + 0.5 * delta_heading

        cos_mid = torch.cos(heading_mid)
        sin_mid = torch.sin(heading_mid)
        vx_b = delta_positions[..., 0] * cos_mid + delta_positions[..., 1] * sin_mid
        vy_b = -delta_positions[..., 0] * sin_mid + delta_positions[..., 1] * cos_mid
        omega = delta_heading / dt
        return {"vx_b": vx_b, "vy_b": vy_b, "omega": omega}

    @staticmethod
    def _compute_signed_speed_and_slip_components(
        vx_b: Tensor,
        vy_b: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """속도 크기만 다시 제한할 수 있도록 방향 성분을 분리합니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 또는 ``[n_valid_anchor]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``vx_b`` 와 같습니다.
            eps: 수치 안정용 작은 값입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - ``s_signed``: 부호가 붙은 속도 크기입니다. shape은 ``vx_b`` 와 같습니다.
                - ``c``: ``|vx| / speed`` 입니다. shape은 ``vx_b`` 와 같습니다.
                - ``u``: ``vy / speed`` 입니다. shape은 ``vx_b`` 와 같습니다.
        """
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + float(eps))
        sign = torch.where(vx_b >= 0.0, torch.ones_like(vx_b), -torch.ones_like(vx_b))
        s_signed = sign * speed
        denom = speed + float(eps)
        c = vx_b.abs() / denom
        u = vy_b / denom
        return s_signed, c, u

    def _apply_sideslip_limit(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        beta_max_rad: Tensor,
        is_nonholonomic: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """side-slip angle 제약으로 ``vy`` 만 줄입니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            beta_max_rad: slip angle 상한입니다. shape은 ``[n_valid_anchor]`` 입니다.
            is_nonholonomic: 차/자전거 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: 제약이 적용된 ``vx_b``, ``vy_b`` 입니다.
        """
        beta = beta_max_rad.to(dtype=vy_b.dtype, device=vy_b.device).unsqueeze(-1)
        vy_limit = (vx_b.abs() + float(self.hparams.eps)) * torch.tan(beta.clamp_min(0.0))
        clipped_vy = torch.clamp(vy_b, min=-vy_limit, max=vy_limit)
        active_mask = is_nonholonomic.unsqueeze(-1) & (beta_max_rad > 0.0).unsqueeze(-1)
        vy_out = torch.where(active_mask, clipped_vy, vy_b)
        return vx_b, vy_out

    def _apply_speed_limit(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        v_max: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """속도 벡터의 크기를 클래스별 최대 속도 안으로 넣습니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            v_max: 최대 속도입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: 제약이 적용된 ``vx_b``, ``vy_b`` 입니다.
        """
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + float(self.hparams.eps))
        scale = torch.clamp(v_max.unsqueeze(-1) / speed.clamp_min(float(self.hparams.eps)), max=1.0)
        return scale * vx_b, scale * vy_b

    def _apply_accel_and_alpha_limit(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        omega: Tensor,
        a_max: Tensor,
        alpha_max: Tensor,
        is_nonholonomic: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """속도 변화량과 yaw-rate 변화량을 시간축 전체에서 제한합니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            omega: yaw-rate 입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            a_max: 최대 종가속도입니다. shape은 ``[n_valid_anchor]`` 입니다.
            alpha_max: 최대 yaw 가속도입니다. shape은 ``[n_valid_anchor]`` 입니다.
            is_nonholonomic: 차/자전거 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: 위 control이 믿을 수 있는지 나타냅니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]: 제약이 적용된 ``vx_b``, ``vy_b``, ``omega`` 입니다.
        """
        if vx_b.shape[1] == 0:
            return vx_b, vy_b, omega

        dt = float(self.hparams.dt)
        accel_limit = a_max.unsqueeze(-1) * dt
        alpha_limit = alpha_max.unsqueeze(-1) * dt
        s_signed, c, u = self._compute_signed_speed_and_slip_components(
            vx_b=vx_b,
            vy_b=vy_b,
            eps=float(self.hparams.eps),
        )

        if vx_b.shape[1] == 1:
            s_old = s_signed
            omega_old = omega
        else:
            ds = s_signed[:, 1:] - s_signed[:, :-1]
            ds = torch.clamp(ds, min=-accel_limit, max=accel_limit)
            s_old = torch.cat([s_signed[:, :1], s_signed[:, :1] + torch.cumsum(ds, dim=1)], dim=1)

            d_omega = omega[:, 1:] - omega[:, :-1]
            d_omega = torch.clamp(
                d_omega,
                min=-alpha_limit,
                max=alpha_limit,
            )
            omega_old = torch.cat(
                [omega[:, :1], omega[:, :1] + torch.cumsum(d_omega, dim=1)],
                dim=1,
            )

        if current_control is None:
            s_selected = s_old
            omega_selected = omega_old
        else:
            if current_control.dim() != 2 or int(current_control.shape[-1]) != 3:
                raise ValueError(
                    "current_control must have shape [n_valid_anchor, 3], "
                    f"got {tuple(current_control.shape)}"
                )
            if tuple(current_control.shape[:1]) != tuple(vx_b.shape[:1]):
                raise ValueError(
                    "current_control first dimension must match vx_b, "
                    f"got {tuple(current_control.shape)} and {tuple(vx_b.shape)}"
                )

            if current_control_valid is None:
                use_prev = torch.ones(vx_b.shape[0], device=vx_b.device, dtype=torch.bool)
            else:
                use_prev = current_control_valid.to(device=vx_b.device, dtype=torch.bool)

            prev_vx = current_control[:, 0]
            prev_vy = current_control[:, 1]
            prev_omega = current_control[:, 2]
            prev_signed_speed, _, _ = self._compute_signed_speed_and_slip_components(
                vx_b=prev_vx,
                vy_b=prev_vy,
                eps=float(self.hparams.eps),
            )

            ds0 = s_signed[:, :1] - prev_signed_speed.unsqueeze(-1)
            ds_rest = s_signed[:, 1:] - s_signed[:, :-1]
            ds_all = torch.cat([ds0, ds_rest], dim=1)
            ds_all = torch.clamp(ds_all, min=-accel_limit, max=accel_limit)
            s_new = prev_signed_speed.unsqueeze(-1) + torch.cumsum(ds_all, dim=1)

            d_omega0 = omega[:, :1] - prev_omega.unsqueeze(-1)
            d_omega_rest = omega[:, 1:] - omega[:, :-1]
            d_omega_all = torch.cat([d_omega0, d_omega_rest], dim=1)
            d_omega_all = torch.clamp(d_omega_all, min=-alpha_limit, max=alpha_limit)
            omega_new = prev_omega.unsqueeze(-1) + torch.cumsum(d_omega_all, dim=1)

            use_prev_mask = use_prev.unsqueeze(-1)
            s_selected = torch.where(use_prev_mask, s_new, s_old)
            omega_selected = torch.where(use_prev_mask, omega_new, omega_old)

        vx_new = s_selected * c
        vy_new = s_selected.abs() * u
        nonholonomic_mask = is_nonholonomic.unsqueeze(-1)
        vx_out = torch.where(nonholonomic_mask, vx_new, vx_b)
        vy_out = torch.where(nonholonomic_mask, vy_new, vy_b)
        omega_out = torch.where(nonholonomic_mask, omega_selected, omega)
        return vx_out, vy_out, omega_out

    def _apply_omega_limit(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        omega: Tensor,
        a_lat_max: Tensor,
        r_min: Tensor,
        omega_abs_max: Tensor,
        is_nonholonomic: Tensor,
    ) -> Tensor:
        """횡가속, 최소 반경, 절대 상한을 이용해 yaw-rate 를 제한합니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            omega: yaw-rate 입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            a_lat_max: 최대 횡가속도입니다. shape은 ``[n_valid_anchor]`` 입니다.
            r_min: 최소 반경입니다. shape은 ``[n_valid_anchor]`` 입니다.
            omega_abs_max: 절대 yaw-rate 상한입니다. shape은 ``[n_valid_anchor]`` 입니다.
            is_nonholonomic: 차/자전거 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Tensor: 제약이 적용된 yaw-rate 입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
        """
        eps = float(self.hparams.eps)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)
        allow_lat = a_lat_max.unsqueeze(-1) / (speed + eps)
        allow_radius = vx_b.abs() / (r_min.unsqueeze(-1) + eps)
        allow_abs = omega_abs_max.unsqueeze(-1)

        allow_nonholonomic = torch.minimum(torch.minimum(allow_lat, allow_radius), allow_abs)
        allow_holonomic = allow_abs
        allow = torch.where(is_nonholonomic.unsqueeze(-1), allow_nonholonomic, allow_holonomic)
        return torch.clamp(omega, min=-allow, max=allow)

    def project_controls(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        omega: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """raw control 시퀀스에 hand-crafted constraint filter 를 적용합니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            omega: yaw-rate 입니다. shape은 ``[n_valid_anchor, 20]`` 입니다.
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
                projector 출력 ``vx_b``, ``vy_b``, ``omega`` 와
                    - shape: ( ``[n_valid_anchor, 20, 3]`` )
                limits: 정규화에 쓸 제한값 사전
        """
        limits = self._build_per_anchor_limits(
            agent_type=agent_type,
            device=vx_b.device,
            dtype=vx_b.dtype,
        )
        vx_proj, vy_proj = self._apply_sideslip_limit(
            vx_b=vx_b,
            vy_b=vy_b,
            beta_max_rad=limits["beta_max_rad"],
            is_nonholonomic=limits["is_nonholonomic"],
        )
        vx_proj, vy_proj = self._apply_speed_limit(
            vx_b=vx_proj,
            vy_b=vy_proj,
            v_max=limits["v_max"],
        )
        vx_proj, vy_proj, omega_proj = self._apply_accel_and_alpha_limit(
            vx_b=vx_proj,
            vy_b=vy_proj,
            omega=omega,
            a_max=limits["a_max"],
            alpha_max=limits["alpha_max"],
            is_nonholonomic=limits["is_nonholonomic"],
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        omega_proj = self._apply_omega_limit(
            vx_b=vx_proj,
            vy_b=vy_proj,
            omega=omega_proj,
            a_lat_max=limits["a_lat_max"],
            r_min=limits["r_min"],
            omega_abs_max=limits["omega_abs_max"],
            is_nonholonomic=limits["is_nonholonomic"],
        )
        return vx_proj, vy_proj, omega_proj, limits

    def _smooth_abs(self, value: Tensor) -> Tensor:
        """절대값 대신 쓰는 부드러운 크기 함수를 계산합니다.

        Args:
            value: 입력 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 부드러운 크기입니다. shape은 입력과 같습니다.
        """
        tau = float(self.smooth_deadzone_tau)
        return torch.sqrt(value * value + tau * tau)

    def _smooth_deadzone(self, value: Tensor) -> Tensor:
        """작은 projector gap 은 눌러 주고, 큰 gap 은 부드럽게 남깁니다.

        Args:
            value: 정규화된 gap 입니다. shape은 ``[n_valid_anchor, 20, 3]`` 입니다.

        Returns:
            Tensor: smooth dead-zone 이 적용된 gap 입니다. shape은 입력과 같습니다.
                ``[n_valid_anchor, 20, 3]``
        """
        tau = float(self.smooth_deadzone_tau)
        epsilon = self.smooth_deadzone_epsilon.view(1, 1, 3).to(value)
        return tau * F.softplus((self._smooth_abs(value) - epsilon) / tau)

    @staticmethod
    def prefix_metric_keys(prefix: str, metrics: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Projector metric dict에 접두사를 붙여 logging 키로 바꿉니다."""
        return {f"{prefix}_{name}": value.detach() for name, value in metrics.items()}

    def compute_terminal_cost(
        self,
        pred_clean_norm: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """class-wise smooth control projection gap terminal cost 를 계산합니다.

        Args:
            pred_clean_norm: rollout 마지막 정규화 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            agent_type: anchor별 객체 종류 번호. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 body control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            terminal_cost : Tensor : shape : [ ]
                batch 평균 terminal cost (가중치 같은거 곱해짐)
            metrics : Dict[str, Tensor]
                logging용 스칼라 사전입니다.
                "terminal_cost" : 값 1개
                "projection_gap" : [n_valid_anchor, 20, 3] 의 평균
                "projection_gap_vx_b_mps" : body x 속도 gap 평균 절대값
                "projection_gap_vy_b_mps" : body y 속도 gap 평균 절대값
                "projection_gap_yaw_rate_degps" : yaw-rate gap 평균 절대값
        """
        if pred_clean_norm.numel() == 0:
            zero = pred_clean_norm.sum() * 0.0
            return zero, {
                "terminal_cost": zero.detach(),
                "projection_gap": zero.detach(),
                "projection_gap_vx_b_mps": zero.detach(),
                "projection_gap_vy_b_mps": zero.detach(),
                "projection_gap_yaw_rate_degps": zero.detach(),
            }
        """ controls Dict[str, Tensor]
            : 아래 텐서를 담은 사전입니다. ( 비 정규화)
                - ``vx_b``: body x 속도. shape은 ``[n_valid_anchor, 20]`` 
                - ``vy_b``: body y 속도. shape은 ``[n_valid_anchor, 20]`` 
                - ``omega``: yaw-rate. shape은 ``[n_valid_anchor, 20]``
        
        """
        controls = self.trajectory_to_controls(pred_clean_norm)
        """
        vx_proj, vy_proj, omega_proj: ``[n_valid_anchor, 20, 3]`` 
        limits : Dict[str, Tensor]
        """
        vx_proj, vy_proj, omega_proj, limits = self.project_controls(
            vx_b=controls["vx_b"],
            vy_b=controls["vy_b"],
            omega=controls["omega"],
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        # gap : ``[n_valid_anchor, 20, 3]``
        gap = torch.stack(
            [
                controls["vx_b"] - vx_proj,
                controls["vy_b"] - vy_proj,
                controls["omega"] - omega_proj,
            ],
            dim=-1,
        )
        # scale: ``[n_valid_anchor, 3]``
        scale = torch.stack(
            [
                limits["v_max"],
                limits["v_b_y_max"],
                limits["omega_abs_max"],
            ],
            dim=-1,
        ).unsqueeze(1)
        # normalized_gap: ``[n_valid_anchor, 20, 3]``
        normalized_gap = gap / scale.clamp_min(float(self.hparams.eps))
        # smooth_gap: ``[n_valid_anchor, 20, 3]``d
        smooth_gap = self._smooth_deadzone(normalized_gap)
        # per_anchor_cost: ``[n_valid_anchor]``
        per_anchor_cost = smooth_gap.pow(2).sum(dim=-1).mean(dim=-1)
        gap_abs_mean = gap.abs().mean(dim=(0, 1))
        # terminal_cost: shape: ``[1]``
        terminal_cost = self.feasible_weight * per_anchor_cost.mean()
        metrics = {
            "terminal_cost": terminal_cost.detach(), #
            "projection_gap": normalized_gap.abs().mean().detach(), # [n_valid_anchor, 20, 3]
            "projection_gap_vx_b_mps": gap_abs_mean[0].detach(),
            "projection_gap_vy_b_mps": gap_abs_mean[1].detach(),
            "projection_gap_yaw_rate_degps": gap_abs_mean[2].mul(180.0 / torch.pi).detach(),
        }
        return terminal_cost, metrics


class AdjointMatchingLoss(nn.Module):
    """Teacher/student suffix-only Adjoint Matching loss 입니다."""

    def __init__(
        self,
        rollout_steps: int = 4,
        rollout_noise_scale: float = 1.0,
        feasible_weight: float = 1.0,
        smooth_deadzone_epsilon: Sequence[float] = (0.01, 0.01, 0.01),
        smooth_deadzone_tau: float = 0.002,
    ) -> None:
        super().__init__()
        self.rollout_steps = int(rollout_steps)
        self.rollout_noise_scale = float(rollout_noise_scale)
        self.projector = SmoothControlProjector(
            feasible_weight=feasible_weight,
            smooth_deadzone_epsilon=smooth_deadzone_epsilon,
            smooth_deadzone_tau=smooth_deadzone_tau,
        )

    @staticmethod
    def _zero_loss_with_trainable_dependency(reference: Tensor, module: nn.Module) -> Tensor:
        """빈 anchor batch에서도 trainable parameter graph를 유지하는 0 loss를 만듭니다."""
        zero = reference.sum() * 0.0
        for parameter in module.parameters():
            if parameter.requires_grad:
                zero = zero + parameter.sum() * 0.0
        return zero

    @staticmethod
    def _assert_finite_tensor(name: str, value: Tensor) -> None:
        """중간 텐서가 NaN/Inf면 바로 실패시킵니다."""
        if value.numel() == 0:
            return
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return
        bad_values = value.detach()[~finite_mask].flatten()[:8].cpu().tolist()
        raise RuntimeError(f"{name} contains non-finite values: {bad_values}")

    @classmethod
    def _assert_finite_tensor_list(cls, name: str, values: Sequence[Tensor]) -> None:
        """여러 rollout 텐서를 순서대로 검사합니다."""
        for idx, value in enumerate(values):
            cls._assert_finite_tensor(f"{name}[{idx}]", value)

    def _build_suffix_rollout_schedule(
        self,
        flow_ode: nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[List[Tensor], int, float]:
        """전체 16-step grid 안에서 suffix-only rollout 시간표를 만듭니다.

        Args:
            flow_ode: ODE helper 입니다.
            batch_size: 유효 anchor 개수입니다.
            device: 시간 텐서를 둘 장치입니다.
            dtype: 시간 텐서 자료형입니다.

        Returns:
            tuple[List[Tensor], int, float]:
                - suffix 시작부터 끝까지의 시간 텐서 목록입니다.
                  길이는 ``rollout_steps + 1`` 이고 각 원소 shape은 ``[n_valid_anchor]`` 입니다.
                - 전체 16-step grid에서 suffix가 시작하는 step 번호입니다.
                - 전체 grid의 고정 step 간격입니다.
        """
        total_steps = int(flow_ode.solver_steps)
        if self.rollout_steps > total_steps:
            raise ValueError(
                "AdjointMatching rollout_steps must be smaller than or equal to flow_ode.solver_steps. "
                f"Got rollout_steps={self.rollout_steps}, flow_ode.solver_steps={total_steps}."
            )
        suffix_start_step = total_steps - self.rollout_steps
        t0 = float(flow_ode.eps)
        dt = (1.0 - t0) / float(total_steps)
        times = [
            torch.full(
                (batch_size,),
                t0 + (suffix_start_step + step_idx) * dt,
                device=device,
                dtype=dtype,
            )
            for step_idx in range(self.rollout_steps + 1)
        ]
        return times, suffix_start_step, dt

    @torch.no_grad()
    def _rollout_suffix_only_memoryless_sde(
        self,
        teacher_flow_decoder: nn.Module,
        student_flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> tuple[List[Tensor], List[Tensor]]:
        """Teacher prefix 뒤에 student suffix-only SDE rollout을 만듭니다.

        Args:
            teacher_flow_decoder: frozen teacher local decoder 입니다.
            student_flow_decoder: fine-tuning 대상 student local decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.

        Returns:
            tuple[List[Tensor], List[Tensor]]:
                - suffix 상태 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
                - suffix 시간 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor]`` 입니다.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        dtype = anchor_hidden_valid.dtype
        device = anchor_hidden_valid.device

        times, suffix_start_step, dt = self._build_suffix_rollout_schedule(
            flow_ode=flow_ode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        x_init = torch.randn(
            batch_size,
            20,
            4,
            device=device,
            dtype=dtype,
        ) * self.rollout_noise_scale

        if suffix_start_step > 0:
            current_state = flow_ode.generate(
                x_init=x_init,
                model_fn=lambda x_t, tau: teacher_flow_decoder(anchor_hidden_valid, x_t, tau),
                steps=suffix_start_step,
                start_step=0,
                total_steps=int(flow_ode.solver_steps),
            )
        else:
            current_state = x_init

        states: List[Tensor] = [current_state.detach()]
        for local_step in range(self.rollout_steps):
            tau = times[local_step]
            student_velocity = student_flow_decoder(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=current_state,
                tau=tau,
            )
            drift = flow_ode.drift_from_velocity(
                x_t=current_state,
                velocity=student_velocity,
                tau=tau,
            )
            noise = torch.randn_like(current_state)
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)
            current_state = current_state + dt * drift + (dt ** 0.5) * sigma * noise
            states.append(current_state.detach())

        return states, times

    def _compute_terminal_gradient(
        self,
        final_state: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """마지막 feasible cost 와 그 gradient 를 계산합니다."""
        final_state_for_grad = final_state.detach().requires_grad_(True)
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state_for_grad,
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        terminal_grad = torch.autograd.grad(terminal_cost, final_state_for_grad)[0].detach()
        return terminal_cost, terminal_grad, metrics

    def _build_teacher_base_drift(
        self,
        teacher_flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        x_state: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Frozen teacher velocity만 사용한 drift 를 계산합니다."""
        teacher_velocity = teacher_flow_decoder(
            anchor_hidden=anchor_hidden_valid,
            x_t_norm=x_state,
            tau=tau,
        )
        return flow_ode.drift_from_velocity(
            x_t=x_state,
            velocity=teacher_velocity,
            tau=tau,
        )

    def _build_lean_adjoints(
        self,
        teacher_flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],
        times: Sequence[Tensor],
        terminal_grad: Tensor,
    ) -> List[Tensor]:
        """Frozen teacher drift 로 suffix lean adjoint 를 뒤로 풉니다."""
        total_steps = int(flow_ode.solver_steps)
        dt = (1.0 - float(flow_ode.eps)) / float(total_steps)
        adjoints: List[Tensor] = [terminal_grad]

        for step_idx in range(self.rollout_steps - 1, -1, -1):
            next_state = states[step_idx + 1].detach().requires_grad_(True)
            tau_next = times[step_idx + 1]
            with torch.enable_grad():
                base_drift = self._build_teacher_base_drift(
                    teacher_flow_decoder=teacher_flow_decoder,
                    flow_ode=flow_ode,
                    anchor_hidden_valid=anchor_hidden_valid,
                    x_state=next_state,
                    tau=tau_next,
                )
                j_t_a = torch.autograd.grad(
                    outputs=base_drift,
                    inputs=next_state,
                    grad_outputs=adjoints[-1],
                    retain_graph=False,
                    create_graph=False,
                )[0]
            adjoints.append((adjoints[-1] + dt * j_t_a).detach())

        adjoints.reverse()
        return adjoints[:-1]

    def _build_regression_loss(
        self,
        teacher_flow_decoder: nn.Module,
        student_flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],
        times: Sequence[Tensor],
        lean_adjoints: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Student-teacher velocity 차이를 lean adjoint target 에 맞춥니다."""
        step_losses: List[Tensor] = []
        delta_velocity_norms: List[Tensor] = []

        for step_idx in range(self.rollout_steps):
            x_state = states[step_idx].detach()
            tau = times[step_idx]
            teacher_velocity = teacher_flow_decoder(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_state,
                tau=tau,
            ).detach()
            student_velocity = student_flow_decoder(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_state,
                tau=tau,
            )
            delta_velocity = student_velocity - teacher_velocity
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)
            delta_velocity_norms.append(delta_velocity.pow(2).mean())
            regression_target = (2.0 / sigma) * delta_velocity + sigma * lean_adjoints[step_idx]
            step_losses.append(regression_target.flatten(1).pow(2).mean(dim=1).mean())

        if len(step_losses) == 0:
            zero = self._zero_loss_with_trainable_dependency(
                reference=anchor_hidden_valid,
                module=student_flow_decoder,
            )
            return zero, zero

        return torch.stack(step_losses).mean(), torch.stack(delta_velocity_norms).mean()

    def forward(
        self,
        teacher_flow_decoder: nn.Module,
        student_flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> AdjointMatchingResult:
        """Teacher prefix + suffix-only student AM fine-tuning loss 를 계산합니다."""
        if anchor_hidden_valid.numel() == 0:
            zero = self._zero_loss_with_trainable_dependency(
                reference=anchor_hidden_valid,
                module=student_flow_decoder,
            )
            empty_sample = anchor_hidden_valid.new_zeros((0, 20, 4))
            return AdjointMatchingResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                projection_gap_vx_b_mps=zero.detach(),
                projection_gap_vy_b_mps=zero.detach(),
                projection_gap_yaw_rate_degps=zero.detach(),
                delta_velocity_norm=zero.detach(),
                final_sample=empty_sample,
                diagnostic_metrics=self.projector.prefix_metric_keys(
                    "stochastic",
                    {
                        "terminal_cost": zero.detach(),
                        "projection_gap": zero.detach(),
                        "projection_gap_vx_b_mps": zero.detach(),
                        "projection_gap_vy_b_mps": zero.detach(),
                        "projection_gap_yaw_rate_degps": zero.detach(),
                    },
                ),
            )

        device_type = anchor_hidden_valid.device.type if anchor_hidden_valid.device.type else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
            if current_control is not None:
                current_control = current_control.to(
                    device=anchor_hidden_valid.device,
                    dtype=torch.float32,
                )
            states, times = self._rollout_suffix_only_memoryless_sde(
                teacher_flow_decoder=teacher_flow_decoder,
                student_flow_decoder=student_flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
            )
            self._assert_finite_tensor_list("am/states", states)
            terminal_cost, terminal_grad, metrics = self._compute_terminal_gradient(
                final_state=states[-1],
                agent_type=agent_type,
                current_control=current_control,
                current_control_valid=current_control_valid,
            )
            self._assert_finite_tensor("am/terminal_cost", terminal_cost)
            self._assert_finite_tensor("am/terminal_grad", terminal_grad)
            self._assert_finite_tensor("am/projection_gap", metrics["projection_gap"])
            self._assert_finite_tensor("am/projection_gap_vx_b_mps", metrics["projection_gap_vx_b_mps"])
            self._assert_finite_tensor("am/projection_gap_vy_b_mps", metrics["projection_gap_vy_b_mps"])
            self._assert_finite_tensor(
                "am/projection_gap_yaw_rate_degps",
                metrics["projection_gap_yaw_rate_degps"],
            )
            lean_adjoints = self._build_lean_adjoints(
                teacher_flow_decoder=teacher_flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                states=states,
                times=times,
                terminal_grad=terminal_grad,
            )
            self._assert_finite_tensor_list("am/lean_adjoints", lean_adjoints)
            regression_loss, delta_velocity_norm = self._build_regression_loss(
                teacher_flow_decoder=teacher_flow_decoder,
                student_flow_decoder=student_flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                states=states,
                times=times,
                lean_adjoints=lean_adjoints,
            )
            self._assert_finite_tensor("am/regression_loss", regression_loss)
            self._assert_finite_tensor("am/delta_velocity_norm", delta_velocity_norm)
            self._assert_finite_tensor("am/final_sample", states[-1])

            return AdjointMatchingResult(
                loss=regression_loss,
                terminal_cost=metrics["terminal_cost"],
                projection_gap=metrics["projection_gap"],
                projection_gap_vx_b_mps=metrics["projection_gap_vx_b_mps"],
                projection_gap_vy_b_mps=metrics["projection_gap_vy_b_mps"],
                projection_gap_yaw_rate_degps=metrics["projection_gap_yaw_rate_degps"],
                delta_velocity_norm=delta_velocity_norm.detach(),
                final_sample=states[-1],
                diagnostic_metrics=self.projector.prefix_metric_keys("stochastic", metrics),
            )
