from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.utils import wrap_angle


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
        terminal_cost: 마지막 feasible cost 평균입니다. shape은 ``[]`` 입니다.
        projection_gap: 정규화된 projector gap의 평균 절대값입니다. shape은 ``[]`` 입니다.
        residual_norm: residual velocity의 평균 제곱 크기입니다. shape은 ``[]`` 입니다.
        final_sample: rollout 마지막 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
    """

    loss: Tensor
    terminal_cost: Tensor
    projection_gap: Tensor
    residual_norm: Tensor
    final_sample: Tensor


class SmoothControlProjector(nn.Module):
    """미분 가능한 feasible terminal cost를 만드는 projector 입니다.

    이 클래스는 외부 feasible 구현의 제한값과 제약 순서를 그대로 가져오되,
    reward 쪽에서는 detach 기반 STE를 쓰지 않고 실제 미분이 흐르는 연산만 사용합니다.
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
            Dict[str, Tensor]: 아래 텐서를 담은 사전입니다.
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
            ds = torch.clamp(ds, min=-accel_limit[:, :-1], max=accel_limit[:, :-1])
            s_old = torch.cat([s_signed[:, :1], s_signed[:, :1] + torch.cumsum(ds, dim=1)], dim=1)

            d_omega = omega[:, 1:] - omega[:, :-1]
            d_omega = torch.clamp(
                d_omega,
                min=-alpha_limit[:, :-1],
                max=alpha_limit[:, :-1],
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
                정규화에 쓸 제한값 사전입니다.
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
        """
        tau = float(self.smooth_deadzone_tau)
        epsilon = self.smooth_deadzone_epsilon.view(1, 1, 3).to(value)
        return tau * torch.log1p(torch.exp((self._smooth_abs(value) - epsilon) / tau))

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
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 body control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Dict[str, Tensor]]:
                batch 평균 terminal cost 와 logging용 스칼라 사전입니다.
        """
        if pred_clean_norm.numel() == 0:
            zero = pred_clean_norm.sum() * 0.0
            return zero, {"terminal_cost": zero.detach(), "projection_gap": zero.detach()}

        controls = self.trajectory_to_controls(pred_clean_norm)
        vx_proj, vy_proj, omega_proj, limits = self.project_controls(
            vx_b=controls["vx_b"],
            vy_b=controls["vy_b"],
            omega=controls["omega"],
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )

        gap = torch.stack(
            [
                controls["vx_b"] - vx_proj,
                controls["vy_b"] - vy_proj,
                controls["omega"] - omega_proj,
            ],
            dim=-1,
        )
        scale = torch.stack(
            [
                limits["v_max"],
                limits["v_b_y_max"],
                limits["omega_abs_max"],
            ],
            dim=-1,
        ).unsqueeze(1)
        normalized_gap = gap / scale.clamp_min(float(self.hparams.eps))
        smooth_gap = self._smooth_deadzone(normalized_gap)

        per_anchor_cost = smooth_gap.pow(2).sum(dim=-1).mean(dim=-1)
        terminal_cost = self.feasible_weight * per_anchor_cost.mean()
        metrics = {
            "terminal_cost": terminal_cost.detach(),
            "projection_gap": normalized_gap.abs().mean().detach(),
        }
        return terminal_cost, metrics


class AdjointMatchingLoss(nn.Module):
    """Residual velocity head 전용 Adjoint Matching loss 입니다."""

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

    def _build_step_times(self, flow_ode: nn.Module, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Tensor]:
        """Euler–Maruyama rollout 과 adjoint 계산에 쓸 시간값을 만듭니다.

        Args:
            flow_ode: ODE helper 입니다. ``eps`` 값을 읽습니다.
            batch_size: 유효 anchor 개수입니다.
            device: 시간 텐서를 둘 장치입니다.
            dtype: 시간 텐서 자료형입니다.

        Returns:
            List[Tensor]: ``t_0`` 부터 ``t_K`` 까지의 시간 텐서 목록입니다.
                각 원소 shape은 ``[n_valid_anchor]`` 입니다.
        """
        t0 = float(flow_ode.eps)
        dt = (1.0 - t0) / float(self.rollout_steps)
        return [
            torch.full((batch_size,), t0 + step_idx * dt, device=device, dtype=dtype)
            for step_idx in range(self.rollout_steps + 1)
        ]

    @torch.no_grad()
    def _rollout_memoryless_sde(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> tuple[List[Tensor], List[Tensor]]:
        """Memoryless Euler–Maruyama SDE로 학습용 rollout 을 만듭니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.

        Returns:
            tuple[List[Tensor], List[Tensor]]:
                - 상태 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
                - 시간 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor]`` 입니다.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        dtype = anchor_hidden_valid.dtype
        device = anchor_hidden_valid.device
        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(
            flow_ode=flow_ode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        current_state = torch.randn(
            batch_size,
            20,
            4,
            device=device,
            dtype=dtype,
        ) * self.rollout_noise_scale
        states: List[Tensor] = [current_state.detach()]

        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]
            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=current_state,
                tau=tau,
            )
            drift = flow_ode.drift_from_velocity(
                x_t=current_state,
                velocity=velocity_dict["velocity"],
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
        """마지막 feasible cost 와 그 gradient 를 계산합니다.

        Args:
            final_state: rollout 마지막 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Dict[str, Tensor]]:
                평균 terminal cost, 마지막 상태에 대한 gradient,
                그리고 logging용 스칼라 사전입니다.
        """
        final_state_for_grad = final_state.detach().requires_grad_(True)
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state_for_grad,
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        terminal_grad = torch.autograd.grad(terminal_cost, final_state_for_grad)[0].detach()
        return terminal_cost, terminal_grad, metrics

    def _build_base_drift(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        x_state: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Base velocity head 만 사용한 drift 를 계산합니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            x_state: 특정 시간의 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            tau: 그 상태의 시간값입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Tensor: base drift 입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        velocity_dict = flow_decoder.forward_components(
            anchor_hidden=anchor_hidden_valid,
            x_t_norm=x_state,
            tau=tau,
        )
        return flow_ode.drift_from_velocity(
            x_t=x_state,
            velocity=velocity_dict["base_velocity"],
            tau=tau,
        )

    def _build_lean_adjoints(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],
        times: Sequence[Tensor],
        terminal_grad: Tensor,
    ) -> List[Tensor]:
        """Base drift 로 lean adjoint 를 뒤로 풉니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            states: rollout 상태 목록입니다. 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            times: 상태별 시간 목록입니다. 각 원소 shape은 ``[n_valid_anchor]`` 입니다.
            terminal_grad: 마지막 상태에 대한 terminal gradient 입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            List[Tensor]: 각 rollout step의 lean adjoint 입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        adjoints: List[Tensor] = [terminal_grad]

        for step_idx in range(self.rollout_steps - 1, -1, -1):
            next_state = states[step_idx + 1].detach().requires_grad_(True)
            tau_next = times[step_idx + 1]
            with torch.enable_grad():
                base_drift = self._build_base_drift(
                    flow_decoder=flow_decoder,
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
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],
        times: Sequence[Tensor],
        lean_adjoints: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Residual velocity 를 lean adjoint target 에 맞춥니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            states: rollout 상태 목록입니다. 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            times: 상태별 시간 목록입니다. 각 원소 shape은 ``[n_valid_anchor]`` 입니다.
            lean_adjoints: 각 rollout step의 lean adjoint 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: 평균 regression loss 와 평균 residual norm 입니다.
        """
        step_losses: List[Tensor] = []
        residual_norms: List[Tensor] = []

        for step_idx in range(self.rollout_steps):
            x_state = states[step_idx].detach()
            tau = times[step_idx]
            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_state,
                tau=tau,
            )
            residual_velocity = velocity_dict["residual_velocity"]
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)
            residual_norms.append(residual_velocity.pow(2).mean())

            regression_target = (2.0 / sigma) * residual_velocity + sigma * lean_adjoints[step_idx]
            step_losses.append(regression_target.flatten(1).pow(2).sum(dim=1).mean())

        if len(step_losses) == 0:
            zero = anchor_hidden_valid.sum() * 0.0
            return zero, zero

        return torch.stack(step_losses).mean(), torch.stack(residual_norms).mean()

    def forward(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> AdjointMatchingResult:
        """Adjoint Matching fine-tuning loss 를 계산합니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 control 입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            AdjointMatchingResult: loss, terminal cost, gap, residual norm, 최종 sample 입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            zero = anchor_hidden_valid.sum() * 0.0
            empty_sample = anchor_hidden_valid.new_zeros((0, 20, 4))
            return AdjointMatchingResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                residual_norm=zero.detach(),
                final_sample=empty_sample,
            )

        device_type = anchor_hidden_valid.device.type if anchor_hidden_valid.device.type else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
            if current_control is not None:
                current_control = current_control.to(
                    device=anchor_hidden_valid.device,
                    dtype=torch.float32,
                )

            states, times = self._rollout_memoryless_sde(
                flow_decoder=flow_decoder,
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

            lean_adjoints = self._build_lean_adjoints(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                states=states,
                times=times,
                terminal_grad=terminal_grad,
            )
            self._assert_finite_tensor_list("am/lean_adjoints", lean_adjoints)

            regression_loss, residual_norm = self._build_regression_loss(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                states=states,
                times=times,
                lean_adjoints=lean_adjoints,
            )
            self._assert_finite_tensor("am/regression_loss", regression_loss)
            self._assert_finite_tensor("am/residual_norm", residual_norm)
            self._assert_finite_tensor("am/final_sample", states[-1])

            return AdjointMatchingResult(
                loss=regression_loss,
                terminal_cost=metrics["terminal_cost"],
                projection_gap=metrics["projection_gap"],
                residual_norm=residual_norm.detach(),
                final_sample=states[-1],
            )
