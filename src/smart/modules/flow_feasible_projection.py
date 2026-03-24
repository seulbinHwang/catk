from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.smart.tokens.agent_token_matching import build_agent_type_masks
from src.smart.utils import wrap_angle


@dataclass(frozen=True)
class FlowConstraintParams:
    """제약 계산에 쓰는 고정 수치를 묶어 둡니다.

    Attributes:
        dt: 시퀀스 시간 간격입니다. 단위는 초입니다.
        eps: 0으로 나누는 일을 막기 위한 작은 값입니다.
        eta_slip: 원본 feasible.py와 같은 S0 설정값입니다.
        eta_speed: 원본 feasible.py와 같은 S1 설정값입니다.
        eta_inc: 원본 feasible.py와 같은 S2 설정값입니다.
        eta_yaw: 원본 feasible.py와 같은 S3 설정값입니다.
    """

    dt: float = 0.1
    eps: float = 1e-6
    eta_slip: float = 0.07
    eta_speed: float = 0.05
    eta_inc: float = 0.10
    eta_yaw: float = 0.05


@dataclass(frozen=True)
class FlowDynamicLimits:
    """차종별 동작 한계를 담습니다.

    Attributes:
        v_max_mps: 최대 속도입니다. 단위는 m/s 입니다.
        a_max_mps2: 최대 앞뒤 가속도 절대값입니다. 단위는 m/s^2 입니다.
        alpha_max_radps2: 최대 방향 변화 가속도입니다. 단위는 rad/s^2 입니다.
        a_lat_max_mps2: 최대 옆가속도입니다. 단위는 m/s^2 입니다.
        R_min_m: 최소 회전 반경입니다. 단위는 m 입니다.
        omega_max_abs_radps: 최대 방향 변화 속도 절대값입니다. 단위는 rad/s 입니다.
        v_b_y_max: 바디 y축 속도 정규화에 쓸 기준값입니다. 단위는 m/s 입니다.
        beta_max_rad: 사이드슬립 각 최대값입니다. 단위는 rad 입니다.
    """

    v_max_mps: float
    a_max_mps2: float
    alpha_max_radps2: float
    a_lat_max_mps2: float
    R_min_m: float
    omega_max_abs_radps: float
    v_b_y_max: float
    beta_max_rad: float


class FlowFeasibleProjector(nn.Module):
    """2초 미래를 control로 바꾼 뒤 feasible gap을 계산합니다.

    이 모듈은 Diffusion-Planner의 feasible.py에서 쓰는 차종별 제한값을 그대로 가져오고,
    forward 값 기준으로 S0 → S1 → S2 → S3 순서의 제약을 적용합니다.
    학습 때는 projector 출력 쪽을 detach해서 gap만 loss에 반영합니다.
    """

    def __init__(self, deadzone: float = 0.01) -> None:
        """projector를 초기화합니다.

        Args:
            deadzone: 정규화된 control gap에서 무시할 작은 구간 크기입니다.
        """
        super().__init__()
        self.params = FlowConstraintParams()
        self.register_buffer(
            "deadzone",
            torch.full((3,), float(deadzone), dtype=torch.float32),
            persistent=False,
        )
        self.constraints: Dict[str, FlowDynamicLimits] = {
            "veh": FlowDynamicLimits(
                v_max_mps=35.0,
                a_max_mps2=8.0,
                alpha_max_radps2=1.75,
                a_lat_max_mps2=4.2,
                R_min_m=4.50,
                omega_max_abs_radps=0.9,
                v_b_y_max=1.0,
                beta_max_rad=0.27,
            ),
            "ped": FlowDynamicLimits(
                v_max_mps=5.0,
                a_max_mps2=4.7,
                alpha_max_radps2=14.0,
                a_lat_max_mps2=3.2,
                R_min_m=0.00001,
                omega_max_abs_radps=3.3,
                v_b_y_max=1.3,
                beta_max_rad=10.0,
            ),
            "cyc": FlowDynamicLimits(
                v_max_mps=22.0,
                a_max_mps2=5.5,
                alpha_max_radps2=6.0,
                a_lat_max_mps2=4.4,
                R_min_m=0.5,
                omega_max_abs_radps=2.0,
                v_b_y_max=1.3,
                beta_max_rad=0.7,
            ),
        }

    def build_limits(self, actor_type: Tensor, dtype: torch.dtype) -> Dict[str, Tensor]:
        """packed anchor 순서의 차종별 제한값을 만듭니다.

        Args:
            actor_type: 차종 번호입니다. shape은 ``[n_anchor]`` 입니다.
            dtype: 반환 텐서 자료형입니다.

        Returns:
            Dict[str, Tensor]:
                차종별 제한값을 모은 사전입니다. 각 값의 shape은 ``[n_anchor]`` 입니다.
        """
        device = actor_type.device
        masks = build_agent_type_masks(actor_type)
        limits: Dict[str, Tensor] = {}
        for key, attr in {
            "v_max": "v_max_mps",
            "a_max": "a_max_mps2",
            "alpha_max": "alpha_max_radps2",
            "a_lat_max": "a_lat_max_mps2",
            "R_min": "R_min_m",
            "omega_abs_max": "omega_max_abs_radps",
            "v_b_y_max": "v_b_y_max",
            "beta_max_rad": "beta_max_rad",
        }.items():
            out = torch.empty(actor_type.shape[0], device=device, dtype=dtype)
            for token_key, mask in masks.items():
                if not bool(mask.any()):
                    continue
                value = getattr(self.constraints[token_key], attr)
                out[mask] = float(value)
            limits[key] = out
        limits["is_nonholonomic"] = ~masks["ped"]
        return limits

    def trajectory_to_body_control(self, traj_norm: Tensor) -> Tensor:
        """정규화된 2초 미래를 body control 시퀀스로 바꿉니다.

        Args:
            traj_norm: 정규화된 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
                마지막 축은 ``[x, y, cos, sin]`` 순서입니다.

        Returns:
            Tensor:
                body control 시퀀스입니다. shape은 ``[n_anchor, 20, 3]`` 입니다.
                마지막 축은 ``[v_x^b, v_y^b, omega]`` 순서입니다.
        """
        if traj_norm.numel() == 0:
            return traj_norm.new_zeros((0, 20, 3))

        dt = float(self.params.dt)
        pos = traj_norm[..., :2] * 20.0
        cos_sin = F.normalize(traj_norm[..., 2:4], dim=-1, eps=float(self.params.eps))
        yaw = torch.atan2(cos_sin[..., 1], cos_sin[..., 0])

        pos_prev = torch.cat([pos.new_zeros((pos.shape[0], 1, 2)), pos[:, :-1]], dim=1)
        yaw_prev = torch.cat([yaw.new_zeros((yaw.shape[0], 1)), yaw[:, :-1]], dim=1)
        dyaw = wrap_angle(yaw - yaw_prev)
        yaw_mid = yaw_prev + 0.5 * dyaw
        vel_world = (pos - pos_prev) / dt

        cos_mid = yaw_mid.cos()
        sin_mid = yaw_mid.sin()
        vx_b = vel_world[..., 0] * cos_mid + vel_world[..., 1] * sin_mid
        vy_b = -vel_world[..., 0] * sin_mid + vel_world[..., 1] * cos_mid
        omega = dyaw / dt
        return torch.stack([vx_b, vy_b, omega], dim=-1)

    def projection_gap(
        self,
        traj_norm: Tensor,
        actor_type: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> Tensor:
        """dead-zone control projection gap을 계산합니다.

        Args:
            traj_norm: 생성된 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
            actor_type: 차종 번호입니다. shape은 ``[n_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 구간 body control 입니다.
                shape은 ``[n_anchor, 3]`` 입니다.
            current_control_valid: ``current_control`` 이 유효한지 나타냅니다.
                shape은 ``[n_anchor]`` 입니다.

        Returns:
            Tensor:
                anchor별 feasible gap 값입니다. shape은 ``[n_anchor]`` 입니다.
        """
        if traj_norm.numel() == 0:
            return traj_norm.new_zeros((0,))

        raw_control = self.trajectory_to_body_control(traj_norm)
        projected_control = self.project_control(
            raw_control=raw_control,
            actor_type=actor_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        limits = self.build_limits(actor_type, dtype=traj_norm.dtype)
        norm_scale = torch.stack(
            [
                limits["v_max"],
                limits["v_b_y_max"],
                limits["omega_abs_max"],
            ],
            dim=-1,
        ).unsqueeze(1)
        normalized_gap = (raw_control - projected_control.detach()) / norm_scale.clamp_min(
            float(self.params.eps)
        )
        deadzone_gap = self._apply_deadzone(normalized_gap)
        return deadzone_gap.square().sum(dim=-1).mean(dim=-1)

    def project_control(
        self,
        raw_control: Tensor,
        actor_type: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> Tensor:
        """body control 시퀀스에 S0~S3 제약을 순서대로 적용합니다.

        Args:
            raw_control: raw body control 입니다. shape은 ``[n_anchor, 20, 3]`` 입니다.
            actor_type: 차종 번호입니다. shape은 ``[n_anchor]`` 입니다.
            current_control: anchor 직전 0.1초 구간 body control 입니다.
                shape은 ``[n_anchor, 3]`` 입니다.
            current_control_valid: ``current_control`` 유효 여부입니다.
                shape은 ``[n_anchor]`` 입니다.

        Returns:
            Tensor:
                projector를 지난 control 입니다. shape은 ``[n_anchor, 20, 3]`` 입니다.
        """
        if raw_control.numel() == 0:
            return raw_control.new_zeros((0, 20, 3))

        limits = self.build_limits(actor_type, dtype=raw_control.dtype)
        vx_raw = raw_control[..., 0]
        vy_raw = raw_control[..., 1]
        omega_raw = raw_control[..., 2]
        vx_proj, vy_proj, omega_proj = self._apply_constraints_batch(
            vx_b_raw=vx_raw,
            vy_b_raw=vy_raw,
            omega_raw=omega_raw,
            limits=limits,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        return torch.stack([vx_proj, vy_proj, omega_proj], dim=-1)

    def _apply_deadzone(self, control_gap: Tensor) -> Tensor:
        """작은 오차는 0으로 지우고 바깥쪽만 남깁니다.

        Args:
            control_gap: 정규화된 control gap 입니다.
                shape은 ``[n_anchor, 20, 3]`` 입니다.

        Returns:
            Tensor:
                dead-zone이 적용된 gap 입니다. shape은 ``[n_anchor, 20, 3]`` 입니다.
        """
        deadzone = self.deadzone.to(device=control_gap.device, dtype=control_gap.dtype)
        return torch.sign(control_gap) * (control_gap.abs() - deadzone.view(1, 1, 3)).clamp_min(0.0)

    def _apply_constraints_batch(
        self,
        vx_b_raw: Tensor,
        vy_b_raw: Tensor,
        omega_raw: Tensor,
        limits: Dict[str, Tensor],
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """feasible.py와 같은 순서로 S0~S3를 한 번에 적용합니다.

        Args:
            vx_b_raw: raw body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b_raw: raw body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            omega_raw: raw yaw-rate 입니다. shape은 ``[n_anchor, 20]`` 입니다.
            limits: 차종별 제한값 사전입니다. 각 값의 shape은 ``[n_anchor]`` 입니다.
            current_control: anchor 직전 body control 입니다. shape은 ``[n_anchor, 3]`` 입니다.
            current_control_valid: ``current_control`` 유효 여부입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                제약을 거친 ``vx_b``, ``vy_b``, ``omega`` 입니다.
                각 텐서 shape은 ``[n_anchor, 20]`` 입니다.
        """
        vx_after, vy_after = self._apply_s0_sideslip_angle_limit(
            vx_b=vx_b_raw,
            vy_b=vy_b_raw,
            beta_max_rad=limits["beta_max_rad"],
            is_nonholonomic=limits["is_nonholonomic"],
        )
        vx_after, vy_after = self._apply_s1_speed_limit(
            vx_b=vx_after,
            vy_b=vy_after,
            v_max=limits["v_max"],
        )
        vx_after, vy_after, omega_after = self._apply_s2_accel_alpha_limits_batch(
            vx_b=vx_after,
            vy_b=vy_after,
            omega=omega_raw,
            a_max=limits["a_max"],
            alpha_max=limits["alpha_max"],
            is_nonholonomic=limits["is_nonholonomic"],
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        omega_after = self._apply_s3_omega_clip(
            vx_b=vx_after,
            vy_b=vy_after,
            omega=omega_after,
            a_lat_max=limits["a_lat_max"],
            R_min=limits["R_min"],
            omega_abs_max=limits["omega_abs_max"],
            is_nonholonomic=limits["is_nonholonomic"],
        )
        return vx_after, vy_after, omega_after

    def _apply_s0_sideslip_angle_limit(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        beta_max_rad: Tensor,
        is_nonholonomic: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """사이드슬립 각 상한으로 ``v_y^b`` 만 줄입니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            beta_max_rad: 차종별 slip 각 상한입니다. shape은 ``[n_anchor]`` 입니다.
            is_nonholonomic: 비보행자 여부입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:
                slip 제한 뒤의 ``vx_b``, ``vy_b`` 입니다.
        """
        beta = beta_max_rad.to(device=vy_b.device, dtype=vy_b.dtype).unsqueeze(-1)
        enabled = beta > 0.0
        active = enabled & is_nonholonomic.unsqueeze(-1)
        vy_limit = (vx_b.abs() + float(self.params.eps)) * torch.tan(beta.clamp_min(0.0))
        vy_limit = torch.where(
            torch.isfinite(vy_limit),
            vy_limit,
            torch.full_like(vy_limit, float("inf")),
        )
        vy_new = vy_b.clamp(min=-vy_limit, max=vy_limit)
        return vx_b, torch.where(active, vy_new, vy_b)

    def _apply_s1_speed_limit(self, vx_b: Tensor, vy_b: Tensor, v_max: Tensor) -> tuple[Tensor, Tensor]:
        """속도 벡터 크기가 차종별 최대값을 넘지 않게 줄입니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            v_max: 차종별 최대 속도입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:
                속도 제한 뒤의 ``vx_b``, ``vy_b`` 입니다.
        """
        v_max_ex = v_max.to(device=vx_b.device, dtype=vx_b.dtype).unsqueeze(-1)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + float(self.params.eps))
        scale = torch.clamp(v_max_ex / speed.clamp_min(float(self.params.eps)), max=1.0)
        return scale * vx_b, scale * vy_b

    def _apply_s2_accel_alpha_limits_batch(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        omega: Tensor,
        a_max: Tensor,
        alpha_max: Tensor,
        is_nonholonomic: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """앞뒤 가속도와 방향 변화 가속도를 차종별 한계 안으로 맞춥니다.

        Args:
            vx_b: S0/S1 뒤의 body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b: S0/S1 뒤의 body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            omega: raw yaw-rate 입니다. shape은 ``[n_anchor, 20]`` 입니다.
            a_max: 최대 앞뒤 가속도입니다. shape은 ``[n_anchor]`` 입니다.
            alpha_max: 최대 yaw 가속도입니다. shape은 ``[n_anchor]`` 입니다.
            is_nonholonomic: 비보행자 여부입니다. shape은 ``[n_anchor]`` 입니다.
            current_control: anchor 직전 body control 입니다. shape은 ``[n_anchor, 3]`` 입니다.
            current_control_valid: ``current_control`` 유효 여부입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                S2 뒤의 ``vx_b``, ``vy_b``, ``omega`` 입니다.
                각 텐서 shape은 ``[n_anchor, 20]`` 입니다.
        """
        if vx_b.numel() == 0:
            return vx_b, vy_b, omega

        nonh = is_nonholonomic.to(device=vx_b.device)
        if not bool(nonh.any()):
            return vx_b, vy_b, omega

        dt = float(self.params.dt)
        eps = float(self.params.eps)
        s_signed, c, u = self._compute_signed_speed_and_slip_components(vx_b=vx_b, vy_b=vy_b)
        a_limit = a_max.to(device=vx_b.device, dtype=vx_b.dtype).unsqueeze(-1) * dt
        alpha_limit = alpha_max.to(device=omega.device, dtype=omega.dtype).unsqueeze(-1) * dt

        if vx_b.shape[-1] == 1:
            s_old = s_signed
            omega_old = omega
        else:
            ds = s_signed[:, 1:] - s_signed[:, :-1]
            ds_clamped = ds.clamp(min=-a_limit, max=a_limit)
            s0 = s_signed[:, :1]
            s_old = torch.cat([s0, s0 + torch.cumsum(ds_clamped, dim=1)], dim=1)

            dw = omega[:, 1:] - omega[:, :-1]
            dw_clamped = dw.clamp(min=-alpha_limit, max=alpha_limit)
            w0 = omega[:, :1]
            omega_old = torch.cat([w0, w0 + torch.cumsum(dw_clamped, dim=1)], dim=1)

        if current_control is None:
            s_sel = s_old
            omega_sel = omega_old
        else:
            if current_control.shape != (vx_b.shape[0], 3):
                raise ValueError(
                    "current_control shape은 [n_anchor, 3] 이어야 합니다. "
                    f"got {tuple(current_control.shape)}"
                )
            if current_control_valid is None:
                use_prev = torch.ones(vx_b.shape[0], device=vx_b.device, dtype=torch.bool)
            else:
                use_prev = current_control_valid.to(device=vx_b.device, dtype=torch.bool)
            if not bool(use_prev.any()):
                s_sel = s_old
                omega_sel = omega_old
            else:
                current_control = current_control.to(device=vx_b.device, dtype=vx_b.dtype)
                vx_prev0 = current_control[:, 0]
                vy_prev0 = current_control[:, 1]
                w_prev0 = current_control[:, 2]
                s_prev0, _, _ = self._compute_signed_speed_and_slip_components(
                    vx_b=vx_prev0.unsqueeze(-1),
                    vy_b=vy_prev0.unsqueeze(-1),
                )
                s_prev0 = s_prev0.squeeze(-1)

                ds0 = s_signed[:, :1] - s_prev0.unsqueeze(-1)
                ds_rest = s_signed[:, 1:] - s_signed[:, :-1]
                ds_all = torch.cat([ds0, ds_rest], dim=1)
                ds_clamped_all = ds_all.clamp(min=-a_limit, max=a_limit)
                s_new = s_prev0.unsqueeze(-1) + torch.cumsum(ds_clamped_all, dim=1)

                dw0 = omega[:, :1] - w_prev0.unsqueeze(-1)
                dw_rest = omega[:, 1:] - omega[:, :-1]
                dw_all = torch.cat([dw0, dw_rest], dim=1)
                dw_clamped_all = dw_all.clamp(min=-alpha_limit, max=alpha_limit)
                omega_new = w_prev0.unsqueeze(-1) + torch.cumsum(dw_clamped_all, dim=1)

                use_prev_ex = use_prev.unsqueeze(-1)
                s_sel = torch.where(use_prev_ex, s_new, s_old)
                omega_sel = torch.where(use_prev_ex, omega_new, omega_old)

        vx_new = s_sel * c
        vy_new = s_sel.abs() * u
        nonh_ex = nonh.unsqueeze(-1)
        vx_out = torch.where(nonh_ex, vx_new, vx_b)
        vy_out = torch.where(nonh_ex, vy_new, vy_b)
        omega_out = torch.where(nonh_ex, omega_sel, omega)
        _ = eps
        return vx_out, vy_out, omega_out

    def _apply_s3_omega_clip(
        self,
        vx_b: Tensor,
        vy_b: Tensor,
        omega: Tensor,
        a_lat_max: Tensor,
        R_min: Tensor,
        omega_abs_max: Tensor,
        is_nonholonomic: Tensor,
    ) -> Tensor:
        """옆가속도, 최소 회전 반경, 절대 yaw-rate 상한을 함께 적용합니다.

        Args:
            vx_b: S2 뒤의 body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b: S2 뒤의 body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            omega: S2 뒤의 yaw-rate 입니다. shape은 ``[n_anchor, 20]`` 입니다.
            a_lat_max: 차종별 최대 옆가속도입니다. shape은 ``[n_anchor]`` 입니다.
            R_min: 차종별 최소 회전 반경입니다. shape은 ``[n_anchor]`` 입니다.
            omega_abs_max: 차종별 최대 yaw-rate 절대값입니다. shape은 ``[n_anchor]`` 입니다.
            is_nonholonomic: 비보행자 여부입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            Tensor:
                S3 뒤의 yaw-rate 입니다. shape은 ``[n_anchor, 20]`` 입니다.
        """
        eps = float(self.params.eps)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)
        vx_abs = vx_b.abs()
        a_lat_max_ex = a_lat_max.to(device=vx_b.device, dtype=vx_b.dtype).unsqueeze(-1)
        r_min_ex = R_min.to(device=vx_b.device, dtype=vx_b.dtype).unsqueeze(-1)
        omega_abs_ex = omega_abs_max.to(device=omega.device, dtype=omega.dtype).unsqueeze(-1)

        allow_lat = a_lat_max_ex / (speed + eps)
        allow_radius = vx_abs / (r_min_ex + eps)
        allow_nonh = torch.minimum(torch.minimum(allow_lat, allow_radius), omega_abs_ex)
        allow_holo = omega_abs_ex
        allow = torch.where(is_nonholonomic.unsqueeze(-1), allow_nonh, allow_holo)
        return omega.clamp(min=-allow, max=allow)

    def _compute_signed_speed_and_slip_components(self, vx_b: Tensor, vy_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """속도 크기만 줄일 때 방향 느낌이 유지되도록 보조 값을 만듭니다.

        Args:
            vx_b: body x 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            vy_b: body y 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                순서대로 ``s_signed``, ``c``, ``u`` 입니다.
                각 텐서 shape은 모두 ``[n_anchor, 20]`` 입니다.
        """
        eps = float(self.params.eps)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)
        sign = torch.where(vx_b >= 0.0, torch.ones_like(vx_b), -torch.ones_like(vx_b))
        s_signed = sign * speed
        denom = speed + eps
        c = vx_b.abs() / denom
        u = vy_b / denom
        s_signed = torch.where(torch.isfinite(s_signed), s_signed, torch.zeros_like(s_signed))
        c = torch.where(torch.isfinite(c), c, torch.zeros_like(c))
        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        return s_signed, c, u
