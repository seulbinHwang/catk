from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class DynamicLimitTable:
    """에이전트 종류별 물리 제한값을 보관합니다.

    Attributes:
        v_max_mps: 최고 속도 제한입니다. shape은 ``[3]`` 입니다.
        v_reverse_max_mps: 후진 속도 크기 제한입니다. shape은 ``[3]`` 입니다.
        a_max_mps2: 앞방향 가감속 제한입니다. shape은 ``[3]`` 입니다.
        alpha_max_radps2: 회전 변화 제한입니다. shape은 ``[3]`` 입니다.
        a_lat_max_mps2: 횡가속 제한입니다. shape은 ``[3]`` 입니다.
        r_min_m: 최소 선회 반경 제한입니다. shape은 ``[3]`` 입니다.
        omega_max_abs_radps: 절대 회전속도 제한입니다. shape은 ``[3]`` 입니다.
        beta_max_rad: 사이드슬립 각도 제한입니다. shape은 ``[3]`` 입니다.
    """

    v_max_mps: Tuple[float, float, float]
    v_reverse_max_mps: Tuple[float, float, float]
    a_max_mps2: Tuple[float, float, float]
    alpha_max_radps2: Tuple[float, float, float]
    a_lat_max_mps2: Tuple[float, float, float]
    r_min_m: Tuple[float, float, float]
    omega_max_abs_radps: Tuple[float, float, float]
    beta_max_rad: Tuple[float, float, float]


DEFAULT_LIMITS = DynamicLimitTable(
    # CAT-K repo의 agent type 인덱스: vehicle=0, pedestrian=1, bicycle=2
    # 값은 Diffusion-Planner feasible.py의 제한값을 그대로 옮겼습니다.
    v_max_mps=(35.0, 5.0, 22.0),
    v_reverse_max_mps=(1.5, 0.0, 0.5),
    a_max_mps2=(8.0, 4.7, 5.5),
    alpha_max_radps2=(1.75, 14.0, 6.0),
    a_lat_max_mps2=(4.2, 3.2, 4.4),
    r_min_m=(4.50, 0.00001, 0.5),
    omega_max_abs_radps=(0.9, 3.3, 2.0),
    beta_max_rad=(0.27, 10.0, 0.7),
)


DRAFT_PHYSICS_COMPONENT_KEYS = (
    "speed",
    "slip",
    "accel",
    "yaw_accel",
    "turn",
)

DRAFT_PHYSICS_ACTUAL_UNIT_KEYS = (
    "speed_excess_mps",
    "slip_beta_excess_deg",
    "accel_excess_mps2",
    "yaw_accel_excess_degps2",
    "turn_yaw_rate_excess_degps",
    "turn_lat_accel_excess_mps2",
    "turn_radius_shortfall_m",
)


def _build_zero_output(reference: Tensor) -> Dict[str, Tensor]:
    zero = reference.new_zeros(())
    output = {
        "loss": zero,
        "raw_pred_loss": zero,
    }
    for key in DRAFT_PHYSICS_COMPONENT_KEYS:
        output[key] = zero
        output[f"pred_{key}"] = zero
        output[f"gt_{key}"] = zero
    for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
        output[key] = zero
        output[f"pred_{key}"] = zero
        output[f"gt_{key}"] = zero
    return output


class DraftPhysicsRegularizer(nn.Module):
    """최종 샘플 기준 physics penalty를 계산합니다.

    이 모듈은 DRaFT의 핵심처럼, 중간 noisy state가 아니라
    "최종으로 생성된 2초 미래"에 대해 물리 penalty를 계산합니다.
    penalty는 GT와 같은 기준으로 같이 계산한 뒤,
    필요하면 "GT보다 더 나쁜 만큼만" 남기도록 만들 수 있습니다.

    Args:
        dt: 미래 점 간 시간 간격입니다. 기본값은 ``0.1`` 초입니다.
        pos_scale_m: 정규화된 ``x, y`` 를 meter로 되돌릴 때 쓸 배율입니다.
            기본값은 ``20.0`` 입니다.
        deadzone_ratio: 작은 초과량은 바로 크게 벌주지 않기 위한 여유 폭입니다.
        deadzone_softness: dead-zone 경계를 부드럽게 만들기 위한 값입니다.
        gt_excess_only: ``True`` 이면 GT보다 더 나쁜 만큼만 loss에 넣습니다.
        speed_weight: 최고 속도 항 가중치입니다.
        slip_weight: slip angle 위반 항 가중치입니다.
        accel_weight: step 간 앞방향 속도 변화 항 가중치입니다.
            첫 delta는 직전 anchor 제어와의 차이로 함께 계산합니다.
        yaw_accel_weight: step 간 회전 변화 항 가중치입니다.
            첫 delta는 직전 anchor 제어와의 차이로 함께 계산합니다.
        turn_weight: 선회 가능성 항 가중치입니다.
        eps: 수치 안정용 작은 값입니다.
    """

    def __init__(
        self,
        dt: float = 0.1,
        pos_scale_m: float = 20.0,
        deadzone_ratio: float = 0.02,
        deadzone_softness: float = 0.02,
        gt_excess_only: bool = True,
        speed_weight: float = 1.0,
        slip_weight: float = 1.0,
        accel_weight: float = 1.0,
        yaw_accel_weight: float = 1.0,
        turn_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.pos_scale_m = float(pos_scale_m)
        self.deadzone_ratio = float(deadzone_ratio)
        self.deadzone_softness = float(deadzone_softness)
        self.gt_excess_only = bool(gt_excess_only)
        self.speed_weight = float(speed_weight)
        self.slip_weight = float(slip_weight)
        self.accel_weight = float(accel_weight)
        self.yaw_accel_weight = float(yaw_accel_weight)
        self.turn_weight = float(turn_weight)
        self.eps = float(eps)

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """생성 미래와 GT 미래의 physics penalty를 계산합니다.

        Args:
            pred_future_norm: 모델이 실제 샘플러로 만든 정규화 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            packed_agent_type: anchor 순서대로 압축한 에이전트 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 구간의 제어입니다.
                마지막 차원은 ``[v_x^b, v_y^b, omega]`` 이고,
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 구간 제어를 믿을 수 있는지 표시합니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                총 physics loss와 각 세부 항의 평균값을 담은 사전입니다.
        """
        if pred_future_norm.numel() == 0:
            return _build_zero_output(pred_future_norm)

        limits = self._gather_limits(
            packed_agent_type=packed_agent_type,
            device=pred_future_norm.device,
            dtype=pred_future_norm.dtype,
        )
        pred_stats = self._compute_proxy_penalties(
            future_norm=pred_future_norm,
            limits=limits,
            prev_control=packed_prev_control,
            prev_control_valid=packed_prev_control_valid,
        )
        gt_stats = self._compute_proxy_penalties(
            future_norm=target_future_norm.detach(),
            limits=limits,
            prev_control=packed_prev_control.detach(),
            prev_control_valid=packed_prev_control_valid,
        )

        component_to_weight = {
            "speed": self.speed_weight,
            "slip": self.slip_weight,
            "accel": self.accel_weight,
            "yaw_accel": self.yaw_accel_weight,
            "turn": self.turn_weight,
        }

        loss_terms: Dict[str, Tensor] = {}
        pred_means: Dict[str, Tensor] = {}
        gt_means: Dict[str, Tensor] = {}
        actual_unit_terms: Dict[str, Tensor] = {}
        pred_actual_unit_means: Dict[str, Tensor] = {}
        gt_actual_unit_means: Dict[str, Tensor] = {}
        raw_pred_loss = pred_future_norm.new_zeros(())
        total_loss = pred_future_norm.new_zeros(())

        for name, weight in component_to_weight.items():
            pred_value = pred_stats[name]
            gt_value = gt_stats[name].detach()
            pred_mean = pred_value.mean()
            gt_mean = gt_value.mean()
            pred_means[f"pred_{name}"] = pred_mean
            gt_means[f"gt_{name}"] = gt_mean

            if self.gt_excess_only:
                effective = torch.relu(pred_value - gt_value)
            else:
                effective = pred_value
            effective_mean = effective.mean()
            loss_terms[name] = effective_mean

            total_loss = total_loss + weight * effective_mean
            raw_pred_loss = raw_pred_loss + weight * pred_mean

        for name in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            pred_value = pred_stats[name]
            gt_value = gt_stats[name].detach()
            pred_actual_unit_means[f"pred_{name}"] = pred_value.mean()
            gt_actual_unit_means[f"gt_{name}"] = gt_value.mean()

            if self.gt_excess_only:
                effective = torch.relu(pred_value - gt_value)
            else:
                effective = pred_value
            actual_unit_terms[name] = effective.mean()

        return {
            "loss": total_loss,
            "raw_pred_loss": raw_pred_loss,
            **loss_terms,
            **pred_means,
            **gt_means,
            **actual_unit_terms,
            **pred_actual_unit_means,
            **gt_actual_unit_means,
        }

    def _compute_proxy_penalties(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        prev_control: Tensor,
        prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """한 미래 궤적 묶음의 physics proxy penalty를 계산합니다.

        Args:
            future_norm: 정규화 미래 궤적입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            limits: anchor별 제한값 사전입니다.
                각 값의 shape은 ``[n_valid_anchor]`` 입니다.
            prev_control: anchor 직전 구간 제어입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            prev_control_valid: 직전 구간 제어 유효 여부입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                각 항목별 anchor 단위 penalty입니다.
                각 값의 shape은 ``[n_valid_anchor]`` 입니다.
        """
        pos_local_m, heading_local = self._denormalize_future(future_norm)
        vx_body, vy_body, omega = self._trajectory_to_body_controls(pos_local_m, heading_local)

        speed = torch.sqrt(vx_body.square() + vy_body.square() + self.eps)
        nonholonomic = limits["is_nonholonomic"].unsqueeze(-1)
        accel_limit = limits["a_max_mps2"] * self.dt
        yaw_accel_limit = limits["alpha_max_radps2"] * self.dt

        speed_pen = self._mean_over_time(
            self._normalized_square_penalty(
                value=speed,
                limit=limits["v_max_mps"].unsqueeze(-1),
            )
        )
        speed_excess_mps = self._mean_over_time(
            torch.relu(speed - limits["v_max_mps"].unsqueeze(-1))
        )

        beta = torch.atan2(vy_body.abs(), vx_body.abs() + self.eps)
        beta_limit = limits["beta_max_rad"].unsqueeze(-1)
        beta_pen = self._normalized_square_penalty(
            value=beta,
            limit=beta_limit,
            enabled=beta_limit > 0.0,
        )
        slip_pen = self._mean_over_time(torch.where(nonholonomic, beta_pen, torch.zeros_like(beta_pen)))
        beta_excess_deg = torch.rad2deg(torch.relu(beta - beta_limit))
        slip_beta_excess_deg = self._mean_over_time(
            torch.where(
                nonholonomic & (beta_limit > 0.0),
                beta_excess_deg,
                torch.zeros_like(beta_excess_deg),
            )
        )

        if vx_body.shape[1] > 0:
            speed_x = vx_body
            # 첫 delta는 직전 anchor 제어와 이어 붙인 경계차분으로 계산합니다.
            dv = torch.empty_like(speed_x)
            dw = torch.empty_like(omega)
            dv[:, 0] = (speed_x[:, 0] - prev_control[:, 0]).abs()
            dw[:, 0] = (omega[:, 0] - prev_control[:, 2]).abs()
            if speed_x.shape[1] > 1:
                dv[:, 1:] = (speed_x[:, 1:] - speed_x[:, :-1]).abs()
                dw[:, 1:] = (omega[:, 1:] - omega[:, :-1]).abs()
            delta_enabled = nonholonomic.expand(-1, dv.shape[1]).clone()
            delta_enabled[:, 0] = delta_enabled[:, 0] & prev_control_valid

            accel_pen = self._masked_mean_over_time(
                self._normalized_square_penalty(dv, accel_limit.unsqueeze(-1)),
                enabled=delta_enabled,
            )
            yaw_accel_pen = self._masked_mean_over_time(
                self._normalized_square_penalty(dw, yaw_accel_limit.unsqueeze(-1)),
                enabled=delta_enabled,
            )
            accel_excess_mps2 = self._masked_mean_over_time(
                torch.relu(dv / self.dt - limits["a_max_mps2"].unsqueeze(-1)),
                enabled=delta_enabled,
            )
            yaw_accel_excess_degps2 = self._masked_mean_over_time(
                torch.rad2deg(torch.relu(dw / self.dt - limits["alpha_max_radps2"].unsqueeze(-1))),
                enabled=delta_enabled,
            )
        else:
            accel_pen = speed_pen.new_zeros(speed_pen.shape)
            yaw_accel_pen = speed_pen.new_zeros(speed_pen.shape)
            accel_excess_mps2 = speed_pen.new_zeros(speed_pen.shape)
            yaw_accel_excess_degps2 = speed_pen.new_zeros(speed_pen.shape)

        omega_abs_pen = self._normalized_square_penalty(
            value=omega.abs(),
            limit=limits["omega_max_abs_radps"].unsqueeze(-1),
        )
        turn_yaw_rate_excess_degps = self._mean_over_time(
            torch.rad2deg(torch.relu(omega.abs() - limits["omega_max_abs_radps"].unsqueeze(-1)))
        )
        lat_acc_value = speed * omega.abs()
        lat_acc_pen = self._normalized_square_penalty(
            value=lat_acc_value,
            limit=limits["a_lat_max_mps2"].unsqueeze(-1),
        )
        turn_lat_accel_excess_mps2 = self._mean_over_time(
            torch.where(
                nonholonomic,
                torch.relu(lat_acc_value - limits["a_lat_max_mps2"].unsqueeze(-1)),
                torch.zeros_like(lat_acc_value),
            )
        )
        radius_value = vx_body.abs() / (omega.abs() + self.eps)
        radius_shortfall = torch.relu(limits["r_min_m"].unsqueeze(-1) - radius_value)
        turn_radius_shortfall_m = self._mean_over_time(
            torch.where(nonholonomic, radius_shortfall, torch.zeros_like(radius_shortfall))
        )
        radius_pen = self._square_from_normalized_excess(
            radius_shortfall / (limits["r_min_m"].unsqueeze(-1) + self.eps)
        )
        turn_pen = self._mean_over_time(
            omega_abs_pen
            + torch.where(nonholonomic, lat_acc_pen + radius_pen, torch.zeros_like(lat_acc_pen))
        )

        return {
            "speed": speed_pen,
            "slip": slip_pen,
            "accel": accel_pen,
            "yaw_accel": yaw_accel_pen,
            "turn": turn_pen,
            "speed_excess_mps": speed_excess_mps,
            "slip_beta_excess_deg": slip_beta_excess_deg,
            "accel_excess_mps2": accel_excess_mps2,
            "yaw_accel_excess_degps2": yaw_accel_excess_degps2,
            "turn_yaw_rate_excess_degps": turn_yaw_rate_excess_degps,
            "turn_lat_accel_excess_mps2": turn_lat_accel_excess_mps2,
            "turn_radius_shortfall_m": turn_radius_shortfall_m,
        }

    def _gather_limits(
        self,
        packed_agent_type: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """anchor별 에이전트 종류에 맞는 제한값을 펼칩니다.

        Args:
            packed_agent_type: anchor 순서대로 압축한 종류 인덱스입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            device: 제한값을 올릴 장치입니다.
            dtype: 제한값 자료형입니다.

        Returns:
            Dict[str, Tensor]:
                anchor별 제한값 사전입니다.
                각 값의 shape은 ``[n_valid_anchor]`` 입니다.
        """
        agent_type = packed_agent_type.to(device=device, dtype=torch.long).clamp(min=0, max=2)

        def _select(values: Tuple[float, float, float]) -> Tensor:
            table = torch.tensor(values, device=device, dtype=dtype)
            return table[agent_type]

        return {
            "v_max_mps": _select(DEFAULT_LIMITS.v_max_mps),
            "a_max_mps2": _select(DEFAULT_LIMITS.a_max_mps2),
            "alpha_max_radps2": _select(DEFAULT_LIMITS.alpha_max_radps2),
            "a_lat_max_mps2": _select(DEFAULT_LIMITS.a_lat_max_mps2),
            "r_min_m": _select(DEFAULT_LIMITS.r_min_m),
            "omega_max_abs_radps": _select(DEFAULT_LIMITS.omega_max_abs_radps),
            "beta_max_rad": _select(DEFAULT_LIMITS.beta_max_rad),
            "is_nonholonomic": agent_type != 1,
        }

    def _denormalize_future(self, future_norm: Tensor) -> Tuple[Tensor, Tensor]:
        """정규화 미래를 meter 단위 local 궤적으로 바꿉니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                meter 단위 local 위치 ``[n_valid_anchor, 20, 2]`` 와
                local heading ``[n_valid_anchor, 20]`` 입니다.
        """
        pos_local_m = future_norm[..., :2] * self.pos_scale_m
        heading_local = torch.atan2(future_norm[..., 3], future_norm[..., 2])
        return pos_local_m, heading_local

    def _trajectory_to_body_controls(
        self,
        pos_local_m: Tensor,
        heading_local: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """local 미래 궤적을 몸체 기준 제어 시퀀스로 바꿉니다.

        현재 시점은 ``[0, 0, 0]`` 으로 두고,
        각 구간마다 ``[v_x^b, v_y^b, omega]`` 를 단순 차분으로 구합니다.

        Args:
            pos_local_m: meter 단위 local 위치입니다.
                shape은 ``[n_valid_anchor, 20, 2]`` 입니다.
            heading_local: local heading입니다.
                shape은 ``[n_valid_anchor, 20]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                앞방향 속도, 옆방향 속도, 회전속도입니다.
                각 shape은 ``[n_valid_anchor, 20]`` 입니다.
        """
        num_anchor = pos_local_m.shape[0]
        pos_zero = pos_local_m.new_zeros((num_anchor, 1, 2))
        heading_zero = heading_local.new_zeros((num_anchor, 1))

        pos_seq = torch.cat([pos_zero, pos_local_m], dim=1)
        heading_seq = torch.cat([heading_zero, heading_local], dim=1)

        delta_pos = pos_seq[:, 1:] - pos_seq[:, :-1]
        heading_start = heading_seq[:, :-1]
        delta_heading = self._wrap_angle(heading_seq[:, 1:] - heading_seq[:, :-1])

        cos_head = heading_start.cos()
        sin_head = heading_start.sin()
        vx_body = (delta_pos[..., 0] * cos_head + delta_pos[..., 1] * sin_head) / self.dt
        vy_body = (-delta_pos[..., 0] * sin_head + delta_pos[..., 1] * cos_head) / self.dt
        omega = delta_heading / self.dt
        return vx_body, vy_body, omega

    def _normalized_square_penalty(
        self,
        value: Tensor,
        limit: Tensor,
        enabled: Tensor | None = None,
    ) -> Tensor:
        """값이 제한을 넘은 정도를 부드럽게 제곱 penalty로 바꿉니다.

        Args:
            value: 실제값입니다. shape은 ``[...,]`` 입니다.
            limit: 제한값입니다. shape은 ``[...,]`` 또는 브로드캐스트 가능한 shape입니다.
            enabled: ``True`` 인 위치만 penalty를 켭니다.
                shape은 ``[...,]`` 또는 브로드캐스트 가능한 shape입니다.

        Returns:
            Tensor:
                같은 shape의 penalty입니다.
        """
        normalized_excess = torch.relu(value - limit) / (limit.abs() + self.eps)
        penalty = self._square_from_normalized_excess(normalized_excess)
        if enabled is None:
            return penalty
        return torch.where(enabled, penalty, penalty.new_zeros(()))

    def _square_from_normalized_excess(self, normalized_excess: Tensor) -> Tensor:
        """정규화 초과량을 부드러운 dead-zone 제곱 penalty로 바꿉니다.

        Args:
            normalized_excess: 0 이상 정규화 초과량입니다.
                shape은 ``[...,]`` 입니다.

        Returns:
            Tensor:
                같은 shape의 penalty입니다.
        """
        shifted = (normalized_excess - self.deadzone_ratio) / max(self.deadzone_softness, self.eps)
        smooth = torch.nn.functional.softplus(shifted) * max(self.deadzone_softness, self.eps)
        return smooth.square()

    def _mean_over_time(self, value: Tensor) -> Tensor:
        """시간축 평균을 계산합니다.

        Args:
            value: 마지막 축이 시간인 텐서입니다. shape은 ``[n_valid_anchor, T]`` 입니다.

        Returns:
            Tensor:
                anchor별 평균값입니다. shape은 ``[n_valid_anchor]`` 입니다.
        """
        if value.dim() == 1:
            return value
        return value.mean(dim=-1)

    def _masked_mean_over_time(self, value: Tensor, enabled: Tensor) -> Tensor:
        """시간축에서 활성화된 위치만 평균합니다."""
        masked_value = torch.where(enabled, value, torch.zeros_like(value))
        if value.dim() == 1:
            return masked_value
        enabled_count = enabled.to(dtype=value.dtype).sum(dim=-1).clamp_min(1.0)
        return masked_value.sum(dim=-1) / enabled_count

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
