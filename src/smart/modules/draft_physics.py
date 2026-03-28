from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class DynamicLimitTable:
    """에이전트 종류별 물리 제한값을 보관합니다.

    Attributes:
        v_max_mps: 최고 속도 제한입니다. shape은 ``[3]`` 입니다.
        a_max_mps2: 가속도 제한입니다. shape은 ``[3]`` 입니다.
        alpha_max_radps2: 회전 변화 제한입니다. shape은 ``[3]`` 입니다.
        a_lat_max_mps2: 횡가속 제한입니다. shape은 ``[3]`` 입니다.
        r_min_m: 최소 선회 반경 제한입니다. shape은 ``[3]`` 입니다.
        omega_max_abs_radps: 절대 회전속도 제한입니다. shape은 ``[3]`` 입니다.
        beta_max_rad: 기존 slip 로그와 호환하기 위해 남겨 둔 값입니다.
            현재 새 penalty 본체에서는 직접 쓰지 않습니다. shape은 ``[3]`` 입니다.
    """

    v_max_mps: Tuple[float, float, float]
    a_max_mps2: Tuple[float, float, float]
    alpha_max_radps2: Tuple[float, float, float]
    a_lat_max_mps2: Tuple[float, float, float]
    r_min_m: Tuple[float, float, float]
    omega_max_abs_radps: Tuple[float, float, float]
    beta_max_rad: Tuple[float, float, float]


DEFAULT_LIMITS = DynamicLimitTable(
    # CAT-K repo의 agent type 인덱스: vehicle=0, pedestrian=1, bicycle=2
    # 값은 사용자가 WOMD에서 정리한 99.3% percentile 제한값을 그대로 씁니다.
    v_max_mps=(35.0, 5.0, 22.0),
    a_max_mps2=(8.0, 4.7, 5.5),
    alpha_max_radps2=(1.75, 14.0, 6.0),
    a_lat_max_mps2=(4.2, 3.2, 4.4),
    r_min_m=(4.50, 0.00001, 0.5),
    omega_max_abs_radps=(0.9, 3.3, 2.0),
    beta_max_rad=(0.27, 10.0, 0.7),
)


DRAFT_PHYSICS_COMPONENT_KEYS = (
    "veh_track",
    "veh_limit",
    "ped_track",
    "ped_limit",
    "ped_heading",
)

DRAFT_PHYSICS_ACTUAL_UNIT_KEYS = (
    "veh_track_mse_norm",
    "veh_speed_excess_mps",
    "veh_yaw_rate_excess_degps",
    "veh_accel_excess_mps2",
    "veh_yaw_accel_excess_degps2",
    "veh_lat_accel_excess_mps2",
    "veh_radius_shortfall_m",
    "ped_track_mse_norm",
    "ped_speed_excess_mps",
    "ped_accel_excess_mps2",
    "ped_yaw_rate_excess_degps",
    "ped_yaw_accel_excess_degps2",
)


def _build_zero_output(reference: Tensor) -> Dict[str, Tensor]:
    """0으로 채운 기본 출력 사전을 만듭니다.

    Args:
        reference: 자료형과 device를 맞추기 위한 참조 텐서입니다.
            shape은 임의입니다.

    Returns:
        Dict[str, Tensor]: 모든 metric이 0으로 채워진 사전입니다.
    """
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
    """최종 샘플 기준의 동역학 추종 penalty를 계산합니다.

    기존 구현은 예측 궤적을 10Hz 차분한 뒤 speed/slip/accel/turn proxy를
    직접 더하는 방식이었습니다. 이 구현은 그 대신
    "예측 궤적을 4개의 반초 knot가 만든 단순한 운동으로 얼마나 잘 설명할 수 있는가"
    를 직접 묻습니다.

    - 차량/자전거는 ``[v, omega]`` knot 4개를 찾습니다.
    - 보행자는 ``[v_x, v_y]`` knot 4개를 찾습니다.
    - 각 knot 사이는 10Hz로 선형 보간합니다.
    - limit penalty는 기존 repo의 dead-zone 제곱 형태와 같은 shape를 유지합니다.

    Args:
        dt: 10Hz 시간 간격입니다. 기본값은 ``0.1`` 초입니다.
        pos_scale_m: 정규화된 ``x, y`` 를 meter로 되돌릴 때 쓰는 배율입니다.
        deadzone_ratio: 작은 초과량을 바로 크게 벌주지 않기 위한 여유 비율입니다.
        deadzone_softness: dead-zone 경계를 부드럽게 만들기 위한 값입니다.
        gt_excess_only: ``True`` 이면 GT보다 더 나쁜 만큼만 loss에 남깁니다.
        track_weight: 추종 항 가중치입니다.
        limit_weight: 이동 limit 항 가중치입니다.
        ped_heading_weight: 보행자 heading 부드러움 항 가중치입니다.
        num_chunks: 2초를 몇 개 knot 구간으로 나눌지 정합니다. 기본값은 ``4`` 입니다.
        chunk_size: 한 knot 구간이 몇 개 10Hz step으로 구성되는지 정합니다. 기본값은 ``5`` 입니다.
        inner_steps: 안쪽 gradient descent 반복 횟수입니다.
        inner_step_size: 안쪽 gradient descent step 크기입니다.
        eps: 수치 안정용 작은 값입니다.
    """

    def __init__(
        self,
        dt: float = 0.1,
        pos_scale_m: float = 20.0,
        deadzone_ratio: float = 0.02,
        deadzone_softness: float = 0.02,
        gt_excess_only: bool = True,
        track_weight: float = 1.0,
        limit_weight: float = 1.0,
        ped_heading_weight: float = 1.0,
        num_chunks: int = 4,
        chunk_size: int = 5,
        inner_steps: int = 5,
        inner_step_size: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.pos_scale_m = float(pos_scale_m)
        self.deadzone_ratio = float(deadzone_ratio)
        self.deadzone_softness = float(deadzone_softness)
        self.gt_excess_only = bool(gt_excess_only)
        self.track_weight = float(track_weight)
        self.limit_weight = float(limit_weight)
        self.ped_heading_weight = float(ped_heading_weight)
        self.num_chunks = int(num_chunks)
        self.chunk_size = int(chunk_size)
        self.num_steps = self.num_chunks * self.chunk_size
        self.inner_steps = int(inner_steps)
        self.inner_step_size = float(inner_step_size)
        self.eps = float(eps)

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
        packed_prev_vel_local_xy: Tensor,
    ) -> Dict[str, Tensor]:
        """생성 미래와 GT 미래의 동역학 penalty를 계산합니다.

        Args:
            pred_future_norm: 모델이 실제 샘플러로 만든 정규화 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            packed_agent_type: anchor 순서대로 압축한 에이전트 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 제어입니다.
                마지막 차원은 ``[v_x^b, v_y^b, omega]`` 입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 제어가 믿을 만한지 나타냅니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_vel_local_xy: anchor 직전의 local-frame 2D 속도입니다.
                마지막 차원은 ``[v_x^{local}, v_y^{local}]`` 입니다.
                shape은 ``[n_valid_anchor, 2]`` 입니다.

        Returns:
            Dict[str, Tensor]: 총 loss와 component logging 사전입니다.
        """
        if pred_future_norm.numel() == 0:
            return _build_zero_output(pred_future_norm)

        limits = self._gather_limits(
            packed_agent_type=packed_agent_type,
            device=pred_future_norm.device,
            dtype=pred_future_norm.dtype,
        )
        pred_stats = self._compute_dynamic_penalties(
            future_norm=pred_future_norm,
            limits=limits,
            prev_control=packed_prev_control,
            prev_control_valid=packed_prev_control_valid,
            prev_vel_local_xy=packed_prev_vel_local_xy,
            differentiable=True,
        )
        gt_stats = self._compute_dynamic_penalties(
            future_norm=target_future_norm.detach(),
            limits=limits,
            prev_control=packed_prev_control.detach(),
            prev_control_valid=packed_prev_control_valid,
            prev_vel_local_xy=packed_prev_vel_local_xy.detach(),
            differentiable=False,
        )

        component_to_weight = {
            "veh_track": self.track_weight,
            "veh_limit": self.limit_weight,
            "ped_track": self.track_weight,
            "ped_limit": self.limit_weight,
            "ped_heading": self.ped_heading_weight,
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

    def _compute_dynamic_penalties(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        prev_control: Tensor,
        prev_control_valid: Tensor,
        prev_vel_local_xy: Tensor,
        differentiable: bool,
    ) -> Dict[str, Tensor]:
        """예측 또는 GT 미래 하나에 대해 새 penalty를 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            limits: anchor별 제한값 사전입니다. 각 값의 shape은 ``[n_valid_anchor]`` 입니다.
            prev_control: 직전 제어입니다. shape은 ``[n_valid_anchor, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_valid_anchor]`` 입니다.
            prev_vel_local_xy: 직전 local 2D 속도입니다. shape은 ``[n_valid_anchor, 2]`` 입니다.
            differentiable: ``True`` 이면 안쪽 solver까지 포함해 gradient를 유지합니다.

        Returns:
            Dict[str, Tensor]: anchor별 penalty와 실제 단위 위반량 사전입니다.
                각 값의 shape은 ``[n_valid_anchor]`` 입니다.
        """
        num_anchor = future_norm.shape[0]
        zero_vec = future_norm.new_zeros((num_anchor,))
        stats: Dict[str, Tensor] = {name: zero_vec.clone() for name in DRAFT_PHYSICS_COMPONENT_KEYS}
        stats.update({name: zero_vec.clone() for name in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS})

        nonholonomic_mask = limits["is_nonholonomic"]
        pedestrian_mask = ~nonholonomic_mask

        if bool(nonholonomic_mask.any()):
            veh_indices = nonholonomic_mask.nonzero(as_tuple=False).squeeze(-1)
            veh_stats = self._compute_vehicle_group_penalties(
                future_norm=future_norm[veh_indices],
                limits={key: value[veh_indices] for key, value in limits.items()},
                prev_control=prev_control[veh_indices],
                prev_control_valid=prev_control_valid[veh_indices],
                differentiable=differentiable,
            )
            for key, value in veh_stats.items():
                stats[key][veh_indices] = value

        if bool(pedestrian_mask.any()):
            ped_indices = pedestrian_mask.nonzero(as_tuple=False).squeeze(-1)
            ped_stats = self._compute_pedestrian_group_penalties(
                future_norm=future_norm[ped_indices],
                limits={key: value[ped_indices] for key, value in limits.items()},
                prev_control=prev_control[ped_indices],
                prev_control_valid=prev_control_valid[ped_indices],
                prev_vel_local_xy=prev_vel_local_xy[ped_indices],
                differentiable=differentiable,
            )
            for key, value in ped_stats.items():
                stats[key][ped_indices] = value

        return stats

    def _compute_vehicle_group_penalties(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        prev_control: Tensor,
        prev_control_valid: Tensor,
        differentiable: bool,
    ) -> Dict[str, Tensor]:
        """차량/자전거 그룹의 새 penalty를 계산합니다.

        Args:
            future_norm: 차량/자전거의 정규화 미래입니다.
                shape은 ``[n_group, 20, 4]`` 입니다.
            limits: 그룹 anchor별 제한값 사전입니다.
                각 값의 shape은 ``[n_group]`` 입니다.
            prev_control: 그룹 직전 제어입니다. shape은 ``[n_group, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_group]`` 입니다.
            differentiable: 안쪽 solver를 gradient graph 안에 둘지 정합니다.

        Returns:
            Dict[str, Tensor]: 그룹 anchor별 metric 사전입니다.
        """
        pos_local_m, heading_local = self._denormalize_future(future_norm)
        vx_body, _, omega = self._trajectory_to_body_controls(pos_local_m, heading_local)

        fallback_start = torch.stack([vx_body[:, 0], omega[:, 0]], dim=-1)
        start_control = torch.stack([prev_control[:, 0], prev_control[:, 2]], dim=-1)
        start_control = torch.where(
            prev_control_valid.unsqueeze(-1),
            start_control,
            fallback_start,
        )

        init_knots = self._build_end_knot_init_from_steps(
            step_values=torch.stack([vx_body, omega], dim=-1),
        )
        solved_knots = self._solve_inner_knots(
            init_knots=init_knots,
            objective_fn=lambda knot: self._vehicle_objective(
                future_norm=future_norm,
                limits=limits,
                start_control=start_control,
                prev_control_valid=prev_control_valid,
                knots=knot,
            ),
            differentiable=differentiable,
        )
        objective = self._vehicle_objective(
            future_norm=future_norm,
            limits=limits,
            start_control=start_control,
            prev_control_valid=prev_control_valid,
            knots=solved_knots,
        )
        return {
            "veh_track": objective["track"],
            "veh_limit": objective["limit"],
            "veh_track_mse_norm": objective["track_mse_norm"],
            "veh_speed_excess_mps": objective["speed_excess_mps"],
            "veh_yaw_rate_excess_degps": objective["yaw_rate_excess_degps"],
            "veh_accel_excess_mps2": objective["accel_excess_mps2"],
            "veh_yaw_accel_excess_degps2": objective["yaw_accel_excess_degps2"],
            "veh_lat_accel_excess_mps2": objective["lat_accel_excess_mps2"],
            "veh_radius_shortfall_m": objective["radius_shortfall_m"],
        }

    def _compute_pedestrian_group_penalties(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        prev_control: Tensor,
        prev_control_valid: Tensor,
        prev_vel_local_xy: Tensor,
        differentiable: bool,
    ) -> Dict[str, Tensor]:
        """보행자 그룹의 새 penalty를 계산합니다.

        Args:
            future_norm: 보행자의 정규화 미래입니다. shape은 ``[n_group, 20, 4]`` 입니다.
            limits: 그룹 anchor별 제한값 사전입니다. 각 값의 shape은 ``[n_group]`` 입니다.
            prev_control: 직전 제어입니다. shape은 ``[n_group, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_group]`` 입니다.
            prev_vel_local_xy: 직전 local 2D 속도입니다. shape은 ``[n_group, 2]`` 입니다.
            differentiable: 안쪽 solver를 gradient graph 안에 둘지 정합니다.

        Returns:
            Dict[str, Tensor]: 그룹 anchor별 metric 사전입니다.
        """
        pos_local_m, heading_local = self._denormalize_future(future_norm)
        step_vel_local = self._trajectory_to_local_velocity(pos_local_m)

        start_vel = torch.where(
            prev_control_valid.unsqueeze(-1),
            prev_vel_local_xy,
            step_vel_local[:, 0],
        )
        init_knots = self._build_end_knot_init_from_steps(step_values=step_vel_local)
        solved_knots = self._solve_inner_knots(
            init_knots=init_knots,
            objective_fn=lambda knot: self._pedestrian_motion_objective(
                future_norm=future_norm,
                limits=limits,
                start_vel=start_vel,
                prev_vel_valid=prev_control_valid,
                knots=knot,
            ),
            differentiable=differentiable,
        )
        motion_objective = self._pedestrian_motion_objective(
            future_norm=future_norm,
            limits=limits,
            start_vel=start_vel,
            prev_vel_valid=prev_control_valid,
            knots=solved_knots,
        )
        heading_objective = self._pedestrian_heading_objective(
            future_norm=future_norm,
            limits=limits,
            prev_control=prev_control,
            prev_control_valid=prev_control_valid,
        )
        return {
            "ped_track": motion_objective["track"],
            "ped_limit": motion_objective["limit"],
            "ped_heading": heading_objective["heading"],
            "ped_track_mse_norm": motion_objective["track_mse_norm"],
            "ped_speed_excess_mps": motion_objective["speed_excess_mps"],
            "ped_accel_excess_mps2": motion_objective["accel_excess_mps2"],
            "ped_yaw_rate_excess_degps": heading_objective["yaw_rate_excess_degps"],
            "ped_yaw_accel_excess_degps2": heading_objective["yaw_accel_excess_degps2"],
        }

    def _solve_inner_knots(
        self,
        init_knots: Tensor,
        objective_fn: Callable[[Tensor], Dict[str, Tensor]],
        differentiable: bool,
    ) -> Tensor:
        """작은 gradient descent로 knot를 맞춥니다.

        Args:
            init_knots: 초기 knot 값입니다.
                shape은 ``[n_anchor, num_chunks, dim]`` 입니다.
            objective_fn: knot를 넣으면 anchor별 objective 사전을 돌려주는 함수입니다.
            differentiable: ``True`` 이면 create_graph=True로 업데이트를 기록합니다.

        Returns:
            Tensor: 최종 knot입니다. shape은 ``[n_anchor, num_chunks, dim]`` 입니다.
        """
        knots = init_knots if differentiable else init_knots.detach()
        if self.inner_steps <= 0:
            return knots

        for _ in range(self.inner_steps):
            knots = knots.requires_grad_(True)
            objective = objective_fn(knots)["objective"].mean()
            grad = torch.autograd.grad(
                objective,
                knots,
                create_graph=differentiable,
                retain_graph=differentiable,
            )[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = grad.view(grad.shape[0], -1).norm(dim=1, keepdim=True).clamp_min(1.0)
            grad = grad / grad_norm.view(grad.shape[0], 1, 1)
            knots = torch.nan_to_num(knots - self.inner_step_size * grad, nan=0.0, posinf=0.0, neginf=0.0)
            if not differentiable:
                knots = knots.detach()
        return knots

    def _vehicle_objective(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        start_control: Tensor,
        prev_control_valid: Tensor,
        knots: Tensor,
    ) -> Dict[str, Tensor]:
        """차량/자전거 knot의 추종/제한 objective를 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
            limits: anchor별 제한값 사전입니다. 각 값의 shape은 ``[n_anchor]`` 입니다.
            start_control: 시작 ``[v, omega]`` 입니다. shape은 ``[n_anchor, 2]`` 입니다.
            prev_control_valid: 시작 제어 유효 여부입니다. shape은 ``[n_anchor]`` 입니다.
            knots: 미래 knot입니다. shape은 ``[n_anchor, 4, 2]`` 입니다.

        Returns:
            Dict[str, Tensor]: anchor별 objective와 logging 값 사전입니다.
        """
        step_values = self._interpolate_knots(start_value=start_control, future_knots=knots)
        step_v = step_values[..., 0]
        step_omega = step_values[..., 1]
        rollout_norm = self._rollout_vehicle_norm(step_v=step_v, step_omega=step_omega)

        track = ((rollout_norm - future_norm) ** 2).mean(dim=(1, 2))

        prev_v = torch.cat([start_control[:, 0:1], step_v[:, :-1]], dim=1)
        prev_omega = torch.cat([start_control[:, 1:2], step_omega[:, :-1]], dim=1)
        accel_value = (step_v - prev_v).abs() / self.dt
        yaw_accel_value = (step_omega - prev_omega).abs() / self.dt
        delta_enabled = torch.ones_like(step_v, dtype=torch.bool)
        delta_enabled[:, 0] = prev_control_valid

        speed_abs = step_v.abs()
        omega_abs = step_omega.abs()
        lat_acc_value = speed_abs * omega_abs
        turn_active = omega_abs > self.eps
        radius_value = speed_abs / (omega_abs + self.eps)
        radius_shortfall = torch.where(
            turn_active,
            torch.relu(limits["r_min_m"].unsqueeze(-1) - radius_value),
            torch.zeros_like(radius_value),
        )

        speed_pen = self._mean_over_time(
            self._normalized_square_penalty(speed_abs, limits["v_max_mps"].unsqueeze(-1))
        )
        omega_pen = self._mean_over_time(
            self._normalized_square_penalty(omega_abs, limits["omega_max_abs_radps"].unsqueeze(-1))
        )
        accel_pen = self._masked_mean_over_time(
            self._normalized_square_penalty(accel_value, limits["a_max_mps2"].unsqueeze(-1)),
            enabled=delta_enabled,
        )
        yaw_accel_pen = self._masked_mean_over_time(
            self._normalized_square_penalty(yaw_accel_value, limits["alpha_max_radps2"].unsqueeze(-1)),
            enabled=delta_enabled,
        )
        lat_acc_pen = self._mean_over_time(
            self._normalized_square_penalty(lat_acc_value, limits["a_lat_max_mps2"].unsqueeze(-1))
        )
        radius_pen = self._mean_over_time(
            self._square_from_normalized_excess(
                radius_shortfall / (limits["r_min_m"].unsqueeze(-1) + self.eps)
            )
        )
        limit = speed_pen + omega_pen + accel_pen + yaw_accel_pen + lat_acc_pen + radius_pen

        speed_excess = self._mean_over_time(
            torch.relu(speed_abs - limits["v_max_mps"].unsqueeze(-1))
        )
        yaw_rate_excess = self._mean_over_time(
            torch.rad2deg(torch.relu(omega_abs - limits["omega_max_abs_radps"].unsqueeze(-1)))
        )
        accel_excess = self._masked_mean_over_time(
            torch.relu(accel_value - limits["a_max_mps2"].unsqueeze(-1)),
            enabled=delta_enabled,
        )
        yaw_accel_excess = self._masked_mean_over_time(
            torch.rad2deg(torch.relu(yaw_accel_value - limits["alpha_max_radps2"].unsqueeze(-1))),
            enabled=delta_enabled,
        )
        lat_acc_excess = self._mean_over_time(
            torch.relu(lat_acc_value - limits["a_lat_max_mps2"].unsqueeze(-1))
        )
        radius_shortfall_m = self._mean_over_time(radius_shortfall)

        return {
            "objective": self.track_weight * track + self.limit_weight * limit,
            "track": track,
            "limit": limit,
            "track_mse_norm": track,
            "speed_excess_mps": speed_excess,
            "yaw_rate_excess_degps": yaw_rate_excess,
            "accel_excess_mps2": accel_excess,
            "yaw_accel_excess_degps2": yaw_accel_excess,
            "lat_accel_excess_mps2": lat_acc_excess,
            "radius_shortfall_m": radius_shortfall_m,
        }

    def _pedestrian_motion_objective(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        start_vel: Tensor,
        prev_vel_valid: Tensor,
        knots: Tensor,
    ) -> Dict[str, Tensor]:
        """보행자 2D 속도 knot의 추종/제한 objective를 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
            limits: anchor별 제한값 사전입니다. 각 값의 shape은 ``[n_anchor]`` 입니다.
            start_vel: 시작 2D 속도입니다. shape은 ``[n_anchor, 2]`` 입니다.
            prev_vel_valid: 시작 속도 유효 여부입니다. shape은 ``[n_anchor]`` 입니다.
            knots: 미래 2D 속도 knot입니다. shape은 ``[n_anchor, 4, 2]`` 입니다.

        Returns:
            Dict[str, Tensor]: anchor별 objective와 logging 값 사전입니다.
        """
        step_vel = self._interpolate_knots(start_value=start_vel, future_knots=knots)
        rollout_pos_norm = self._rollout_pedestrian_pos_norm(step_vel=step_vel)
        target_pos_norm = future_norm[..., :2]
        track = ((rollout_pos_norm - target_pos_norm) ** 2).mean(dim=(1, 2))

        prev_step_vel = torch.cat([start_vel.unsqueeze(1), step_vel[:, :-1]], dim=1)
        accel_value = (step_vel - prev_step_vel).norm(dim=-1) / self.dt
        delta_enabled = torch.ones_like(accel_value, dtype=torch.bool)
        delta_enabled[:, 0] = prev_vel_valid

        speed_value = step_vel.norm(dim=-1)
        speed_pen = self._mean_over_time(
            self._normalized_square_penalty(speed_value, limits["v_max_mps"].unsqueeze(-1))
        )
        accel_pen = self._masked_mean_over_time(
            self._normalized_square_penalty(accel_value, limits["a_max_mps2"].unsqueeze(-1)),
            enabled=delta_enabled,
        )
        limit = speed_pen + accel_pen

        speed_excess = self._mean_over_time(
            torch.relu(speed_value - limits["v_max_mps"].unsqueeze(-1))
        )
        accel_excess = self._masked_mean_over_time(
            torch.relu(accel_value - limits["a_max_mps2"].unsqueeze(-1)),
            enabled=delta_enabled,
        )
        return {
            "objective": self.track_weight * track + self.limit_weight * limit,
            "track": track,
            "limit": limit,
            "track_mse_norm": track,
            "speed_excess_mps": speed_excess,
            "accel_excess_mps2": accel_excess,
        }

    def _pedestrian_heading_objective(
        self,
        future_norm: Tensor,
        limits: Dict[str, Tensor],
        prev_control: Tensor,
        prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """보행자 heading 자체의 부드러움을 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
            limits: anchor별 제한값 사전입니다. 각 값의 shape은 ``[n_anchor]`` 입니다.
            prev_control: 직전 제어입니다. shape은 ``[n_anchor, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]: anchor별 heading penalty와 실제 단위 위반량 사전입니다.
        """
        _, heading_local = self._denormalize_future(future_norm)
        heading_zero = heading_local.new_zeros((heading_local.shape[0], 1))
        heading_seq = torch.cat([heading_zero, heading_local], dim=1)
        omega = self._wrap_angle(heading_seq[:, 1:] - heading_seq[:, :-1]) / self.dt

        prev_omega = torch.cat([prev_control[:, 2:3], omega[:, :-1]], dim=1)
        yaw_accel_value = (omega - prev_omega).abs() / self.dt
        delta_enabled = torch.ones_like(omega, dtype=torch.bool)
        delta_enabled[:, 0] = prev_control_valid

        yaw_rate_pen = self._mean_over_time(
            self._normalized_square_penalty(
                omega.abs(),
                limits["omega_max_abs_radps"].unsqueeze(-1),
            )
        )
        yaw_accel_pen = self._masked_mean_over_time(
            self._normalized_square_penalty(
                yaw_accel_value,
                limits["alpha_max_radps2"].unsqueeze(-1),
            ),
            enabled=delta_enabled,
        )
        heading = yaw_rate_pen + yaw_accel_pen
        return {
            "heading": heading,
            "yaw_rate_excess_degps": self._mean_over_time(
                torch.rad2deg(
                    torch.relu(omega.abs() - limits["omega_max_abs_radps"].unsqueeze(-1))
                )
            ),
            "yaw_accel_excess_degps2": self._masked_mean_over_time(
                torch.rad2deg(
                    torch.relu(yaw_accel_value - limits["alpha_max_radps2"].unsqueeze(-1))
                ),
                enabled=delta_enabled,
            ),
        }

    def _build_end_knot_init_from_steps(self, step_values: Tensor) -> Tensor:
        """20 step 값을 4개의 chunk 끝 knot 초기값으로 바꿉니다.

        Args:
            step_values: 10Hz step 값입니다.
                shape은 ``[n_anchor, 20, dim]`` 입니다.

        Returns:
            Tensor: knot 초기값입니다. shape은 ``[n_anchor, 4, dim]`` 입니다.
        """
        end_indices = torch.arange(
            self.chunk_size - 1,
            self.num_steps,
            self.chunk_size,
            device=step_values.device,
        )
        return step_values.index_select(dim=1, index=end_indices)

    def _interpolate_knots(self, start_value: Tensor, future_knots: Tensor) -> Tensor:
        """시작값과 미래 knot를 이어 10Hz 선형 보간 시퀀스를 만듭니다.

        Args:
            start_value: 현재 anchor 시점 값입니다. shape은 ``[n_anchor, dim]`` 입니다.
            future_knots: 미래 knot 값입니다. shape은 ``[n_anchor, 4, dim]`` 입니다.

        Returns:
            Tensor: 10Hz step 값입니다. shape은 ``[n_anchor, 20, dim]`` 입니다.
        """
        knot_seq = torch.cat([start_value.unsqueeze(1), future_knots], dim=1)
        weights = torch.linspace(
            1.0 / float(self.chunk_size),
            1.0,
            self.chunk_size,
            device=future_knots.device,
            dtype=future_knots.dtype,
        )
        segments = []
        for chunk_idx in range(self.num_chunks):
            start = knot_seq[:, chunk_idx : chunk_idx + 1]
            end = knot_seq[:, chunk_idx + 1 : chunk_idx + 2]
            segment = start + (end - start) * weights.view(1, self.chunk_size, 1)
            segments.append(segment)
        return torch.cat(segments, dim=1)

    def _rollout_vehicle_norm(self, step_v: Tensor, step_omega: Tensor) -> Tensor:
        """차량/자전거용 ``[v, omega]`` 시퀀스로 정규화 미래를 다시 만듭니다.

        Args:
            step_v: 10Hz 앞방향 속도입니다. shape은 ``[n_anchor, 20]`` 입니다.
            step_omega: 10Hz 회전속도입니다. shape은 ``[n_anchor, 20]`` 입니다.

        Returns:
            Tensor: 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.
                마지막 차원은 ``[x/20, y/20, cos, sin]`` 입니다.
        """
        num_anchor = step_v.shape[0]
        x = step_v.new_zeros((num_anchor,))
        y = step_v.new_zeros((num_anchor,))
        heading = step_v.new_zeros((num_anchor,))
        pos_out = []
        head_out = []
        for step_idx in range(self.num_steps):
            x = x + step_v[:, step_idx] * heading.cos() * self.dt
            y = y + step_v[:, step_idx] * heading.sin() * self.dt
            heading = heading + step_omega[:, step_idx] * self.dt
            pos_out.append(torch.stack([x, y], dim=-1))
            head_out.append(heading)
        pos_local = torch.stack(pos_out, dim=1)
        heading_local = torch.stack(head_out, dim=1)
        return torch.stack(
            [
                pos_local[..., 0] / self.pos_scale_m,
                pos_local[..., 1] / self.pos_scale_m,
                heading_local.cos(),
                heading_local.sin(),
            ],
            dim=-1,
        )

    def _rollout_pedestrian_pos_norm(self, step_vel: Tensor) -> Tensor:
        """보행자 2D 속도 시퀀스로 정규화 위치를 다시 만듭니다.

        Args:
            step_vel: 10Hz local 2D 속도입니다. shape은 ``[n_anchor, 20, 2]`` 입니다.

        Returns:
            Tensor: 정규화 위치입니다. shape은 ``[n_anchor, 20, 2]`` 입니다.
        """
        pos = step_vel.new_zeros((step_vel.shape[0], 2))
        pos_out = []
        for step_idx in range(self.num_steps):
            pos = pos + step_vel[:, step_idx] * self.dt
            pos_out.append(pos)
        pos_local = torch.stack(pos_out, dim=1)
        return pos_local / self.pos_scale_m

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
            Dict[str, Tensor]: anchor별 제한값 사전입니다.
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
            future_norm: 정규화 미래입니다. shape은 ``[n_anchor, 20, 4]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                meter 단위 local 위치 ``[n_anchor, 20, 2]`` 와
                local heading ``[n_anchor, 20]`` 입니다.
        """
        pos_local_m = future_norm[..., :2] * self.pos_scale_m
        heading_local = torch.atan2(future_norm[..., 3], future_norm[..., 2])
        return pos_local_m, heading_local

    def _trajectory_to_body_controls(
        self,
        pos_local_m: Tensor,
        heading_local: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """미래 궤적을 10Hz body-frame 제어 시퀀스로 바꿉니다.

        Args:
            pos_local_m: meter 단위 local 위치입니다. shape은 ``[n_anchor, 20, 2]`` 입니다.
            heading_local: local heading입니다. shape은 ``[n_anchor, 20]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                앞방향 속도 ``[n_anchor, 20]``, 옆방향 속도 ``[n_anchor, 20]``,
                회전속도 ``[n_anchor, 20]`` 입니다.
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

    def _trajectory_to_local_velocity(self, pos_local_m: Tensor) -> Tensor:
        """미래 위치를 10Hz local 2D 속도 시퀀스로 바꿉니다.

        Args:
            pos_local_m: meter 단위 local 위치입니다. shape은 ``[n_anchor, 20, 2]`` 입니다.

        Returns:
            Tensor: local 2D 속도입니다. shape은 ``[n_anchor, 20, 2]`` 입니다.
        """
        num_anchor = pos_local_m.shape[0]
        pos_zero = pos_local_m.new_zeros((num_anchor, 1, 2))
        pos_seq = torch.cat([pos_zero, pos_local_m], dim=1)
        return (pos_seq[:, 1:] - pos_seq[:, :-1]) / self.dt

    def _normalized_square_penalty(
        self,
        value: Tensor,
        limit: Tensor,
        enabled: Tensor | None = None,
    ) -> Tensor:
        """값이 제한을 넘은 정도를 부드러운 제곱 penalty로 바꿉니다.

        Args:
            value: 실제값입니다. shape은 ``[...,]`` 입니다.
            limit: 제한값입니다. shape은 브로드캐스트 가능하면 됩니다.
            enabled: ``True`` 인 위치만 penalty를 켭니다.
                shape은 브로드캐스트 가능하면 됩니다.

        Returns:
            Tensor: 같은 shape의 penalty입니다.
        """
        normalized_excess = torch.relu(value - limit) / (limit.abs() + self.eps)
        penalty = self._square_from_normalized_excess(normalized_excess)
        if enabled is None:
            return penalty
        return torch.where(enabled, penalty, penalty.new_zeros(()))

    def _square_from_normalized_excess(self, normalized_excess: Tensor) -> Tensor:
        """정규화 초과량을 부드러운 dead-zone 제곱 penalty로 바꿉니다.

        Args:
            normalized_excess: 0 이상 정규화 초과량입니다. shape은 ``[...,]`` 입니다.

        Returns:
            Tensor: 같은 shape의 penalty입니다.
        """
        shifted = (normalized_excess - self.deadzone_ratio) / max(self.deadzone_softness, self.eps)
        smooth = torch.nn.functional.softplus(shifted) * max(self.deadzone_softness, self.eps)
        return smooth.square()

    def _mean_over_time(self, value: Tensor) -> Tensor:
        """시간축 평균을 계산합니다.

        Args:
            value: 마지막 축이 시간인 텐서입니다. shape은 ``[n_anchor, T]`` 입니다.

        Returns:
            Tensor: anchor별 평균값입니다. shape은 ``[n_anchor]`` 입니다.
        """
        if value.dim() == 1:
            return value
        return value.mean(dim=-1)

    def _masked_mean_over_time(self, value: Tensor, enabled: Tensor) -> Tensor:
        """시간축에서 활성화된 위치만 평균합니다.

        Args:
            value: 마지막 축이 시간인 텐서입니다. shape은 ``[n_anchor, T]`` 입니다.
            enabled: 활성화 마스크입니다. shape은 ``[n_anchor, T]`` 입니다.

        Returns:
            Tensor: anchor별 평균값입니다. shape은 ``[n_anchor]`` 입니다.
        """
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
