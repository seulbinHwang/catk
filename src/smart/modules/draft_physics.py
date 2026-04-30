from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


VEHICLE_TYPE = 0
PEDESTRIAN_TYPE = 1
BICYCLE_TYPE = 2


@dataclass(frozen=True)
class DynamicLimitTable:
    """에이전트 종류별 제한값을 묶어 둡니다.

    Attributes:
        v_max_mps: 최고 속도 제한입니다. shape은 ``[3]`` 입니다.
            순서는 ``[vehicle, pedestrian, bicycle]`` 입니다.
        a_max_mps2: 가속도 제한입니다. shape은 ``[3]`` 입니다.
            차량과 자전거는 앞방향 가속도, 사람은 2차원 가속도 크기를 뜻합니다.
        a_lat_max_mps2: 횡가속 제한입니다. shape은 ``[3]`` 입니다.
            사람에는 쓰지 않으므로 0이어도 됩니다.
        beta_max_rad: body-frame slip angle 제한입니다. shape은 ``[3]`` 입니다.
            차량과 자전거의 옆방향 미끄러짐을 제한할 때 씁니다.
        alpha_max_radps2: 회전 변화 제한입니다. shape은 ``[3]`` 입니다.
            LQR commit bridge 의 곡률 변화율 clip 에 쓰입니다.
        r_min_m: 최소 선회 반경 제한입니다. shape은 ``[3]`` 입니다.
            LQR commit bridge 의 곡률 크기 clip 에 쓰입니다.
        omega_max_abs_radps: 절대 회전속도 제한입니다. shape은 ``[3]`` 입니다.
            LQR commit bridge 의 속도-기반 곡률 clip 에 쓰입니다.
    """

    v_max_mps: Tuple[float, float, float]
    a_max_mps2: Tuple[float, float, float]
    a_lat_max_mps2: Tuple[float, float, float]
    beta_max_rad: Tuple[float, float, float] = (0.27, 10.0, 0.70)
    alpha_max_radps2: Tuple[float, float, float] = (1.75, 14.0, 6.0)
    r_min_m: Tuple[float, float, float] = (4.50, 1.0e-5, 0.5)
    omega_max_abs_radps: Tuple[float, float, float] = (0.9, 3.3, 2.0)


DEFAULT_LIMITS = DynamicLimitTable(
    v_max_mps=(35.0, 5.0, 22.0),
    a_max_mps2=(8.0, 4.7, 5.5),
    a_lat_max_mps2=(4.2, 0.0, 4.4),
    beta_max_rad=(0.27, 10.0, 0.70),
    alpha_max_radps2=(1.75, 14.0, 6.0),
    r_min_m=(4.50, 1.0e-5, 0.5),
    omega_max_abs_radps=(0.9, 3.3, 2.0),
)


DRAFT_PHYSICS_COMPONENT_KEYS = (
    "vehicle_hard",
    "vehicle_slip",
    "vehicle_soft",
    "vehicle_total",
    "bicycle_hard",
    "bicycle_slip",
    "bicycle_soft",
    "bicycle_total",
    "pedestrian_hard",
    "pedestrian_soft",
    "pedestrian_head",
    "pedestrian_total",
)

DRAFT_PHYSICS_ACTUAL_UNIT_KEYS = (
    "speed_excess_mps",
    "slip_beta_excess_deg",
    "accel_excess_mps2",
    "steer_excess_deg",
    "steer_rate_excess_degps",
    "lat_accel_excess_mps2",
    "heading_error_deg",
)


def _build_zero_output(reference: Tensor) -> Dict[str, Tensor]:
    """출력이 없을 때 쓸 0 스칼라 사전을 만듭니다.

    Args:
        reference: device와 dtype를 맞추기 위한 기준 텐서입니다.
            shape은 임의입니다.

    Returns:
        Dict[str, Tensor]:
            모든 값이 0인 스칼라 사전입니다.
    """
    zero = reference.new_zeros(())
    output = {
        "loss": zero,
        "raw_pred_loss": zero,
    }
    for key in DRAFT_PHYSICS_COMPONENT_KEYS:
        output[key] = zero
    for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
        output[key] = zero
        output[f"pred_{key}"] = zero
        output[f"gt_{key}"] = zero
    return output


class DraftPhysicsRegularizer(nn.Module):
    """inverse feasibility 기반 DRaFT penalty를 계산합니다.

    이 모듈은 20개 미래 점을 다시 속도, 가속도, 조향각 같은 값으로 바꾼 뒤,
    각 에이전트 종류가 실제로 낼 수 있는 범위를 벗어나는지 계산합니다.

    차량과 자전거는 자전거 모델에 맞춘 역추론 값을 쓰고,
    사람은 2차원 속도와 2차원 가속도로 계산합니다.
    차량과 자전거에는 body-frame slip angle 벌점을 추가해 heading 방향과
    실제 이동 방향이 크게 어긋나는 옆방향 미끄러짐을 줄입니다.

    Args:
        dt: 미래 점 간 시간 간격입니다. 기본값은 ``0.1`` 초입니다.
        pos_scale_m: 정규화된 ``x, y`` 를 meter로 되돌릴 때 쓸 배율입니다.
            기본값은 ``20.0`` 입니다.
        speed_floor_mps: 곡률 계산에서 저속 불안정을 막기 위한 최소 속도입니다.
        vehicle_v_max_mps: 차량 최고 속도 제한입니다.
        vehicle_a_max_mps2: 차량 앞방향 가속도 제한입니다.
        vehicle_lat_accel_max_mps2: 차량 횡가속 제한입니다.
        vehicle_beta_max_rad: 차량 slip angle 제한입니다.
        bicycle_v_max_mps: 자전거 최고 속도 제한입니다.
        bicycle_a_max_mps2: 자전거 앞방향 가속도 제한입니다.
        bicycle_lat_accel_max_mps2: 자전거 횡가속 제한입니다.
        bicycle_beta_max_rad: 자전거 slip angle 제한입니다.
        pedestrian_v_max_mps: 사람 속도 제한입니다.
        pedestrian_a_max_mps2: 사람 2차원 가속도 제한입니다.
        vehicle_wheelbase_scale: 차량 wheelbase 비율입니다.
        bicycle_wheelbase_scale: 자전거 wheelbase 비율입니다.
        vehicle_steer_max_rad: 차량 최대 조향각입니다.
        bicycle_steer_max_rad: 자전거 최대 조향각입니다.
        vehicle_steer_rate_max_radps: 차량 최대 조향각 변화율입니다.
        bicycle_steer_rate_max_radps: 자전거 최대 조향각 변화율입니다.
        soft_weight: 모든 class에 공통으로 쓰는 roughness 항 가중치입니다.
        compare_softness_to_gt: ``True`` 이면 soft roughness를 GT보다 더 큰 만큼만
            반영하고, ``False`` 이면 prediction roughness 자체를 그대로 반영합니다.
        slip_deadzone_ratio: slip angle 초과량 중 무시할 정규화 여유 폭입니다.
            ``semi_continuous_lqr`` 의 기본값 ``0.02`` 를 그대로 사용합니다.
        slip_deadzone_softness: slip dead-zone 경계를 부드럽게 만드는 값입니다.
            ``semi_continuous_lqr`` 의 기본값 ``0.02`` 를 그대로 사용합니다.
        pedestrian_heading_weight: 사람 heading 약한 정렬 항 가중치입니다.
        pedestrian_heading_speed_threshold_mps: 사람 heading 항을 켜는 최소 속도입니다.
        eps: 수치 안정용 작은 값입니다.
    """

    def __init__(
        self,
        dt: float = 0.1,
        pos_scale_m: float = 20.0,
        speed_floor_mps: float = 0.5,
        vehicle_v_max_mps: float = DEFAULT_LIMITS.v_max_mps[VEHICLE_TYPE],
        vehicle_a_max_mps2: float = DEFAULT_LIMITS.a_max_mps2[VEHICLE_TYPE],
        vehicle_lat_accel_max_mps2: float = DEFAULT_LIMITS.a_lat_max_mps2[VEHICLE_TYPE],
        vehicle_beta_max_rad: float = DEFAULT_LIMITS.beta_max_rad[VEHICLE_TYPE],
        bicycle_v_max_mps: float = DEFAULT_LIMITS.v_max_mps[BICYCLE_TYPE],
        bicycle_a_max_mps2: float = DEFAULT_LIMITS.a_max_mps2[BICYCLE_TYPE],
        bicycle_lat_accel_max_mps2: float = DEFAULT_LIMITS.a_lat_max_mps2[BICYCLE_TYPE],
        bicycle_beta_max_rad: float = DEFAULT_LIMITS.beta_max_rad[BICYCLE_TYPE],
        pedestrian_v_max_mps: float = DEFAULT_LIMITS.v_max_mps[PEDESTRIAN_TYPE],
        pedestrian_a_max_mps2: float = DEFAULT_LIMITS.a_max_mps2[PEDESTRIAN_TYPE],
        vehicle_wheelbase_scale: float = 0.60,
        bicycle_wheelbase_scale: float = 0.85,
        vehicle_steer_max_rad: float = 0.55,
        bicycle_steer_max_rad: float = 1.00,
        vehicle_steer_rate_max_radps: float = 0.8,
        bicycle_steer_rate_max_radps: float = 1.5,
        soft_weight: float = 0.25,
        compare_softness_to_gt: bool = True,
        slip_deadzone_ratio: float = 0.02,
        slip_deadzone_softness: float = 0.02,
        pedestrian_heading_weight: float = 0.05,
        pedestrian_heading_speed_threshold_mps: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.pos_scale_m = float(pos_scale_m)
        self.speed_floor_mps = float(speed_floor_mps)
        self.limit_table = DynamicLimitTable(
            v_max_mps=(
                float(vehicle_v_max_mps),
                float(pedestrian_v_max_mps),
                float(bicycle_v_max_mps),
            ),
            a_max_mps2=(
                float(vehicle_a_max_mps2),
                float(pedestrian_a_max_mps2),
                float(bicycle_a_max_mps2),
            ),
            a_lat_max_mps2=(
                float(vehicle_lat_accel_max_mps2),
                0.0,
                float(bicycle_lat_accel_max_mps2),
            ),
            beta_max_rad=(
                float(vehicle_beta_max_rad),
                10.0,
                float(bicycle_beta_max_rad),
            ),
        )
        self.vehicle_wheelbase_scale = float(vehicle_wheelbase_scale)
        self.bicycle_wheelbase_scale = float(bicycle_wheelbase_scale)
        self.vehicle_steer_max_rad = float(vehicle_steer_max_rad)
        self.bicycle_steer_max_rad = float(bicycle_steer_max_rad)
        self.vehicle_steer_rate_max_radps = float(vehicle_steer_rate_max_radps)
        self.bicycle_steer_rate_max_radps = float(bicycle_steer_rate_max_radps)
        self.soft_weight = float(soft_weight)
        self.compare_softness_to_gt = bool(compare_softness_to_gt)
        self.slip_deadzone_ratio = float(slip_deadzone_ratio)
        self.slip_deadzone_softness = float(slip_deadzone_softness)
        self.pedestrian_heading_weight = float(pedestrian_heading_weight)
        self.pedestrian_heading_speed_threshold_mps = float(pedestrian_heading_speed_threshold_mps)
        self.eps = float(eps)

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_agent_length: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """예측 미래와 GT 미래의 inverse feasibility 값을 계산합니다.

        Args:
            pred_future_norm: 모델이 생성한 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            packed_agent_type: anchor 순서대로 압축한 에이전트 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_agent_length: anchor 순서대로 압축한 agent box length입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 구간의 제어입니다.
                마지막 차원은 ``[v_x^b, v_y^b, omega]`` 이고,
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 구간 제어 유효 마스크입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                최종 loss, raw loss, class별 세부 항, 실제 단위 평균값을 담은 사전입니다.
        """
        if pred_future_norm.numel() == 0:
            return _build_zero_output(pred_future_norm)

        agent_type = packed_agent_type.to(device=pred_future_norm.device, dtype=torch.long).clamp(min=0, max=2)
        agent_length = packed_agent_length.to(device=pred_future_norm.device, dtype=pred_future_norm.dtype)
        prev_control = packed_prev_control.to(device=pred_future_norm.device, dtype=pred_future_norm.dtype)
        prev_control_valid = packed_prev_control_valid.to(device=pred_future_norm.device, dtype=torch.bool)

        output = _build_zero_output(pred_future_norm)
        pred_class_losses: List[Tensor] = []
        raw_pred_class_losses: List[Tensor] = []
        pred_actual_buckets: Dict[str, List[Tensor]] = {
            key: [] for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS
        }
        gt_actual_buckets: Dict[str, List[Tensor]] = {
            key: [] for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS
        }

        class_specs = (
            (VEHICLE_TYPE, "vehicle"),
            (PEDESTRIAN_TYPE, "pedestrian"),
            (BICYCLE_TYPE, "bicycle"),
        )
        for class_id, class_name in class_specs:
            class_mask = agent_type == class_id
            if not class_mask.any():
                continue

            pred_class_future = pred_future_norm[class_mask]
            gt_class_future = target_future_norm[class_mask].detach()
            class_prev_control = prev_control[class_mask]
            class_prev_valid = prev_control_valid[class_mask]
            class_length = agent_length[class_mask]

            if class_id == PEDESTRIAN_TYPE:
                pred_stats = self._compute_pedestrian_stats(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                )
                gt_stats = self._compute_pedestrian_stats(
                    future_norm=gt_class_future,
                    prev_control=class_prev_control.detach(),
                    prev_control_valid=class_prev_valid,
                )
                if self.compare_softness_to_gt:
                    soft_effective = torch.relu(pred_stats["soft"] - gt_stats["soft"])
                else:
                    soft_effective = pred_stats["soft"]
                effective_total = (
                    pred_stats["hard"]
                    + self.soft_weight * soft_effective
                    + self.pedestrian_heading_weight * pred_stats["head"]
                )
                raw_total = (
                    pred_stats["hard"]
                    + self.soft_weight * pred_stats["soft"]
                    + self.pedestrian_heading_weight * pred_stats["head"]
                )
                output["pedestrian_hard"] = pred_stats["hard"].mean()
                output["pedestrian_soft"] = soft_effective.mean()
                output["pedestrian_head"] = pred_stats["head"].mean()
                output["pedestrian_total"] = effective_total.mean()
                pred_actual_buckets["speed_excess_mps"].append(pred_stats["speed_excess_mps"].mean())
                gt_actual_buckets["speed_excess_mps"].append(gt_stats["speed_excess_mps"].mean())
                pred_actual_buckets["accel_excess_mps2"].append(pred_stats["accel_excess_mps2"].mean())
                gt_actual_buckets["accel_excess_mps2"].append(gt_stats["accel_excess_mps2"].mean())
                pred_actual_buckets["heading_error_deg"].append(pred_stats["heading_error_deg"].mean())
                gt_actual_buckets["heading_error_deg"].append(gt_stats["heading_error_deg"].mean())
            else:
                pred_stats = self._compute_vehicle_like_stats(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                    agent_length=class_length,
                    class_id=class_id,
                )
                gt_stats = self._compute_vehicle_like_stats(
                    future_norm=gt_class_future,
                    prev_control=class_prev_control.detach(),
                    prev_control_valid=class_prev_valid,
                    agent_length=class_length,
                    class_id=class_id,
                )
                if self.compare_softness_to_gt:
                    soft_effective = torch.relu(pred_stats["soft"] - gt_stats["soft"])
                else:
                    soft_effective = pred_stats["soft"]
                # slip은 semi_continuous_lqr의 proxy penalty처럼 예측 궤적의
                # 비홀로노믹 위반 자체를 직접 더합니다. 전체 draft.max_weight는
                # 기존 학습 파이프라인이 그대로 적용합니다.
                effective_total = (
                    pred_stats["hard"]
                    + pred_stats["slip"]
                    + self.soft_weight * soft_effective
                )
                raw_total = (
                    pred_stats["hard"]
                    + pred_stats["slip"]
                    + self.soft_weight * pred_stats["soft"]
                )

                output[f"{class_name}_hard"] = pred_stats["hard"].mean()
                output[f"{class_name}_slip"] = pred_stats["slip"].mean()
                output[f"{class_name}_soft"] = soft_effective.mean()
                output[f"{class_name}_total"] = effective_total.mean()
                pred_actual_buckets["speed_excess_mps"].append(pred_stats["speed_excess_mps"].mean())
                gt_actual_buckets["speed_excess_mps"].append(gt_stats["speed_excess_mps"].mean())
                pred_actual_buckets["slip_beta_excess_deg"].append(pred_stats["slip_beta_excess_deg"].mean())
                gt_actual_buckets["slip_beta_excess_deg"].append(gt_stats["slip_beta_excess_deg"].mean())
                pred_actual_buckets["accel_excess_mps2"].append(pred_stats["accel_excess_mps2"].mean())
                gt_actual_buckets["accel_excess_mps2"].append(gt_stats["accel_excess_mps2"].mean())
                pred_actual_buckets["steer_excess_deg"].append(pred_stats["steer_excess_deg"].mean())
                gt_actual_buckets["steer_excess_deg"].append(gt_stats["steer_excess_deg"].mean())
                pred_actual_buckets["steer_rate_excess_degps"].append(pred_stats["steer_rate_excess_degps"].mean())
                gt_actual_buckets["steer_rate_excess_degps"].append(gt_stats["steer_rate_excess_degps"].mean())
                pred_actual_buckets["lat_accel_excess_mps2"].append(pred_stats["lat_accel_excess_mps2"].mean())
                gt_actual_buckets["lat_accel_excess_mps2"].append(gt_stats["lat_accel_excess_mps2"].mean())

            pred_class_losses.append(effective_total.mean())
            raw_pred_class_losses.append(raw_total.mean())

        output["loss"] = self._mean_list_or_zero(pred_class_losses, pred_future_norm)
        output["raw_pred_loss"] = self._mean_list_or_zero(raw_pred_class_losses, pred_future_norm)
        for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            output[f"pred_{key}"] = self._mean_list_or_zero(pred_actual_buckets[key], pred_future_norm)
            output[f"gt_{key}"] = self._mean_list_or_zero(gt_actual_buckets[key], pred_future_norm)
            output[key] = output[f"pred_{key}"]
        return output

    def _compute_vehicle_like_stats(
        self,
        future_norm: Tensor,
        prev_control: Tensor,
        prev_control_valid: Tensor,
        agent_length: Tensor,
        class_id: int,
    ) -> Dict[str, Tensor]:
        """차량 또는 자전거의 역추론 물리량을 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_agent, T, 4]`` 입니다.
            prev_control: 직전 구간 제어입니다.
                shape은 ``[n_agent, 3]`` 이고 마지막 차원은 ``[v_x^b, v_y^b, omega]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_agent]`` 입니다.
            agent_length: agent box length입니다. shape은 ``[n_agent]`` 입니다.
            class_id: ``vehicle`` 또는 ``bicycle``의 종류 번호입니다.

        Returns:
            Dict[str, Tensor]:
                hard, slip, soft, 실제 단위 초과량을 담은 사전입니다.
                각 값의 shape은 ``[n_agent]`` 입니다.
        """
        pos_local_m, heading_local = self._denormalize_future(future_norm)
        pos_seq, heading_seq = self._prepend_virtual_start(pos_local_m, heading_local)
        delta_pos = pos_seq[:, 1:] - pos_seq[:, :-1]
        heading_prev = heading_seq[:, :-1]
        delta_heading = self._wrap_angle(heading_seq[:, 1:] - heading_seq[:, :-1])

        cos_head = heading_prev.cos()
        sin_head = heading_prev.sin()
        vx_body = (delta_pos[..., 0] * cos_head + delta_pos[..., 1] * sin_head) / self.dt
        vy_body = (-delta_pos[..., 0] * sin_head + delta_pos[..., 1] * cos_head) / self.dt
        speed = vx_body

        speed_floor = speed.abs().clamp_min(self.speed_floor_mps)
        curvature = delta_heading / (speed_floor * self.dt)

        wheelbase_scale = self.vehicle_wheelbase_scale if class_id == VEHICLE_TYPE else self.bicycle_wheelbase_scale
        steer_max_rad = self.vehicle_steer_max_rad if class_id == VEHICLE_TYPE else self.bicycle_steer_max_rad
        steer_rate_max_radps = (
            self.vehicle_steer_rate_max_radps
            if class_id == VEHICLE_TYPE
            else self.bicycle_steer_rate_max_radps
        )
        wheelbase = wheelbase_scale * agent_length.clamp_min(0.1)
        steer = torch.atan(wheelbase.unsqueeze(-1) * curvature)

        v_pre = prev_control[:, 0]
        omega_pre = prev_control[:, 2]
        steer_pre = torch.atan(
            wheelbase * omega_pre / v_pre.abs().clamp_min(self.speed_floor_mps)
        )
        prev_valid = prev_control_valid.to(dtype=future_norm.dtype)

        accel = speed.new_zeros(speed.shape)
        accel[:, 0] = prev_valid * (speed[:, 0] - v_pre) / self.dt
        if speed.shape[1] > 1:
            accel[:, 1:] = (speed[:, 1:] - speed[:, :-1]) / self.dt

        steer_rate = steer.new_zeros(steer.shape)
        steer_rate[:, 0] = prev_valid * (steer[:, 0] - steer_pre) / self.dt
        if steer.shape[1] > 1:
            steer_rate[:, 1:] = (steer[:, 1:] - steer[:, :-1]) / self.dt

        lat_accel = speed.abs().square() * curvature.abs()

        v_max = self._select_limit(self.limit_table.v_max_mps, class_id, future_norm)
        a_max = self._select_limit(self.limit_table.a_max_mps2, class_id, future_norm)
        a_lat_max = self._select_limit(self.limit_table.a_lat_max_mps2, class_id, future_norm)
        beta_max = self._select_limit(self.limit_table.beta_max_rad, class_id, future_norm)

        beta = torch.atan2(vy_body.abs(), vx_body.abs() + self.eps)
        beta_pen = self._normalized_slip_square_penalty(
            value=beta,
            limit=beta_max,
            enabled=beta_max > 0.0,
        )
        slip = self._mean_over_time(beta_pen)
        slip_beta_excess_deg = self._mean_over_time(
            torch.rad2deg(torch.relu(beta - beta_max))
        )

        hard = self._mean_over_time(
            self._phi(speed.abs() / v_max - 1.0)
            + self._phi(accel.abs() / a_max - 1.0)
            + self._phi(steer.abs() / steer_max_rad - 1.0)
            + self._phi(steer_rate.abs() / steer_rate_max_radps - 1.0)
            + self._phi(lat_accel / a_lat_max - 1.0)
        )

        if accel.shape[1] > 1:
            accel_delta = accel[:, 1:] - accel[:, :-1]
            steer_rate_delta = steer_rate[:, 1:] - steer_rate[:, :-1]
            soft = self._mean_over_time(
                (accel_delta / a_max).square()
                + (steer_rate_delta / steer_rate_max_radps).square()
            )
        else:
            soft = hard.new_zeros(hard.shape)

        return {
            "hard": hard,
            "slip": slip,
            "soft": soft,
            "speed_excess_mps": self._mean_over_time(torch.relu(speed.abs() - v_max)),
            "slip_beta_excess_deg": slip_beta_excess_deg,
            "accel_excess_mps2": self._mean_over_time(torch.relu(accel.abs() - a_max)),
            "steer_excess_deg": self._mean_over_time(torch.rad2deg(torch.relu(steer.abs() - steer_max_rad))),
            "steer_rate_excess_degps": self._mean_over_time(
                torch.rad2deg(torch.relu(steer_rate.abs() - steer_rate_max_radps))
            ),
            "lat_accel_excess_mps2": self._mean_over_time(torch.relu(lat_accel - a_lat_max)),
        }

    def _compute_pedestrian_stats(
        self,
        future_norm: Tensor,
        prev_control: Tensor,
        prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """사람의 2차원 속도와 2차원 가속도를 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_agent, T, 4]`` 입니다.
            prev_control: 직전 구간 제어입니다.
                shape은 ``[n_agent, 3]`` 이고 마지막 차원은 ``[v_x^b, v_y^b, omega]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                hard, soft, head, 실제 단위 값을 담은 사전입니다.
                각 값의 shape은 ``[n_agent]`` 입니다.
        """
        pos_local_m, heading_local = self._denormalize_future(future_norm)
        pos_seq, _ = self._prepend_virtual_start(pos_local_m, heading_local)
        vel_vec = (pos_seq[:, 1:] - pos_seq[:, :-1]) / self.dt
        speed = torch.linalg.norm(vel_vec, dim=-1)

        prev_vel = self._prev_body_velocity_to_anchor_local(prev_control)
        prev_valid = prev_control_valid.to(dtype=future_norm.dtype).unsqueeze(-1)
        accel_vec = vel_vec.new_zeros(vel_vec.shape)
        accel_vec[:, 0] = prev_valid * (vel_vec[:, 0] - prev_vel) / self.dt
        if vel_vec.shape[1] > 1:
            accel_vec[:, 1:] = (vel_vec[:, 1:] - vel_vec[:, :-1]) / self.dt
        accel = torch.linalg.norm(accel_vec, dim=-1)

        v_max = self._select_limit(self.limit_table.v_max_mps, PEDESTRIAN_TYPE, future_norm)
        a_max = self._select_limit(self.limit_table.a_max_mps2, PEDESTRIAN_TYPE, future_norm)
        hard = self._mean_over_time(
            self._phi(speed / v_max - 1.0)
            + self._phi(accel / a_max - 1.0)
        )

        if accel_vec.shape[1] > 1:
            accel_delta = accel_vec[:, 1:] - accel_vec[:, :-1]
            accel_delta_norm = torch.linalg.norm(accel_delta, dim=-1)
            soft = self._mean_over_time((accel_delta_norm / a_max).square())
        else:
            soft = hard.new_zeros(hard.shape)

        vel_angle = self._safe_angle_from_xy(vel_vec)
        heading_gap = self._wrap_angle(heading_local - vel_angle)
        heading_mask = speed > self.pedestrian_heading_speed_threshold_mps
        head = self._mean_over_time(
            torch.where(heading_mask, heading_gap.square(), torch.zeros_like(heading_gap))
        )
        heading_error_deg = self._masked_mean_over_time(
            torch.rad2deg(heading_gap.abs()),
            heading_mask,
        )

        return {
            "hard": hard,
            "soft": soft,
            "head": head,
            "speed_excess_mps": self._mean_over_time(torch.relu(speed - v_max)),
            "accel_excess_mps2": self._mean_over_time(torch.relu(accel - a_max)),
            "heading_error_deg": heading_error_deg,
        }

    def _prev_body_velocity_to_anchor_local(self, prev_control: Tensor) -> Tensor:
        """직전 body-frame 속도를 현재 anchor-local 2D 속도로 회전합니다.

        Args:
            prev_control: anchor 직전 구간의 제어입니다.
                마지막 차원은 ``[v_x^b, v_y^b, omega]`` 이고,
                shape은 ``[n_agent, 3]`` 입니다.

        Returns:
            Tensor:
                현재 anchor-local 기준 2D 속도입니다. shape은 ``[n_agent, 2]`` 입니다.
        """
        prev_body_vel = prev_control[:, :2]
        prev_head_local = -prev_control[:, 2] * self.dt
        cos_prev = prev_head_local.cos()
        sin_prev = prev_head_local.sin()
        return torch.stack(
            [
                prev_body_vel[:, 0] * cos_prev - prev_body_vel[:, 1] * sin_prev,
                prev_body_vel[:, 0] * sin_prev + prev_body_vel[:, 1] * cos_prev,
            ],
            dim=-1,
        )

    def _select_limit(
        self,
        values: Tuple[float, float, float],
        class_id: int,
        reference: Tensor,
    ) -> Tensor:
        """클래스 하나에 대응하는 제한값 스칼라를 텐서로 만듭니다.

        Args:
            values: ``[vehicle, pedestrian, bicycle]`` 순서의 제한값입니다.
            class_id: 읽고 싶은 종류 번호입니다.
            reference: device와 dtype를 맞출 기준 텐서입니다.
                shape은 임의입니다.

        Returns:
            Tensor:
                스칼라 제한값 텐서입니다. shape은 ``[]`` 입니다.
        """
        return reference.new_tensor(float(values[class_id]))

    def _denormalize_future(self, future_norm: Tensor) -> Tuple[Tensor, Tensor]:
        """정규화 미래를 meter 단위 위치와 heading으로 바꿉니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_agent, T, 4]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                meter 단위 위치 ``[n_agent, T, 2]`` 와 heading ``[n_agent, T]`` 입니다.
        """
        pos_local_m = future_norm[..., :2] * self.pos_scale_m
        heading_local = self._safe_angle_from_xy(future_norm[..., 2:4])
        return pos_local_m, heading_local

    def _safe_angle_from_xy(self, xy: Tensor) -> Tensor:
        """거의 0인 2D 벡터에서도 backward가 NaN이 되지 않도록 각도를 구합니다.

        Args:
            xy: heading 2D 벡터입니다. shape은 ``[..., 2]`` 입니다.

        Returns:
            Tensor:
                안전하게 복원한 각도입니다. shape은 ``[...]`` 입니다.
        """
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True)
        default_xy = torch.zeros_like(xy)
        default_xy[..., 0] = 1.0
        safe_xy = torch.where(
            xy_norm > self.eps,
            xy / xy_norm.clamp_min(self.eps),
            default_xy,
        )
        return torch.atan2(safe_xy[..., 1], safe_xy[..., 0])

    def _prepend_virtual_start(
        self,
        pos_local_m: Tensor,
        heading_local: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """가상 0번째 step을 앞에 붙입니다.

        Args:
            pos_local_m: meter 단위 위치입니다. shape은 ``[n_agent, T, 2]`` 입니다.
            heading_local: heading입니다. shape은 ``[n_agent, T]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                0번째 step이 붙은 위치 ``[n_agent, T+1, 2]`` 와
                heading ``[n_agent, T+1]`` 입니다.
        """
        num_agent = pos_local_m.shape[0]
        pos_zero = pos_local_m.new_zeros((num_agent, 1, 2))
        heading_zero = heading_local.new_zeros((num_agent, 1))
        return torch.cat([pos_zero, pos_local_m], dim=1), torch.cat([heading_zero, heading_local], dim=1)

    def _normalized_slip_square_penalty(
        self,
        value: Tensor,
        limit: Tensor,
        enabled: Tensor | None = None,
    ) -> Tensor:
        """LQR 브랜치와 같은 slip 초과량 제곱 penalty를 계산합니다.

        Args:
            value: slip angle입니다. shape은 ``[n_agent, T]`` 입니다.
            limit: slip angle 제한값입니다. shape은 ``[]`` 이며 broadcast됩니다.
            enabled: ``True``인 위치만 penalty를 켭니다.
                shape은 ``[]`` 또는 ``[n_agent, T]`` 로 broadcast 가능합니다.

        Returns:
            Tensor:
                slip penalty입니다. shape은 ``[n_agent, T]`` 입니다.
        """
        normalized_excess = torch.relu(value - limit) / (limit.abs() + self.eps)
        shifted = (normalized_excess - self.slip_deadzone_ratio) / max(
            self.slip_deadzone_softness,
            self.eps,
        )
        smooth = torch.nn.functional.softplus(shifted) * max(
            self.slip_deadzone_softness,
            self.eps,
        )
        penalty = smooth.square()
        if enabled is None:
            return penalty
        return torch.where(enabled, penalty, penalty.new_zeros(()))

    def _phi(self, value: Tensor) -> Tensor:
        """hard 위반 함수입니다.

        Args:
            value: 제한값 대비 초과 비율입니다. shape은 임의입니다.

        Returns:
            Tensor:
                양수 초과량만 부드럽게 반영한 penalty입니다. shape은 입력과 같습니다.
        """
        return torch.relu(value).square()

    def _mean_over_time(self, value: Tensor) -> Tensor:
        """시간축 평균을 계산합니다.

        Args:
            value: 마지막 축이 시간인 텐서입니다. shape은 ``[n_agent, T]`` 입니다.
                이미 anchor별 값이면 shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor:
                anchor별 평균값입니다. shape은 ``[n_agent]`` 입니다.
        """
        if value.dim() == 1:
            return value
        return value.mean(dim=-1)

    def _masked_mean_over_time(self, value: Tensor, mask: Tensor) -> Tensor:
        """시간축에서 유효한 위치만 평균합니다.

        Args:
            value: 평균할 값입니다. shape은 ``[n_agent, T]`` 입니다.
            mask: 유효 위치입니다. shape은 ``[n_agent, T]`` 입니다.

        Returns:
            Tensor:
                anchor별 masked 평균입니다. shape은 ``[n_agent]`` 입니다.
        """
        if value.dim() == 1:
            return torch.where(mask, value, torch.zeros_like(value))
        mask_f = mask.to(dtype=value.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        return (value * mask_f).sum(dim=-1) / denom

    def _mean_list_or_zero(self, values: List[Tensor], reference: Tensor) -> Tensor:
        """스칼라 목록의 평균을 계산하고, 비어 있으면 0을 돌려줍니다.

        Args:
            values: 평균할 스칼라 텐서 목록입니다.
            reference: 빈 목록일 때 device와 dtype를 맞출 기준 텐서입니다.
                shape은 임의입니다.

        Returns:
            Tensor:
                스칼라 평균값입니다. shape은 ``[]`` 입니다.
        """
        if len(values) == 0:
            return reference.new_zeros(())
        return torch.stack(values).mean()

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor:
                같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
