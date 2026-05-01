from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from src.smart.modules.draft_physics import (
    BICYCLE_TYPE,
    DRAFT_PHYSICS_COMPONENT_KEYS,
    DraftPhysicsRegularizer,
    PEDESTRIAN_TYPE,
    VEHICLE_TYPE,
    _build_zero_output,
)


class TopKDraftPhysicsRegularizer(DraftPhysicsRegularizer):
    """상위 위반 중심 DRaFT 물리 손실을 계산합니다.

    기존 물리 손실은 각 agent의 미래 2초를 시간 평균으로 줄입니다. 이 클래스는
    기존 평균 손실을 그대로 계산한 뒤, 시간축에서 큰 위반만 고른 손실을 추가로
    계산하고 두 값을 절반씩 섞습니다. 기본 ``topk_violation_k=20``에서는 20개
    미래 시점을 모두 평균하므로 기존 로직과 같은 값이 됩니다.

    Args:
        *args: 기존 ``DraftPhysicsRegularizer``에 그대로 넘길 위치 인자입니다.
        topk_violation_k: 한 agent 안에서 가장 큰 위반을 몇 개 시점까지 볼지 정합니다.
            20이면 2초 미래 전체를 보므로 기존 평균 손실과 같습니다.
        **kwargs: 기존 ``DraftPhysicsRegularizer``에 그대로 넘길 이름 인자입니다.
    """

    def __init__(
        self,
        *args: object,
        topk_violation_k: int = 20,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(topk_violation_k) < 1:
            raise ValueError("topk_violation_k must be >= 1.")
        self.topk_violation_k = int(topk_violation_k)

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_agent_length: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """기존 평균 손실과 상위 위반 손실을 절반씩 섞어 반환합니다.

        Args:
            pred_future_norm: 모델이 생성한 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            packed_agent_type: anchor별 agent 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_agent_length: anchor별 agent 길이입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 제어값입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 제어값 유효 여부입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                기존 출력 사전과 같은 구조입니다. ``loss``와 class별 total 항은
                ``0.5 * (기존 평균 손실 + 상위 위반 손실)`` 값입니다.
        """
        mean_output = super().forward(
            pred_future_norm=pred_future_norm,
            target_future_norm=target_future_norm,
            packed_agent_type=packed_agent_type,
            packed_agent_length=packed_agent_length,
            packed_prev_control=packed_prev_control,
            packed_prev_control_valid=packed_prev_control_valid,
        )

        if pred_future_norm.numel() == 0:
            return mean_output
        if self.topk_violation_k >= int(pred_future_norm.shape[1]):
            return mean_output

        topk_output = self._compute_topk_output(
            pred_future_norm=pred_future_norm,
            target_future_norm=target_future_norm,
            packed_agent_type=packed_agent_type,
            packed_agent_length=packed_agent_length,
            packed_prev_control=packed_prev_control,
            packed_prev_control_valid=packed_prev_control_valid,
        )
        mean_output["loss"] = 0.5 * (mean_output["loss"] + topk_output["loss"])
        mean_output["raw_pred_loss"] = 0.5 * (
            mean_output["raw_pred_loss"] + topk_output["raw_pred_loss"]
        )
        for key in DRAFT_PHYSICS_COMPONENT_KEYS:
            mean_output[key] = 0.5 * (mean_output[key] + topk_output[key])
        return mean_output

    def _compute_topk_output(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_agent_length: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """상위 위반만 모은 물리 손실 사전을 만듭니다.

        Args:
            pred_future_norm: 모델이 생성한 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            packed_agent_type: anchor별 agent 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_agent_length: anchor별 agent 길이입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 제어값입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 제어값 유효 여부입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]: 상위 위반 집계만 사용한 손실 사전입니다.
        """
        agent_type = packed_agent_type.to(
            device=pred_future_norm.device,
            dtype=torch.long,
        ).clamp(min=0, max=2)
        agent_length = packed_agent_length.to(
            device=pred_future_norm.device,
            dtype=pred_future_norm.dtype,
        )
        prev_control = packed_prev_control.to(
            device=pred_future_norm.device,
            dtype=pred_future_norm.dtype,
        )
        prev_control_valid = packed_prev_control_valid.to(
            device=pred_future_norm.device,
            dtype=torch.bool,
        )

        output = _build_zero_output(pred_future_norm)
        pred_class_losses: List[Tensor] = []
        raw_pred_class_losses: List[Tensor] = []

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
                pred_stats = self._compute_pedestrian_topk_stats(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                )
                gt_stats = self._compute_pedestrian_topk_stats(
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
            else:
                pred_stats = self._compute_vehicle_like_topk_stats(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                    agent_length=class_length,
                    class_id=class_id,
                )
                gt_stats = self._compute_vehicle_like_topk_stats(
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

            pred_class_losses.append(effective_total.mean())
            raw_pred_class_losses.append(raw_total.mean())

        output["loss"] = self._mean_list_or_zero(pred_class_losses, pred_future_norm)
        output["raw_pred_loss"] = self._mean_list_or_zero(raw_pred_class_losses, pred_future_norm)
        return output

    def _topk_mean_over_time(self, value: Tensor) -> Tensor:
        """시간축에서 큰 값 K개만 골라 평균합니다.

        Args:
            value: 시점별 위반량입니다. shape은 ``[n_agent, T]`` 입니다.

        Returns:
            Tensor: agent별 상위 위반 평균입니다. shape은 ``[n_agent]`` 입니다.
        """
        if value.shape[1] == 0:
            return value.new_zeros((value.shape[0],))
        topk = min(self.topk_violation_k, int(value.shape[1]))
        if topk >= int(value.shape[1]):
            return self._mean_over_time(value)
        return value.topk(topk, dim=1, largest=True, sorted=False).values.mean(dim=1)

    def _compute_vehicle_like_topk_stats(
        self,
        future_norm: Tensor,
        prev_control: Tensor,
        prev_control_valid: Tensor,
        agent_length: Tensor,
        class_id: int,
    ) -> Dict[str, Tensor]:
        """차량/자전거의 상위 위반 물리량을 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_agent, T, 4]`` 입니다.
            prev_control: 직전 구간 제어입니다. shape은 ``[n_agent, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_agent]`` 입니다.
            agent_length: agent 길이입니다. shape은 ``[n_agent]`` 입니다.
            class_id: 차량 또는 자전거 종류 번호입니다.

        Returns:
            Dict[str, Tensor]: ``hard``, ``slip``, ``soft`` 상위 위반 평균입니다.
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

        wheelbase_scale = (
            self.vehicle_wheelbase_scale if class_id == VEHICLE_TYPE else self.bicycle_wheelbase_scale
        )
        steer_max_rad = (
            self.vehicle_steer_max_rad if class_id == VEHICLE_TYPE else self.bicycle_steer_max_rad
        )
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
        beta_penalty = self._normalized_slip_square_penalty(
            value=beta,
            limit=beta_max,
            enabled=beta_max > 0.0,
        )
        hard_penalty = (
            self._phi(speed.abs() / v_max - 1.0)
            + self._phi(accel.abs() / a_max - 1.0)
            + self._phi(steer.abs() / steer_max_rad - 1.0)
            + self._phi(steer_rate.abs() / steer_rate_max_radps - 1.0)
            + self._phi(lat_accel / a_lat_max - 1.0)
        )

        if accel.shape[1] > 1:
            accel_delta = accel[:, 1:] - accel[:, :-1]
            steer_rate_delta = steer_rate[:, 1:] - steer_rate[:, :-1]
            soft_penalty = (
                (accel_delta / a_max).square()
                + (steer_rate_delta / steer_rate_max_radps).square()
            )
        else:
            soft_penalty = hard_penalty.new_zeros((hard_penalty.shape[0], 0))

        return {
            "hard": self._topk_mean_over_time(hard_penalty),
            "slip": self._topk_mean_over_time(beta_penalty),
            "soft": self._topk_mean_over_time(soft_penalty),
        }

    def _compute_pedestrian_topk_stats(
        self,
        future_norm: Tensor,
        prev_control: Tensor,
        prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """보행자의 상위 위반 물리량을 계산합니다.

        Args:
            future_norm: 정규화 미래입니다. shape은 ``[n_agent, T, 4]`` 입니다.
            prev_control: 직전 구간 제어입니다. shape은 ``[n_agent, 3]`` 입니다.
            prev_control_valid: 직전 제어 유효 여부입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]: ``hard``, ``soft``, ``head`` 상위 위반 평균입니다.
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
        hard_penalty = self._phi(speed / v_max - 1.0) + self._phi(accel / a_max - 1.0)

        if accel_vec.shape[1] > 1:
            accel_delta = accel_vec[:, 1:] - accel_vec[:, :-1]
            accel_delta_norm = torch.linalg.norm(accel_delta, dim=-1)
            soft_penalty = (accel_delta_norm / a_max).square()
        else:
            soft_penalty = hard_penalty.new_zeros((hard_penalty.shape[0], 0))

        vel_angle = self._safe_angle_from_xy(vel_vec)
        heading_gap = self._wrap_angle(heading_local - vel_angle)
        heading_mask = speed > self.pedestrian_heading_speed_threshold_mps
        head_penalty = torch.where(
            heading_mask,
            heading_gap.square(),
            torch.zeros_like(heading_gap),
        )

        return {
            "hard": self._topk_mean_over_time(hard_penalty),
            "soft": self._topk_mean_over_time(soft_penalty),
            "head": self._topk_mean_over_time(head_penalty),
        }
