"""Agent-type-aware kinematic feasibility projection for flow inference.

Agent type 별로 다른 post-processing을 적용합니다:

- Vehicle (type=0) / Cyclist (type=2) : Non-holonomic (Bicycle Model)
    Heading 방향으로만 이동 가능 → heading projection + deadzone

- Pedestrian (type=1) : Holonomic (Point-mass)
    방향 무관하게 이동 가능 → magnitude-based deadzone + velocity clipping
    (heading projection 금지 — 보행자 게걸음 제거 시 정상 lateral step까지 삭제됨)

Agent type encoding (Waymo / SMART):
    0 = Vehicle, 1 = Pedestrian, 2 = Cyclist
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KinematicProjection(nn.Module):
    """Per-agent-type kinematic projection applied after each ODE step.

    Args:
        coord_scale: 모델이 예측한 (Δx, Δy) 정규화 값을 미터로 바꾸는 스케일.
            내부 물리 제약(deadzone/조향 제한/속도 클리핑)은 미터 단위로 계산하고,
            반환은 다시 정규화 단위로 되돌립니다.
        dt: 한 스텝 시간 간격(초). (Δx, Δy)를 "스텝당 변위"로 해석할 때 사용합니다.
        vehicle_deadzone: Vehicle/Cyclist 종방향 변위(스텝당, meter)의 deadzone threshold.
            절댓값이 이 값 미만이면 0으로 강제 (jitter 제거).
        ped_deadzone: 보행자 이동 벡터 크기(스텝당, meter)의 deadzone threshold.
            크기가 이 값 미만이면 정지로 간주.
        ped_max_speed: 보행자 이동 벡터 최대 크기(스텝당, meter).
            이 값을 초과하는 이동은 스케일링으로 잘라냅니다.
        lat_accel_limit: 차량/자전거의 횡가속 한계(m/s^2). yaw 변화량을 속도에 따라 제한합니다.
        max_dyaw_static: 저속에서도 무한정 회전하지 않도록 하는 기하학적 dyaw 상한(라디안/스텝).
        eps: heading 벡터 정규화 시 0 나누기 방지 epsilon.
    """

    def __init__(
        self,
        coord_scale: float = 20.0,
        dt: float = 0.1,
        vehicle_deadzone: float = 0.05,
        ped_deadzone: float = 0.025,
        ped_max_speed: float = 0.5,
        lat_accel_limit: float = 4.0,
        max_dyaw_static: float = 0.35,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.dt = float(dt)
        self.vehicle_deadzone = float(vehicle_deadzone)
        self.ped_deadzone = float(ped_deadzone)
        self.ped_max_speed = float(ped_max_speed)
        self.lat_accel_limit = float(lat_accel_limit)
        self.max_dyaw_static = float(max_dyaw_static)
        self.eps = float(eps)

    def forward(
        self,
        x: Tensor,
        agent_type: Tensor | None = None,
        proj_weight: float = 1.0,
    ) -> Tensor:
        """kinematic projection을 적용합니다.

        Args:
            x: ODE step 출력. shape ``[n, 20, 4]``, dim 4 = (Δx, Δy, cos, sin).
            agent_type: 각 agent의 type. shape ``[n]``.
                None이면 전체를 vehicle로 처리합니다.
            proj_weight: 0.0 = projection 없음(x 그대로), 1.0 = 완전 projection.
                PPR 루프에서 ODE 시간 t_next를 넘겨 초반 step은 약하게,
                마지막 step(t_next=1)은 완전하게 projection 합니다.

        Returns:
            Tensor: kinematically feasible 궤적. shape ``[n, 20, 4]``.
        """
        if agent_type is None:
            projected = self._project_nonholonomic(x)
        else:
            agent_type = agent_type.to(device=x.device)
            ped_mask = (agent_type == 1)   # pedestrian
            veh_mask = ~ped_mask           # vehicle(0) + cyclist(2)

            projected = x.clone()
            if veh_mask.any():
                projected[veh_mask] = self._project_nonholonomic(x[veh_mask])
            if ped_mask.any():
                projected[ped_mask] = self._project_pointmass(x[ped_mask])

        if proj_weight >= 1.0 - self.eps:
            return projected
        return proj_weight * projected + (1.0 - proj_weight) * x

    # ------------------------------------------------------------------
    # Non-holonomic: heading projection + deadzone  (Vehicle / Cyclist)
    # ------------------------------------------------------------------

    def _project_nonholonomic(self, x: Tensor) -> Tensor:
        # x shape: [Batch, Seq, 4] (normalized Δx, Δy, cos, sin)
        pred_dx_n = x[..., 0]
        pred_dy_n = x[..., 1]
        pred_cos = x[..., 2]
        pred_sin = x[..., 3]

        # 0) Heading 벡터 정규화
        norm = torch.sqrt(pred_cos**2 + pred_sin**2 + self.eps)
        pred_cos = pred_cos / norm
        pred_sin = pred_sin / norm

        # 1) Meter 변환 및 속도 계산
        pred_dx_m = pred_dx_n * self.coord_scale
        pred_dy_m = pred_dy_n * self.coord_scale
        step_dist_m = torch.sqrt(pred_dx_m**2 + pred_dy_m**2 + self.eps)
        v_ms = step_dist_m / max(self.dt, self.eps)

        # 2) Raw yaw 및 dyaw 계산
        raw_yaw = torch.atan2(pred_sin, pred_cos)
        yaw_prev = torch.cat([raw_yaw[..., :1], raw_yaw[..., :-1]], dim=-1)
        dyaw = torch.atan2(torch.sin(raw_yaw - yaw_prev), torch.cos(raw_yaw - yaw_prev))
        # 첫 번째 스텝의 dyaw는 0으로 시작
        dyaw = torch.cat([torch.zeros_like(dyaw[..., :1]), dyaw[..., 1:]], dim=-1)

        # 3) 속도 기반 dyaw 제한 + 정지 상태 판별
        v_ms_safe = v_ms.clamp_min(1.0)
        max_dyaw_dynamic = (self.lat_accel_limit / v_ms_safe) * self.dt
        allowed_dyaw = torch.minimum(
            max_dyaw_dynamic,
            torch.full_like(max_dyaw_dynamic, self.max_dyaw_static),
        )

        # [핵심] 움직임이 deadzone보다 작으면 회전(dyaw)을 0으로 고정
        is_moving = (step_dist_m >= self.vehicle_deadzone).float()
        clipped_dyaw = torch.clamp(dyaw, -allowed_dyaw, allowed_dyaw)
        clipped_dyaw = clipped_dyaw * is_moving

        # 4) 보정된 heading 계산
        # 첫 스텝이 정지 상태면 raw_yaw[0](예측 heading)을 버리고 0(= previous head 유지)으로 시작
        init_yaw = raw_yaw[..., :1] * is_moving[..., :1]
        feasible_yaw = init_yaw + torch.cumsum(clipped_dyaw, dim=-1)
        c = torch.cos(feasible_yaw)
        s = torch.sin(feasible_yaw)

        # 5) Heading 방향 성분 투영 (Non-holonomic 제약) + 위치 deadzone
        v_signed_m = pred_dx_m * c + pred_dy_m * s
        v_signed_m = v_signed_m * is_moving  # 위치도 동일하게 deadzone 적용

        dx_feas_m = v_signed_m * c
        dy_feas_m = v_signed_m * s

        # 6) 다시 normalized 단위로 복귀
        dx_feas_n = dx_feas_m / self.coord_scale
        dy_feas_n = dy_feas_m / self.coord_scale

        return torch.stack([dx_feas_n, dy_feas_n, c, s], dim=-1)

    # ------------------------------------------------------------------
    # Holonomic: magnitude deadzone + velocity clipping  (Pedestrian)
    # ------------------------------------------------------------------
    def _project_pointmass(self, x: Tensor) -> Tensor:
        pred_dx_n = x[..., 0]
        pred_dy_n = x[..., 1]
        pred_cos = x[..., 2]
        pred_sin = x[..., 3]

        # 내부 계산은 meter에서
        pred_dx_m = pred_dx_n * self.coord_scale
        pred_dy_m = pred_dy_n * self.coord_scale
        step_dist_m = torch.sqrt(pred_dx_m**2 + pred_dy_m**2 + self.eps)

        # 1. Magnitude deadzone: 미세 jitter 제거
        moving = (step_dist_m >= self.ped_deadzone).float()
        dx_feas_m = pred_dx_m * moving
        dy_feas_m = pred_dy_m * moving

        # 2. Velocity clipping: 순간이동 방지
        scale = (self.ped_max_speed / step_dist_m.clamp_min(self.eps)).clamp_max(1.0)
        dx_feas_m = dx_feas_m * scale
        dy_feas_m = dy_feas_m * scale

        # 다시 normalized 단위로 복귀
        dx_feas_n = dx_feas_m / self.coord_scale
        dy_feas_n = dy_feas_m / self.coord_scale

        # heading은 그대로 유지 (보행자는 방향과 이동이 독립적)
        return torch.stack([dx_feas_n, dy_feas_n, pred_cos, pred_sin], dim=-1)
