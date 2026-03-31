"""Agent-type-aware kinematic feasibility projection for flow inference.

두 가지 projection 모드:

1. **Bicycle Model** (``use_bicycle_model=True``, ``v_init`` 제공 시):
   물리 모델 기반 projection.
   - Vehicle/Cyclist: Kinematic Bicycle Model
       State: (v, θ),  Controls: (a, δ)
       Constraints: |a| ≤ a_max/d_max,  |Δθ| ≤ v·tan(δ_max)/L·dt
   - Pedestrian: Point-mass (holonomic)
       Constraint: |Δv| ≤ ped_a_max·dt, v ≥ 0

2. **Heuristic** (``v_init`` 없을 때 fallback):
   - Vehicle/Cyclist: non-holonomic heading projection + deadzone
   - Pedestrian: magnitude deadzone + speed clipping

Agent type encoding (Waymo / SMART):
    0 = Vehicle, 1 = Pedestrian, 2 = Cyclist
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class KinematicProjection(nn.Module):
    """Per-agent-type kinematic projection.

    Args:
        coord_scale: 정규화 (Δx, Δy) → 미터 변환 스케일 (기본 20.0).
        dt: 한 스텝 시간 간격 (초, 기본 0.1).
        use_bicycle_model: True면 bicycle model 사용 (``v_init`` 파라미터 필요).
            False면 heuristic projection만 사용.

        --- Bicycle model 파라미터 ---
        wheelbase: 차량 축간거리 (m). 기본 2.7.
        delta_max: 최대 조향각 (rad). 기본 0.52 ≈ 30°.
        a_max: 최대 가속도 (m/s²). 기본 4.0.
        d_max: 최대 감속도 (m/s²). 기본 8.0.
        ped_a_max: 보행자 최대 가속/감속 (m/s²). 기본 2.0.

        --- Heuristic 파라미터 (fallback) ---
        vehicle_deadzone: Vehicle/Cyclist 정지 판별 threshold (m/step). 기본 0.05.
        ped_deadzone: 보행자 정지 판별 threshold (m/step). 기본 0.025.
        ped_max_speed: 보행자 최대 속도 (m/step). 기본 0.5.
        lat_accel_limit: heuristic 횡가속 한계 (m/s²). 기본 4.0.
        max_dyaw_static: heuristic dyaw 상한 (rad/step). 기본 0.35.
        eps: 수치 안정성 epsilon.
    """

    def __init__(
        self,
        coord_scale: float = 20.0,
        dt: float = 0.1,
        use_bicycle_model: bool = False,
        # Bicycle model params
        wheelbase: float = 2.7,
        delta_max: float = 0.52,
        a_max: float = 4.0,
        d_max: float = 8.0,
        ped_a_max: float = 2.0,
        # Heuristic fallback params
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
        self.use_bicycle_model = bool(use_bicycle_model)
        # Bicycle
        self.wheelbase = float(wheelbase)
        self.delta_max = float(delta_max)
        self.tan_delta_max = math.tan(delta_max)
        self.a_max = float(a_max)
        self.d_max = float(d_max)
        self.ped_a_max = float(ped_a_max)
        # Heuristic
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
        v_init: Tensor | None = None,
    ) -> Tensor:
        """Kinematic projection을 적용합니다.

        Args:
            x: shape ``[n, T, 4]``, (Δx_norm, Δy_norm, cos_δ, sin_δ).
                cos_δ / sin_δ 는 현재 heading 기준 **delta heading** 입니다.
            agent_type: shape ``[n]``. None이면 전체 vehicle 처리.
            proj_weight: 0.0 = identity, 1.0 = 완전 projection.
                PPR에서 t_next 를 넘겨 초반 step은 약하게 projection 합니다.
            v_init: 각 agent의 청크 시작 속도 (m/s). shape ``[n]``.
                제공되고 ``use_bicycle_model=True`` 이면 bicycle model을 사용합니다.
                None이면 heuristic projection으로 fallback 합니다.

        Returns:
            Tensor: shape ``[n, T, 4]``.
        """
        use_bm = self.use_bicycle_model and v_init is not None

        if agent_type is None:
            if use_bm:
                projected = self._project_bicycle_vehicle(x, v_init)
            else:
                projected = self._project_nonholonomic(x)
        else:
            agent_type = agent_type.to(device=x.device)
            ped_mask = (agent_type == 1)
            veh_mask = ~ped_mask
            projected = x.clone()
            if veh_mask.any():
                _v = v_init[veh_mask] if use_bm else None
                if use_bm:
                    projected[veh_mask] = self._project_bicycle_vehicle(x[veh_mask], _v)
                else:
                    projected[veh_mask] = self._project_nonholonomic(x[veh_mask])
            if ped_mask.any():
                _v = v_init[ped_mask] if use_bm else None
                if use_bm:
                    projected[ped_mask] = self._project_bicycle_ped(x[ped_mask], _v)
                else:
                    projected[ped_mask] = self._project_pointmass(x[ped_mask])

        if proj_weight >= 1.0 - self.eps:
            return projected
        return proj_weight * projected + (1.0 - proj_weight) * x

    # ------------------------------------------------------------------
    # Bicycle Model: Vehicle / Cyclist
    # ------------------------------------------------------------------

    def _project_bicycle_vehicle(self, x: Tensor, v_init: Tensor) -> Tensor:
        """Kinematic Bicycle Model projection.

        Args:
            x: shape ``[n, T, 4]``.
            v_init: 청크 시작 속도 (m/s). shape ``[n]``.

        Returns:
            Tensor: shape ``[n, T, 4]``.
        """
        n, T, _ = x.shape
        device, dtype = x.device, x.dtype

        dx_m = x[..., 0] * self.coord_scale   # [n, T]
        dy_m = x[..., 1] * self.coord_scale
        cos_d = x[..., 2]
        sin_d = x[..., 3]

        # 예측 속도 (m/s)
        v_pred = torch.sqrt(dx_m**2 + dy_m**2 + self.eps) / self.dt  # [n, T]

        # 순간 delta heading: raw_yaw 는 청크 시작 기준 누적 delta
        raw_yaw = torch.atan2(sin_d, cos_d)  # [n, T]
        yaw_prev = torch.cat(
            [torch.zeros(n, 1, device=device, dtype=dtype), raw_yaw[:, :-1]], dim=1
        )
        dθ_pred = torch.atan2(
            torch.sin(raw_yaw - yaw_prev), torch.cos(raw_yaw - yaw_prev)
        )  # [n, T]

        # Sequential bicycle model
        v_cur = v_init.to(device=device, dtype=dtype)
        v_feas_list = []
        dθ_feas_list = []

        tan_dm = self.tan_delta_max
        L = self.wheelbase
        dt = self.dt
        a_hi_step = self.a_max * dt
        d_hi_step = self.d_max * dt

        for t in range(T):
            # 가속도 제약
            v_lo = (v_cur - d_hi_step).clamp_min(0.0)
            v_hi = v_cur + a_hi_step
            v_t = torch.max(torch.min(v_pred[:, t], v_hi), v_lo)

            # Yaw rate 제약: v_cur 기반 (예측값이 아님)
            dθ_max = v_cur * tan_dm / L * dt            # [n]
            dθ_t = torch.clamp(dθ_pred[:, t], min=-dθ_max, max=dθ_max)

            v_feas_list.append(v_t)
            dθ_feas_list.append(dθ_t)
            v_cur = v_t  # 다음 step의 초기 속도

        v_feas = torch.stack(v_feas_list, dim=1)    # [n, T]
        dθ_feas = torch.stack(dθ_feas_list, dim=1)  # [n, T]

        # 누적 delta heading → cos/sin 재구성
        θ_feas = torch.cumsum(dθ_feas, dim=1)       # [n, T]
        c_out = torch.cos(θ_feas)
        s_out = torch.sin(θ_feas)

        # 이동 거리 재구성 (non-holonomic: heading 방향으로만)
        disp_feas = v_feas * dt                      # [n, T]  m/step
        dx_out = disp_feas * c_out / self.coord_scale
        dy_out = disp_feas * s_out / self.coord_scale

        return torch.stack([dx_out, dy_out, c_out, s_out], dim=-1)

    # ------------------------------------------------------------------
    # Bicycle Model: Pedestrian (Point-mass, holonomic)
    # ------------------------------------------------------------------

    def _project_bicycle_ped(self, x: Tensor, v_init: Tensor) -> Tensor:
        """Point-mass model projection for pedestrians.

        이동 방향은 유지하고, 속도 크기만 가속도 제약으로 정제합니다.
        Heading (cos/sin)은 별도 facing direction이므로 그대로 유지합니다.

        Args:
            x: shape ``[n, T, 4]``.
            v_init: 청크 시작 속도 (m/s). shape ``[n]``.

        Returns:
            Tensor: shape ``[n, T, 4]``.
        """
        n, T, _ = x.shape
        device, dtype = x.device, x.dtype

        dx_m = x[..., 0] * self.coord_scale
        dy_m = x[..., 1] * self.coord_scale
        disp_pred = torch.sqrt(dx_m**2 + dy_m**2 + self.eps)  # [n, T]
        v_pred = disp_pred / self.dt

        v_cur = v_init.to(device=device, dtype=dtype)
        v_feas_list = []
        a_hi_step = self.ped_a_max * self.dt

        for t in range(T):
            v_lo = (v_cur - a_hi_step).clamp_min(0.0)
            v_hi = v_cur + a_hi_step
            v_t = torch.max(torch.min(v_pred[:, t], v_hi), v_lo)
            v_feas_list.append(v_t)
            v_cur = v_t

        v_feas = torch.stack(v_feas_list, dim=1)    # [n, T]
        disp_feas = v_feas * self.dt                 # [n, T]

        # 방향 유지, 크기만 조정
        scale = disp_feas / disp_pred.clamp_min(self.eps)
        dx_out = dx_m * scale / self.coord_scale
        dy_out = dy_m * scale / self.coord_scale

        return torch.stack([dx_out, dy_out, x[..., 2], x[..., 3]], dim=-1)

    # ------------------------------------------------------------------
    # Heuristic fallback: Vehicle / Cyclist
    # ------------------------------------------------------------------

    def _project_nonholonomic(self, x: Tensor) -> Tensor:
        pred_dx_n = x[..., 0]
        pred_dy_n = x[..., 1]
        pred_cos = x[..., 2]
        pred_sin = x[..., 3]

        norm = torch.sqrt(pred_cos**2 + pred_sin**2 + self.eps)
        pred_cos = pred_cos / norm
        pred_sin = pred_sin / norm

        pred_dx_m = pred_dx_n * self.coord_scale
        pred_dy_m = pred_dy_n * self.coord_scale
        step_dist_m = torch.sqrt(pred_dx_m**2 + pred_dy_m**2 + self.eps)
        v_ms = step_dist_m / max(self.dt, self.eps)

        raw_yaw = torch.atan2(pred_sin, pred_cos)
        yaw_prev = torch.cat([raw_yaw[..., :1], raw_yaw[..., :-1]], dim=-1)
        dyaw = torch.atan2(torch.sin(raw_yaw - yaw_prev), torch.cos(raw_yaw - yaw_prev))
        dyaw = torch.cat([torch.zeros_like(dyaw[..., :1]), dyaw[..., 1:]], dim=-1)

        v_ms_safe = v_ms.clamp_min(1.0)
        max_dyaw_dynamic = (self.lat_accel_limit / v_ms_safe) * self.dt
        allowed_dyaw = torch.minimum(
            max_dyaw_dynamic,
            torch.full_like(max_dyaw_dynamic, self.max_dyaw_static),
        )

        is_moving = (step_dist_m >= self.vehicle_deadzone).float()
        clipped_dyaw = torch.clamp(dyaw, -allowed_dyaw, allowed_dyaw)
        clipped_dyaw = clipped_dyaw * is_moving

        # 첫 스텝이 정지면 prev heading (delta=0) 유지
        init_yaw = raw_yaw[..., :1] * is_moving[..., :1]
        feasible_yaw = init_yaw + torch.cumsum(clipped_dyaw, dim=-1)
        c = torch.cos(feasible_yaw)
        s = torch.sin(feasible_yaw)

        v_signed_m = pred_dx_m * c + pred_dy_m * s
        v_signed_m = v_signed_m * is_moving

        dx_feas_n = (v_signed_m * c) / self.coord_scale
        dy_feas_n = (v_signed_m * s) / self.coord_scale

        return torch.stack([dx_feas_n, dy_feas_n, c, s], dim=-1)

    # ------------------------------------------------------------------
    # Heuristic fallback: Pedestrian
    # ------------------------------------------------------------------

    def _project_pointmass(self, x: Tensor) -> Tensor:
        pred_dx_n = x[..., 0]
        pred_dy_n = x[..., 1]
        pred_cos = x[..., 2]
        pred_sin = x[..., 3]

        pred_dx_m = pred_dx_n * self.coord_scale
        pred_dy_m = pred_dy_n * self.coord_scale
        step_dist_m = torch.sqrt(pred_dx_m**2 + pred_dy_m**2 + self.eps)

        moving = (step_dist_m >= self.ped_deadzone).float()
        dx_feas_m = pred_dx_m * moving
        dy_feas_m = pred_dy_m * moving

        scale = (self.ped_max_speed / step_dist_m.clamp_min(self.eps)).clamp_max(1.0)
        dx_feas_m = dx_feas_m * scale
        dy_feas_m = dy_feas_m * scale

        return torch.stack([
            dx_feas_m / self.coord_scale,
            dy_feas_m / self.coord_scale,
            pred_cos,
            pred_sin,
        ], dim=-1)
