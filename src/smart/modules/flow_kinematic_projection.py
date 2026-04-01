"""TV-LQR Kinematic Bicycle Model projection for flow model inference.

토큰 포맷 (agent당, step당) — Flow 학습 타겟(`flow_token_processor`)과 동일:
    x[..., 0:2] : (x_cum_norm, y_cum_norm)
        coarse anchor **고정 LOCAL** 프레임에서의 **누적** 위치 (정규화, m 복원: ×coord_scale).
        각 시점 t의 값 = anchor 시점 대비 t까지의 오프셋이 아니라, 해당 10Hz 시각의
        중심점 로컬 좌표 (연속 시점으로 보면 사실상 누적 경로).
        내부적으로는 스텝 변위는 인접 시점 차분으로 구함.
    x[..., 2:4] : (cos_θ, sin_θ)
        해당 시점의 **로컬 절대 헤딩** (anchor heading 대비, Flow 타겟과 동일).

commit_bridge 처리 (ContinuousCommitBridge.commit):
    pos_local = y_hat_norm[..., :2] * 20          → 누적 LOCAL 위치 (m)
    transform_to_global(pos_local, current_head)  → 현재 heading으로 회전 후 world 좌표 변환
    commit_head = current_head + atan2(sin_θ, cos_θ) (토큰별 상대 회전 누적)

Agent type (Waymo/SMART):
    0 = Vehicle, 1 = Pedestrian, 2 = Cyclist

성능:
    모든 연산이 agent 차원에서 완전 벡터화됨 (Python for-loop은 T 축만).
    LQR 역방향 Riccati도 B개 agent를 배치 matmul로 동시 처리.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _wrap_to_pi(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class KinematicProjection(nn.Module):
    """TV-LQR + Kinematic Bicycle Model projection (fully vectorized over agents).

    LOCAL chunk frame에서 동작 (Flow 타겟과 동일 frame):
      - 좌표 채널은 **누적** 로컬 위치(정규화); 프로젝션은 차분해 자전거/보행자를 적용한 뒤
        다시 누적 형식으로 내보냄.
      - heading 채널은 해당 시점의 로컬 절대 헤딩(기존과 동일).
    """

    def __init__(
        self,
        coord_scale: float = 20.0,
        dt: float = 0.1,
        # Bicycle model
        wheelbase: float = 2.7,
        delta_max: float = 0.52,
        a_max: float = 4.0,
        d_max: float = 8.0,
        delta_rate_max: float = 0.6,
        # Pedestrian
        ped_a_max: float = 2.0,
        eps: float = 1e-6,
        # TV-LQR
        use_lqr: bool = True,
        lqr_q_xy: float = 2.0,
        lqr_q_yaw: float = 2.0,
        lqr_q_v: float = 0.5,
        lqr_q_delta: float = 0.2,
        lqr_r_a: float = 0.2,
        lqr_r_delta_rate: float = 0.2,
        lqr_qf_scale: float = 2.0,
        **_unused: object,
    ) -> None:
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.dt = float(dt)
        self.wheelbase = float(wheelbase)
        self.delta_max = float(delta_max)
        self.a_max = float(a_max)
        self.d_max = float(d_max)
        self.delta_rate_max = float(delta_rate_max)
        self.ped_a_max = float(ped_a_max)
        self.eps = float(eps)
        self.use_lqr = bool(use_lqr)
        self.lqr_q_xy = float(lqr_q_xy)
        self.lqr_q_yaw = float(lqr_q_yaw)
        self.lqr_q_v = float(lqr_q_v)
        self.lqr_q_delta = float(lqr_q_delta)
        self.lqr_r_a = float(lqr_r_a)
        self.lqr_r_delta_rate = float(lqr_r_delta_rate)
        self.lqr_qf_scale = float(lqr_qf_scale)

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        agent_type: Tensor | None = None,
        proj_weight: float = 1.0,
        v_init: Tensor | None = None,
    ) -> Tensor:
        """Kinematic projection (PPR 및 단일 후처리용).

        Args:
            x:          [n, T, 4]. (x_cum_norm, y_cum_norm, cos θ, sin θ).
            agent_type: [n]. 0=Vehicle, 1=Pedestrian, 2=Cyclist. None → 전체 vehicle.
            proj_weight: 0=identity, 1=완전 projection.
            v_init:     [n]. chunk 시작 속도 (m/s).

        Returns:
            Tensor [n, T, 4].
        """
        if x.numel() == 0:
            return x

        projected = x.clone()
        n = x.shape[0]

        if agent_type is None:
            if n > 0:
                projected = self._project_bicycle_batch(x, v_init, delta_init=None)
        else:
            ped_mask = (agent_type == 1) # pedestrian
            veh_mask = ~ped_mask

            if veh_mask.any():
                v0 = v_init[veh_mask] if v_init is not None else None
                projected[veh_mask] = self._project_bicycle_batch(x[veh_mask], v0)

            if ped_mask.any():
                v0 = v_init[ped_mask] if v_init is not None else None
                projected[ped_mask] = self._project_ped_batch(x[ped_mask], v0)

        if proj_weight >= 1.0 - self.eps:
            return projected
        return proj_weight * projected + (1.0 - proj_weight) * x

    def project_with_state(
        self,
        x: Tensor,
        agent_type: Tensor | None = None,
        v_init: Tensor | None = None,
        delta_init: Tensor | None = None,
        commit_steps: int = 5,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project and return kinematic state at chunk commit boundary.

        Args:
            x:           [n, T, 4].
            agent_type:  [n].
            v_init:      [n]. chunk 시작 속도 (m/s).
            delta_init:  [n]. chunk 시작 조향각 (rad). None = 곡률에서 추정.
            commit_steps: committed chunk 길이 (기본 5 = shift).

        Returns:
            (projected [n, T, 4], v_final [n], delta_final [n])
            v_final[i]:     step commit_steps-1 종료 시 속도 (다음 chunk 시작).
            delta_final[i]: 동일 시점 조향각.
        """
        if x.numel() == 0:
            n = x.shape[0]
            zeros = x.new_zeros(n)
            return x, zeros, zeros

        n, T = x.shape[0], x.shape[1]
        commit_idx = min(commit_steps, T)
        projected = x.clone()
        v_final = x.new_zeros(n)
        delta_final = x.new_zeros(n)

        if agent_type is None:
            if n > 0:
                out, v_seq, d_seq = self._project_bicycle_batch_with_state(
                    x, v_init, delta_init
                )
                projected = out
                v_final = v_seq[:, commit_idx]
                delta_final = d_seq[:, commit_idx]
        else:
            ped_mask = (agent_type == 1)
            veh_mask = ~ped_mask

            if veh_mask.any():
                v0 = v_init[veh_mask] if v_init is not None else None
                d0 = delta_init[veh_mask] if delta_init is not None else None
                out_v, v_seq_v, d_seq_v = self._project_bicycle_batch_with_state(
                    x[veh_mask], v0, d0
                )
                projected[veh_mask] = out_v
                v_final[veh_mask] = v_seq_v[:, commit_idx]
                delta_final[veh_mask] = d_seq_v[:, commit_idx]

            if ped_mask.any():
                v0 = v_init[ped_mask] if v_init is not None else None
                out_p, v_seq_p = self._project_ped_batch_with_state(x[ped_mask], v0)
                projected[ped_mask] = out_p
                v_final[ped_mask] = v_seq_p[:, commit_idx]
                # pedestrian: delta = 0 (holonomic)

        return projected, v_final, delta_final

    def _step_delta_m_from_cumulative_xy(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Flow 타겟과 동일: x[...,0:2]는 누적 로컬 위치(정규화) → 스텝 변위 (m)."""
        pos_cum = x[:, :, :2]
        prev = torch.cat(
            [pos_cum.new_zeros(pos_cum.shape[0], 1, 2), pos_cum[:, :-1]],
            dim=1,
        )
        d_norm = pos_cum - prev
        return d_norm[..., 0] * self.coord_scale, d_norm[..., 1] * self.coord_scale

    # ──────────────────────────────────────────────────────────────────────
    # Pedestrian: Point-mass speed clamp (벡터화)
    # ──────────────────────────────────────────────────────────────────────

    def _project_ped_batch(self, x: Tensor, v_init: Optional[Tensor]) -> Tensor:
        out, _ = self._project_ped_batch_with_state(x, v_init)
        return out

    def _project_ped_batch_with_state(
        self, x: Tensor, v_init: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Pedestrian batch projection.

        Args:
            x:      [B, T, 4].
            v_init: [B] or None.

        Returns:
            (out [B, T, 4], v_seq [B, T+1]).
            v_seq[:, 0] = v_init, v_seq[:, t+1] = velocity after step t.
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        dx_m, dy_m = self._step_delta_m_from_cumulative_xy(x)
        disp = (dx_m ** 2 + dy_m ** 2 + self.eps).sqrt()   # [B, T]
        v_pred = disp / self.dt                              # [B, T]

        v_seq = torch.empty(B, T + 1, device=device, dtype=dtype)
        if v_init is not None:
            v_seq[:, 0] = v_init.to(device=device, dtype=dtype).clamp_min(0.0)
        else:
            v_seq[:, 0] = v_pred[:, 0].clamp_min(0.0)

        a_hi = self.ped_a_max * self.dt
        for t in range(T):
            v_lo = (v_seq[:, t] - a_hi).clamp_min(0.0)     # [B]
            v_hi = v_seq[:, t] + a_hi                       # [B]
            v_seq[:, t + 1] = torch.max(torch.min(v_pred[:, t], v_hi), v_lo)

        scale = (v_seq[:, 1:] * self.dt) / disp.clamp_min(self.eps)   # [B, T]
        out = x.clone()
        step_x = dx_m * scale / self.coord_scale
        step_y = dy_m * scale / self.coord_scale
        out[:, :, 0] = torch.cumsum(step_x, dim=1)
        out[:, :, 1] = torch.cumsum(step_y, dim=1)
        return out, v_seq

    # ──────────────────────────────────────────────────────────────────────
    # Vehicle / Cyclist: TV-LQR + Kinematic Bicycle (배치 벡터화)
    # ──────────────────────────────────────────────────────────────────────

    def _project_bicycle_batch(
        self,
        x: Tensor,
        v_init: Optional[Tensor],
        delta_init: Optional[Tensor] = None,
    ) -> Tensor:
        out, _, _ = self._project_bicycle_batch_with_state(x, v_init, delta_init)
        return out

    def _project_bicycle_batch_with_state(
        self,
        x: Tensor,
        v_init: Optional[Tensor],
        delta_init: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """TV-LQR bicycle model projection (B agents, 완전 벡터화).

        LOCAL chunk frame에서 동작:
          - 시작: (px=0, py=0, yaw=0), velocity=v_init, steering=delta_init.
          - 출력 변위/heading은 동일 LOCAL frame 기준.

        Args:
            x:          [B, T, 4]. 입력 토큰.
            v_init:     [B] or None. chunk 시작 속도 (m/s).
            delta_init: [B] or None. chunk 시작 조향각 (rad).
                        None → steering_target[:, 0] 으로 초기화.

            x[B, T, 0] = 시작점 anchor를 기준으로 현재 시점 x좌표 기준 얼마나 떨어져 있는가. 를 /20으로 normalized 된 값.
            x[B, T, 1] = 시작점 anchor를 기준으로 현재 시점 y좌표 기준 얼마나 떨어져 있는가. 를 /20으로 normalized 된 값.
            x[B, T, 2] = 시작점 anchor를 기준으로 현재 시점 heading 차이의 cos값
            x[B, T, 3] = 시작점 anchor를 기준으로 현재 시점 heading 차이의 sin값

        Returns:
            (out [B, T, 4], v_seq [B, T+1], delta_seq [B, T+1])
            v_seq[:, 0] = v_init, v_seq[:, t+1] = velocity after step t.
            delta_seq[:, 0] = initial steering, delta_seq[:, t+1] = steering after step t.
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        dt, L, eps = self.dt, self.wheelbase, self.eps

        # ── 1. 토큰 디코딩 (좌표는 누적 로컬 → 스텝 변위) ────────────────────
        dx_m, dy_m = self._step_delta_m_from_cumulative_xy(x)
        disp_m = (dx_m ** 2 + dy_m ** 2 + eps).sqrt() #매 스텝마다 이동한 거리.
        v_pred = disp_m / dt                     # [B, T], m/s

        norm_h = (x[:, :, 2] ** 2 + x[:, :, 3] ** 2 + eps).sqrt() #model output cos, sin의 norm이 1이 아닐 수 있으므로 보정용.
        theta_cum = torch.atan2(x[:, :, 3] / norm_h, x[:, :, 2] / norm_h)   # [B, T] #보정한 cos, sin을 기반으로 heading 계산
        theta_prev = torch.cat([theta_cum.new_zeros(B, 1), theta_cum[:, :-1]], dim=1) #-1시점씩 sliding시킨 뒤
        dtheta_pred = _wrap_to_pi(theta_cum - theta_prev)   # [B, T] # 현재 시점 heading과 이전 시점 heading의 차이를 빼서 step단위 \theta 차이 계산.

        # ── 2. 참조 속도 v_ref [B, T+1] ─────────────────────────────────
        v_ref = torch.empty(B, T + 1, device=device, dtype=dtype)
        if v_init is not None:
            v_ref[:, 0] = v_init.to(device=device, dtype=dtype).clamp_min(0.0)
        else:
            v_ref[:, 0] = v_pred[:, 0].clamp_min(0.0)

        a_hi = self.a_max * dt
        d_hi = self.d_max * dt
        for t in range(T):
            v_lo = (v_ref[:, t] - d_hi).clamp_min(0.0)
            v_hi = v_ref[:, t] + a_hi
            v_ref[:, t + 1] = torch.max(torch.min(v_pred[:, t], v_hi), v_lo)

        # ── 3. 참조 조향각 delta_ref [B, T+1] ───────────────────────────
        # gate: 저속 시 곡률 영향 억제 → 정지 차량 crab-walk 방지
        v_at_step = v_ref[:, :T]                                           # [B, T]
        gate = (v_at_step / (a_hi + eps)).clamp(0.0, 1.0)
        kappa = dtheta_pred / (v_at_step * dt + eps) * gate
        steering_target = torch.atan(L * kappa).clamp(-self.delta_max, self.delta_max)

        delta_ref = torch.empty(B, T + 1, device=device, dtype=dtype)
        if delta_init is not None:
            delta_ref[:, 0] = delta_init.to(device=device, dtype=dtype).clamp(
                -self.delta_max, self.delta_max
            )
        else:
            delta_ref[:, 0] = steering_target[:, 0]

        for t in range(T):
            tgt = steering_target[:, t]
            dr = ((tgt - delta_ref[:, t]) / dt).clamp(-self.delta_rate_max, self.delta_rate_max)
            delta_ref[:, t + 1] = (delta_ref[:, t] + dr * dt).clamp(
                -self.delta_max, self.delta_max
            )

        # ── 4. 참조 궤적 전향 적분 [B, T+1] ─────────────────────────────
        px_ref = torch.zeros(B, T + 1, device=device, dtype=dtype)
        py_ref = torch.zeros(B, T + 1, device=device, dtype=dtype)
        yaw_ref = torch.zeros(B, T + 1, device=device, dtype=dtype)
        for t in range(T):
            yaw_ref[:, t + 1] = yaw_ref[:, t] + v_ref[:, t] * torch.tan(delta_ref[:, t]) / L * dt
            px_ref[:, t + 1] = px_ref[:, t] + v_ref[:, t] * torch.cos(yaw_ref[:, t]) * dt
            py_ref[:, t + 1] = py_ref[:, t] + v_ref[:, t] * torch.sin(yaw_ref[:, t]) * dt

        # Feedforward controls [B, T]
        a_ff = ((v_ref[:, 1:] - v_ref[:, :-1]) / dt).clamp(-self.d_max, self.a_max)
        dr_ff = ((delta_ref[:, 1:] - delta_ref[:, :-1]) / dt).clamp(
            -self.delta_rate_max, self.delta_rate_max
        )

        # ── 5. TV-LQR 게인 [B, 2, 5] × T ────────────────────────────────
        if self.use_lqr:
            k_list = self._tv_lqr_gains_batch(
                v_ref[:, :T], delta_ref[:, :T], yaw_ref[:, :T]
            )
        else:
            k_list = []

        # ── 6. 전향 시뮬레이션 (벡터화) ──────────────────────────────────
        px = px_ref[:, 0].clone()       # [B]
        py = py_ref[:, 0].clone()
        yaw = yaw_ref[:, 0].clone()
        v_s = v_ref[:, 0].clone()
        delta_s = delta_ref[:, 0].clone()

        px_out = torch.empty(B, T + 1, device=device, dtype=dtype)
        py_out = torch.empty(B, T + 1, device=device, dtype=dtype)
        yaw_out = torch.empty(B, T + 1, device=device, dtype=dtype)
        v_out = torch.empty(B, T + 1, device=device, dtype=dtype)
        delta_out = torch.empty(B, T + 1, device=device, dtype=dtype)

        px_out[:, 0] = px; py_out[:, 0] = py; yaw_out[:, 0] = yaw
        v_out[:, 0] = v_s; delta_out[:, 0] = delta_s

        for t in range(T):
            a_cmd = a_ff[:, t].clone()    # [B]
            dr_cmd = dr_ff[:, t].clone()  # [B]

            if t < len(k_list):
                # error state [B, 5]
                e = torch.stack([
                    px - px_ref[:, t],
                    py - py_ref[:, t],
                    _wrap_to_pi(yaw - yaw_ref[:, t]),
                    v_s - v_ref[:, t],
                    delta_s - delta_ref[:, t],
                ], dim=1)  # [B, 5]
                # u_fb = -(K @ e):  K [B,2,5] × e [B,5,1] → [B,2]
                u_fb = -(k_list[t] @ e.unsqueeze(-1)).squeeze(-1)
                a_cmd = (a_cmd + u_fb[:, 0]).clamp(-self.d_max, self.a_max)
                dr_cmd = (dr_cmd + u_fb[:, 1]).clamp(-self.delta_rate_max, self.delta_rate_max)

            # Kinematic bicycle (rear-axle center 기준)
            v_next = (v_s + a_cmd * dt).clamp_min(0.0)
            delta_next = (delta_s + dr_cmd * dt).clamp(-self.delta_max, self.delta_max)
            yaw_next = yaw + v_s * torch.tan(delta_s) / L * dt
            px_next = px + v_s * torch.cos(yaw) * dt
            py_next = py + v_s * torch.sin(yaw) * dt

            px = px_next; py = py_next; yaw = yaw_next
            v_s = v_next; delta_s = delta_next

            px_out[:, t + 1] = px; py_out[:, t + 1] = py; yaw_out[:, t + 1] = yaw
            v_out[:, t + 1] = v_s; delta_out[:, t + 1] = delta_s

        # ── 7. 토큰 포맷 재인코딩 (Flow / commit_bridge: 누적 로컬 xy) ───────
        step_x_norm = (px_out[:, 1:] - px_out[:, :-1]) / self.coord_scale   # [B, T]
        step_y_norm = (py_out[:, 1:] - py_out[:, :-1]) / self.coord_scale
        cum_x_norm = torch.cumsum(step_x_norm, dim=1)
        cum_y_norm = torch.cumsum(step_y_norm, dim=1)

        # 누적 delta heading: yaw_out[:, 0] = 0 (LOCAL frame 시작)
        theta_cum_out = yaw_out[:, 1:]   # [B, T]
        cos_out = torch.cos(theta_cum_out)
        sin_out = torch.sin(theta_cum_out)

        out = torch.stack([cum_x_norm, cum_y_norm, cos_out, sin_out], dim=-1)
        return out, v_out, delta_out

    # ──────────────────────────────────────────────────────────────────────
    # TV-LQR: 역방향 Riccati (B agents 배치 벡터화)
    # ──────────────────────────────────────────────────────────────────────

    def _tv_lqr_gains_batch(
        self,
        v_ref: Tensor,
        delta_ref: Tensor,
        yaw_ref: Tensor,
    ) -> list[Tensor]:
        """배치 역방향 Riccati sweep → T개의 LQR 게인 K[0..T-1] 반환.

        State error:  e = [Δx, Δy, Δyaw, Δv, Δdelta]  (5차원)
        Control:      u = [a, delta_rate]               (2차원)

        Args:
            v_ref:     [B, T]. 참조 속도.
            delta_ref: [B, T]. 참조 조향각.
            yaw_ref:   [B, T]. 참조 heading (LOCAL frame 절대 yaw).

        Returns:
            List[Tensor] len=T, 각 원소 [B, 2, 5].
        """
        B, T = v_ref.shape
        device, dtype = v_ref.device, v_ref.dtype
        dt, L = self.dt, self.wheelbase
        eps = self.eps

        Q, R, Qf = self._build_lqr_weights(device=device, dtype=dtype)
        # P: [B, 5, 5]
        P = Qf.unsqueeze(0).expand(B, -1, -1).contiguous()

        I5 = torch.eye(5, device=device, dtype=dtype)
        I2 = torch.eye(2, device=device, dtype=dtype)
        Q_b = Q.unsqueeze(0)   # [1, 5, 5] for broadcast
        R_b = R.unsqueeze(0)   # [1, 2, 2]
        I2_b = I2.unsqueeze(0) * eps

        k_list: list[Tensor] = [torch.empty(B, 2, 5, device=device, dtype=dtype) for _ in range(T)]

        for t in range(T - 1, -1, -1):
            v = v_ref[:, t].clamp_min(0.0)                                    # [B]
            d = delta_ref[:, t].clamp(-self.delta_max, self.delta_max)        # [B]
            yaw = yaw_ref[:, t]                                                # [B]

            cy = torch.cos(yaw); sy = torch.sin(yaw)
            cos_d = torch.cos(d)
            cos_d_sq = (cos_d * cos_d).clamp_min(eps)
            tan_d = torch.tan(d)

            # 선형화된 A: [B, 5, 5]
            A_t = I5.unsqueeze(0).expand(B, -1, -1).clone()
            A_t[:, 0, 2] = -dt * v * sy
            A_t[:, 0, 3] = dt * cy
            A_t[:, 1, 2] = dt * v * cy
            A_t[:, 1, 3] = dt * sy
            A_t[:, 2, 3] = dt * tan_d / L
            A_t[:, 2, 4] = dt * v / (L * cos_d_sq)

            # 입력 B: [B, 5, 2]
            B_t = torch.zeros(B, 5, 2, device=device, dtype=dtype)
            B_t[:, 3, 0] = dt
            B_t[:, 4, 1] = dt

            # K = (R + B^T P B)^{-1} B^T P A
            Bt_P = torch.bmm(B_t.transpose(-2, -1), P)          # [B, 2, 5]
            S = R_b + torch.bmm(Bt_P, B_t) + I2_b               # [B, 2, 2]
            K = torch.linalg.solve(S, torch.bmm(Bt_P, A_t))     # [B, 2, 5]
            k_list[t] = K

            # P = Q + A^T P (A - BK)
            BK = torch.bmm(B_t, K)                               # [B, 5, 5]
            P = Q_b + torch.bmm(A_t.transpose(-2, -1), torch.bmm(P, A_t - BK))

        return k_list

    def _build_lqr_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        q = torch.tensor(
            [self.lqr_q_xy, self.lqr_q_xy, self.lqr_q_yaw, self.lqr_q_v, self.lqr_q_delta],
            device=device, dtype=dtype,
        )
        r = torch.tensor([self.lqr_r_a, self.lqr_r_delta_rate], device=device, dtype=dtype)
        return torch.diag(q), torch.diag(r), torch.diag(self.lqr_qf_scale * q)
