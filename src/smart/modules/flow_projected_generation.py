"""Projected Diffusion-style OT-ODE generation with feasibility projection.

각 ODE step 후 kinematic feasibility gap에 대한 gradient descent를 수행해
생성된 trajectory가 feasible region으로 수렴하도록 유도합니다.

Reference: "Projected Diffusion Models" (Fishman et al.)
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ProjectedFlowGenerator(nn.Module):
    """OT-ODE generation with per-step feasibility projection.

    알고리즘:
        for step t=0..T-1:
            x_{t+dt} = x_t + dt * v_θ(x_t, tau)       # ODE step
            for k in range(n_proj_steps):               # feasibility grad descent
                gap = projector.compute_terminal_cost(x_{t+dt})
                x_{t+dt} = x_{t+dt} - proj_lr * ∇_x gap

    Args:
        projector: kinematic feasibility gap를 계산하는 모듈입니다.
        n_proj_steps: 매 ODE step 후 gradient descent 반복 횟수입니다.
        proj_lr: gradient descent step size입니다.
    """

    def __init__(
        self,
        projector: nn.Module,
        n_proj_steps: int = 3,
        proj_lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.n_proj_steps = n_proj_steps
        self.proj_lr = proj_lr

    def _feasibility_grad_step(
        self,
        x: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> Tensor:
        """kinematic gap에 대해 n_proj_steps번 gradient descent를 수행합니다.

        Args:
            x: 현재 normalized trajectory입니다. shape ``[n, 20, 4]``.
            agent_type: agent 종류입니다. shape ``[n]``.
            current_control: 직전 body control입니다. shape ``[n, 3]`` 또는 None.
            current_control_valid: current_control 유효 여부입니다. shape ``[n]`` 또는 None.

        Returns:
            Tensor: projection 후 normalized trajectory. shape ``[n, 20, 4]``.
        """
        for step_idx in range(self.n_proj_steps):
            x_req = x.detach().requires_grad_(True)
            with torch.enable_grad():
                gap, metrics = self.projector.compute_terminal_cost(
                    pred_clean_norm=x_req,
                    agent_type=agent_type,
                    current_control=current_control,
                    current_control_valid=current_control_valid,
                )
            if not gap.requires_grad:
                # gap이 0 (이미 feasible)이면 grad 없음 → 조기 종료
                break
            (grad,) = torch.autograd.grad(gap, x_req)
            x = (x - self.proj_lr * grad).detach()
        return x

    @torch.no_grad()
    def generate(
        self,
        flow_ode: nn.Module,
        model_fn: Callable[[Tensor, Tensor], Tensor],
        x_init: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
        steps: int = 16,
    ) -> Tensor:
        """Projected OT-ODE generation을 수행합니다.

        Args:
            flow_ode: ``generate`` / ``eps`` 속성을 가진 FlowODE 모듈입니다.
            model_fn: ``(x_t: [n,20,4], tau: [n]) -> velocity: [n,20,4]`` 인 callable입니다.
            x_init: 시작 noise입니다. shape ``[n, 20, 4]``.
            agent_type: agent 종류입니다. shape ``[n]``.
            current_control: 직전 body control입니다. shape ``[n, 3]`` 또는 None.
            current_control_valid: current_control 유효 여부입니다.
            steps: ODE 적분 스텝 수입니다.

        Returns:
            Tensor: 생성된 normalized trajectory. shape ``[n, 20, 4]``.
        """
        x_t = x_init
        t0 = float(getattr(flow_ode, "eps", 1e-3))
        dt = (1.0 - t0) / float(steps)

        for i in range(steps):
            tau_val = t0 + i * dt
            tau = x_t.new_full((x_t.shape[0],), tau_val)
            tau_mid = x_t.new_full((x_t.shape[0],), tau_val + 0.5 * dt)

            # midpoint method (2nd-order RK) — inference와 동일
            v1 = model_fn(x_t, tau)
            x_mid = (x_t + 0.5 * dt * v1).detach()
            v2 = model_fn(x_mid, tau_mid)
            x_t = (x_t + dt * v2).detach()

            # Feasibility grad descent
            if self.n_proj_steps > 0:
                x_t = self._feasibility_grad_step(
                    x_t,
                    agent_type=agent_type.to(x_t.device),
                    current_control=(
                        current_control.to(x_t.device, dtype=torch.float32)
                        if current_control is not None else None
                    ),
                    current_control_valid=(
                        current_control_valid.to(x_t.device)
                        if current_control_valid is not None else None
                    ),
                )

        return x_t
