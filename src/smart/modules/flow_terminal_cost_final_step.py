from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.smart.modules.flow_adjoint_matching import SmoothControlProjector
from src.smart.utils import angle_between_2d_vectors


@dataclass
class TerminalCostFinalStepResult:
    loss: Tensor
    terminal_cost: Tensor
    projection_gap: Tensor
    final_count: Tensor
    flow_reg_loss: Tensor = None


class TerminalCostFinalStepLoss(nn.Module):
    """
    Feasibility 기반 projector terminal_cost를 연속값 reward(최소화 loss)로 쓰고,
    memoryless Euler-Maruyama rollout에서 마지막 diffusion step에서만 gradient가 흐르도록 합니다.
    """

    def __init__(
        self,
        rollout_steps: int = 4,
        rollout_noise_scale: float = 1.0,
        feasible_weight: float = 1.0,
        smooth_deadzone_epsilon: Tuple[float, float, float] = (0.01, 0.01, 0.01),
        smooth_deadzone_tau: float = 0.002,
        flow_reg_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.rollout_steps = int(rollout_steps)
        self.rollout_noise_scale = float(rollout_noise_scale)
        self.flow_reg_lambda = float(flow_reg_lambda)
        self.projector = SmoothControlProjector(
            feasible_weight=feasible_weight,
            smooth_deadzone_epsilon=smooth_deadzone_epsilon,
            smooth_deadzone_tau=smooth_deadzone_tau,
        )

    def _build_step_times(
        self,
        flow_ode: nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        """
        t0=tau_start=flow_ode.eps로부터 rollout_steps 구간을 나눕니다.
        """
        t0 = float(flow_ode.eps)
        dt = (1.0 - t0) / float(self.rollout_steps)
        return [
            torch.full((batch_size,), t0 + step_idx * dt, device=device, dtype=dtype)
            for step_idx in range(self.rollout_steps)
        ]

    def _rollout_memoryless_sde_last_step_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        x_init_norm: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list[Tensor]]:
        """
        memoryless Euler-Maruyama SDE 롤아웃을 만들되,
        중간 step은 `detach`해서 마지막 step에서만 autograd graph를 유지합니다.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        device = anchor_hidden_valid.device
        dtype = anchor_hidden_valid.dtype

        if x_init_norm is None:
            x_init_norm = torch.randn(
                batch_size,
                20,
                4,
                device=device,
                dtype=dtype,
            ) * self.rollout_noise_scale
        else:
            if x_init_norm.dim() != 3 or tuple(x_init_norm.shape[-2:]) != (20, 4):
                raise ValueError(
                    f"x_init_norm must have shape [N,20,4], got {tuple(x_init_norm.shape)}"
                )
            x_init_norm = x_init_norm.to(device=device, dtype=dtype)

        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(
            flow_ode=flow_ode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        current_state = x_init_norm
        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]

            # DDP unused-parameter를 피하기 위해 매 step에서 파라미터가 그래프에 연결되도록
            # forward는 항상 autograd를 켠 채로 수행합니다.
            # 그래프 폭증은 아래에서 state detach로 제어합니다.
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

            next_state = current_state + dt * drift + (dt**0.5) * sigma * noise

            if step_idx < self.rollout_steps - 1:
                current_state = next_state.detach()
            else:
                current_state = next_state

        self._assert_finite_tensor("final_state", current_state)
        return current_state, times

    def _rollout_ode_last_step_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """결정론적 ODE rollout. 마지막 step에서만 autograd graph를 유지합니다.

        SDE 노이즈 없이 순수 drift로 적분하므로 trajectory가 안정적입니다.
        gradient는 마지막 step의 forward_components를 통해서만 전파됩니다.

        Returns:
            Tuple[Tensor, Tensor]:
                - final_state: [n_anchor, 20, 4]
                - residual_velocity: 마지막 step의 residual_velocity_head 출력 [n_anchor, 20, 4]
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        device = anchor_hidden_valid.device
        dtype = anchor_hidden_valid.dtype

        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(flow_ode, batch_size, device, dtype)

        x_t = torch.randn(batch_size, 20, 4, device=device, dtype=dtype)

        # OT flow ODE — midpoint method (inference와 동일)
        # DDP unused-parameter 방지: 매 step에서 autograd 활성 상태로 forward합니다.
        last_residual: torch.Tensor | None = None
        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]
            tau_mid = tau + 0.5 * dt

            v1_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_t,
                tau=tau,
            )
            x_mid = (x_t + 0.5 * dt * v1_dict["velocity"])
            if step_idx < self.rollout_steps - 1:
                x_mid = x_mid.detach()

            v2_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_mid,
                tau=tau_mid,
            )
            next_x_t = x_t + dt * v2_dict["velocity"]
            last_residual = v2_dict["residual_velocity"]

            if step_idx < self.rollout_steps - 1:
                x_t = next_x_t.detach()
            else:
                x_t = next_x_t

        self._assert_finite_tensor("ode_final_state", x_t)
        return x_t, last_residual

    def forward_open_loop(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> TerminalCostFinalStepResult:
        """
        GT context에서 anchors를 독립적으로 샘플링하는 open-loop penalty입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            # DDP: residual_velocity_head 파라미터를 gradient graph에 연결해야 함
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        if current_control is not None:
            current_control = current_control.to(dtype=torch.float32, device=anchor_hidden_valid.device)

        final_state, residual_velocity = self._rollout_ode_last_step_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
        )
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state,
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        self._assert_finite_tensor("open_loop/terminal_cost", terminal_cost)
        self._assert_finite_tensor("open_loop/projection_gap", metrics["projection_gap"])

        # flow regularization: residual_velocity → 0 으로 당겨 pretrained 분포 유지
        flow_reg_loss = self.flow_reg_lambda * residual_velocity.pow(2).mean()
        loss = terminal_cost + flow_reg_loss

        return TerminalCostFinalStepResult(
            loss=loss,
            terminal_cost=terminal_cost.detach(),
            projection_gap=metrics["projection_gap"].detach(),
            final_count=torch.tensor(anchor_hidden_valid.shape[0], device=terminal_cost.device),
            flow_reg_loss=flow_reg_loss.detach(),
        )

    def _rollout_ode_from_noisy_gt(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        gt_clean_norm: Tensor,
    ) -> Tensor:
        """GT에 noise를 추가한 x_t에서 ODE를 시작합니다.

        flow_ode.sample로 GT → (x_t, tau_start)를 만들고,
        각 sample마다 dt = (1 - tau_start) / rollout_steps로
        tau_start → 1까지 OT-ODE를 적분합니다.
        마지막 step에서만 gradient graph를 유지합니다.

        Args:
            gt_clean_norm: GT normalized trajectory입니다. shape은 ``[batch, 20, 4]``.

        Returns:
            Tensor: ODE final state, shape ``[batch, 20, 4]``.
        """
        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        gt_clean_norm = gt_clean_norm.to(dtype=torch.float32, device=anchor_hidden_valid.device)

        # GT → x_t 샘플링 (random tau per sample)
        flow_sample = flow_ode.sample(gt_clean_norm, target_type="velocity")
        x_t = flow_sample.x_t.detach()       # [batch, 20, 4]
        tau_start = flow_sample.tau.detach()  # [batch]

        # per-sample dt: (1 - tau_start) / rollout_steps
        dt = (1.0 - tau_start) / float(self.rollout_steps)  # [batch]
        dt_view = dt.view(-1, 1, 1)

        for step_idx in range(self.rollout_steps):
            tau = tau_start + step_idx * dt           # [batch]
            tau_mid = tau + 0.5 * dt                  # [batch]

            # midpoint method
            v1_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_t,
                tau=tau,
            )
            x_mid = x_t + 0.5 * dt_view * v1_dict["velocity"]
            if step_idx < self.rollout_steps - 1:
                x_mid = x_mid.detach()

            v2_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_mid,
                tau=tau_mid,
            )
            next_x_t = x_t + dt_view * v2_dict["velocity"]
            if step_idx < self.rollout_steps - 1:
                x_t = next_x_t.detach()
            else:
                x_t = next_x_t  # 마지막 step: gradient 유지

        self._assert_finite_tensor("noisy_gt_ode/final_state", x_t)
        return x_t

    def _rollout_ode_full_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """전체 rollout_steps에 걸쳐 gradient가 흐르는 OT-ODE rollout입니다.

        모든 step에서 state를 detach하지 않으므로 BPTT로 모든 step의
        velocity 파라미터에 gradient가 전달됩니다.

        Args:
            flow_decoder: flow decoder 모듈입니다.
            flow_ode: ODE/flow 모듈입니다.
            anchor_hidden_valid: GT history 컨텍스트입니다. shape ``[n, hidden_dim]``.
            noise: 시작 noise입니다. shape ``[n, 20, 4]``.

        Returns:
            Tensor: 최종 ODE state. shape ``[n, 20, 4]``.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        device = anchor_hidden_valid.device
        dtype = anchor_hidden_valid.dtype

        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(flow_ode, batch_size, device, dtype)

        x_t = noise.to(device=device, dtype=dtype)
        t0 = float(flow_ode.eps)
        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]
            tau_mid = tau + 0.5 * dt  # midpoint time

            # midpoint method (2nd-order RK) — inference와 동일
            v1 = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_t,
                tau=tau,
            )["velocity"]
            x_mid = x_t + 0.5 * dt * v1
            v2 = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_mid,
                tau=tau_mid,
            )["velocity"]
            # OT-ODE midpoint step: NO detach — full BPTT
            x_t = x_t + dt * v2

        self._assert_finite_tensor("full_grad_ode/final_state", x_t)
        return x_t

    def _bc_loss_flowmatching(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        gt_clean_norm: Tensor,
    ) -> Tensor:
        """Flow Matching BC loss를 계산합니다.

        표준 OT-Flow Matching 목적함수입니다::

            L_BC = E_{t, x0, x1}[ ||v_θ(x_t, t) - (x1 - x0)||² ]

        여기서 x0 ~ N(0,I), x1 = GT, x_t = (1-t)*x0 + t*x1 입니다.

        Args:
            flow_decoder: flow decoder 모듈입니다.
            flow_ode: ODE/flow 모듈입니다.
            anchor_hidden_valid: shape ``[n, hidden_dim]``.
            gt_clean_norm: GT normalized trajectory. shape ``[n, 20, 4]``.

        Returns:
            Tensor: scalar BC loss.
        """
        device = gt_clean_norm.device
        dtype = gt_clean_norm.dtype
        n = gt_clean_norm.shape[0]

        t0 = float(flow_ode.eps)
        t = t0 + (1.0 - t0) * torch.rand(1, device=device, dtype=dtype)  # scalar
        tau = t.expand(n)  # [n]

        x0 = torch.randn_like(gt_clean_norm)  # independent noise for BC
        x_t = (1.0 - t) * x0 + t * gt_clean_norm  # OT interpolation

        velocity_dict = flow_decoder.forward_components(
            anchor_hidden=anchor_hidden_valid,
            x_t_norm=x_t,
            tau=tau,
        )
        v_pred = velocity_dict["velocity"]
        v_target = (gt_clean_norm - x0).detach()  # conditional vector field

        return F.mse_loss(v_pred, v_target)

    # ──────────────────────────────────────────────────────────────────────
    # Public wrappers for DICE / external callers
    # ──────────────────────────────────────────────────────────────────────

    def generate_policy_trajectory(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> Tensor:
        """Sample Gaussian noise and run full-BPTT ODE to produce a policy trajectory.

        Gradient flows through all rollout steps, enabling BPTT-based actor updates.

        Args:
            flow_decoder: flow decoder module.
            flow_ode: ODE/flow module.
            anchor_hidden_valid: [n, hidden_dim].

        Returns:
            Tensor: policy trajectory with full gradient graph. shape [n, 20, 4].
        """
        n = anchor_hidden_valid.shape[0]
        noise = (
            torch.randn(n, 20, 4, device=anchor_hidden_valid.device, dtype=anchor_hidden_valid.dtype)
            * self.rollout_noise_scale
        )
        return self._rollout_ode_full_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
            noise=noise,
        )

    def bc_flow_loss(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        gt_clean_norm: Tensor,
    ) -> Tensor:
        """Public wrapper: flow-matching BC loss against GT.

        Args:
            flow_decoder: flow decoder module.
            flow_ode: ODE/flow module.
            anchor_hidden_valid: [n, hidden_dim].
            gt_clean_norm: GT trajectory. [n, 20, 4].

        Returns:
            Scalar BC loss (gradient-enabled for actor update).
        """
        return self._bc_loss_flowmatching(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
            gt_clean_norm=gt_clean_norm,
        )

    def forward_feasibility_with_bc(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        gt_clean_norm: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> "TerminalCostFinalStepResult":
        """Feasibility loss + BC (Flow Matching) 정규화를 결합한 open-loop fine-tuning입니다.

        전체 rollout_steps에 걸쳐 gradient가 흐르도록 BPTT를 사용하며,
        추가 head 없이 flow_decoder 파라미터에 직접 gradient를 전달합니다.

        loss = feasibility_cost + flow_reg_lambda * bc_loss

        Args:
            flow_decoder: flow decoder 모듈입니다.
            flow_ode: ODE/flow 모듈입니다.
            anchor_hidden_valid: shape ``[n_valid, hidden_dim]``.
            gt_clean_norm: GT normalized trajectory. shape ``[n_valid, 20, 4]``.
            agent_type: shape ``[n_valid]``.
            current_control: shape ``[n_valid, 3]`` 또는 None.
            current_control_valid: shape ``[n_valid]`` 또는 None.

        Returns:
            TerminalCostFinalStepResult: loss와 logging용 스칼라들입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        gt_clean_norm = gt_clean_norm.to(dtype=torch.float32, device=anchor_hidden_valid.device)
        if current_control is not None:
            current_control = current_control.to(
                dtype=torch.float32, device=anchor_hidden_valid.device
            )

        # 1. Pure noise (scale 적용)
        noise = (
            torch.randn(
                anchor_hidden_valid.shape[0], 20, 4,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            * self.rollout_noise_scale
        )

        # 2. 전체 step BPTT OT-ODE rollout
        final_state = self._rollout_ode_full_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
            noise=noise,
        )

        # 3. Feasibility loss
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state,
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        self._assert_finite_tensor("feasibility_bc/terminal_cost", terminal_cost)

        # 4. BC loss (Flow Matching) — flow_reg_lambda=0이면 스킵
        if self.flow_reg_lambda > 0.0:
            bc_loss = self._bc_loss_flowmatching(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                gt_clean_norm=gt_clean_norm,
            )
            self._assert_finite_tensor("feasibility_bc/bc_loss", bc_loss)
            flow_reg_loss = self.flow_reg_lambda * bc_loss
        else:
            bc_loss = torch.zeros((), device=terminal_cost.device, dtype=torch.float32)
            flow_reg_loss = bc_loss

        total_loss = terminal_cost + flow_reg_loss

        return TerminalCostFinalStepResult(
            loss=total_loss,
            terminal_cost=terminal_cost.detach(),
            projection_gap=metrics["projection_gap"].detach(),
            final_count=torch.tensor(
                anchor_hidden_valid.shape[0],
                device=total_loss.device,
                dtype=torch.float32,
            ),
            flow_reg_loss=bc_loss.detach(),
        )

    def forward_reward_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        reward_fn,
        gt_clean_norm: Optional[Tensor] = None,
        **reward_kwargs,
    ) -> "TerminalCostFinalStepResult":
        """Full-BPTT ODE rollout followed by a pluggable reward function.

        The reward is computed from the final ODE state and gradient flows
        back through all rollout steps into ``flow_decoder`` (BPTT).  The
        reward function itself should use soft operations (no hard clamps) so
        that gradients are non-zero everywhere.

        Args:
            flow_decoder: flow decoder module (must have ``requires_grad=True``).
            flow_ode: ODE / flow module.
            anchor_hidden_valid: encoded context. shape ``[n, hidden_dim]``.
            reward_fn: callable ``(y_hat, **reward_kwargs) -> (loss, metrics_dict)``.
                ``loss`` is a scalar to minimise; ``metrics_dict`` may contain
                ``"projection_gap"`` for logging.
            gt_clean_norm: GT normalised trajectory for optional BC regularisation.
                shape ``[n, 20, 4]`` or None.
            **reward_kwargs: forwarded verbatim to ``reward_fn``.

        Returns:
            TerminalCostFinalStepResult: loss and logging scalars.
        """
        if anchor_hidden_valid.numel() == 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        if gt_clean_norm is not None:
            gt_clean_norm = gt_clean_norm.to(
                dtype=torch.float32, device=anchor_hidden_valid.device
            )

        noise = (
            torch.randn(
                anchor_hidden_valid.shape[0], 20, 4,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            * self.rollout_noise_scale
        )

        # Full BPTT OT-ODE rollout — gradient flows through all steps
        final_state = self._rollout_ode_full_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
            noise=noise,
        )

        # Pluggable reward: soft operations, no hard clamp
        reward_loss, reward_metrics = reward_fn(final_state, **reward_kwargs)
        self._assert_finite_tensor("reward_grad/reward_loss", reward_loss)

        # Optional BC regularisation (keeps model close to pretrained distribution)
        if self.flow_reg_lambda > 0.0 and gt_clean_norm is not None:
            bc_loss = self._bc_loss_flowmatching(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                gt_clean_norm=gt_clean_norm,
            )
            self._assert_finite_tensor("reward_grad/bc_loss", bc_loss)
            flow_reg_loss = self.flow_reg_lambda * bc_loss
        else:
            bc_loss = torch.zeros((), device=reward_loss.device, dtype=torch.float32)
            flow_reg_loss = bc_loss

        total_loss = reward_loss + flow_reg_loss

        projection_gap = reward_metrics.get(
            "projection_gap",
            torch.zeros((), device=reward_loss.device, dtype=torch.float32),
        )

        return TerminalCostFinalStepResult(
            loss=total_loss,
            terminal_cost=reward_loss.detach(),
            projection_gap=projection_gap.detach(),
            final_count=torch.tensor(
                anchor_hidden_valid.shape[0],
                device=total_loss.device,
                dtype=torch.float32,
            ),
            flow_reg_loss=bc_loss.detach(),
        )

    def forward_l2(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        gt_clean_norm: Tensor,
    ) -> TerminalCostFinalStepResult:
        """GT trajectory와의 L2 거리를 loss로 사용합니다.

        OT-ODE rollout으로 생성한 final state와 GT normalized trajectory 사이의
        MSE를 최소화합니다. gradient는 마지막 step에서만 흐릅니다.

        Args:
            flow_decoder: flow decoder 모듈입니다.
            flow_ode: ODE/flow 모듈입니다.
            anchor_hidden_valid: GT history로 인코딩된 anchor 컨텍스트입니다.
                shape은 ``[n_valid, hidden_dim]`` 입니다.
            gt_clean_norm: normalized GT future trajectory입니다.
                shape은 ``[n_valid, 20, 4]`` 입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        gt_clean_norm = gt_clean_norm.to(dtype=torch.float32, device=anchor_hidden_valid.device)

        final_state, _ = self._rollout_ode_last_step_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
        )
        self._assert_finite_tensor("l2/final_state", final_state)

        loss = F.mse_loss(final_state, gt_clean_norm)
        return TerminalCostFinalStepResult(
            loss=loss,
            terminal_cost=loss.detach(),
            projection_gap=torch.zeros((), device=loss.device),
            final_count=torch.tensor(
                anchor_hidden_valid.shape[0], device=loss.device, dtype=torch.float32
            ),
        )

    @staticmethod
    def _assert_finite_tensor(name: str, value: Tensor) -> None:
        """NaN/Inf가 있으면 즉시 실패시킵니다."""
        if value.numel() == 0:
            return
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return
        bad_values = value.detach()[~finite_mask].flatten()[:8].cpu().tolist()
        raise RuntimeError(f"{name} contains non-finite values: {bad_values}")

    def _zero_loss_with_trainable_dependency(
        self,
        flow_decoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        빈 anchor에서도 graph 연결을 유지한 0-loss를 만듭니다.
        """
        # DDP에서 "unused parameter"를 피하려면 trainable 파라미터 전부가
        # 그래프에 연결되어야 합니다. 첫 파라미터 하나만 연결하면 나머지가
        # unused로 판정될 수 있습니다.
        zero = torch.zeros((), device=device, dtype=dtype)
        has_trainable = False
        for p in flow_decoder.parameters():
            if p.requires_grad:
                zero = zero + p.sum() * 0.0
                has_trainable = True
        if has_trainable:
            return zero
        return torch.zeros((), device=device, dtype=dtype)

    def forward_closed_loop(
        self,
        *,
        agent_decoder: nn.Module,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_seed: Optional[int] = None,
    ) -> TerminalCostFinalStepResult:
        """
        closed-loop receding horizon에서 각 step의 2초 terminal penalty를 누적합니다.

        중요한 점:
        - context/commit/retokenize는 autograd 그래프를 만들 필요가 없으므로 no_grad로 처리합니다.
        - diffusion 내부에서 memoryless SDE rollback은 마지막 step에서만 graph를 유지합니다.
        - 다음 horizon update는 생성된 상태를 detach해서 그래프 폭증을 방지합니다.
        """
        # rollout_cache/노이즈 테이프는 그래프가 필요 없습니다.
        with torch.no_grad():
            rollout_cache = agent_decoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        state = agent_decoder._clone_rollout_cache(rollout_cache)
        n_agent = int(state["n_agent"])
        n_step_future_10hz = int(state["n_step_future_10hz"])
        n_step_future_2hz = int(state["n_step_future_2hz"])
        max_context_steps = int(state["max_context_steps"])

        pos_window = state["pos_window"]
        head_window = state["head_window"]
        head_vector_window = state["head_vector_window"]
        valid_window = state["valid_window"]
        pred_idx_window = state["pred_idx_window"]
        feat_a = state["feat_a"]
        agent_token_emb = state["agent_token_emb"]
        agent_token_emb_veh = state["agent_token_emb_veh"]
        agent_token_emb_ped = state["agent_token_emb_ped"]
        agent_token_emb_cyc = state["agent_token_emb_cyc"]
        veh_mask = state["veh_mask"]
        ped_mask = state["ped_mask"]
        cyc_mask = state["cyc_mask"]
        categorical_embs = state["categorical_embs"]
        feat_a_now = state["feat_a_now"]
        feat_a_t_dict: Dict[int, Tensor] = state["feat_a_t_dict"]

        if n_agent == 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=pos_window.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        # finetune_config.rollout_noise_scale을 rollout_from_cache의 x_init_norm 테이프로 재사용합니다.
        sampling_noise = SimpleNamespace(noise_scale=self.rollout_noise_scale)
        with torch.no_grad():
            rollout_noise_tape = agent_decoder._build_rollout_noise_tape(
                num_agent=n_agent,
                tape_steps=n_step_future_10hz + 20 - agent_decoder.shift,
                device=feat_a_now.device,
                dtype=feat_a_now.dtype,
                sampling_noise=sampling_noise,
                sampling_seed=sampling_seed,
                scenario_sampling_seeds=None,
                agent_batch=tokenized_agent.get("batch", None),
            )

        total_loss_sum = torch.zeros(
            (), device=feat_a_now.device, dtype=torch.float32
        )
        total_count = torch.zeros((), device=feat_a_now.device, dtype=torch.float32)
        projection_gap_sum = torch.zeros((), device=feat_a_now.device, dtype=torch.float32)

        for t in range(n_step_future_2hz):
            n_step = pos_window.shape[1]

            # context computation
            if t == 0:
                current_hidden = feat_a_now
            else:
                with torch.no_grad():
                    inference_mask = valid_window.clone()
                    inference_mask[:, :-1] = False

                    edge_index_t, r_t = agent_decoder.build_temporal_edge(
                        pos_a=pos_window,
                        head_a=head_window,
                        head_vector_a=head_vector_window,
                        mask=valid_window,
                        inference_mask=inference_mask,
                    )
                    edge_index_t = torch.stack(
                        (
                            edge_index_t[0],
                            (edge_index_t[1] + 1) // n_step - 1,
                        ),
                        dim=0,
                    )

                    edge_index_pl2a, r_pl2a = agent_decoder.build_map2agent_edge(
                        pos_pl=map_feature["position"],
                        orient_pl=map_feature["orientation"],
                        pos_a=pos_window[:, -1:],
                        head_a=head_window[:, -1:],
                        head_vector_a=head_vector_window[:, -1:],
                        mask=inference_mask[:, -1:],
                        batch_s=tokenized_agent["batch"],
                        batch_pl=map_feature["batch"],
                    )

                    recent_motion = agent_decoder._build_recent_coarse_motion(
                        pos_window=pos_window,
                        valid_window=valid_window,
                    )
                    edge_index_a2a, r_a2a = agent_decoder.build_interaction_edge(
                        pos_a=pos_window[:, -1:],
                        head_a=head_window[:, -1:],
                        head_vector_a=head_vector_window[:, -1:],
                        batch_s=tokenized_agent["batch"],
                        mask=inference_mask[:, -1:],
                        motion_a=recent_motion.unsqueeze(1),
                    )

                    for i in range(agent_decoder.num_layers):
                        temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
                        current_hidden = agent_decoder.t_attn_layers[i](
                            (temporal_feat.flatten(0, 1), temporal_feat[:, -1]),
                            r_t,
                            edge_index_t,
                        )
                        current_hidden = agent_decoder.pt2a_attn_layers[i](
                            (map_feature["pt_token"], current_hidden),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                        current_hidden = agent_decoder.a2a_attn_layers[i](
                            current_hidden, r_a2a, edge_index_a2a
                        )
                        if i + 1 < agent_decoder.num_layers:
                            feat_a_t_dict[i + 1] = torch.cat(
                                [feat_a_t_dict[i + 1], current_hidden.unsqueeze(1)],
                                dim=1,
                            )

            active_mask = valid_window[:, -1]
            if not bool(active_mask.any()):
                # next state update should still advance context windows
                with torch.no_grad():
                    next_pos = pos_window[:, -1].clone()
                    next_head = head_window[:, -1].clone()
                    next_token_idx = pred_idx_window[:, -1].clone()

                    next_valid = active_mask.clone()
                    pos_window = torch.cat([pos_window, next_pos.unsqueeze(1)], dim=1)
                    head_window = torch.cat([head_window, next_head.unsqueeze(1)], dim=1)
                    valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
                    pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)

                    head_vector_next = torch.stack(
                        [next_head.cos(), next_head.sin()], dim=-1
                    )
                    head_vector_window = torch.cat(
                        [head_vector_window, head_vector_next.unsqueeze(1)], dim=1
                    )

                    agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
                    agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
                    agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
                    agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
                    agent_token_emb = torch.cat(
                        [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
                    )

                    motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
                    x_a = torch.stack(
                        [
                            torch.norm(motion_vector_a, p=2, dim=-1),
                        angle_between_2d_vectors(
                            ctr_vector=head_vector_window[:, -1],
                            nbr_vector=motion_vector_a,
                        ),
                        ],
                        dim=-1,
                    )
                    x_a = agent_decoder.x_a_emb(
                        continuous_inputs=x_a, categorical_embs=categorical_embs
                    )
                    feat_a_next = agent_decoder.fusion_emb(
                        torch.cat([agent_token_emb_next, x_a], dim=-1).unsqueeze(1)
                    )
                    feat_a = torch.cat([feat_a, feat_a_next], dim=1)

                    if pos_window.shape[1] > max_context_steps:
                        pos_window = pos_window[:, -max_context_steps:]
                        head_window = head_window[:, -max_context_steps:]
                        head_vector_window = head_vector_window[:, -max_context_steps:]
                        valid_window = valid_window[:, -max_context_steps:]
                        pred_idx_window = pred_idx_window[:, -max_context_steps:]
                        agent_token_emb = agent_token_emb[:, -max_context_steps:]
                        feat_a = feat_a[:, -max_context_steps:]
                        for key in feat_a_t_dict:
                            feat_a_t_dict[key] = feat_a_t_dict[key][
                                :, -max_context_steps:
                            ]
                continue

            # diffusion + terminal cost (여기만 gradient 유지)
            active_hidden = current_hidden[active_mask]
            noise_start = t * agent_decoder.shift
            x_init_norm = rollout_noise_tape[
                active_mask, noise_start : noise_start + 20
            ].contiguous()

            with torch.autocast(
                device_type=active_hidden.device.type if active_hidden.device.type else "cpu",
                enabled=False,
            ):
                final_state = self._rollout_memoryless_sde_last_step_grad(
                    flow_decoder=flow_decoder,
                    flow_ode=flow_ode,
                    anchor_hidden_valid=active_hidden,
                    x_init_norm=x_init_norm,
                )[0]

                self._assert_finite_tensor(f"closed_loop/t{t}/final_state", final_state)
                # closed-loop에서는 이전 continuity control을 쉽게 만들기 어려우므로 None 처리합니다.
                agent_type = tokenized_agent["type"][active_mask]
                terminal_cost, metrics = self.projector.compute_terminal_cost(
                    pred_clean_norm=final_state,
                    agent_type=agent_type,
                    current_control=None,
                    current_control_valid=None,
                )

            active_count = active_mask.sum().to(dtype=torch.float32, device=terminal_cost.device)
            total_loss_sum = total_loss_sum + terminal_cost.to(dtype=torch.float32) * active_count
            total_count = total_count + active_count
            projection_gap_sum = projection_gap_sum + metrics["projection_gap"].to(dtype=torch.float32) * active_count

            # commit + retokenize는 다음 horizon update만 위한 값이므로 detach합니다.
            with torch.no_grad():
                commit_pos_act, commit_head_act, next_pos_act, next_head_act = agent_decoder.commit_bridge.commit(
                    y_hat_norm=final_state.detach(),
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                )
                next_token_idx_act = agent_decoder.commit_bridge.retokenize(
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                    commit_pos=commit_pos_act,
                    commit_head=commit_head_act,
                    agent_type=tokenized_agent["type"][active_mask],
                    token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                    token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                    token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                    token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                )

                next_pos = pos_window[:, -1].clone()
                next_head = head_window[:, -1].clone()
                next_token_idx = pred_idx_window[:, -1].clone()
                next_pos[active_mask] = next_pos_act
                next_head[active_mask] = next_head_act
                next_token_idx[active_mask] = next_token_idx_act

                next_valid = active_mask.clone()
                pos_window = torch.cat([pos_window, next_pos.unsqueeze(1)], dim=1)
                head_window = torch.cat([head_window, next_head.unsqueeze(1)], dim=1)
                valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
                pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)

                head_vector_next = torch.stack([next_head.cos(), next_head.sin()], dim=-1)
                head_vector_window = torch.cat(
                    [head_vector_window, head_vector_next.unsqueeze(1)], dim=1
                )

                agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
                agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
                agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
                agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
                agent_token_emb = torch.cat(
                    [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
                )

                motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
                x_a = torch.stack(
                    [
                        torch.norm(motion_vector_a, p=2, dim=-1),
                        angle_between_2d_vectors(
                            ctr_vector=head_vector_window[:, -1],
                            nbr_vector=motion_vector_a,
                        ),
                    ],
                    dim=-1,
                )

                x_a = agent_decoder.x_a_emb(
                    continuous_inputs=x_a, categorical_embs=categorical_embs
                )
                feat_a_next = agent_decoder.fusion_emb(
                    torch.cat([agent_token_emb_next, x_a], dim=-1).unsqueeze(1)
                )
                feat_a = torch.cat([feat_a, feat_a_next], dim=1)

                if pos_window.shape[1] > max_context_steps:
                    pos_window = pos_window[:, -max_context_steps:]
                    head_window = head_window[:, -max_context_steps:]
                    head_vector_window = head_vector_window[:, -max_context_steps:]
                    valid_window = valid_window[:, -max_context_steps:]
                    pred_idx_window = pred_idx_window[:, -max_context_steps:]
                    agent_token_emb = agent_token_emb[:, -max_context_steps:]
                    feat_a = feat_a[:, -max_context_steps:]
                    for key in feat_a_t_dict:
                        feat_a_t_dict[key] = feat_a_t_dict[key][
                            :, -max_context_steps:
                        ]

        if total_count.item() <= 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=total_loss_sum.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        avg_loss = total_loss_sum / total_count
        avg_projection_gap = projection_gap_sum / total_count.clamp_min(1.0)
        return TerminalCostFinalStepResult(
            loss=avg_loss,
            terminal_cost=avg_loss.detach(),
            projection_gap=avg_projection_gap.detach(),
            final_count=total_count.detach(),
        )

