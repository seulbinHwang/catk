from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.utils import wrap_angle
import torch.nn.functional as F


@dataclass
class ValueTrainingResult:
    """Value Training 학습 한 번의 결과를 모아 둡니다.

    Attributes:
        loss: 실제로 역전파할 regression loss 입니다. shape은 ``[]`` 입니다.
    """

    loss: Tensor


class ValueRegressionLoss(nn.Module):
    """Value regression loss 입니다."""

    def __init__(
        self,
        rollout_steps: int = 4,
        rollout_noise_scale: float = 1.0,
        rollout_time_grid: str = "logspace",
    ) -> None:
        super().__init__()
        self.rollout_steps = int(rollout_steps)
        self.rollout_noise_scale = float(rollout_noise_scale)
        self.rollout_time_grid = str(rollout_time_grid)
        if self.rollout_time_grid not in {"uniform", "logspace"}:
            raise ValueError(
                "rollout_time_grid must be either 'uniform' or 'logspace'. "
                f"Got: {self.rollout_time_grid}"
            )

    @staticmethod
    def _zero_loss_with_trainable_dependency(reference: Tensor, module: nn.Module) -> Tensor:
        """빈 anchor batch에서도 trainable parameter graph를 유지하는 0 loss를 만듭니다."""
        zero = reference.sum() * 0.0
        for parameter in module.parameters():
            if parameter.requires_grad:
                zero = zero + parameter.sum() * 0.0
        return zero

    @staticmethod
    def _assert_finite_tensor(name: str, value: Tensor) -> None:
        """중간 텐서가 NaN/Inf면 바로 실패시킵니다."""
        if value.numel() == 0:
            return
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return
        bad_values = value.detach()[~finite_mask].flatten()[:8].cpu().tolist()
        raise RuntimeError(f"{name} contains non-finite values: {bad_values}")

    @classmethod
    def _assert_finite_tensor_list(cls, name: str, values: Sequence[Tensor]) -> None:
        """여러 rollout 텐서를 순서대로 검사합니다."""
        for idx, value in enumerate(values):
            cls._assert_finite_tensor(f"{name}[{idx}]", value)

    def _build_step_schedule(
        self,
        flow_ode: nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """학습 rollout에 쓸 시간축과 구간별 가중치를 만듭니다.

        Args:
            flow_ode: ODE helper 입니다. 시작 시각 ``eps`` 를 읽습니다.
            batch_size: 유효 anchor 개수입니다.
            device: 시간 텐서를 둘 장치입니다.
            dtype: 시간 텐서 자료형입니다.

        Returns:
            tuple[List[Tensor], List[Tensor], List[Tensor]]:
                1. 시간 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                   각 원소 shape은 ``[n_valid_anchor]`` 입니다.
                2. 구간 길이 목록입니다. 길이는 ``rollout_steps`` 이고,
                   각 원소 shape은 ``[]`` 입니다.
                3. step 손실 가중치 목록입니다. 길이는 ``rollout_steps`` 이고,
                   각 원소 shape은 ``[]`` 입니다.

        이 함수의 목적은 0 근처 구간을 더 촘촘하게 쪼개는 것입니다.
        ``logspace`` 모드에서는 시작 시각은 그대로 두고, 초반에는 짧은 구간을,
        뒤로 갈수록 긴 구간을 배치합니다. 이렇게 하면 ``1 / t`` 와 memoryless
        noise가 강한 초반 구간을 더 부드럽게 적분할 수 있습니다.
        """
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be positive, got {self.rollout_steps}")

        t0 = float(flow_ode.eps)
        if not 0.0 < t0 < 1.0:
            raise ValueError(f"flow_ode.eps must satisfy 0 < eps < 1, got {t0}")

        schedule_dtype = torch.float64
        if self.rollout_time_grid == "uniform":
            scalar_times = torch.linspace(
                t0,
                1.0,
                self.rollout_steps + 1,
                device=device,
                dtype=schedule_dtype,
            )
        else:
            normalized = torch.linspace(
                0.0,
                1.0,
                self.rollout_steps + 1,
                device=device,
                dtype=schedule_dtype,
            )
            scalar_times = torch.exp(math.log(t0) * (1.0 - normalized))
            scalar_times[0] = t0
            scalar_times[-1] = 1.0

        scalar_step_sizes = scalar_times[1:] - scalar_times[:-1]
        scalar_step_weights = scalar_step_sizes / scalar_step_sizes.sum().clamp_min(1e-12)

        times = [
            torch.full(
                (batch_size,),
                float(scalar_times[step_idx].item()),
                device=device,
                dtype=dtype,
            )
            for step_idx in range(self.rollout_steps + 1)
        ]
        step_sizes = [
            torch.tensor(
                float(scalar_step_sizes[step_idx].item()),
                device=device,
                dtype=dtype,
            )
            for step_idx in range(self.rollout_steps)
        ]
        step_weights = [
            torch.tensor(
                float(scalar_step_weights[step_idx].item()),
                device=device,
                dtype=dtype,
            )
            for step_idx in range(self.rollout_steps)
        ]
        return times, step_sizes, step_weights

    @torch.no_grad()
    def _rollout_memoryless_sde(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Memoryless Euler–Maruyama SDE로 학습용 rollout 을 만듭니다.

        Args:
            flow_decoder:
                velocity field decoder 입니다.
            flow_ode:
                OT path helper 입니다.
            anchor_hidden_valid:
                유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]``

        여기서 말하는 상태 (state)
            “정규화된 미래 궤적”

        Returns:
            tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
                1. 상태 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                   각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
                2. 시간 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                   각 원소 shape은 ``[n_valid_anchor]`` 입니다.
                3. 구간 길이 목록입니다. 길이는 ``rollout_steps`` 이고,
                   각 원소 shape은 ``[]`` 입니다.
                4. 구간 가중치 목록입니다. 길이는 ``rollout_steps`` 이고,
                   각 원소 shape은 ``[]`` 입니다.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        dtype = anchor_hidden_valid.dtype
        device = anchor_hidden_valid.device

        """ times : List[Tensor]
            ``t_0`` 부터 ``t_K`` 까지의 시간 텐서 목록
            각 원소 shape은 ``[n_valid_anchor]`` 입니다.
        step_sizes : List[Tensor]
            각 구간 길이 목록입니다.
            길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[]`` 입니다.
        step_weights : List[Tensor]
            구간 길이를 전체 길이로 나눈 loss 가중치입니다.
            길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[]`` 입니다.
        """
        times, step_sizes, step_weights = self._build_step_schedule(
            flow_ode=flow_ode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        current_state = torch.randn(
            batch_size,
            20,
            4,
            device=device,
            dtype=dtype,
        ) * self.rollout_noise_scale
        states: List[Tensor] = [current_state.detach()]

        for step_idx in range(self.rollout_steps):
            tau = times[step_idx] # shape : ``[n_valid_anchor]
            """ velocity_dict
            base_velocity : "기존 모델이 원래 가고 싶어하는 방향"
            residual_velocity : "fine-tuning이 추가하는 보정 "
            velocity : " 둘을 합친 최종 이동 방향 "
            
            3개 다 전부 shape : [n_valid_anchor, 20, 4]
            """

            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=current_state,
                tau=tau,
            )
            # drift : [batch, 20, 4]
            drift = flow_ode.drift_from_velocity(
                x_t=current_state, # [n_valid_anchor, 20, 4]
                velocity=velocity_dict["velocity"], # [n_valid_anchor, 20, 4]
                tau=tau, # [n_valid_anchor]
            )
            noise = torch.randn_like(current_state)
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)
            step_size = step_sizes[step_idx]
            current_state = current_state + step_size * drift + torch.sqrt(step_size) * sigma * noise
            states.append(current_state.detach())

        return states, times, step_sizes, step_weights

    def _compute_terminal_gradient(
        self,
        final_state: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor],
        current_control_valid: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """마지막 feasible cost 와 그 gradient 를 계산합니다.

        Args:
            final_state: rollout 마지막 상태입니다. shape은 ``[n_valid_anchor, 20, 4]``
            agent_type: anchor별 객체 종류 번호입니다. shape은 ``[n_valid_anchor]``
            current_control: anchor 직전 0.1초 control 입니다. shape은 ``[n_valid_anchor, 3]``
            current_control_valid: current control 유효 여부입니다. shape은 ``[n_valid_anchor]``

        Returns:
            terminal_cost : Tensor  : shape : [ ]
                batch 평균 terminal cost (마지막 궤적과 projector 간의 gap)
            terminal_grad : Tensor : shape : [n_valid_anchor, 20, 4]
            metrics : Dict[str, Tensor]
                logging용 스칼라 사전입니다.
                "terminal_cost" : 값 1개
                "projection_gap" : [n_valid_anchor, 20, 3]
                    : terminal_cost 를 평균하기 전 값
                "projection_gap_vx_b_mps" : body x 속도 gap 평균 절대값
                "projection_gap_vy_b_mps" : body y 속도 gap 평균 절대값
                "projection_gap_yaw_rate_degps" : yaw-rate gap 평균 절대값
        """
        final_state_for_grad = final_state.detach().requires_grad_(True)
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state_for_grad, # [n_valid_anchor, 20, 4]
            agent_type=agent_type, # [n_valid_anchor]
            current_control=current_control, # [n_valid_anchor, 3]
            current_control_valid=current_control_valid, # [n_valid_anchor]
        )
        # terminal_grad : shape : [n_valid_anchor, 20, 4]
        terminal_grad = torch.autograd.grad(terminal_cost, final_state_for_grad)[0].detach()
        return terminal_cost, terminal_grad, metrics

    def _build_base_drift(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        x_state: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Base velocity head 만 사용한 drift 를 계산합니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            x_state: 특정 시간의 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            tau: 그 상태의 시간값입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Tensor: base drift 입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        velocity_dict = flow_decoder.forward_components(
            anchor_hidden=anchor_hidden_valid,
            x_t_norm=x_state,
            tau=tau,
        )
        return flow_ode.drift_from_velocity(
            x_t=x_state,
            velocity=velocity_dict["base_velocity"],
            tau=tau,
        )

    def _build_lean_adjoints(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],
        times: Sequence[Tensor],
        terminal_grad: Tensor,
        step_sizes: Sequence[Tensor],
    ) -> List[Tensor]:
        """Base drift 로 lean adjoint 를 뒤로 풉니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            states: rollout 상태 목록입니다. 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            times: 상태별 시간 목록입니다. 각 원소 shape은 ``[n_valid_anchor]`` 입니다.
            terminal_grad: 마지막 상태에 대한 terminal gradient 입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            step_sizes: rollout 구간 길이 목록입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[]`` 입니다.

        Returns:
            List[Tensor]: 각 rollout step의 lean adjoint 입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """

        adjoints: List[Tensor] = [terminal_grad]
        """
        if self.rollout_steps = 16, step_idx : 15, 14, ..., 0
        """
        for step_idx in range(self.rollout_steps - 1, -1, -1):
            # next_state: shape ( n_valid_anchor, 20, 4 )
            next_state = states[step_idx + 1].detach().requires_grad_(True)
            # tau_next: shape ( n_valid_anchor, )
            tau_next = times[step_idx + 1]
            with torch.enable_grad():
                # base_drift: shape ( n_valid_anchor, 20, 4 )
                base_drift = self._build_base_drift(
                    flow_decoder=flow_decoder,
                    flow_ode=flow_ode,
                    anchor_hidden_valid=anchor_hidden_valid,
                    x_state=next_state,
                    tau=tau_next,
                )
                j_t_a = torch.autograd.grad(
                    outputs=base_drift, # ( n_valid_anchor, 20, 4 )
                    inputs=next_state, # ( n_valid_anchor, 20, 4 )
                    grad_outputs=adjoints[-1], # ( n_valid_anchor, 20, 4 )
                    retain_graph=False,
                    create_graph=False,
                )[0]
            step_size = step_sizes[step_idx].to(device=j_t_a.device, dtype=j_t_a.dtype)
            adjoints.append((adjoints[-1] + step_size * j_t_a).detach())

        adjoints.reverse()
        return adjoints[:-1]

    def _build_regression_loss(
        self,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        states: Sequence[Tensor],  # List[Tensor] len: rollout_steps + 1 # shape : [, n_valid_anchor, 20, 4]
        times: Sequence[Tensor],
        lean_adjoints: Sequence[Tensor],
        step_weights: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Residual velocity 를 lean adjoint target 에 맞춥니다.

        Args:
            flow_decoder: velocity field decoder 입니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid: 유효 anchor 문맥입니다. shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            states: rollout 상태 목록입니다. 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            times: 상태별 시간 목록입니다. 각 원소 shape은 ``[n_valid_anchor]`` 입니다.
            lean_adjoints: List[Tensor]: 각 rollout step의 lean adjoint 입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            step_weights: 각 rollout step의 시간 길이 비율입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:

            1. 평균 regression loss
                shape : ( )
            2. 평균 residual norm
                    shape : ( )
        """
        step_losses: List[Tensor] = []
        residual_norms: List[Tensor] = []

        for step_idx in range(self.rollout_steps):
            x_state = states[step_idx].detach()
            tau = times[step_idx]
            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_state,
                tau=tau,
            )
            residual_velocity = velocity_dict["residual_velocity"] # ( n_valid_anchor, 20, 4 )
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)
            step_weight = step_weights[step_idx].to(device=residual_velocity.device, dtype=residual_velocity.dtype)
            residual_norms.append(step_weight * residual_velocity.pow(2).mean())

            # regression_target: ( n_valid_anchor, 20, 4 )
            regression_target = (2.0 / sigma) * residual_velocity + sigma * lean_adjoints[step_idx]
            # 비균일 시간축에서는 각 step loss를 같은 비율로 더하면 안 됩니다.
            # 연속시간 적분을 더 가깝게 따르도록, 구간 길이 비율만큼 가중 평균합니다.
            step_loss = regression_target.flatten(1).pow(2).mean(dim=1).mean()
            step_losses.append(step_weight * step_loss)

        if len(step_losses) == 0:
            zero = self._zero_loss_with_trainable_dependency(
                reference=anchor_hidden_valid,
                module=flow_decoder,
            )
            return zero, zero

        return torch.stack(step_losses).sum(), torch.stack(residual_norms).sum()

    def forward(
        self,
        flow_decoder: nn.Module, #원본 Flow_decoder와 동일한 모델의 Deepcopy 버전으로, Value 예측을 위해 사용됩니다. 
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        agent_type: Tensor, # [n_valid_anchor]
        current_control: Optional[Tensor], # [n_valid_anchor, 3]
        current_control_valid: Optional[Tensor], # [n_valid_anchor]
    ) -> ValueTrainingResult:
        """Value regression loss 를 계산합니다.

        Args:
            flow_decoder: 원본 Flow_decoder와 동일한 모델의 Deepcopy 버전으로, Value 예측을 위해 사용됩니다.
            flow_ode: OT path helper 입니다.
            anchor_hidden_valid:
                유효 anchor 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            agent_type:
                anchor별 객체 종류 번호입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            current_control:
                “anchor 직전 0.1초 동안의 현재 운동 상태를 body frame으로 표현한 값”
                정규화된 값도 아니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid:
                current control 유효 여부입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            ValueTrainingResult:
                loss: 실제로 역전파할 regression loss 입니다. shape은 ``[]`` 입니다.
        """

        if anchor_hidden_valid.numel() == 0:
            zero = self._zero_loss_with_trainable_dependency(
                reference=anchor_hidden_valid,
                module=flow_decoder,
            )
            empty_sample = anchor_hidden_valid.new_zeros((0, 20, 4))
            return ValueTrainingResult(
                loss=zero,
            )

        device_type = anchor_hidden_valid.device.type if anchor_hidden_valid.device.type else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
            if current_control is not None:
                current_control = current_control.to(
                    device=anchor_hidden_valid.device,
                    dtype=torch.float32,
                )
            """
            states : List[Tensor]
                - 상태 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            times : List[Tensor]
                - 시간 목록입니다. 길이는 ``rollout_steps + 1`` 이고,
                  각 원소 shape은 ``[n_valid_anchor]`` 입니다.
            """
            states, times, step_sizes, step_weights = self._rollout_memoryless_sde(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
            )
            self._assert_finite_tensor_list("am/states", states)
            """
            terminal_cost : Tensor  : shape : [ ]
                batch 평균 terminal cost
            terminal_grad : Tensor : shape : [n_valid_anchor, 20, 4]
            metrics : Dict[str, Tensor]
                logging용 스칼라 사전입니다.
                "terminal_cost" : 값 1개
                "projection_gap" : [n_valid_anchor, 20, 3]
                "projection_gap_vx_b_mps" : body x 속도 gap 평균 절대값
                "projection_gap_vy_b_mps" : body y 속도 gap 평균 절대값
                "projection_gap_yaw_rate_degps" : yaw-rate gap 평균 절대값
            """
            terminal_cost, terminal_grad, metrics = self._compute_terminal_gradient(
                final_state=states[-1], # shape : [n_valid_anchor, 20, 4]
                agent_type=agent_type, # shape : [n_valid_anchor]
                current_control=current_control, # shape : [n_valid_anchor, 3]
                current_control_valid=current_control_valid, # shape : [n_valid_anchor]
            )
            self._assert_finite_tensor("am/terminal_cost", terminal_cost)
            self._assert_finite_tensor("am/terminal_grad", terminal_grad)
            self._assert_finite_tensor("am/projection_gap", metrics["projection_gap"])
            self._assert_finite_tensor("am/projection_gap_vx_b_mps", metrics["projection_gap_vx_b_mps"])
            self._assert_finite_tensor("am/projection_gap_vy_b_mps", metrics["projection_gap_vy_b_mps"])
            self._assert_finite_tensor(
                "am/projection_gap_yaw_rate_degps",
                metrics["projection_gap_yaw_rate_degps"],
            )
            """ lean_adjoints
            List[Tensor]: 각 rollout step의 lean adjoint 입니다.
                길이는 ``rollout_steps`` 이고, 각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            """
            lean_adjoints = self._build_lean_adjoints(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid, # shape : [n_valid_anchor, hidden_dim]
                states=states, # List[Tensor] len: rollout_steps + 1 # shape : [, n_valid_anchor, 20, 4]
                times=times, # List[Tensor] len: rollout_steps + 1 # shape : [, n_valid_anchor]
                terminal_grad=terminal_grad, # [n_valid_anchor, 20, 4]
                step_sizes=step_sizes,
            )
            self._assert_finite_tensor_list("am/lean_adjoints", lean_adjoints)
            """
            1. 평균 regression loss
                shape : ( ) 
            2. 평균 residual norm
                    shape : ( )
            """
            regression_loss, residual_norm = self._build_regression_loss(
                flow_decoder=flow_decoder,
                flow_ode=flow_ode,
                anchor_hidden_valid=anchor_hidden_valid,
                states=states,  # List[Tensor] len: rollout_steps + 1 # shape : [, n_valid_anchor, 20, 4]
                times=times,
                lean_adjoints=lean_adjoints,
                step_weights=step_weights,
            )
            self._assert_finite_tensor("am/regression_loss", regression_loss)
            self._assert_finite_tensor("am/residual_norm", residual_norm)
            self._assert_finite_tensor("am/final_sample", states[-1])

            return AdjointMatchingResult(
                loss=regression_loss, # shape : ( )
                # terminal_cost: (마지막 궤적과 projector 간의 gap)
                terminal_cost=metrics["terminal_cost"], # shape : ( )
                # projection_gap: terminal_cost 를 평균하기 전 값
                projection_gap=metrics["projection_gap"], # [n_valid_anchor, 20, 3]
                projection_gap_vx_b_mps=metrics["projection_gap_vx_b_mps"],
                projection_gap_vy_b_mps=metrics["projection_gap_vy_b_mps"],
                projection_gap_yaw_rate_degps=metrics["projection_gap_yaw_rate_degps"],
                residual_norm=residual_norm.detach(), # shape : ( )
                final_sample=states[-1], # shape : [n_valid_anchor, 20, 4]
                diagnostic_metrics=self.projector.prefix_metric_keys("stochastic", metrics),
            )
