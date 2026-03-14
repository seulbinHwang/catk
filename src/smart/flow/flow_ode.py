from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor


@dataclass
class FlowSample:
    """학습용 flow 샘플 묶음이다.

    Attributes:
        noised: [*, n_future_step, 4] 모양의 섞인 미래이다.
        target: [*, n_future_step, 4] 모양의 맞춰야 할 속도이다.
        tau: [*] 모양의 시간 값이다.
        noise: [*, n_future_step, 4] 모양의 시작 잡음이다.
    """

    noised: Tensor
    target: Tensor
    tau: Tensor
    noise: Tensor


class FlowODE:
    """학습용 섞기와 추론용 고정 ODE 적분을 담당한다.

    이 구현은 Flow-Planner가 쓰는 `sample(...)`, `generate(...)` 흐름을
    SMART에 맞게 단순하게 옮긴 것이다. 학습 때는 깨끗한 미래와 랜덤 잡음을
    직선으로 섞고, 추론 때는 잡음에서 시작해 작은 고정 step 수로 끝까지 간다.
    """

    def __init__(self, tau_eps: float = 1e-3) -> None:
        self.tau_eps = tau_eps

    @staticmethod
    def _expand_tau(tau: Tensor, target: Tensor) -> Tensor:
        """`tau`를 target과 같은 차원 수로 늘린다.

        Args:
            tau: [*] 모양의 시간 값이다.
            target: [*, n_future_step, 4] 모양의 기준 텐서이다.

        Returns:
            [*, 1, 1] 모양으로 늘어난 시간 값이다.
        """
        while tau.dim() < target.dim():
            tau = tau.unsqueeze(-1)
        return tau

    def sample(self, x_data: Tensor) -> FlowSample:
        """학습용 noised sample과 target flow를 만든다.

        Args:
            x_data: [*, n_future_step, 4] 모양의 깨끗한 미래이다.

        Returns:
            FlowSample: noised 미래, target velocity, 시간 값, 시작 잡음을 담는다.
        """
        tau = torch.rand(*x_data.shape[:-2], device=x_data.device, dtype=x_data.dtype)
        tau = tau * (1.0 - self.tau_eps) + self.tau_eps
        noise = torch.randn_like(x_data)
        tau_expand = self._expand_tau(tau, x_data)
        noised = (1.0 - tau_expand) * noise + tau_expand * x_data
        target = x_data - noise
        return FlowSample(noised=noised, target=target, tau=tau, noise=noise)

    def reconstruct_start(self, x_t: Tensor, velocity: Tensor, tau: Tensor) -> Tensor:
        """예측한 속도에서 깨끗한 미래를 복원한다.

        Args:
            x_t: [*, n_future_step, 4] 모양의 섞인 미래이다.
            velocity: [*, n_future_step, 4] 모양의 예측 속도이다.
            tau: [*] 모양의 시간 값이다.

        Returns:
            [*, n_future_step, 4] 모양의 복원된 깨끗한 미래이다.
        """
        tau_expand = self._expand_tau(tau, x_t)
        return x_t + (1.0 - tau_expand) * velocity

    @torch.no_grad()
    def generate(
        self,
        x_init: Tensor,
        model_fn: Callable[[Tensor, Tensor], Tensor],
        sample_steps: int,
        sample_temperature: float = 1.0,
        sample_method: str = "euler",
    ) -> Tensor:
        """랜덤 잡음에서 시작해 미래 샘플을 만든다.

        Args:
            x_init: [*, n_future_step, 4] 모양의 시작 잡음이다.
            model_fn: `(x_t, tau) -> velocity` 꼴의 함수이다.
            sample_steps: 고정 적분 step 수이다.
            sample_temperature: 시작 잡음 크기 조절 값이다.
            sample_method: `euler` 또는 `heun`을 받는다.

        Returns:
            [*, n_future_step, 4] 모양의 최종 샘플이다.
        """
        x_t = x_init * sample_temperature
        dt = 1.0 / float(sample_steps)
        batch_shape = x_init.shape[:-2]

        for step in range(sample_steps):
            tau_value = self.tau_eps + (1.0 - self.tau_eps) * (step / sample_steps)
            tau = torch.full(
                batch_shape,
                fill_value=tau_value,
                device=x_init.device,
                dtype=x_init.dtype,
            )
            velocity = model_fn(x_t, tau)

            if sample_method == "heun":
                x_euler = x_t + dt * velocity
                tau_next_value = self.tau_eps + (1.0 - self.tau_eps) * (
                    (step + 1) / sample_steps
                )
                tau_next = torch.full(
                    batch_shape,
                    fill_value=tau_next_value,
                    device=x_init.device,
                    dtype=x_init.dtype,
                )
                velocity_next = model_fn(x_euler, tau_next)
                x_t = x_t + 0.5 * dt * (velocity + velocity_next)
            elif sample_method == "euler":
                x_t = x_t + dt * velocity
            else:
                raise ValueError(f"Unsupported sample_method: {sample_method}")
        return x_t
