from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.utils import (
    cal_polygon_contour,
    safe_angle_from_2d_vector,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.modules.dynamic_limits import DEFAULT_LIMITS
from src.smart.modules.kinematic_control import (
    control_norm_to_pose_norm,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)


@dataclass(frozen=True)
class LQRCommitBridgeConfig:
    """Closed-loop LQR bridge 설정을 담습니다.

    Attributes:
        dt: 10Hz fine step 길이입니다.
        history_steps: 제어 참조를 만들 때 쓸 실제 fine history 길이입니다.
            shape 의미는 ``6`` 이면 최근 0.5초 + 현재까지의 6개 점입니다.
        horizon_steps: LQR가 직접 볼 미래 길이입니다.
        velocity_smooth_lambda: 속도 곡선 매끈함 가중치입니다.
        curvature_smooth_lambda: 곡률 곡선 매끈함 가중치입니다.
        curvature_init_reg: 저속에서 곡률 추정이 깨지지 않게 하는 작은 값입니다.
        stop_speed_mps: 저속 종방향 제어로 넘길 기준 속도입니다.
        stop_speed_kp: 저속 종방향 비례 제어 gain입니다.
        longitudinal_q: 1초 뒤 속도 오차 가중치입니다.
        longitudinal_r: 종방향 제어 크기 가중치입니다.
        lateral_q_lat: 횡방향 위치 오차 가중치입니다.
        lateral_q_head: 진행 방향 오차 가중치입니다.
        lateral_q_kappa: 현재 곡률 상태 가중치입니다.
        lateral_r: 곡률 변화율 제어 크기 가중치입니다.
        accel_tau_s: 가속 입력 1차 지연 시간입니다.
        curvature_tau_s: 곡률 입력 1차 지연 시간입니다.
        min_speed_for_curvature_clip_mps: 곡률 clip 계산에서 쓸 최소 속도입니다.
    """

    dt: float = 0.1
    history_steps: int = 6
    horizon_steps: int = 10
    velocity_smooth_lambda: float = 1.0e-4
    curvature_smooth_lambda: float = 1.0e-2
    curvature_init_reg: float = 1.0e-10
    stop_speed_mps: float = 0.2
    stop_speed_kp: float = 0.5
    longitudinal_q: float = 10.0
    longitudinal_r: float = 1.0
    lateral_q_lat: float = 1.0
    lateral_q_head: float = 10.0
    lateral_q_kappa: float = 0.1
    lateral_r: float = 1.0
    accel_tau_s: float = 0.2
    curvature_tau_s: float = 0.05
    min_speed_for_curvature_clip_mps: float = 0.5


# On sm_80 (A100) the flash / memory-efficient SDPA kernels inside
# `nn.MultiheadAttention` can exceed their grid-dim limit on large
# batches with many agents, causing
# `RuntimeError: CUDA error: invalid configuration argument` even when VRAM
# is not close to full. `ChunkStepRefiner` only attends over seq_len=5, so
# forcing the math SDPA kernel here is cheap and avoids that failure mode.
_SDPA_SAFE_BACKENDS = [SDPBackend.MATH]


@dataclass
class FlowSample:
    x_t: torch.Tensor
    target: torch.Tensor
    tau: torch.Tensor


class FlowODE:
    """Flow matching helper with backward-compatible linear/OT paths.

    Notes:
        - ``path_type="linear"`` reproduces the current repo behavior:
          ``x_t = (1 - t) * x_0 + t * x_1`` and ``v = x_1 - x_0``.
        - ``path_type="ot"`` uses the affine OT path used in FM papers:
          ``x_t = sigma_t * x_0 + t * x_1``,
          ``sigma_t = 1 - (1 - sigma_min) * t``,
          ``v = x_1 - (1 - sigma_min) * x_0``.

    With ``sigma_min = 0``, the OT path reduces exactly to the current linear path.
    """

    def __init__(
        self,
        eps: float = 1e-3,
        solver_steps: int = 4,
        solver_method: str = "midpoint",
        path_type: str = "ot",
        sigma_min: float = 1e-3,
    ) -> None:
        if path_type not in {"linear", "ot"}:
            raise ValueError(f"Unsupported path_type: {path_type}")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1")

        self.eps = eps
        self.solver_steps = solver_steps
        self.solver_method = solver_method
        self.path_type = path_type
        self.sigma_min = sigma_min

    def _beta(self) -> float:
        if self.path_type == "linear":
            return 1.0
        return 1.0 - self.sigma_min

    def _sigma_t(self, tau: torch.Tensor) -> torch.Tensor:
        beta = self._beta()
        return 1.0 - beta * tau

    def tau_interval_for_terminal_step(
        self,
        steps: int,
        terminal_step: int,
    ) -> tuple[float, float]:
        """terminal denoising step에 대응되는 tau 구간을 계산합니다.

        Args:
            steps: 전체 denoising grid 개수입니다. 예를 들어 shape과 무관한 scalar 값
                ``32`` 입니다.
            terminal_step: 실제로 실행할 마지막 step 번호입니다. ``1``이면 noise에
                가장 가까운 첫 step이고, ``steps``이면 clean에 가장 가까운 마지막
                step입니다.

        Returns:
            tuple[float, float]: tau 하한과 상한입니다. 각 값은 scalar입니다.
        """
        steps = int(steps)
        terminal_step = int(terminal_step)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if terminal_step < 1 or terminal_step > steps:
            raise ValueError(
                "terminal_step must be in [1, steps], "
                f"got terminal_step={terminal_step}, steps={steps}."
            )
        dt = (1.0 - float(self.eps)) / float(steps)
        tau_low = float(self.eps) + float(terminal_step - 1) * dt
        tau_high = float(self.eps) + float(terminal_step) * dt
        return tau_low, min(tau_high, 1.0)

    def _expand_tau_bound(
        self,
        bound: float | torch.Tensor | None,
        clean: torch.Tensor,
        default_value: float,
        name: str,
    ) -> torch.Tensor:
        """tau 하한이나 상한을 batch 길이에 맞춥니다.

        Args:
            bound: scalar 값 또는 path별 값입니다. tensor인 경우 shape은 ``[]`` 또는
                ``[n_path]`` 입니다.
            clean: clean path입니다. shape은 ``[n_path, n_step, 4]`` 입니다.
            default_value: ``bound`` 가 없을 때 쓸 scalar 값입니다.
            name: 오류 메시지에 사용할 이름입니다.

        Returns:
            torch.Tensor: path별 tau 경계입니다. shape은 ``[n_path]`` 입니다.
        """
        batch_size = int(clean.shape[0])
        if bound is None:
            return clean.new_full((batch_size,), float(default_value))
        if torch.is_tensor(bound):
            bound_tensor = bound.to(device=clean.device, dtype=clean.dtype)
            if bound_tensor.ndim == 0:
                return bound_tensor.expand(batch_size)
            if tuple(bound_tensor.shape) != (batch_size,):
                raise ValueError(
                    f"{name} must have shape [] or [n_path], "
                    f"got {tuple(bound_tensor.shape)} for n_path={batch_size}."
                )
            return bound_tensor
        return clean.new_full((batch_size,), float(bound))

    def _sample_tau(
        self,
        clean: torch.Tensor,
        tau_low: float | torch.Tensor | None = None,
        tau_high: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """clean path별 tau를 지정 구간에서 샘플링합니다.

        Args:
            clean: clean path입니다. shape은 ``[n_path, n_step, 4]`` 입니다.
            tau_low: tau 하한입니다. ``None`` 이면 ``eps`` 를 사용합니다. tensor인 경우
                shape은 ``[]`` 또는 ``[n_path]`` 입니다.
            tau_high: tau 상한입니다. ``None`` 이면 ``1`` 을 사용합니다. tensor인 경우
                shape은 ``[]`` 또는 ``[n_path]`` 입니다.

        Returns:
            torch.Tensor: path별 tau입니다. shape은 ``[n_path]`` 입니다.
        """
        low = self._expand_tau_bound(
            bound=tau_low,
            clean=clean,
            default_value=float(self.eps),
            name="tau_low",
        )
        high = self._expand_tau_bound(
            bound=tau_high,
            clean=clean,
            default_value=1.0,
            name="tau_high",
        )
        low = low.clamp(min=float(self.eps), max=1.0)
        high = high.clamp(min=float(self.eps), max=1.0)
        if torch.any(high <= low):
            raise ValueError("tau_high must be larger than tau_low for every path.")
        unit = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        return low + unit * (high - low)

    def sample(
        self,
        clean: torch.Tensor,
        target_type: str = "velocity",
        tau_low: float | torch.Tensor | None = None,
        tau_high: float | torch.Tensor | None = None,
    ) -> FlowSample:
        """clean path에 noise를 섞어 flow matching 학습 샘플을 만듭니다.

        Args:
            clean: clean path입니다. shape은 ``[n_path, n_step, 4]`` 입니다.
            target_type: 현재는 ``"velocity"`` 만 지원합니다.
            tau_low: tau 하한입니다. ``None`` 이면 전체 구간의 하한 ``eps`` 를 씁니다.
                tensor인 경우 shape은 ``[]`` 또는 ``[n_path]`` 입니다.
            tau_high: tau 상한입니다. ``None`` 이면 전체 구간의 상한 ``1`` 을 씁니다.
                tensor인 경우 shape은 ``[]`` 또는 ``[n_path]`` 입니다.

        Returns:
            FlowSample: noisy path, target velocity, tau를 담습니다. ``x_t`` 와
            ``target`` shape은 ``[n_path, n_step, 4]`` 이고, ``tau`` shape은
            ``[n_path]`` 입니다.
        """
        if target_type != "velocity":
            raise ValueError(f"Unsupported target_type: {target_type}")

        tau = self._sample_tau(clean=clean, tau_low=tau_low, tau_high=tau_high)
        noise = torch.randn_like(clean)
        view_tau = tau.view(-1, 1, 1)
        view_sigma = self._sigma_t(tau).view(-1, 1, 1)
        beta = self._beta()
        x_t = view_sigma * noise + view_tau * clean
        target = clean - beta * noise
        return FlowSample(x_t=x_t, target=target, tau=tau)

    def predict_clean_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        beta = self._beta()
        sigma_t = self._sigma_t(tau).view(-1, 1, 1)
        return beta * x_t + sigma_t * velocity

    def generate(
        self,
        x_init: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        steps: Optional[int] = None,
        method: Optional[str] = None,
        backprop_last_k: Optional[int] = None,
        terminal_step: Optional[int] = None,
        return_terminal_clean: bool = False,
    ) -> torch.Tensor:
        """ODE 샘플링으로 정규화 미래를 만듭니다.

        Args:
            x_init: 시작 잡음 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            model_fn: 현재 상태와 시간 ``tau`` 를 받아 속도를 돌려주는 함수입니다.
                입력 shape은 ``x_t=[n_valid_anchor, 20, 4]``, ``tau=[n_valid_anchor]`` 입니다.
            steps: 전체 denoising grid 개수입니다. ``None`` 이면 기본 solver step을 씁니다.
            method: 적분 방식입니다. ``None`` 이면 기본 solver 방식을 씁니다.
            backprop_last_k: 마지막 몇 step에만 gradient를 남길지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다. ``return_terminal_clean=True`` 일 때는
                terminal step 하나만 gradient를 남깁니다.
            terminal_step: 전체 grid 중 실제로 실행할 마지막 step 번호입니다. ``None`` 이면
                ``steps`` 를 끝까지 실행합니다.
            return_terminal_clean: ``True`` 면 마지막 noisy 상태를 그대로 반환하지 않고,
                terminal step에서 예측한 clean estimate를 반환합니다.

        Returns:
            torch.Tensor: 정규화 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        steps = self.solver_steps if steps is None else int(steps)
        method = self.solver_method if method is None else method
        max_step = steps if terminal_step is None else int(terminal_step)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if max_step < 1 or max_step > steps:
            raise ValueError(
                "terminal_step must be in [1, steps], "
                f"got terminal_step={max_step}, steps={steps}."
            )

        x_t = x_init
        t0 = self.eps
        dt = (1.0 - t0) / float(steps)

        if return_terminal_clean:
            for i in range(max_step - 1):
                t = t0 + i * dt
                tau = x_t.new_full((x_t.shape[0],), t)
                with torch.no_grad():
                    x_t = self._integrate_one_step(
                        x_t=x_t,
                        tau=tau,
                        dt=dt,
                        method=method,
                        model_fn=model_fn,
                    )
                x_t = x_t.detach()
            terminal_tau = x_t.new_full((x_t.shape[0],), t0 + (max_step - 1) * dt)
            velocity = model_fn(x_t, terminal_tau)
            return self.predict_clean_from_velocity(x_t, velocity, terminal_tau)

        if backprop_last_k is None or int(backprop_last_k) >= int(max_step):
            grad_start_step = 0
        else:
            grad_start_step = max(0, int(max_step) - max(0, int(backprop_last_k)))

        for i in range(max_step):
            t = t0 + i * dt
            tau = x_t.new_full((x_t.shape[0],), t)
            use_grad = i >= grad_start_step
            if use_grad:
                x_t = self._integrate_one_step(
                    x_t=x_t,
                    tau=tau,
                    dt=dt,
                    method=method,
                    model_fn=model_fn,
                )
            else:
                with torch.no_grad():
                    x_t = self._integrate_one_step(
                        x_t=x_t,
                        tau=tau,
                        dt=dt,
                        method=method,
                        model_fn=model_fn,
                    )
                x_t = x_t.detach()
        return x_t

    def _integrate_one_step(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        dt: float,
        method: str,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """한 ODE step만 적분합니다.

        Args:
            x_t: 현재 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            tau: 현재 시간입니다. shape은 ``[n_valid_anchor]`` 입니다.
            dt: 이번 step 길이입니다.
            method: ``midpoint`` 또는 ``euler`` 입니다.
            model_fn: 속도 예측 함수입니다.

        Returns:
            torch.Tensor: 다음 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        if method == "midpoint":
            v1 = model_fn(x_t, tau)
            x_mid = x_t + 0.5 * dt * v1
            tau_mid = tau + 0.5 * dt
            v2 = model_fn(x_mid, tau_mid)
            return x_t + dt * v2
        if method == "euler":
            v = model_fn(x_t, tau)
            return x_t + dt * v
        raise ValueError(f"Unsupported solver method: {method}")


class AnchorContextProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, anchor_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(anchor_hidden)


class NormalizedNoisyFutureEncoder(nn.Module):
    def __init__(
        self,
        flow_dim: int,
        num_chunks: int = 4,
        chunk_size: int = 5,
        flow_state_dim: int = 4,
    ) -> None:
        super().__init__()
        self.flow_dim = flow_dim
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.num_steps = num_chunks * chunk_size
        self.flow_state_dim = int(flow_state_dim)

        self.step_proj = nn.Linear(self.flow_state_dim, flow_dim)
        self.step_embed = nn.Embedding(self.num_steps, flow_dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(1, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim),
        )
        self.chunk_pool = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim),
        )

    def forward(
        self,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
        future_valid_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size = x_t_norm.shape[0]
        if x_t_norm.shape[1] != self.num_steps:
            raise ValueError(
                "NormalizedNoisyFutureEncoder expected "
                f"{self.num_steps} future steps, got {x_t_norm.shape[1]}."
            )
        if x_t_norm.shape[-1] != self.flow_state_dim:
            raise ValueError(
                "NormalizedNoisyFutureEncoder expected last dim "
                f"{self.flow_state_dim}, got {x_t_norm.shape[-1]}."
            )

        tau_emb = self.tau_mlp(tau.unsqueeze(-1))
        step_tokens = self.step_proj(x_t_norm)
        step_ids = torch.arange(self.num_steps, device=x_t_norm.device)
        step_tokens = step_tokens + self.step_embed(step_ids).unsqueeze(0)
        step_tokens = step_tokens + tau_emb.unsqueeze(1)

        if future_valid_mask is not None:
            if tuple(future_valid_mask.shape) != tuple(x_t_norm.shape[:2]):
                raise ValueError(
                    "future_valid_mask shape must match x_t_norm first two dimensions: "
                    f"expected={tuple(x_t_norm.shape[:2])}, actual={tuple(future_valid_mask.shape)}."
                )
            future_valid_mask = future_valid_mask.to(device=x_t_norm.device, dtype=torch.bool)

        step_tokens = step_tokens.view(
            batch_size,
            self.num_chunks,
            self.chunk_size,
            self.flow_dim,
        )
        if future_valid_mask is None:
            chunk_tokens = self.chunk_pool(step_tokens.mean(dim=2))
            return step_tokens, chunk_tokens, tau_emb, None, None

        step_valid_mask = future_valid_mask.view(batch_size, self.num_chunks, self.chunk_size)
        step_valid_float = step_valid_mask.to(dtype=step_tokens.dtype).unsqueeze(-1)
        step_tokens = step_tokens * step_valid_float

        valid_count = step_valid_float.sum(dim=2).clamp_min(1.0)
        chunk_tokens = self.chunk_pool(step_tokens.sum(dim=2) / valid_count)
        chunk_valid_mask = step_valid_mask.any(dim=2)
        chunk_tokens = chunk_tokens * chunk_valid_mask.to(dtype=chunk_tokens.dtype).unsqueeze(-1)
        return step_tokens, chunk_tokens, tau_emb, future_valid_mask, chunk_valid_mask


class HalfSecondChunkMixerBlock(nn.Module):
    """2초 전체 의도 토큰과 0.5초 묶음 토큰을 함께 갱신하는 블록입니다."""

    def __init__(self, flow_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(flow_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=flow_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(flow_dim * 2, flow_dim * 2),
            nn.SiLU(),
            nn.Linear(flow_dim * 2, flow_dim * 3),
        )

        self.mlp_norm = nn.LayerNorm(flow_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flow_dim, flow_dim * 2),
            nn.SiLU(),
            nn.Linear(flow_dim * 2, flow_dim),
        )

    def _modulate(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """장면 문맥과 잡음 제거 단계 정보를 토큰에 주입합니다.

        Args:
            x: 갱신할 토큰입니다. shape은 ``[n_path, 1 + n_chunk, flow_dim]`` 입니다.
            cond: 문맥에서 만든 조절값입니다. shape은 ``[n_path, 3 * flow_dim]`` 입니다.

        Returns:
            torch.Tensor: 문맥 정보가 섞인 토큰입니다. shape은 ``x`` 와 같습니다.
        """
        scale, bias, gate = cond.chunk(3, dim=-1)
        # scale/bias/gate: [n_path, flow_dim]
        return x + torch.sigmoid(gate).unsqueeze(1) * (
            x * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        )

    def _build_safe_key_padding_mask(
        self,
        chunk_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """전체 의도 토큰은 항상 보이도록 attention mask를 만듭니다.

        Args:
            chunk_valid_mask: 유효한 0.5초 묶음 표시입니다. shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: attention에서 숨길 위치입니다.
                shape은 ``[n_path, 1 + n_chunk]`` 입니다.
        """
        # plan_padding_mask: [n_path, 1]
        plan_padding_mask = torch.zeros(
            chunk_valid_mask.shape[0],
            1,
            dtype=torch.bool,
            device=chunk_valid_mask.device,
        )
        # chunk_padding_mask: [n_path, n_chunk]
        chunk_padding_mask = ~chunk_valid_mask.bool()
        return torch.cat([plan_padding_mask, chunk_padding_mask], dim=1)

    def forward(
        self,
        plan_token: torch.Tensor,
        chunk_tokens: torch.Tensor,
        context: torch.Tensor,
        tau_emb: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """2초 전체 의도와 0.5초 묶음 정보를 한 번에 갱신합니다.

        Args:
            plan_token: 2초 전체 움직임을 대표하는 토큰입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            chunk_tokens: 0.5초 묶음별 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            context: 과거, 지도, 주변 객체 정보를 담은 문맥입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            tau_emb: 현재 잡음 제거 단계 정보입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            chunk_valid_mask: loss에 포함할 수 있는 묶음 표시입니다.
                shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                갱신된 전체 의도 토큰과 묶음 토큰입니다.
                각 shape은 ``[n_path, flow_dim]``, ``[n_path, n_chunk, flow_dim]`` 입니다.
        """
        # seq_tokens: [n_path, 1 + n_chunk, flow_dim]
        seq_tokens = torch.cat([plan_token.unsqueeze(1), chunk_tokens], dim=1)
        attn_in = self.attn_norm(seq_tokens)
        with sdpa_kernel(_SDPA_SAFE_BACKENDS):
            if chunk_valid_mask is None:
                attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            else:
                attn_out, _ = self.attn(
                    attn_in,
                    attn_in,
                    attn_in,
                    key_padding_mask=self._build_safe_key_padding_mask(chunk_valid_mask),
                    need_weights=False,
                )
        seq_tokens = seq_tokens + attn_out

        # cond: [n_path, 3 * flow_dim]
        cond = self.cond_mlp(torch.cat([context, tau_emb], dim=-1))
        mlp_in = self._modulate(self.mlp_norm(seq_tokens), cond)
        seq_tokens = seq_tokens + self.mlp(mlp_in)

        # plan_token: [n_path, flow_dim], chunk_tokens: [n_path, n_chunk, flow_dim]
        plan_token = seq_tokens[:, 0]
        chunk_tokens = seq_tokens[:, 1:]
        if chunk_valid_mask is not None:
            chunk_tokens = chunk_tokens * chunk_valid_mask.to(dtype=chunk_tokens.dtype).unsqueeze(-1)
        return plan_token, chunk_tokens


class ChunkStepRefiner(nn.Module):
    """0.1초 단위 토큰을 자기 묶음과 양옆 묶음을 참고해 보정합니다."""

    def __init__(self, flow_dim: int, num_heads: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(flow_dim, flow_dim)
        self.pre_proj = nn.Linear(flow_dim, flow_dim)

        self.attn_norm = nn.LayerNorm(flow_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=flow_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.mlp_norm = nn.LayerNorm(flow_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flow_dim, flow_dim * 2),
            nn.SiLU(),
            nn.Linear(flow_dim * 2, flow_dim),
        )

    def _build_safe_step_key_padding_mask(
        self,
        step_valid_mask: torch.Tensor,
        batch_size: int,
        num_chunks: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """0.5초 묶음 내부 attention에서 무효 frame을 숨깁니다.

        Args:
            step_valid_mask: 유효한 0.1초 frame 표시입니다.
                shape은 ``[n_path, n_chunk * chunk_size]`` 입니다.
            batch_size: 경로 수입니다.
            num_chunks: 0.5초 묶음 수입니다.
            chunk_size: 묶음 하나 안의 frame 수입니다.

        Returns:
            torch.Tensor: attention에서 숨길 frame 표시입니다.
                shape은 ``[n_path * n_chunk, chunk_size]`` 입니다.
        """
        expected_shape = (batch_size, num_chunks * chunk_size)
        if tuple(step_valid_mask.shape) != expected_shape:
            raise ValueError(
                "step_valid_mask shape must match flattened future steps: "
                f"expected={expected_shape}, actual={tuple(step_valid_mask.shape)}."
            )
        # key_padding_mask: [n_path * n_chunk, chunk_size]
        key_padding_mask = ~step_valid_mask.view(batch_size, num_chunks, chunk_size).reshape(
            batch_size * num_chunks,
            chunk_size,
        ).bool()
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask = key_padding_mask & ~all_masked.unsqueeze(1)
        return key_padding_mask

    def _build_neighbor_chunk_summary(
        self,
        chunk_tokens: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """각 묶음이 이전/현재/다음 묶음 정보를 함께 보도록 요약합니다.

        Args:
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다.
                shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: 양옆 묶음까지 섞은 묶음 요약입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
        """
        batch_size, num_chunks, _ = chunk_tokens.shape
        # zero_token: [n_path, 1, flow_dim]
        zero_token = chunk_tokens.new_zeros(batch_size, 1, chunk_tokens.shape[-1])
        # prev_tokens/current_tokens/next_tokens: [n_path, n_chunk, flow_dim]
        prev_tokens = torch.cat([zero_token, chunk_tokens[:, :-1]], dim=1)
        next_tokens = torch.cat([chunk_tokens[:, 1:], zero_token], dim=1)

        if chunk_valid_mask is None:
            # current_valid: [n_path, n_chunk, 1]
            current_valid = chunk_tokens.new_ones(batch_size, num_chunks, 1)
        else:
            if tuple(chunk_valid_mask.shape) != (batch_size, num_chunks):
                raise ValueError(
                    "chunk_valid_mask shape must match chunk_tokens first two dimensions: "
                    f"expected={(batch_size, num_chunks)}, actual={tuple(chunk_valid_mask.shape)}."
                )
            current_valid = chunk_valid_mask.to(dtype=chunk_tokens.dtype).unsqueeze(-1)
        # zero_valid: [n_path, 1, 1]
        zero_valid = current_valid.new_zeros(batch_size, 1, 1)
        # prev_valid/next_valid: [n_path, n_chunk, 1]
        prev_valid = torch.cat([zero_valid, current_valid[:, :-1]], dim=1)
        next_valid = torch.cat([current_valid[:, 1:], zero_valid], dim=1)

        # denom: [n_path, n_chunk, 1]
        denom = (prev_valid + current_valid + next_valid).clamp_min(1.0)
        return (
            prev_tokens * prev_valid
            + chunk_tokens * current_valid
            + next_tokens * next_valid
        ) / denom

    def forward(
        self,
        step_tokens: torch.Tensor,
        chunk_tokens: torch.Tensor,
        plan_token: torch.Tensor,
        context: torch.Tensor,
        step_valid_mask: torch.Tensor | None = None,
        chunk_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """20개 frame 토큰을 전체 의도와 양옆 묶음 정보를 써서 보정합니다.

        Args:
            step_tokens: 0.1초 단위 토큰입니다.
                shape은 ``[n_path, n_chunk, chunk_size, flow_dim]`` 입니다.
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            plan_token: 2초 전체 움직임을 대표하는 토큰입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            context: 과거, 지도, 주변 객체 정보를 담은 문맥입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            step_valid_mask: 유효한 frame 표시입니다.
                shape은 ``[n_path, n_chunk * chunk_size]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다.
                shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: 보정된 frame 토큰입니다.
                shape은 ``[n_path, n_chunk * chunk_size, flow_dim]`` 입니다.
        """
        batch_size, num_chunks, chunk_size, dim = step_tokens.shape
        # neighbor_chunk_tokens: [n_path, n_chunk, flow_dim]
        neighbor_chunk_tokens = self._build_neighbor_chunk_summary(
            chunk_tokens=chunk_tokens,
            chunk_valid_mask=chunk_valid_mask,
        )
        # global_context: [n_path, 1, 1, flow_dim]
        global_context = self.context_proj(context + plan_token).view(batch_size, 1, 1, dim)

        # step_tokens: [n_path, n_chunk, chunk_size, flow_dim]
        step_tokens = step_tokens + neighbor_chunk_tokens.unsqueeze(2) + global_context
        step_tokens = self.pre_proj(step_tokens)

        # step_tokens_flat: [n_path * n_chunk, chunk_size, flow_dim]
        step_tokens_flat = step_tokens.view(batch_size * num_chunks, chunk_size, dim)
        attn_in = self.attn_norm(step_tokens_flat)
        with sdpa_kernel(_SDPA_SAFE_BACKENDS):
            if step_valid_mask is None:
                attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            else:
                attn_out, _ = self.attn(
                    attn_in,
                    attn_in,
                    attn_in,
                    key_padding_mask=self._build_safe_step_key_padding_mask(
                        step_valid_mask=step_valid_mask,
                        batch_size=batch_size,
                        num_chunks=num_chunks,
                        chunk_size=chunk_size,
                    ),
                    need_weights=False,
                )
        step_tokens_flat = step_tokens_flat + attn_out
        step_tokens_flat = step_tokens_flat + self.mlp(self.mlp_norm(step_tokens_flat))

        # refined_tokens: [n_path, n_chunk * chunk_size, flow_dim]
        refined_tokens = step_tokens_flat.view(batch_size, num_chunks * chunk_size, dim)
        if step_valid_mask is not None:
            refined_tokens = refined_tokens * step_valid_mask.to(dtype=refined_tokens.dtype).unsqueeze(-1)
        return refined_tokens


class FlowVelocityHead(nn.Module):
    def __init__(self, flow_dim: int, flow_state_dim: int = 4) -> None:
        super().__init__()
        self.flow_state_dim = int(flow_state_dim)
        self.net = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, self.flow_state_dim),
        )

    def forward(self, step_tokens: torch.Tensor) -> torch.Tensor:
        """frame 토큰에서 세부 보정 속도를 예측합니다.

        Args:
            step_tokens: 0.1초 단위 토큰입니다. shape은 ``[n_path, n_step, flow_dim]`` 입니다.

        Returns:
            torch.Tensor: frame별 세부 보정 속도입니다.
                shape은 ``[n_path, n_step, flow_state_dim]`` 입니다.
        """
        return self.net(step_tokens)


class PlanFirstResidualVelocityHead(nn.Module):
    """Frame velocity를 주 출력으로 쓰고 plan/chunk 경로는 작은 bias로만 더합니다."""

    def __init__(
        self,
        flow_dim: int,
        flow_state_dim: int = 4,
        plan_bias_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.flow_state_dim = int(flow_state_dim)
        self.plan_bias_scale = float(plan_bias_scale)
        self.frame_velocity_head = FlowVelocityHead(flow_dim=flow_dim, flow_state_dim=self.flow_state_dim)
        self.plan_chunk_bias_head = nn.Linear(flow_dim, self.flow_state_dim)
        nn.init.zeros_(self.plan_chunk_bias_head.weight)
        nn.init.zeros_(self.plan_chunk_bias_head.bias)

    def _fill_invalid_chunk_velocity(
        self,
        chunk_velocity: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Invalid chunk bias는 가장 가까운 valid chunk bias로 채웁니다."""
        if chunk_valid_mask is None:
            return chunk_velocity
        if tuple(chunk_valid_mask.shape) != tuple(chunk_velocity.shape[:2]):
            raise ValueError(
                "chunk_valid_mask shape must match chunk_velocity first two dimensions: "
                f"expected={tuple(chunk_velocity.shape[:2])}, actual={tuple(chunk_valid_mask.shape)}."
            )
        valid_mask = chunk_valid_mask.to(device=chunk_velocity.device, dtype=torch.bool)
        if bool(valid_mask.all()):
            return chunk_velocity

        batch_size, num_chunks = valid_mask.shape
        chunk_idx = torch.arange(num_chunks, device=chunk_velocity.device).view(1, num_chunks)
        previous_idx = torch.where(valid_mask, chunk_idx, chunk_idx.new_zeros(())).cummax(dim=1).values
        has_valid = valid_mask.any(dim=1, keepdim=True)

        reverse_valid = valid_mask.flip(dims=[1])
        reverse_idx = chunk_idx.expand(batch_size, -1)
        reverse_previous = torch.where(
            reverse_valid,
            reverse_idx,
            reverse_idx.new_zeros(()),
        ).cummax(dim=1).values
        next_idx = (num_chunks - 1 - reverse_previous).flip(dims=[1])

        has_previous = valid_mask.cumsum(dim=1) > 0
        fill_idx = torch.where(has_previous, previous_idx, next_idx)
        gather_idx = fill_idx.unsqueeze(-1).expand(-1, -1, chunk_velocity.shape[-1])
        filled = chunk_velocity.gather(dim=1, index=gather_idx)
        filled = torch.where(has_valid.unsqueeze(-1), filled, chunk_velocity.new_zeros(chunk_velocity.shape))
        return torch.where(valid_mask.unsqueeze(-1), chunk_velocity, filled)

    def _expand_chunk_bias(
        self,
        chunk_velocity: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """0.5초 chunk bias를 각 chunk 내부 0.1초 frame에 그대로 반복합니다."""
        num_steps = int(num_steps)
        num_chunks = int(chunk_velocity.shape[1])
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}.")
        if num_steps % num_chunks == 0:
            repeat = num_steps // num_chunks
            return chunk_velocity.repeat_interleave(repeat, dim=1)

        # Defensive fallback for unusual horizons. The normal 20-step/4-chunk path does not use this.
        step_idx = torch.arange(num_steps, device=chunk_velocity.device)
        chunk_idx = torch.div(step_idx * num_chunks, num_steps, rounding_mode="floor").clamp(max=num_chunks - 1)
        gather_idx = chunk_idx.view(1, num_steps, 1).expand(chunk_velocity.shape[0], -1, chunk_velocity.shape[-1])
        return chunk_velocity.gather(dim=1, index=gather_idx)

    def _interpolate_chunk_base(
        self,
        chunk_velocity: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Backward-compatible wrapper: chunk bias는 보간하지 않고 chunk 내부에서 반복합니다."""
        return self._expand_chunk_bias(chunk_velocity=chunk_velocity, num_steps=num_steps)

    def forward(
        self,
        step_tokens: torch.Tensor,
        chunk_tokens: torch.Tensor,
        tau: torch.Tensor,
        step_valid_mask: torch.Tensor | None = None,
        chunk_valid_mask: torch.Tensor | None = None,
        plan_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Frame별 full velocity에 작은 plan/chunk bias를 더합니다.

        Args:
            step_tokens: 0.1초 단위 토큰입니다.
                shape은 ``[n_path, n_step, flow_dim]`` 입니다.
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            tau: Flow Matching 시간입니다. shape은 ``[n_path]`` 입니다.
                FM-safe head에서는 frame velocity를 suppress하지 않기 위해 사용하지 않습니다.
            step_valid_mask: 유효한 frame 표시입니다. shape은 ``[n_path, n_step]`` 입니다.
            chunk_valid_mask: 유효한 0.5초 묶음 표시입니다. shape은 ``[n_path, n_chunk]`` 입니다.
            plan_token: 2초 전체 의도 토큰입니다. shape은 ``[n_path, flow_dim]`` 입니다.

        Returns:
            torch.Tensor: 최종 Flow Matching 속도입니다.
                shape은 ``[n_path, n_step, flow_state_dim]`` 입니다.
        """
        del tau
        frame_velocity = self.frame_velocity_head(step_tokens)

        bias_input = chunk_tokens
        if plan_token is not None:
            if tuple(plan_token.shape) != (chunk_tokens.shape[0], chunk_tokens.shape[-1]):
                raise ValueError(
                    "plan_token shape must be [n_path, flow_dim]: "
                    f"expected={(chunk_tokens.shape[0], chunk_tokens.shape[-1])}, actual={tuple(plan_token.shape)}."
                )
            bias_input = bias_input + plan_token.unsqueeze(1)

        chunk_bias = self.plan_chunk_bias_head(bias_input)
        chunk_bias = self._fill_invalid_chunk_velocity(
            chunk_velocity=chunk_bias,
            chunk_valid_mask=chunk_valid_mask,
        )
        frame_bias = self._expand_chunk_bias(
            chunk_velocity=chunk_bias,
            num_steps=step_tokens.shape[1],
        )
        velocity = frame_velocity + self.plan_bias_scale * frame_bias
        if step_valid_mask is not None:
            velocity = velocity * step_valid_mask.to(dtype=velocity.dtype).unsqueeze(-1)
        return velocity


class HierarchicalFlowDecoder(nn.Module):
    """2초 전체 의도를 먼저 만들고 frame별 세부 보정을 더하는 Flow decoder입니다."""

    def __init__(
        self,
        context_dim: int,
        flow_dim: int,
        num_future_steps: int = 20,
        num_chunk_heads: int = 4,
        num_chunk_layers: int = 2,
        chunk_size: int = 5,
        flow_state_dim: int = 4,
    ) -> None:
        super().__init__()
        if int(num_future_steps) <= 0:
            raise ValueError(f"num_future_steps must be positive, got {num_future_steps}.")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        if int(num_future_steps) % int(chunk_size) != 0:
            raise ValueError(
                "num_future_steps must be divisible by chunk_size, "
                f"got {num_future_steps} and {chunk_size}."
            )
        num_chunks = int(num_future_steps) // int(chunk_size)
        self.context_projector = AnchorContextProjector(context_dim, flow_dim)
        self.flow_state_dim = int(flow_state_dim)
        self.noisy_future_encoder = NormalizedNoisyFutureEncoder(
            flow_dim=flow_dim,
            num_chunks=num_chunks,
            chunk_size=int(chunk_size),
            flow_state_dim=self.flow_state_dim,
        )
        self.chunk_mixers = nn.ModuleList(
            [
                HalfSecondChunkMixerBlock(flow_dim=flow_dim, num_heads=num_chunk_heads)
                for _ in range(num_chunk_layers)
            ]
        )
        self.step_refiner = ChunkStepRefiner(
            flow_dim=flow_dim,
            num_heads=num_chunk_heads,
        )
        self.velocity_head = PlanFirstResidualVelocityHead(
            flow_dim=flow_dim,
            flow_state_dim=self.flow_state_dim,
        )

    def _masked_chunk_mean(
        self,
        chunk_tokens: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """유효한 0.5초 묶음만 사용해 2초 전체 요약을 만듭니다.

        Args:
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다.
                shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: 묶음 평균 요약입니다. shape은 ``[n_path, flow_dim]`` 입니다.
        """
        if chunk_valid_mask is None:
            return chunk_tokens.mean(dim=1)
        if tuple(chunk_valid_mask.shape) != tuple(chunk_tokens.shape[:2]):
            raise ValueError(
                "chunk_valid_mask shape must match chunk_tokens first two dimensions: "
                f"expected={tuple(chunk_tokens.shape[:2])}, actual={tuple(chunk_valid_mask.shape)}."
            )
        # valid_float: [n_path, n_chunk, 1]
        valid_float = chunk_valid_mask.to(dtype=chunk_tokens.dtype).unsqueeze(-1)
        # valid_count: [n_path, 1]
        valid_count = valid_float.sum(dim=1).clamp_min(1.0)
        return (chunk_tokens * valid_float).sum(dim=1) / valid_count

    def _build_global_plan_token(
        self,
        chunk_tokens: torch.Tensor,
        context: torch.Tensor,
        tau_emb: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """장면 문맥, 잡음 제거 단계, 묶음 요약을 합쳐 전체 의도 토큰을 만듭니다.

        Args:
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            context: 과거, 지도, 주변 객체 정보를 담은 문맥입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            tau_emb: 현재 잡음 제거 단계 정보입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다.
                shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: 2초 전체 움직임을 대표하는 토큰입니다.
                shape은 ``[n_path, flow_dim]`` 입니다.
        """
        # chunk_summary: [n_path, flow_dim]
        chunk_summary = self._masked_chunk_mean(
            chunk_tokens=chunk_tokens,
            chunk_valid_mask=chunk_valid_mask,
        )
        return (chunk_summary + context + tau_emb) * (3.0 ** -0.5)

    def _run_chunk_mixer(
        self,
        block: HalfSecondChunkMixerBlock,
        plan_token: torch.Tensor,
        chunk_tokens: torch.Tensor,
        context: torch.Tensor,
        tau_emb: torch.Tensor,
        chunk_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """전체 의도 토큰과 0.5초 묶음 토큰을 같이 갱신합니다.

        Args:
            block: 묶음 처리 블록입니다.
            plan_token: 2초 전체 의도 토큰입니다. shape은 ``[n_path, flow_dim]`` 입니다.
            chunk_tokens: 0.5초 묶음 토큰입니다. shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            context: 장면 문맥입니다. shape은 ``[n_path, flow_dim]`` 입니다.
            tau_emb: 잡음 제거 단계 정보입니다. shape은 ``[n_path, flow_dim]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다. shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 갱신된 전체 의도 토큰과 묶음 토큰입니다.
        """
        return block(
            plan_token=plan_token,
            chunk_tokens=chunk_tokens,
            context=context,
            tau_emb=tau_emb,
            chunk_valid_mask=chunk_valid_mask,
        )

    def _run_step_refiner(
        self,
        step_tokens: torch.Tensor,
        chunk_tokens: torch.Tensor,
        plan_token: torch.Tensor,
        context: torch.Tensor,
        step_valid_mask: torch.Tensor | None,
        chunk_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """frame 토큰을 전체 의도와 양옆 묶음 정보를 써서 보정합니다.

        Args:
            step_tokens: 0.1초 단위 토큰입니다.
                shape은 ``[n_path, n_chunk, chunk_size, flow_dim]`` 입니다.
            chunk_tokens: 0.5초 묶음 토큰입니다.
                shape은 ``[n_path, n_chunk, flow_dim]`` 입니다.
            plan_token: 2초 전체 의도 토큰입니다. shape은 ``[n_path, flow_dim]`` 입니다.
            context: 장면 문맥입니다. shape은 ``[n_path, flow_dim]`` 입니다.
            step_valid_mask: 유효한 frame 표시입니다. shape은 ``[n_path, n_step]`` 입니다.
            chunk_valid_mask: 유효한 묶음 표시입니다. shape은 ``[n_path, n_chunk]`` 입니다.

        Returns:
            torch.Tensor: 보정된 frame 토큰입니다. shape은 ``[n_path, n_step, flow_dim]`` 입니다.
        """
        return self.step_refiner(
            step_tokens=step_tokens,
            chunk_tokens=chunk_tokens,
            plan_token=plan_token,
            context=context,
            step_valid_mask=step_valid_mask,
            chunk_valid_mask=chunk_valid_mask,
        )

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
        future_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow Matching 속도를 예측합니다.

        Args:
            anchor_hidden: 현재 anchor의 장면 문맥입니다.
                shape은 ``[n_path, context_dim]`` 입니다.
            x_t_norm: 잡음이 섞인 2초 미래입니다.
                shape은 ``[n_path, num_future_steps, flow_state_dim]`` 입니다.
            tau: Flow Matching 시간입니다. shape은 ``[n_path]`` 입니다.
            future_valid_mask: loss에 사용할 수 있는 미래 frame 표시입니다.
                shape은 ``[n_path, num_future_steps]`` 입니다.

        Returns:
            torch.Tensor: 예측한 Flow Matching 속도입니다.
                shape은 ``[n_path, num_future_steps, flow_state_dim]`` 입니다.
        """
        # context: [n_path, flow_dim]
        context = self.context_projector(anchor_hidden)
        # step_tokens: [n_path, n_chunk, chunk_size, flow_dim]
        # chunk_tokens: [n_path, n_chunk, flow_dim]
        # tau_emb: [n_path, flow_dim]
        (
            step_tokens,
            chunk_tokens,
            tau_emb,
            step_valid_mask,
            chunk_valid_mask,
        ) = self.noisy_future_encoder(
            x_t_norm=x_t_norm,
            tau=tau,
            future_valid_mask=future_valid_mask,
        )

        # plan_token: [n_path, flow_dim]
        plan_token = self._build_global_plan_token(
            chunk_tokens=chunk_tokens,
            context=context,
            tau_emb=tau_emb,
            chunk_valid_mask=chunk_valid_mask,
        )
        for block in self.chunk_mixers:
            plan_token, chunk_tokens = self._run_chunk_mixer(
                block=block,
                plan_token=plan_token,
                chunk_tokens=chunk_tokens,
                context=context,
                tau_emb=tau_emb,
                chunk_valid_mask=chunk_valid_mask,
            )

        # step_tokens: [n_path, num_future_steps, flow_dim]
        step_tokens = self._run_step_refiner(
            step_tokens=step_tokens,
            chunk_tokens=chunk_tokens,
            plan_token=plan_token,
            context=context,
            step_valid_mask=step_valid_mask,
            chunk_valid_mask=chunk_valid_mask,
        )
        return self.velocity_head(
            step_tokens=step_tokens,
            chunk_tokens=chunk_tokens,
            tau=tau,
            step_valid_mask=step_valid_mask,
            chunk_valid_mask=chunk_valid_mask,
            plan_token=plan_token,
        )

class ContinuousCommitBridge:
    """Continuous FM 출력을 closed-loop 실행 상태로 바꾸는 다리입니다.

    이 클래스는 세 가지 일을 담당합니다.
    1) 6개 점 경로 기준 motion token 재매칭
    2) stop-motion 토큰이 나오면 0.5초 chunk를 완전히 정지로 고정
    3) vehicle / bicycle에 대해서만 curvature-domain LQR과 kinematic bicycle로
       다음 0.5초 5개 fine 상태를 실제 실행
    """

    def __init__(
        self,
        commit_steps: int = 5,
        pos_scale_m: float = 20.0,
        use_lqr: bool = False,
        use_stop_motion: bool = False,
        config: LQRCommitBridgeConfig | None = None,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        control_pos_scale_m: float = 1.0,
        control_vehicle_no_slip_point_ratio: float = 0.0,
        control_cyclist_no_slip_point_ratio: float = 0.0,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
    ) -> None:
        self.commit_steps = int(commit_steps)
        self.pos_scale_m = float(pos_scale_m)
        self.use_lqr = bool(use_lqr)
        # Stop-motion gating is disabled branch-wide. The argument remains only
        # for config/checkpoint compatibility.
        self.use_stop_motion = False
        self.use_kinematic_control_flow = bool(use_kinematic_control_flow)
        self.use_holonomic_model_only = bool(use_holonomic_model_only)
        self.control_pos_scale_m = float(control_pos_scale_m)
        (
            self.control_vehicle_no_slip_point_ratio,
            self.control_cyclist_no_slip_point_ratio,
        ) = validate_control_no_slip_ratio_config(
            vehicle_no_slip_point_ratio=control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=control_cyclist_no_slip_point_ratio,
        )
        self.control_vehicle_yaw_scale_rad = control_vehicle_yaw_scale_rad
        self.control_pedestrian_yaw_scale_rad = control_pedestrian_yaw_scale_rad
        self.control_cyclist_yaw_scale_rad = control_cyclist_yaw_scale_rad
        if self.use_kinematic_control_flow:
            (
                self.control_vehicle_yaw_scale_rad,
                self.control_pedestrian_yaw_scale_rad,
                self.control_cyclist_yaw_scale_rad,
            ) = validate_control_yaw_scale_config(
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            )
        self.config = config if config is not None else LQRCommitBridgeConfig()
        self._difference_gram_cache: dict[tuple[int, str, str], torch.Tensor] = {}

    @staticmethod
    def _select_token_chunk_local(
        next_token_idx: torch.Tensor,
        agent_type: torch.Tensor,
        token_bank_all_veh: torch.Tensor,
        token_bank_all_ped: torch.Tensor,
        token_bank_all_cyc: torch.Tensor,
    ) -> torch.Tensor:
        """선택한 token id에 대응하는 0.5초 local contour chunk를 꺼냅니다."""
        token_chunk_local = token_bank_all_veh.new_zeros((agent_type.shape[0], 6, 4, 2))
        token_banks = {
            "veh": token_bank_all_veh,
            "ped": token_bank_all_ped,
            "cyc": token_bank_all_cyc,
        }

        for token_key, mask in build_agent_type_masks(agent_type).items():
            if not mask.any():
                continue

            token_bank = token_banks[token_key]
            if token_bank.dim() != 4:
                raise ValueError(
                    "Token chunk restore expects full trajectory token banks with shape "
                    f"[n_token, 6, 4, 2], got {tuple(token_bank.shape)} for {token_key}."
                )
            token_chunk_local[mask] = token_bank[next_token_idx[mask]]

        return token_chunk_local

    def commit(
        self,
        y_hat_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        agent_type: torch.Tensor | None = None,
        agent_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_hat_norm = self._flow_output_to_pose_norm(
            y_hat_norm=y_hat_norm,
            agent_type=agent_type,
            agent_length=agent_length,
        )
        first_chunk = y_hat_norm[:, : self.commit_steps].clone()
        first_chunk[..., :2] = first_chunk[..., :2] * self.pos_scale_m

        delta_head = safe_angle_from_2d_vector(first_chunk[..., 2:4])

        commit_pos, _ = transform_to_global(
            pos_local=first_chunk[..., :2],
            head_local=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        commit_head = wrap_angle(current_head.unsqueeze(1) + delta_head)

        next_pos = commit_pos[:, -1]
        next_head = commit_head[:, -1]
        return commit_pos, commit_head, next_pos, next_head

    def _flow_output_to_pose_norm(
        self,
        y_hat_norm: torch.Tensor,
        agent_type: torch.Tensor | None = None,
        agent_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Closed-loop commit paths consume the common pose-space flow view."""
        if not self.use_kinematic_control_flow:
            return y_hat_norm
        if agent_type is None:
            raise ValueError("agent_type is required when use_kinematic_control_flow=True.")
        return control_norm_to_pose_norm(
            control_norm=y_hat_norm,
            agent_type=agent_type,
            agent_length=agent_length,
            pos_scale_m=self.control_pos_scale_m,
            vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            use_holonomic_model_only=self.use_holonomic_model_only,
            vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
        )

    def _build_full_future_from_flow(
        self,
        y_hat_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """정규화 2초 미래 전체를 전역 중심점과 방향으로 바꿉니다.

        Args:
            y_hat_norm: 정규화 2초 미래입니다. shape은 ``[n_agent, 20, 4]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - future_pos: 전역 2초 미래 중심점 ``[n_agent, 20, 2]``
                - future_head: 전역 2초 미래 방향 ``[n_agent, 20]``
        """
        future_local_xy = y_hat_norm[..., :2] * 20.0
        future_pos, _ = transform_to_global(
            pos_local=future_local_xy,
            head_local=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        future_head = wrap_angle(
            current_head.unsqueeze(1) + safe_angle_from_2d_vector(y_hat_norm[..., 2:4])
        )
        return future_pos, future_head


    def _build_local_commit_contour_chunk(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        commit_pos: torch.Tensor,
        commit_head: torch.Tensor,
        token_agent_shape: torch.Tensor,
    ) -> torch.Tensor:
        """현재 coarse 상태를 원점으로 한 6개 점 local 사각형 경로를 만듭니다.

        Args:
            current_pos: 현재 coarse 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse 방향입니다. shape은 ``[n_agent]`` 입니다.
            commit_pos: 이번 0.5초 구간의 10Hz 중심점 예측입니다.
                shape은 ``[n_agent, 5, 2]`` 입니다.
            commit_head: 이번 0.5초 구간의 10Hz 방향 예측입니다.
                shape은 ``[n_agent, 5]`` 입니다.
            token_agent_shape: 토큰 매칭에 쓸 고정 박스 크기입니다.
                shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            torch.Tensor:
                현재 상태를 포함한 local 사각형 경로입니다.
                shape은 ``[n_agent, 6, 4, 2]`` 입니다.
        """
        pos_seq = torch.cat([current_pos.unsqueeze(1), commit_pos], dim=1)
        head_seq = torch.cat([current_head.unsqueeze(1), commit_head], dim=1)
        contour_global = cal_polygon_contour(
            pos=pos_seq,
            head=head_seq,
            width_length=token_agent_shape.unsqueeze(1),
        )
        contour_local_flat, _ = transform_to_local(
            pos_global=contour_global.flatten(1, 2),
            head_global=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        return contour_local_flat.view(pos_seq.shape[0], pos_seq.shape[1], 4, 2)

    def retokenize(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        commit_pos: torch.Tensor,
        commit_head: torch.Tensor,
        agent_type: torch.Tensor,
        token_agent_shape: torch.Tensor,
        token_bank_all_veh: torch.Tensor,
        token_bank_all_ped: torch.Tensor,
        token_bank_all_cyc: torch.Tensor,
    ) -> torch.Tensor:
        """학습과 같은 6개 점 경로 기준으로 다음 coarse 토큰 번호를 다시 고릅니다.

        Args:
            current_pos: 현재 coarse 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse 방향입니다. shape은 ``[n_agent]`` 입니다.
            commit_pos: 이번 0.5초 구간의 10Hz 중심점 예측입니다.
                shape은 ``[n_agent, 5, 2]`` 입니다.
            commit_head: 이번 0.5초 구간의 10Hz 방향 예측입니다.
                shape은 ``[n_agent, 5]`` 입니다.
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            token_agent_shape: 토큰 매칭에 쓸 고정 박스 크기입니다.
                shape은 ``[n_agent, 2]`` 입니다.
            token_bank_all_veh: 차량 토큰 은행입니다.
                shape은 ``[n_token, 6, 4, 2]`` 입니다.
            token_bank_all_ped: 보행자 토큰 은행입니다.
                shape은 ``[n_token, 6, 4, 2]`` 입니다.
            token_bank_all_cyc: 자전거 토큰 은행입니다.
                shape은 ``[n_token, 6, 4, 2]`` 입니다.

        Returns:
            torch.Tensor:
                다음 coarse 상태에 붙일 토큰 번호입니다. shape은 ``[n_agent]`` 입니다.
        """
        contour_chunk_local = self._build_local_commit_contour_chunk(
            current_pos=current_pos,
            current_head=current_head,
            commit_pos=commit_pos,
            commit_head=commit_head,
            token_agent_shape=token_agent_shape,
        )
        return match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=contour_chunk_local,
            token_bank_all_veh=token_bank_all_veh,
            token_bank_all_ped=token_bank_all_ped,
            token_bank_all_cyc=token_bank_all_cyc,
            reduction="sum",
        )

    def restore_token_state(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        next_token_idx: torch.Tensor,
        agent_type: torch.Tensor,
        token_bank_all_veh: torch.Tensor,
        token_bank_all_ped: torch.Tensor,
        token_bank_all_cyc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """고른 coarse 토큰을 학습과 같은 방식으로 pose/head로 복원합니다."""
        next_pos = current_pos.clone()
        next_head = current_head.clone()
        token_banks = {
            "veh": token_bank_all_veh[:, -1],
            "ped": token_bank_all_ped[:, -1],
            "cyc": token_bank_all_cyc[:, -1],
        }

        for token_key, mask in build_agent_type_masks(agent_type).items():
            if not mask.any():
                continue

            token_contour_local = token_banks[token_key][next_token_idx[mask]]
            token_center_local = token_contour_local.mean(dim=1)
            token_center_global, _ = transform_to_global(
                pos_local=token_center_local.unsqueeze(1),
                head_local=None,
                pos_now=current_pos[mask],
                head_now=current_head[mask],
            )
            next_pos[mask] = token_center_global.squeeze(1)

            token_dxy_local = token_contour_local[:, 0] - token_contour_local[:, 3]
            token_head_local = torch.atan2(token_dxy_local[:, 1], token_dxy_local[:, 0])
            next_head[mask] = wrap_angle(current_head[mask] + token_head_local)

        return next_pos, next_head

    def restore_token_chunk(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        next_token_idx: torch.Tensor,
        agent_type: torch.Tensor,
        token_bank_all_veh: torch.Tensor,
        token_bank_all_ped: torch.Tensor,
        token_bank_all_cyc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """고른 coarse 토큰의 전체 0.5초 chunk를 전역 중심점과 방향으로 복원합니다."""
        token_chunk_local = self._select_token_chunk_local(
            next_token_idx=next_token_idx,
            agent_type=agent_type,
            token_bank_all_veh=token_bank_all_veh,
            token_bank_all_ped=token_bank_all_ped,
            token_bank_all_cyc=token_bank_all_cyc,
        )
        token_center_local = token_chunk_local.mean(dim=2)
        token_dxy_local = token_chunk_local[:, :, 0] - token_chunk_local[:, :, 3]
        token_head_local = torch.atan2(token_dxy_local[:, :, 1], token_dxy_local[:, :, 0])
        token_center_global, token_head_global = transform_to_global(
            pos_local=token_center_local,
            head_local=token_head_local,
            pos_now=current_pos,
            head_now=current_head,
        )
        token_head_global = wrap_angle(token_head_global)

        commit_pos = token_center_global[:, 1:]
        commit_head = token_head_global[:, 1:]
        next_pos = commit_pos[:, -1]
        next_head = commit_head[:, -1]
        return commit_pos, commit_head, next_pos, next_head
    def _build_stationary_token_contour(
        self,
        token_agent_shape: torch.Tensor,
    ) -> torch.Tensor:
        """각 차종의 고정 토큰 박스로 정지 6점 contour를 만듭니다.

        Args:
            token_agent_shape: 토큰 매칭에 쓸 고정 가로, 세로 크기입니다.
                shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            torch.Tensor: 정지 6점 local contour 입니다.
                shape은 ``[n_agent, 6, 4, 2]`` 입니다.
        """
        stationary_pos = token_agent_shape.new_zeros((token_agent_shape.shape[0], 6, 2))
        stationary_head = token_agent_shape.new_zeros((token_agent_shape.shape[0], 6))
        return cal_polygon_contour(
            pos=stationary_pos,
            head=stationary_head,
            width_length=token_agent_shape.unsqueeze(1),
        )

    def build_stop_motion_mask(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        commit_pos: torch.Tensor,
        commit_head: torch.Tensor,
        agent_type: torch.Tensor,
        token_agent_shape: torch.Tensor,
        token_bank_all_veh: torch.Tensor,
        token_bank_all_ped: torch.Tensor,
        token_bank_all_cyc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """raw FM 0.5초 chunk가 정지 토큰과 맞는지 판별합니다.

        Args:
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.
            commit_pos: raw FM가 낸 다음 0.5초 중심점입니다. shape은 ``[n_agent, 5, 2]`` 입니다.
            commit_head: raw FM가 낸 다음 0.5초 방향입니다. shape은 ``[n_agent, 5]`` 입니다.
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            token_agent_shape: 고정 토큰 박스 크기입니다. shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - raw_token_idx: raw FM chunk의 토큰 번호 ``[n_agent]``
                - stop_mask: 정지 토큰과 일치하는지 여부 ``[n_agent]``
        """
        raw_token_idx = self.retokenize(
            current_pos=current_pos,
            current_head=current_head,
            commit_pos=commit_pos,
            commit_head=commit_head,
            agent_type=agent_type,
            token_agent_shape=token_agent_shape,
            token_bank_all_veh=token_bank_all_veh,
            token_bank_all_ped=token_bank_all_ped,
            token_bank_all_cyc=token_bank_all_cyc,
        )
        stationary_contour = self._build_stationary_token_contour(token_agent_shape=token_agent_shape)
        stop_token_idx = match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=stationary_contour,
            token_bank_all_veh=token_bank_all_veh,
            token_bank_all_ped=token_bank_all_ped,
            token_bank_all_cyc=token_bank_all_cyc,
            reduction="sum",
        )
        return raw_token_idx, raw_token_idx == stop_token_idx

    def freeze_commit_chunk(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """다음 0.5초 5개 상태를 현재 상태로 완전히 고정합니다."""
        commit_pos = current_pos.unsqueeze(1).expand(-1, 5, -1).clone()
        commit_head = current_head.unsqueeze(1).expand(-1, 5).clone()
        return commit_pos, commit_head, current_pos.clone(), current_head.clone()

    def _get_difference_gram(
        self,
        num_edge: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """1차 차분 제곱합에 쓰는 Gram 행렬을 돌려줍니다.

        Args:
            num_edge: 속도 또는 곡률 edge 개수입니다.
            device: 행렬을 만들 장치입니다.
            dtype: 행렬 자료형입니다.

        Returns:
            torch.Tensor: ``D^T D`` 행렬입니다. shape은 ``[num_edge, num_edge]`` 입니다.
        """
        cache_key = (num_edge, str(device), str(dtype))
        if cache_key in self._difference_gram_cache:
            return self._difference_gram_cache[cache_key]

        if num_edge <= 1:
            gram = torch.zeros((num_edge, num_edge), device=device, dtype=dtype)
        else:
            diff = torch.zeros((num_edge - 1, num_edge), device=device, dtype=dtype)
            diag_idx = torch.arange(num_edge - 1, device=device)
            diff[diag_idx, diag_idx] = -1.0
            diff[diag_idx, diag_idx + 1] = 1.0
            gram = diff.transpose(0, 1) @ diff
        self._difference_gram_cache[cache_key] = gram
        return gram

    def _fit_smoothed_speed_profile(
        self,
        pos_seq: torch.Tensor,
        valid_seq: torch.Tensor,
    ) -> torch.Tensor:
        """위치 시퀀스에서 batched 선형계로 매끈한 속도 곡선을 추정합니다.

        Args:
            pos_seq: 현재까지 실제 이력과 미래 참조를 붙인 중심점입니다.
                shape은 ``[n_agent, n_step, 2]`` 입니다.
            valid_seq: 같은 시퀀스의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            torch.Tensor: edge 기준 속도 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.
        """
        dt = float(self.config.dt)
        edge_valid = valid_seq[:, :-1] & valid_seq[:, 1:]
        ds_over_dt = torch.norm(pos_seq[:, 1:] - pos_seq[:, :-1], dim=-1) / dt
        edge_weight = edge_valid.to(pos_seq.dtype)
        num_edge = ds_over_dt.shape[1]
        eye = torch.eye(num_edge, device=pos_seq.device, dtype=pos_seq.dtype)
        gram = self._get_difference_gram(num_edge, pos_seq.device, pos_seq.dtype)
        system = torch.diag_embed(edge_weight) + self.config.velocity_smooth_lambda * gram.unsqueeze(0)
        rhs = edge_weight * ds_over_dt
        return torch.linalg.solve(system + 1.0e-6 * eye.unsqueeze(0), rhs.unsqueeze(-1)).squeeze(-1)

    def _fit_smoothed_curvature_profile(
        self,
        head_seq: torch.Tensor,
        valid_seq: torch.Tensor,
        speed_profile: torch.Tensor,
    ) -> torch.Tensor:
        """방향 시퀀스와 속도 곡선에서 batched 곡률 곡선을 추정합니다.

        Args:
            head_seq: 현재까지 실제 이력과 미래 참조를 붙인 방향입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            valid_seq: 같은 시퀀스의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            speed_profile: edge 기준 속도 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.

        Returns:
            torch.Tensor: edge 기준 곡률 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.
        """
        dt = float(self.config.dt)
        edge_valid = valid_seq[:, :-1] & valid_seq[:, 1:]
        yaw_rate_obs = wrap_angle(head_seq[:, 1:] - head_seq[:, :-1]) / dt
        num_edge = speed_profile.shape[1]
        eye = torch.eye(num_edge, device=head_seq.device, dtype=head_seq.dtype)
        gram = self._get_difference_gram(num_edge, head_seq.device, head_seq.dtype)
        speed_abs = speed_profile.abs()
        edge_weight = edge_valid.to(head_seq.dtype)
        diag_weight = edge_weight * speed_abs.square()
        rhs = edge_weight * speed_abs * yaw_rate_obs
        system = (
            torch.diag_embed(diag_weight)
            + self.config.curvature_smooth_lambda * gram.unsqueeze(0)
            + self.config.curvature_init_reg * eye.unsqueeze(0)
        )
        return torch.linalg.solve(system + 1.0e-6 * eye.unsqueeze(0), rhs.unsqueeze(-1)).squeeze(-1)

    def _estimate_reference_profiles(
        self,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        exec_pos_history: torch.Tensor,
        exec_head_history: torch.Tensor,
        exec_valid_history: torch.Tensor,
        future_pos: torch.Tensor,
        future_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """과거 0.5초와 2초 미래를 묶어 속도/곡률 참조를 만듭니다.

        Args:
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.
            exec_pos_history: 최근 실제 fine history 입니다. shape은 ``[n_agent, 6, 2]`` 입니다.
            exec_head_history: 최근 실제 fine heading 입니다. shape은 ``[n_agent, 6]`` 입니다.
            exec_valid_history: 최근 실제 fine valid 입니다. shape은 ``[n_agent, 6]`` 입니다.
            future_pos: raw FM 2초 미래 중심점입니다. shape은 ``[n_agent, 20, 2]`` 입니다.
            future_head: raw FM 2초 미래 방향입니다. shape은 ``[n_agent, 20]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - v0: 현재 속도 추정 ``[n_agent]``
                - a_prev: 직전 가속도 추정 ``[n_agent]``
                - kappa0: 현재 곡률 추정 ``[n_agent]``
                - v_ref_horizon: 다음 1초 속도 참조 ``[n_agent, horizon]``
                - kappa_ref_horizon: 다음 1초 곡률 참조 ``[n_agent, horizon]``
        """
        history_steps = min(exec_pos_history.shape[1], int(self.config.history_steps))
        history_pos = exec_pos_history[:, -history_steps:].clone()
        history_head = exec_head_history[:, -history_steps:].clone()
        history_valid = exec_valid_history[:, -history_steps:].clone()
        history_pos[:, -1] = current_pos
        history_head[:, -1] = current_head
        history_valid[:, -1] = True

        pos_seq = torch.cat([history_pos, future_pos], dim=1)
        head_seq = torch.cat([history_head, future_head], dim=1)
        valid_seq = torch.cat(
            [history_valid, torch.ones_like(future_head, dtype=torch.bool)],
            dim=1,
        )

        speed_profile = self._fit_smoothed_speed_profile(pos_seq=pos_seq, valid_seq=valid_seq)
        curvature_profile = self._fit_smoothed_curvature_profile(
            head_seq=head_seq,
            valid_seq=valid_seq,
            speed_profile=speed_profile,
        )
        history_edge_idx = history_steps - 1
        horizon_steps = int(self.config.horizon_steps)
        v0 = speed_profile[:, history_edge_idx - 1].clamp_min(0.0)
        if history_edge_idx >= 2:
            a_prev = (speed_profile[:, history_edge_idx - 1] - speed_profile[:, history_edge_idx - 2]) / self.config.dt
        else:
            a_prev = speed_profile.new_zeros(speed_profile.shape[0])
        kappa0 = curvature_profile[:, history_edge_idx - 1]
        v_ref_horizon = speed_profile[:, history_edge_idx : history_edge_idx + horizon_steps]
        kappa_ref_horizon = curvature_profile[:, history_edge_idx : history_edge_idx + horizon_steps]
        return v0, a_prev, kappa0, v_ref_horizon, kappa_ref_horizon

    def _solve_longitudinal_lqr(
        self,
        v0: torch.Tensor,
        v_ref_target: torch.Tensor,
    ) -> torch.Tensor:
        """1초 뒤 속도를 맞추는 상수 가속도 하나를 닫힌형으로 풉니다."""
        horizon_time = float(self.config.horizon_steps) * float(self.config.dt)
        q = float(self.config.longitudinal_q)
        r = float(self.config.longitudinal_r)
        numerator = -q * horizon_time * (v0 - v_ref_target)
        denominator = q * (horizon_time**2) + r
        return numerator / max(denominator, 1.0e-6)

    def _solve_lateral_lqr(
        self,
        v_profile: torch.Tensor,
        kappa0: torch.Tensor,
        kappa_ref_profile: torch.Tensor,
    ) -> torch.Tensor:
        """1초 회전 계획을 따르는 상수 곡률 변화율 하나를 닫힌형으로 풉니다.

        Args:
            v_profile: 다음 1초 속도 참조입니다. shape은 ``[n_agent, horizon]`` 입니다.
            kappa0: 현재 곡률입니다. shape은 ``[n_agent]`` 입니다.
            kappa_ref_profile: 다음 1초 곡률 참조입니다. shape은 ``[n_agent, horizon]`` 입니다.

        Returns:
            torch.Tensor: horizon 전체에 유지할 곡률 변화율입니다. shape은 ``[n_agent]`` 입니다.
        """
        dt = float(self.config.dt)
        q_diag = kappa_ref_profile.new_tensor(
            [
                float(self.config.lateral_q_lat),
                float(self.config.lateral_q_head),
                float(self.config.lateral_q_kappa),
            ]
        )
        z_no_u = torch.stack(
            [kappa0.new_zeros(kappa0.shape[0]), kappa0.new_zeros(kappa0.shape[0]), kappa0],
            dim=-1,
        )
        gamma = torch.zeros_like(z_no_u)
        b = kappa0.new_tensor([0.0, 0.0, dt]).unsqueeze(0).expand(kappa0.shape[0], -1)

        for step_idx in range(kappa_ref_profile.shape[1]):
            v_step = v_profile[:, step_idx]
            a_mat = torch.zeros((kappa0.shape[0], 3, 3), device=kappa0.device, dtype=kappa0.dtype)
            a_mat[:, 0, 0] = 1.0
            a_mat[:, 0, 1] = v_step * dt
            a_mat[:, 1, 1] = 1.0
            a_mat[:, 1, 2] = v_step * dt
            a_mat[:, 2, 2] = 1.0
            c_vec = torch.stack(
                [
                    v_step.new_zeros(v_step.shape[0]),
                    -(v_step * dt * kappa_ref_profile[:, step_idx]),
                    v_step.new_zeros(v_step.shape[0]),
                ],
                dim=-1,
            )
            z_no_u = torch.bmm(a_mat, z_no_u.unsqueeze(-1)).squeeze(-1) + c_vec
            gamma = torch.bmm(a_mat, gamma.unsqueeze(-1)).squeeze(-1) + b

        numerator = (gamma * q_diag.unsqueeze(0) * z_no_u).sum(dim=-1)
        denominator = (gamma * q_diag.unsqueeze(0) * gamma).sum(dim=-1) + float(self.config.lateral_r)
        return -numerator / denominator.clamp_min(1.0e-6)

    def _gather_dynamic_limits(
        self,
        agent_type: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """차종별 WOMD 통계 기반 물리 제한을 꺼냅니다."""
        device = agent_type.device
        return {
            "v_max": torch.tensor(DEFAULT_LIMITS.v_max_mps, device=device, dtype=dtype)[agent_type.long()],
            "a_max": torch.tensor(DEFAULT_LIMITS.a_max_mps2, device=device, dtype=dtype)[agent_type.long()],
            "alpha_max": torch.tensor(DEFAULT_LIMITS.alpha_max_radps2, device=device, dtype=dtype)[agent_type.long()],
            "a_lat_max": torch.tensor(DEFAULT_LIMITS.a_lat_max_mps2, device=device, dtype=dtype)[agent_type.long()],
            "r_min": torch.tensor(DEFAULT_LIMITS.r_min_m, device=device, dtype=dtype)[agent_type.long()],
            "omega_max": torch.tensor(DEFAULT_LIMITS.omega_max_abs_radps, device=device, dtype=dtype)[agent_type.long()],
        }

    def _clip_curvature_and_rate(
        self,
        kappa_state: torch.Tensor,
        a_value: torch.Tensor,
        u_value: torch.Tensor,
        speed_value: torch.Tensor,
        limits: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 속도와 class별 envelope로 곡률과 곡률 변화율을 제한합니다."""
        speed_floor = float(self.config.min_speed_for_curvature_clip_mps)
        speed_safe = torch.maximum(speed_value.abs(), speed_value.new_full(speed_value.shape, speed_floor))
        inv_radius_limit = 1.0 / limits["r_min"].clamp_min(1.0e-6)
        yaw_rate_limit = limits["omega_max"] / speed_safe
        lat_accel_limit = limits["a_lat_max"] / speed_safe.square().clamp_min(1.0e-6)
        kappa_limit = torch.minimum(inv_radius_limit, torch.minimum(yaw_rate_limit, lat_accel_limit))
        kappa_state = torch.clamp(kappa_state, -kappa_limit, kappa_limit)

        u_low = (-limits["alpha_max"] - a_value * kappa_state) / speed_safe
        u_high = (limits["alpha_max"] - a_value * kappa_state) / speed_safe
        u_clipped = torch.clamp(u_value, torch.minimum(u_low, u_high), torch.maximum(u_low, u_high))
        return kappa_state, u_clipped

    def execute_lqr_commit(
        self,
        y_hat_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        exec_pos_history: torch.Tensor,
        exec_head_history: torch.Tensor,
        exec_valid_history: torch.Tensor,
        agent_type: torch.Tensor,
        agent_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """vehicle / bicycle의 다음 0.5초를 0.1초 receding-horizon LQR로 실행합니다.

        Args:
            y_hat_norm: raw FM 2초 미래입니다. pose-space에서는 ``[n_agent, 20, 4]``,
                control-space에서는 ``[n_agent, 20, 3]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.
            exec_pos_history: 최근 실제 fine history 입니다. shape은 ``[n_agent, 6, 2]`` 입니다.
            exec_head_history: 최근 실제 fine heading 입니다. shape은 ``[n_agent, 6]`` 입니다.
            exec_valid_history: 최근 실제 fine valid 입니다. shape은 ``[n_agent, 6]`` 입니다.
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            agent_length: WOMD box length입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - commit_pos: 실행된 다음 0.5초 중심점 ``[n_agent, 5, 2]``
                - commit_head: 실행된 다음 0.5초 방향 ``[n_agent, 5]``
                - next_pos: 마지막 중심점 ``[n_agent, 2]``
                - next_head: 마지막 방향 ``[n_agent]``
        """
        if y_hat_norm.numel() == 0:
            empty_pos = current_pos.new_zeros((0, 5, 2))
            empty_head = current_head.new_zeros((0, 5))
            return empty_pos, empty_head, current_pos.clone(), current_head.clone()

        y_hat_pose_norm = self._flow_output_to_pose_norm(
            y_hat_norm=y_hat_norm,
            agent_type=agent_type,
            agent_length=agent_length,
        )
        future_pos, future_head = self._build_full_future_from_flow(
            y_hat_norm=y_hat_pose_norm,
            current_pos=current_pos,
            current_head=current_head,
        )
        v0, a_prev, kappa0, _, _ = self._estimate_reference_profiles(
            current_pos=current_pos,
            current_head=current_head,
            exec_pos_history=exec_pos_history,
            exec_head_history=exec_head_history,
            exec_valid_history=exec_valid_history,
            future_pos=future_pos,
            future_head=future_head,
        )

        limits = self._gather_dynamic_limits(agent_type=agent_type, dtype=current_pos.dtype)
        dt = float(self.config.dt)
        accel_alpha = dt / (dt + float(self.config.accel_tau_s))
        curvature_alpha = dt / (dt + float(self.config.curvature_tau_s))
        history_steps = min(exec_pos_history.shape[1], int(self.config.history_steps))

        pos_state = current_pos.clone()
        head_state = current_head.clone()
        speed_state = v0.clamp_min(0.0)
        accel_state = a_prev.clamp(-limits["a_max"], limits["a_max"])
        kappa_state = kappa0.clone()
        exec_pos_history_state = exec_pos_history[:, -history_steps:].clone()
        exec_head_history_state = exec_head_history[:, -history_steps:].clone()
        exec_valid_history_state = exec_valid_history[:, -history_steps:].clone()
        exec_pos_history_state[:, -1] = current_pos
        exec_head_history_state[:, -1] = current_head
        exec_valid_history_state[:, -1] = True

        commit_pos = current_pos.new_zeros((current_pos.shape[0], 5, 2))
        commit_head = current_head.new_zeros((current_head.shape[0], 5))
        for step_idx in range(5):
            # Keep the FM future fixed for the 0.5s block, but re-solve control on the remaining horizon every 0.1s.
            remaining_future_pos = future_pos[:, step_idx:]
            remaining_future_head = future_head[:, step_idx:]
            _, _, _, v_ref_horizon, kappa_ref_horizon = self._estimate_reference_profiles(
                current_pos=pos_state,
                current_head=head_state,
                exec_pos_history=exec_pos_history_state,
                exec_head_history=exec_head_history_state,
                exec_valid_history=exec_valid_history_state,
                future_pos=remaining_future_pos,
                future_head=remaining_future_head,
            )
            v_ref_target = v_ref_horizon[:, -1]
            a_star = self._solve_longitudinal_lqr(v0=speed_state, v_ref_target=v_ref_target)
            u_star = self._solve_lateral_lqr(
                v_profile=v_ref_horizon,
                kappa0=kappa_state,
                kappa_ref_profile=kappa_ref_horizon,
            )
            slow_mask = (speed_state < float(self.config.stop_speed_mps)) & (
                v_ref_target < float(self.config.stop_speed_mps)
            )
            if slow_mask.any():
                a_star = a_star.clone()
                u_star = u_star.clone()
                a_star[slow_mask] = -float(self.config.stop_speed_kp) * (
                    speed_state[slow_mask] - v_ref_target[slow_mask]
                )
                u_star[slow_mask] = 0.0
            a_star = torch.clamp(a_star, -limits["a_max"], limits["a_max"])

            accel_applied = accel_state + accel_alpha * (a_star - accel_state)
            accel_applied = torch.clamp(accel_applied, -limits["a_max"], limits["a_max"])
            kappa_state, u_applied = self._clip_curvature_and_rate(
                kappa_state=kappa_state,
                a_value=accel_applied,
                u_value=u_star,
                speed_value=speed_state,
                limits=limits,
            )
            kappa_ideal = kappa_state + dt * u_applied
            kappa_applied = kappa_state + curvature_alpha * (kappa_ideal - kappa_state)
            kappa_applied, _ = self._clip_curvature_and_rate(
                kappa_state=kappa_applied,
                a_value=accel_applied,
                u_value=u_applied,
                speed_value=speed_state,
                limits=limits,
            )
            pos_state = pos_state + dt * speed_state.unsqueeze(-1) * torch.stack(
                [head_state.cos(), head_state.sin()],
                dim=-1,
            )
            head_state = wrap_angle(head_state + dt * speed_state * kappa_applied)
            speed_state = torch.clamp(
                speed_state + dt * accel_applied,
                min=torch.zeros_like(limits["v_max"]),
                max=limits["v_max"],
            )
            accel_state = accel_applied
            kappa_state = kappa_applied
            commit_pos[:, step_idx] = pos_state
            commit_head[:, step_idx] = head_state
            if history_steps == 1:
                exec_pos_history_state = pos_state.unsqueeze(1)
                exec_head_history_state = head_state.unsqueeze(1)
                exec_valid_history_state = torch.ones_like(head_state, dtype=torch.bool).unsqueeze(1)
            else:
                exec_pos_history_state = torch.cat(
                    [exec_pos_history_state[:, 1:], pos_state.unsqueeze(1)],
                    dim=1,
                )
                exec_head_history_state = torch.cat(
                    [exec_head_history_state[:, 1:], head_state.unsqueeze(1)],
                    dim=1,
                )
                exec_valid_history_state = torch.cat(
                    [
                        exec_valid_history_state[:, 1:],
                        torch.ones_like(head_state, dtype=torch.bool).unsqueeze(1),
                    ],
                    dim=1,
                )

        return commit_pos, commit_head, commit_pos[:, -1], commit_head[:, -1]
