from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.modules.draft_physics import DEFAULT_LIMITS


def _device_prefers_math_sdp(device: torch.device) -> bool:
    """알려진 불안정 GPU에서는 SDPA 고속 커널 대신 math 경로를 먼저 선택합니다."""
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False

    major, _minor = torch.cuda.get_device_capability(device)
    # Blackwell(sm_120) 계열에서 validation 중 SDPA 고속 커널 크래시가 관찰됐습니다.
    return major >= 12


def _is_retryable_sdp_runtime_error(error: RuntimeError) -> bool:
    """SDPA 고속 커널 실패로 판단되면 안전한 math 경로 재시도를 허용합니다."""
    error_message = str(error).lower()
    retryable_patterns = (
        "invalid configuration argument",
        "no kernel image is available",
        "device kernel image is invalid",
        "operation not supported",
    )
    return any(pattern in error_message for pattern in retryable_patterns)


def _run_attention_with_safe_fallback(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """기본은 고속 커널을 쓰고, 알려진/관측된 실패에서만 math 경로로 강등합니다."""
    if _device_prefers_math_sdp(query.device):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attn_out, _ = attn(query, key, value, need_weights=False)
        return attn_out

    try:
        attn_out, _ = attn(query, key, value, need_weights=False)
        return attn_out
    except RuntimeError as error:
        if query.device.type != "cuda" or not _is_retryable_sdp_runtime_error(error):
            raise
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attn_out, _ = attn(query, key, value, need_weights=False)
        return attn_out


@dataclass
class FlowSample:
    x_t: torch.Tensor
    target: torch.Tensor
    tau: torch.Tensor


@dataclass(frozen=True)
class LQRCommitBridgeConfig:
    """Closed-loop LQR bridge 설정을 담습니다.

    Attributes:
        dt: 10Hz fine step 길이입니다.
        history_steps: 제어 참조를 만들 때 쓸 실제 fine history 길이입니다.
            shape 의미는 ``6`` 이면 최근 0.5초 + 현재까지의 6개 점입니다.
        horizon_steps: LQR가 직접 볼 미래 길이입니다.
        replan_every_step: ``True`` 면 0.1초마다 receding-horizon LQR를 다시 풉니다.
            ``False`` 면 chunk 시작에서 한 번만 제어 명령을 풀고 0.5초 동안 유지합니다.
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
        clip_longitudinal_command: 종방향 목표 가속도를 물리 한계로 먼저 자를지 여부입니다.
        clip_lateral_projection_and_final_curvature_state: 현재 속도/동역학 한계 기반
            횡방향 projection과 지연 뒤 최종 곡률 상태 재-clip을 같이 켤지 여부입니다.
        accel_tau_s: 가속 입력 1차 지연 시간입니다.
        curvature_tau_s: 곡률 입력 1차 지연 시간입니다.
        min_speed_for_curvature_clip_mps: 곡률 clip 계산에서 쓸 최소 속도입니다.
    """

    dt: float = 0.1
    history_steps: int = 6
    horizon_steps: int = 10
    replan_every_step: bool = True
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
    clip_longitudinal_command: bool = True
    clip_lateral_projection_and_final_curvature_state: bool = True
    accel_tau_s: float = 0.2
    curvature_tau_s: float = 0.05
    min_speed_for_curvature_clip_mps: float = 0.5


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

    def sample(self, clean: torch.Tensor, target_type: str = "velocity") -> FlowSample:
        if target_type != "velocity":
            raise ValueError(f"Unsupported target_type: {target_type}")

        tau = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        tau = tau * (1.0 - self.eps) + self.eps

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
    ) -> torch.Tensor:
        """ODE 샘플링으로 최종 clean future를 만듭니다.

        Args:
            x_init: 시작 잡음 상태입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            model_fn: 현재 상태와 시간 ``tau`` 를 받아 속도를 돌려주는 함수입니다.
            steps: 샘플링 step 수입니다. ``None`` 이면 기본 solver step을 씁니다.
            method: 적분 방식입니다. ``None`` 이면 기본 solver 방식을 씁니다.
            backprop_last_k: 마지막 몇 step에만 gradient를 남길지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다.

        Returns:
            torch.Tensor: 최종 정규화 미래입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        steps = self.solver_steps if steps is None else steps
        method = self.solver_method if method is None else method

        x_t = x_init
        t0 = self.eps
        dt = (1.0 - t0) / float(steps)

        if backprop_last_k is None or int(backprop_last_k) >= int(steps):
            grad_start_step = 0
        else:
            grad_start_step = max(0, int(steps) - max(0, int(backprop_last_k)))

        for i in range(steps):
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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, anchor_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(anchor_hidden)


class NormalizedNoisyFutureEncoder(nn.Module):
    def __init__(self, flow_dim: int, num_chunks: int = 4, chunk_size: int = 5) -> None:
        super().__init__()
        self.flow_dim = flow_dim
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.num_steps = num_chunks * chunk_size

        self.step_proj = nn.Linear(4, flow_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_t_norm.shape[0]

        tau_emb = self.tau_mlp(tau.unsqueeze(-1))
        step_tokens = self.step_proj(x_t_norm)
        step_ids = torch.arange(self.num_steps, device=x_t_norm.device)
        step_tokens = step_tokens + self.step_embed(step_ids).unsqueeze(0)
        step_tokens = step_tokens + tau_emb.unsqueeze(1)

        step_tokens = step_tokens.view(
            batch_size,
            self.num_chunks,
            self.chunk_size,
            self.flow_dim,
        )
        chunk_tokens = self.chunk_pool(step_tokens.mean(dim=2))
        return step_tokens, chunk_tokens, tau_emb


class HalfSecondChunkMixerBlock(nn.Module):
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
        scale, bias, gate = cond.chunk(3, dim=-1)
        return x + torch.sigmoid(gate).unsqueeze(1) * (
            x * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        )

    def forward(
        self,
        chunk_tokens: torch.Tensor,
        context: torch.Tensor,
        tau_emb: torch.Tensor,
    ) -> torch.Tensor:
        attn_in = self.attn_norm(chunk_tokens)
        attn_out = _run_attention_with_safe_fallback(self.attn, attn_in, attn_in, attn_in)
        chunk_tokens = chunk_tokens + attn_out

        cond = self.cond_mlp(torch.cat([context, tau_emb], dim=-1))
        mlp_in = self._modulate(self.mlp_norm(chunk_tokens), cond)
        chunk_tokens = chunk_tokens + self.mlp(mlp_in)
        return chunk_tokens


class ChunkStepRefiner(nn.Module):
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

    def forward(
        self,
        step_tokens: torch.Tensor,
        chunk_tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_chunks, chunk_size, dim = step_tokens.shape

        step_tokens = step_tokens + chunk_tokens.unsqueeze(2)
        step_tokens = step_tokens + self.context_proj(context).view(batch_size, 1, 1, dim)
        step_tokens = self.pre_proj(step_tokens)

        step_tokens = step_tokens.view(batch_size * num_chunks, chunk_size, dim)
        attn_in = self.attn_norm(step_tokens)
        attn_out = _run_attention_with_safe_fallback(self.attn, attn_in, attn_in, attn_in)
        step_tokens = step_tokens + attn_out
        step_tokens = step_tokens + self.mlp(self.mlp_norm(step_tokens))
        step_tokens = step_tokens.view(batch_size, num_chunks * chunk_size, dim)
        return step_tokens


class FlowVelocityHead(nn.Module):
    def __init__(self, flow_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, 4),
        )

    def forward(self, step_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(step_tokens)


class HierarchicalFlowDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        flow_dim: int,
        num_chunk_heads: int = 4,
        num_chunk_layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_projector = AnchorContextProjector(context_dim, flow_dim)
        self.noisy_future_encoder = NormalizedNoisyFutureEncoder(flow_dim=flow_dim)
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
        self.velocity_head = FlowVelocityHead(flow_dim=flow_dim)

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_projector(anchor_hidden)
        step_tokens, chunk_tokens, tau_emb = self.noisy_future_encoder(x_t_norm, tau)

        for block in self.chunk_mixers:
            chunk_tokens = block(chunk_tokens, context, tau_emb)

        step_tokens = self.step_refiner(step_tokens, chunk_tokens, context)
        return self.velocity_head(step_tokens)


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
        use_lqr: bool = False,
        use_stop_motion: bool = False,
        config: LQRCommitBridgeConfig | None = None,
    ) -> None:
        self.use_lqr = bool(use_lqr)
        self.use_stop_motion = bool(use_stop_motion)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """raw flow 미래에서 바로 첫 0.5초 chunk를 꺼냅니다.

        Args:
            y_hat_norm: 정규화된 2초 미래입니다. shape은 ``[n_agent, 20, 4]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - commit_pos: 다음 0.5초 5개 중심점 ``[n_agent, 5, 2]``
                - commit_head: 다음 0.5초 5개 방향 ``[n_agent, 5]``
                - next_pos: 다음 coarse 상태 중심점 ``[n_agent, 2]``
                - next_head: 다음 coarse 상태 방향 ``[n_agent]``
        """
        first_chunk = y_hat_norm[:, :5].clone()
        first_chunk[..., :2] = first_chunk[..., :2] * 20.0

        cos_sin = F.normalize(first_chunk[..., 2:4], dim=-1, eps=1.0e-6)
        delta_head = torch.atan2(cos_sin[..., 1], cos_sin[..., 0])

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
        future_dir = F.normalize(y_hat_norm[..., 2:4], dim=-1, eps=1.0e-6)
        future_head = wrap_angle(
            current_head.unsqueeze(1) + torch.atan2(future_dir[..., 1], future_dir[..., 0])
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
        """학습과 같은 6개 점 경로 기준으로 다음 coarse 토큰 번호를 다시 고릅니다."""
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
            num_k=1,
            sample_topk=False,
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
            num_k=1,
            sample_topk=False,
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

    def _build_edge_forward_axis(
        self,
        head_seq: torch.Tensor,
    ) -> torch.Tensor:
        """각 edge 시작점의 차체 앞 방향 단위벡터를 만듭니다.

        Args:
            head_seq: 각 시점의 차체 방향입니다.
                shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            torch.Tensor: 각 edge 시작점 기준 앞 방향 단위벡터입니다.
                shape은 ``[n_agent, n_step - 1, 2]`` 입니다.
        """
        return torch.stack([head_seq[:, :-1].cos(), head_seq[:, :-1].sin()], dim=-1)

    def _compute_signed_speed_observation(
        self,
        pos_seq: torch.Tensor,
        head_seq: torch.Tensor,
        valid_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """각 edge 이동을 차체 앞 방향에 투영해 앞뒤 부호가 있는 속도를 만듭니다.

        Args:
            pos_seq: 연속된 위치 시퀀스입니다.
                shape은 ``[n_agent, n_step, 2]`` 입니다.
            head_seq: 각 시점의 차체 방향입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            valid_seq: 같은 시퀀스의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - signed_speed_obs: edge마다 계산한 signed 속도 관측치입니다.
                    shape은 ``[n_agent, n_step - 1]`` 입니다.
                - edge_valid: 인접한 두 점이 모두 유효한 edge 여부입니다.
                    shape은 ``[n_agent, n_step - 1]`` 입니다.
        """
        dt = float(self.config.dt)
        edge_delta = pos_seq[:, 1:] - pos_seq[:, :-1]
        forward_axis = self._build_edge_forward_axis(head_seq=head_seq)
        signed_speed_obs = (edge_delta * forward_axis).sum(dim=-1) / dt
        edge_valid = valid_seq[:, :-1] & valid_seq[:, 1:]
        return signed_speed_obs, edge_valid

    def _apply_signed_speed_floor(
        self,
        speed_value: torch.Tensor,
    ) -> torch.Tensor:
        """속도 크기는 최소값 이상으로 올리되 앞뒤 부호는 그대로 유지합니다.

        Args:
            speed_value: signed 속도 값입니다.
                shape은 임의의 텐서 shape을 가질 수 있습니다.

        Returns:
            torch.Tensor: 절댓값은 최소값 이상이고 부호는 유지된 signed 속도입니다.
                shape은 입력과 같습니다.
        """
        speed_floor = float(self.config.min_speed_for_curvature_clip_mps)
        speed_sign = torch.where(
            speed_value < 0.0,
            -torch.ones_like(speed_value),
            torch.ones_like(speed_value),
        )
        speed_abs = torch.maximum(
            speed_value.abs(),
            speed_value.new_full(speed_value.shape, speed_floor),
        )
        return speed_sign * speed_abs

    def _fit_smoothed_speed_profile(
        self,
        pos_seq: torch.Tensor,
        head_seq: torch.Tensor,
        valid_seq: torch.Tensor,
        fixed_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """위치와 방향 시퀀스에서 앞뒤 부호가 있는 속도 곡선을 추정합니다.

        Args:
            pos_seq: 연속된 위치 시퀀스입니다.
                shape은 ``[n_agent, n_step, 2]`` 입니다.
            head_seq: 각 시점의 차체 방향입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            valid_seq: 같은 시퀀스의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            fixed_prefix: smoothness 시작 조건으로 고정할 과거 edge 값입니다.
                shape은 ``[n_agent, prefix_edge]`` 이고 ``prefix_edge`` 는 0, 1, 2 중 하나입니다.

        Returns:
            torch.Tensor: edge 기준 signed 속도 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.
        """
        signed_speed_obs, edge_valid = self._compute_signed_speed_observation(
            pos_seq=pos_seq,
            head_seq=head_seq,
            valid_seq=valid_seq,
        )
        edge_weight = edge_valid.to(pos_seq.dtype)
        rhs = edge_weight * signed_speed_obs
        return self._solve_smoothed_edge_profile(
            diag_weight=edge_weight,
            rhs=rhs,
            smooth_lambda=float(self.config.velocity_smooth_lambda),
            fixed_prefix=fixed_prefix,
        )

    def _fit_smoothed_curvature_profile(
        self,
        head_seq: torch.Tensor,
        valid_seq: torch.Tensor,
        speed_profile: torch.Tensor,
        fixed_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """방향 시퀀스와 signed 속도 곡선에서 곡률 곡선을 추정합니다.

        Args:
            head_seq: 연속된 방향 시퀀스입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            valid_seq: 같은 시퀀스의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            speed_profile: edge 기준 signed 속도 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.
            fixed_prefix: smoothness 시작 조건으로 고정할 과거 edge 곡률입니다.
                shape은 ``[n_agent, prefix_edge]`` 이고 ``prefix_edge`` 는 0, 1, 2 중 하나입니다.

        Returns:
            torch.Tensor: edge 기준 곡률 곡선입니다.
                shape은 ``[n_agent, n_step - 1]`` 입니다.
        """
        dt = float(self.config.dt)
        edge_valid = valid_seq[:, :-1] & valid_seq[:, 1:]
        yaw_rate_obs = wrap_angle(head_seq[:, 1:] - head_seq[:, :-1]) / dt
        edge_weight = edge_valid.to(head_seq.dtype)
        speed_abs = speed_profile.abs()
        safe_speed = self._apply_signed_speed_floor(speed_profile)
        diag_weight = edge_weight * speed_abs.square()
        kappa_obs = yaw_rate_obs / safe_speed
        rhs = diag_weight * kappa_obs
        return self._solve_smoothed_edge_profile(
            diag_weight=diag_weight,
            rhs=rhs,
            smooth_lambda=float(self.config.curvature_smooth_lambda),
            fixed_prefix=fixed_prefix,
            diag_reg=float(self.config.curvature_init_reg),
        )

    def _solve_smoothed_edge_profile(
        self,
        diag_weight: torch.Tensor,
        rhs: torch.Tensor,
        smooth_lambda: float,
        fixed_prefix: Optional[torch.Tensor] = None,
        diag_reg: float = 0.0,
    ) -> torch.Tensor:
        """고정 prefix를 경계조건으로 둘 수 있는 1차 차분 smoothing 선형계를 풉니다."""
        num_edge = diag_weight.shape[1]
        if num_edge == 0:
            return rhs.new_zeros((rhs.shape[0], 0))

        device = diag_weight.device
        dtype = diag_weight.dtype
        eye = torch.eye(num_edge, device=device, dtype=dtype)

        prefix_len = 0 if fixed_prefix is None else int(fixed_prefix.shape[1])
        if prefix_len == 0:
            gram = self._get_difference_gram(num_edge, device, dtype)
            system = torch.diag_embed(diag_weight) + smooth_lambda * gram.unsqueeze(0)
            adjusted_rhs = rhs
        else:
            full_gram = self._get_difference_gram(prefix_len + num_edge, device, dtype)
            gram_xx = full_gram[prefix_len:, prefix_len:]
            gram_xp = full_gram[prefix_len:, :prefix_len]
            system = torch.diag_embed(diag_weight) + smooth_lambda * gram_xx.unsqueeze(0)
            prefix_term = torch.bmm(
                gram_xp.unsqueeze(0).expand(rhs.shape[0], -1, -1),
                fixed_prefix.unsqueeze(-1),
            ).squeeze(-1)
            adjusted_rhs = rhs - smooth_lambda * prefix_term

        if diag_reg > 0.0:
            system = system + diag_reg * eye.unsqueeze(0)
        return torch.linalg.solve(system + 1.0e-6 * eye.unsqueeze(0), adjusted_rhs.unsqueeze(-1)).squeeze(-1)

    def _build_profile_init_prefix(
        self,
        current_value: torch.Tensor,
        current_rate: torch.Tensor,
    ) -> torch.Tensor:
        """현재/직전 상태를 미래 edge profile의 고정 prefix로 바꿉니다."""
        prev_value = current_value - float(self.config.dt) * current_rate
        return torch.stack([prev_value, current_value], dim=1)

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
        """과거로 현재 상태를 추정하고, 미래는 그 상태에서 시작하는 참조로 만듭니다.

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
        dt = float(self.config.dt)
        history_steps = min(exec_pos_history.shape[1], int(self.config.history_steps))
        history_pos = exec_pos_history[:, -history_steps:].clone()
        history_head = exec_head_history[:, -history_steps:].clone()
        history_valid = exec_valid_history[:, -history_steps:].clone()
        history_pos[:, -1] = current_pos
        history_head[:, -1] = current_head
        history_valid[:, -1] = True

        past_speed_profile = self._fit_smoothed_speed_profile(
            pos_seq=history_pos,
            head_seq=history_head,
            valid_seq=history_valid,
        )
        if past_speed_profile.shape[1] >= 1:
            v0 = past_speed_profile[:, -1]
        else:
            v0 = current_pos.new_zeros(current_pos.shape[0])
        if past_speed_profile.shape[1] >= 2:
            a_prev = (past_speed_profile[:, -1] - past_speed_profile[:, -2]) / dt
        else:
            a_prev = current_pos.new_zeros(current_pos.shape[0])

        past_curvature_profile = self._fit_smoothed_curvature_profile(
            head_seq=history_head,
            valid_seq=history_valid,
            speed_profile=past_speed_profile,
        )
        if past_curvature_profile.shape[1] >= 1:
            kappa0 = past_curvature_profile[:, -1]
        else:
            kappa0 = current_head.new_zeros(current_head.shape[0])
        if past_curvature_profile.shape[1] >= 2:
            kappa_rate0 = (past_curvature_profile[:, -1] - past_curvature_profile[:, -2]) / dt
        else:
            kappa_rate0 = current_head.new_zeros(current_head.shape[0])

        future_valid = torch.ones_like(future_head, dtype=torch.bool)
        future_pos_seq = torch.cat([current_pos.unsqueeze(1), future_pos], dim=1)
        future_head_seq = torch.cat([current_head.unsqueeze(1), future_head], dim=1)
        future_valid_seq = torch.cat(
            [torch.ones_like(current_head, dtype=torch.bool).unsqueeze(1), future_valid],
            dim=1,
        )
        future_speed_profile = self._fit_smoothed_speed_profile(
            pos_seq=future_pos_seq,
            head_seq=future_head_seq,
            valid_seq=future_valid_seq,
            fixed_prefix=self._build_profile_init_prefix(current_value=v0, current_rate=a_prev),
        )
        future_curvature_profile = self._fit_smoothed_curvature_profile(
            head_seq=future_head_seq,
            valid_seq=future_valid_seq,
            speed_profile=future_speed_profile,
            fixed_prefix=self._build_profile_init_prefix(current_value=kappa0, current_rate=kappa_rate0),
        )

        horizon_steps = int(self.config.horizon_steps)
        v_ref_horizon = future_speed_profile[:, :horizon_steps]
        kappa_ref_horizon = future_curvature_profile[:, :horizon_steps]
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
            "v_reverse_max": torch.tensor(DEFAULT_LIMITS.v_reverse_max_mps, device=device, dtype=dtype)[
                agent_type.long()
            ],
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
        speed_safe = self._apply_signed_speed_floor(speed_value)
        speed_safe_abs = speed_safe.abs()
        inv_radius_limit = 1.0 / limits["r_min"].clamp_min(1.0e-6)
        yaw_rate_limit = limits["omega_max"] / speed_safe_abs
        lat_accel_limit = limits["a_lat_max"] / speed_safe_abs.square().clamp_min(1.0e-6)
        kappa_limit = torch.minimum(inv_radius_limit, torch.minimum(yaw_rate_limit, lat_accel_limit))
        kappa_state = torch.clamp(kappa_state, -kappa_limit, kappa_limit)

        u_low = (-limits["alpha_max"] - a_value * kappa_state) / speed_safe
        u_high = (limits["alpha_max"] - a_value * kappa_state) / speed_safe
        u_clipped = torch.clamp(u_value, torch.minimum(u_low, u_high), torch.maximum(u_low, u_high))
        return kappa_state, u_clipped

    def _postprocess_lqr_commands(
        self,
        speed_state: torch.Tensor,
        v_ref_target: torch.Tensor,
        a_star: torch.Tensor,
        u_star: torch.Tensor,
        limits: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """저속 예외 처리와 종방향 명령 clamp를 공통으로 적용합니다."""
        slow_mask = (speed_state.abs() < float(self.config.stop_speed_mps)) & (
            v_ref_target.abs() < float(self.config.stop_speed_mps)
        )
        if slow_mask.any():
            a_star = a_star.clone()
            u_star = u_star.clone()
            a_star[slow_mask] = -float(self.config.stop_speed_kp) * (
                speed_state[slow_mask] - v_ref_target[slow_mask]
            )
            u_star[slow_mask] = 0.0
        if self.config.clip_longitudinal_command:
            a_star = torch.clamp(a_star, -limits["a_max"], limits["a_max"])
        return a_star, u_star

    def execute_lqr_commit(
        self,
        y_hat_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        exec_pos_history: torch.Tensor,
        exec_head_history: torch.Tensor,
        exec_valid_history: torch.Tensor,
        agent_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """vehicle / bicycle의 다음 0.5초를 LQR commit bridge로 실행합니다.

        Args:
            y_hat_norm: raw FM 2초 미래입니다. shape은 ``[n_agent, 20, 4]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.
            exec_pos_history: 최근 실제 fine history 입니다. shape은 ``[n_agent, 6, 2]`` 입니다.
            exec_head_history: 최근 실제 fine heading 입니다. shape은 ``[n_agent, 6]`` 입니다.
            exec_valid_history: 최근 실제 fine valid 입니다. shape은 ``[n_agent, 6]`` 입니다.
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.

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

        future_pos, future_head = self._build_full_future_from_flow(
            y_hat_norm=y_hat_norm,
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
        speed_state = v0.clone()
        accel_state = a_prev.clamp(-limits["a_max"], limits["a_max"])
        kappa_state = kappa0.clone()
        replan_every_step = bool(self.config.replan_every_step)
        exec_pos_history_state = exec_pos_history[:, -history_steps:].clone()
        exec_head_history_state = exec_head_history[:, -history_steps:].clone()
        exec_valid_history_state = exec_valid_history[:, -history_steps:].clone()
        exec_pos_history_state[:, -1] = current_pos
        exec_head_history_state[:, -1] = current_head
        exec_valid_history_state[:, -1] = True

        commit_pos = current_pos.new_zeros((current_pos.shape[0], 5, 2))
        commit_head = current_head.new_zeros((current_head.shape[0], 5))
        if not replan_every_step:
            _, _, _, v_ref_horizon, kappa_ref_horizon = self._estimate_reference_profiles(
                current_pos=pos_state,
                current_head=head_state,
                exec_pos_history=exec_pos_history_state,
                exec_head_history=exec_head_history_state,
                exec_valid_history=exec_valid_history_state,
                future_pos=future_pos,
                future_head=future_head,
            )
            held_v_ref_target = v_ref_horizon[:, -1]
            held_a_star = self._solve_longitudinal_lqr(v0=speed_state, v_ref_target=held_v_ref_target)
            held_u_star = self._solve_lateral_lqr(
                v_profile=v_ref_horizon,
                kappa0=kappa_state,
                kappa_ref_profile=kappa_ref_horizon,
            )
            held_a_star, held_u_star = self._postprocess_lqr_commands(
                speed_state=speed_state,
                v_ref_target=held_v_ref_target,
                a_star=held_a_star,
                u_star=held_u_star,
                limits=limits,
            )
        for step_idx in range(5):
            if replan_every_step:
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
                a_star, u_star = self._postprocess_lqr_commands(
                    speed_state=speed_state,
                    v_ref_target=v_ref_target,
                    a_star=a_star,
                    u_star=u_star,
                    limits=limits,
                )
            else:
                a_star = held_a_star
                u_star = held_u_star

            accel_applied = accel_state + accel_alpha * (a_star - accel_state)
            accel_applied = torch.clamp(accel_applied, -limits["a_max"], limits["a_max"])
            if self.config.clip_lateral_projection_and_final_curvature_state:
                kappa_state, u_applied = self._clip_curvature_and_rate(
                    kappa_state=kappa_state,
                    a_value=accel_applied,
                    u_value=u_star,
                    speed_value=speed_state,
                    limits=limits,
                )
            else:
                u_applied = u_star
            kappa_ideal = kappa_state + dt * u_applied
            kappa_applied = kappa_state + curvature_alpha * (kappa_ideal - kappa_state)
            if self.config.clip_lateral_projection_and_final_curvature_state:
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
                min=-limits["v_reverse_max"],
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
