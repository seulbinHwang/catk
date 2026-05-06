from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smart.tokens.agent_token_matching import match_token_idx_from_local_contour
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


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

    Adjoint / gradient-checkpointing mode (``use_adjoint_for_bptt``):
        When True, each ``model_fn`` call inside ``generate()`` is wrapped with
        ``torch.utils.checkpoint``.  This is the discrete-ODE analogue of the
        Neural ODE adjoint method: intermediate activations inside the velocity
        network are NOT stored during the forward pass and are recomputed on
        demand during the backward pass.  Memory usage drops from
        O(solver_steps × activation_size) to O(activation_size) at the cost
        of one extra forward pass per ODE integration.
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
        # When True, each model_fn call in generate() is wrapped with
        # torch.utils.checkpoint to trade memory for recomputation.
        # Set externally by SMARTFlow._run_flow_bptt_ft_step when bptt_use_adjoint=True.
        self.use_adjoint_for_bptt: bool = False
        # 마지막 N solver step 에서만 velocity output 이 model_fn 파라미터 gradient 를 받습니다.
        # 0 이하면 비활성 (모든 solver step velocity 에 gradient 흐름).
        # early step 의 velocity 는 detach 되지만 x_t chain 은 끊기지 않아
        # coarse step 간 BPTT (x_t 경로) 는 온전히 유지됩니다.
        # bptt_last_n_solver_steps=K 일 때 solver_steps-K 로 설정됩니다.
        self.last_n_grad_solver_steps: int = 0

    def _beta(self) -> float:
        if self.path_type == "linear":
            return 1.0
        return 1.0 - self.sigma_min

    def sigma_t(self, tau: torch.Tensor) -> torch.Tensor:
        """현재 OT path에서 ``x_0`` 앞에 곱해지는 계수를 계산합니다.

        Args:
            tau: 생성 진행률입니다. shape은 ``[batch]`` 입니다.

        Returns:
            torch.Tensor: 각 샘플의 ``sigma_t`` 입니다. shape은 ``[batch]`` 입니다.
        """
        beta = self._beta()
        return 1.0 - beta * tau

    def _sigma_t(self, tau: torch.Tensor) -> torch.Tensor:
        return self.sigma_t(tau)

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

    def drift_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """AM stochastic rollout 에 쓰는 drift 를 계산합니다.

        Args:
            x_t: 현재 상태입니다. shape은 ``[batch, 20, 4]`` 입니다.
            velocity: velocity field 값입니다. shape은 ``[batch, 20, 4]`` 입니다.
            tau: 현재 진행률입니다. shape은 ``[batch]`` 입니다.

        Returns:
            torch.Tensor: drift 입니다. shape은 ``[batch, 20, 4]`` 입니다.
        """
        tau_view = tau.view(-1, 1, 1).clamp_min(self.eps)
        return 2.0 * velocity - x_t / tau_view

    def memoryless_sigma(self, tau: torch.Tensor) -> torch.Tensor:
        """Adjoint Matching fine-tuning에 맞는 memoryless noise 크기를 계산합니다.

        Args:
            tau: 현재 진행률입니다. shape은 ``[batch]`` 입니다.

        Returns:
            torch.Tensor: 각 샘플의 noise scale 입니다. shape은 ``[batch]`` 입니다.
        """

        sigma_t = self._sigma_t(tau)
        return torch.sqrt((2.0 * sigma_t).clamp_min(0.0) / tau.clamp_min(self.eps))

    def generate(
        self,
        x_init: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        steps: Optional[int] = None,
        method: Optional[str] = None,
        post_process_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        steps = self.solver_steps if steps is None else steps
        method = self.solver_method if method is None else method
        use_adjoint = self.use_adjoint_for_bptt
        _last_n_grad = max(0, int(self.last_n_grad_solver_steps))

        if use_adjoint:
            from torch.utils.checkpoint import checkpoint as ckpt

            def _ckpt_call(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
                # use_reentrant=False: safer for non-leaf inputs; avoids version counter issues.
                return ckpt(model_fn, x, tau, use_reentrant=False)
        else:
            _ckpt_call = model_fn  # no-op alias

        x_t = x_init
        t0 = self.eps
        dt = (1.0 - t0) / float(steps)

        if _last_n_grad > 0 and not getattr(self, "_last_n_grad_logged", False):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"[FlowODE] last_n_grad_solver_steps={_last_n_grad}/{steps}: "
                f"solver steps 0~{steps - _last_n_grad - 1} velocity detached (x_t chain intact), "
                f"steps {steps - _last_n_grad}~{steps - 1} → model params gradient."
            )
            self._last_n_grad_logged = True

        for i in range(steps):
            # True 이면 이 step 의 velocity 가 model_fn 파라미터에 gradient 를 전달합니다.
            # False 이면 velocity 를 detach 해 파라미터 업데이트를 차단하지만,
            # x_t chain 자체는 끊기지 않아 coarse step 간 BPTT 는 유지됩니다.
            _keep_vel_grad = (_last_n_grad == 0) or (i >= steps - _last_n_grad)

            t = t0 + i * dt
            tau = x_t.new_full((x_t.shape[0],), t)

            if method == "midpoint":
                v1 = _ckpt_call(x_t, tau)
                if not _keep_vel_grad:
                    v1 = v1.detach()
                x_mid = x_t + 0.5 * dt * v1
                tau_mid = x_t.new_full((x_t.shape[0],), t + 0.5 * dt)
                v2 = _ckpt_call(x_mid, tau_mid)
                if not _keep_vel_grad:
                    v2 = v2.detach()
                x_t = x_t + dt * v2
            elif method == "euler":
                v = _ckpt_call(x_t, tau)
                if not _keep_vel_grad:
                    v = v.detach()
                x_t = x_t + dt * v
            else:
                raise ValueError(f"Unsupported solver method: {method}")

            if post_process_fn is not None:
                x_t = post_process_fn(x_t)

        return x_t

    @torch.no_grad()
    def generate_predict_project_renoise(
        self,
        x_init: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        project_fn: Callable[[torch.Tensor, float], torch.Tensor],
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict-Project-Renoise generation loop.

        매 step:
          1. v_θ(x_t, t) 로 x_1 예측
          2. x_1 을 feasible region으로 projection (강도는 t_next로 결정)
          3. x_0(noise)를 역산하고 projected x_1과 재결합(re-noise)하여 x_{t+dt} 생성

        ODE path: x_t = σ_t · x_0 + t · x_1,  σ_t = 1 - β·t,  β = 1 - σ_min

        Projection 강도(proj_weight):
          ``t_next`` 를 그대로 사용합니다. 마지막 step에서 ``t_next=1.0`` 이므로
          최종 출력은 완전 projection 됩니다. 초반 step은 t_next 가 작아서
          projection 영향이 적습니다. (noisy x_1_pred 를 강제로 project 하는
          부작용을 줄이는 time-adaptive schedule)

        Args:
            x_init: 초기 noise. shape ``[n, 20, 4]``.
            model_fn: ``(x_t, tau) -> velocity`` callable.
            project_fn: ``(x_1, proj_weight: float) -> x_1_feasible`` callable.
                ``proj_weight`` 는 0.0(projection 없음) ~ 1.0(완전 projection).
            steps: ODE step 수. None이면 self.solver_steps 사용.

        Returns:
            Tensor: 생성된 feasible trajectory. shape ``[n, 20, 4]``.
        """
        steps = self.solver_steps if steps is None else steps
        beta = self._beta()

        x_t = x_init
        t0 = self.eps
        dt = (1.0 - t0) / float(steps)

        for i in range(steps):
            t = t0 + i * dt
            t_next = t + dt
            tau = x_t.new_full((x_t.shape[0],), t)
            sigma_t = 1.0 - beta * t

            # 1. velocity 예측 → x_1 예측
            v = model_fn(x_t, tau)
            x_1_pred = beta * x_t + sigma_t * v  # predict_clean_from_velocity

            # 2. x_1 → feasible region으로 projection
            #    t_next 를 proj_weight 로 사용: 마지막 step(t_next=1.0)에서 완전 projection
            x_1_proj = project_fn(x_1_pred, t_next)

            # 3. x_0(noise) 역산: x_t = σ_t·x_0 + t·x_1_pred → x_0 = (x_t - t·x_1_pred) / σ_t
            sigma_t_safe = max(sigma_t, self.eps)
            x_0_est = (x_t - t * x_1_pred) / sigma_t_safe

            # 4. re-noise: x_{t+dt} = σ_{t+dt}·x_0 + (t+dt)·x_1_proj
            sigma_t_next = 1.0 - beta * t_next
            x_t = sigma_t_next * x_0_est + t_next * x_1_proj

        return x_t


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
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
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
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
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


class ResidualFlowVelocityHead(nn.Module):
    """Fine-tuning 때만 움직이는 작은 residual velocity head 입니다."""

    def __init__(self, flow_dim: int, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = flow_dim // 2 if bottleneck_dim is None else bottleneck_dim
        self.net = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

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
        self.residual_velocity_head = ResidualFlowVelocityHead(flow_dim=flow_dim)

    def _build_step_tokens(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity head 직전의 step feature 를 만듭니다.

        Args:
            anchor_hidden: anchor 문맥입니다. shape은 ``[batch, hidden_dim]`` 입니다.
            x_t_norm: 현재 noisy trajectory 입니다. shape은 ``[batch, 20, 4]`` 입니다.
            tau: 생성 진행률입니다. shape은 ``[batch]`` 입니다.

        Returns:
            torch.Tensor: 정제된 step feature 입니다. shape은 ``[batch, 20, flow_dim]`` 입니다.
        """
        context = self.context_projector(anchor_hidden)
        step_tokens, chunk_tokens, tau_emb = self.noisy_future_encoder(x_t_norm, tau)

        for block in self.chunk_mixers:
            chunk_tokens = block(chunk_tokens, context, tau_emb)

        return self.step_refiner(step_tokens, chunk_tokens, context)

    def forward_components(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Base velocity 와 residual velocity 를 함께 계산합니다.

        Args:
            anchor_hidden: anchor 문맥입니다. shape은 ``[batch, hidden_dim]`` 입니다.
            x_t_norm: 현재 noisy trajectory 입니다. shape은 ``[batch, 20, 4]`` 입니다.
            tau: 생성 진행률입니다. shape은 ``[batch]`` 입니다.

        Returns:
            dict[str, torch.Tensor]: 아래 키를 담은 사전입니다.
                - ``velocity``: base와 residual을 더한 최종 velocity 입니다.
                  shape은 ``[batch, 20, 4]`` 입니다.
                - ``base_velocity``: base velocity 입니다. shape은 ``[batch, 20, 4]`` 입니다.
                - ``residual_velocity``: residual velocity 입니다. shape은 ``[batch, 20, 4]`` 입니다.
                - ``step_tokens``: 마지막 step feature 입니다. shape은 ``[batch, 20, flow_dim]`` 입니다.
        """
        # step_tokens: [batch, 20, flow_dim]
        step_tokens = self._build_step_tokens(anchor_hidden, x_t_norm, tau)
        base_velocity = self.velocity_head(step_tokens)
        residual_velocity = self.residual_velocity_head(step_tokens)
        return {
            "velocity": base_velocity + residual_velocity,
            "base_velocity": base_velocity,
            "residual_velocity": residual_velocity,
            "step_tokens": step_tokens,
        }

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(anchor_hidden, x_t_norm, tau)["velocity"]


class ContinuousCommitBridge:
    """Bridge continuous flow output back to SMART coarse rollout state."""

    def commit(
        self,
        y_hat_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        first_chunk = y_hat_norm[:, :5].clone()
        first_chunk[..., :2] = first_chunk[..., :2] * 20.0

        cos_sin = F.normalize(first_chunk[..., 2:4], dim=-1)
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
        """학습과 같은 기준으로 다음 coarse 토큰 번호를 다시 고릅니다.

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
        next_contour = cal_polygon_contour(
            commit_pos[:, -1],
            commit_head[:, -1],
            token_agent_shape,
        )
        next_contour_local, _ = transform_to_local(
            pos_global=next_contour,
            head_global=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        return match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=next_contour_local,
            token_bank_all_veh=token_bank_all_veh,
            token_bank_all_ped=token_bank_all_ped,
            token_bank_all_cyc=token_bank_all_cyc,
            reduction="sum",
            num_k=1,
            sample_topk=False,
        )
