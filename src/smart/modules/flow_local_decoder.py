from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ) -> torch.Tensor:
        steps = self.solver_steps if steps is None else steps
        method = self.solver_method if method is None else method

        x_t = x_init
        t0 = self.eps
        dt = (1.0 - t0) / float(steps)

        for i in range(steps):
            t = t0 + i * dt
            tau = x_t.new_full((x_t.shape[0],), t)

            if method == "midpoint":
                v1 = model_fn(x_t, tau)
                x_mid = x_t + 0.5 * dt * v1
                tau_mid = x_t.new_full((x_t.shape[0],), t + 0.5 * dt)
                v2 = model_fn(x_mid, tau_mid)
                x_t = x_t + dt * v2
            elif method == "euler":
                v = model_fn(x_t, tau)
                x_t = x_t + dt * v
            else:
                raise ValueError(f"Unsupported solver method: {method}")

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
        current_contour = cal_polygon_contour(current_pos, current_head, token_agent_shape)
        future_contours = [
            cal_polygon_contour(commit_pos[:, i], commit_head[:, i], token_agent_shape)
            for i in range(commit_pos.shape[1])
        ]
        contour_global = torch.stack([current_contour] + future_contours, dim=1)

        contour_local, _ = transform_to_local(
            pos_global=contour_global.flatten(1, 2),
            head_global=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        contour_local = contour_local.view(contour_global.shape)

        token_idx = torch.zeros(
            agent_type.shape[0],
            device=agent_type.device,
            dtype=torch.long,
        )

        token_banks = {
            "veh": (agent_type == 0, token_bank_all_veh),
            "ped": (agent_type == 1, token_bank_all_ped),
            "cyc": (agent_type == 2, token_bank_all_cyc),
        }
        for _, (mask, token_bank) in token_banks.items():
            if not mask.any():
                continue

            dist = torch.norm(
                token_bank.unsqueeze(0) - contour_local[mask].unsqueeze(1),
                dim=-1,
            ).mean(dim=(-1, -2))
            token_idx[mask] = torch.argmin(dist, dim=-1)

        return token_idx
