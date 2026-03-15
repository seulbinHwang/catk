from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smart.utils import cal_polygon_contour, transform_to_local, transform_to_global, wrap_angle


@dataclass
class FlowSample:
    x_t: torch.Tensor
    target: torch.Tensor
    tau: torch.Tensor


class FlowODE:
    """Minimal linear-path flow ODE helper in normalized trajectory space."""

    def __init__(
        self,
        eps: float = 1e-3,
        solver_steps: int = 4,
        solver_method: str = "midpoint",
    ) -> None:
        self.eps = eps
        self.solver_steps = solver_steps
        self.solver_method = solver_method

    def sample(self, clean: torch.Tensor, target_type: str = "velocity") -> FlowSample:
        if target_type != "velocity":
            raise ValueError(f"Unsupported target_type: {target_type}")
        tau = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        tau = tau * (1.0 - self.eps) + self.eps
        noise = torch.randn_like(clean)
        view_tau = tau.view(-1, 1, 1)
        x_t = (1.0 - view_tau) * noise + view_tau * clean
        target = clean - noise
        return FlowSample(x_t=x_t, target=target, tau=tau)

    @staticmethod
    def predict_clean_from_velocity(
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return x_t + (1.0 - tau).view(-1, 1, 1) * velocity

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


class AnchorPrefixMemoryBuilder(nn.Module):
    """Builds a short causal prefix memory for each valid anchor.

    Each anchor sees at most the current slot and the six immediately preceding
    coarse slots, giving a fixed 7-token memory with left padding when history
    is shorter.
    """

    def __init__(self, prefix_len: int = 7) -> None:
        super().__init__()
        self.prefix_len = prefix_len

    def forward(
        self,
        ctx_hidden_pack: torch.Tensor,
        ctx_valid: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_agent, num_slots, hidden_dim = ctx_hidden_pack.shape
        num_anchor = num_slots - 1
        prefix_memory_full = ctx_hidden_pack.new_zeros(num_agent, num_anchor, self.prefix_len, hidden_dim)
        prefix_memory_mask_full = torch.zeros(
            num_agent,
            num_anchor,
            self.prefix_len,
            dtype=torch.bool,
            device=ctx_hidden_pack.device,
        )
        for anchor_offset in range(num_anchor):
            current_slot = anchor_offset + 1
            start_slot = max(0, current_slot - self.prefix_len + 1)
            prefix_hidden = ctx_hidden_pack[:, start_slot : current_slot + 1]
            prefix_valid = ctx_valid[:, start_slot : current_slot + 1]
            pad = self.prefix_len - prefix_hidden.shape[1]
            prefix_memory_full[:, anchor_offset, pad:] = prefix_hidden
            prefix_memory_mask_full[:, anchor_offset, pad:] = prefix_valid

        anchor_hidden_full = ctx_hidden_pack[:, 1:, :]
        anchor_hidden = anchor_hidden_full[anchor_mask]
        prefix_memory = prefix_memory_full[anchor_mask]
        prefix_memory_mask = prefix_memory_mask_full[anchor_mask]
        return anchor_hidden, prefix_memory, prefix_memory_mask

    def current(
        self,
        ctx_hidden_cache: torch.Tensor,
        ctx_valid: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if active_mask.sum() == 0:
            empty_hidden = ctx_hidden_cache.new_zeros((0, ctx_hidden_cache.shape[-1]))
            empty_memory = ctx_hidden_cache.new_zeros((0, self.prefix_len, ctx_hidden_cache.shape[-1]))
            empty_mask = torch.zeros((0, self.prefix_len), dtype=torch.bool, device=ctx_hidden_cache.device)
            return empty_hidden, empty_memory, empty_mask

        hidden_active = ctx_hidden_cache[active_mask]
        valid_active = ctx_valid[active_mask]
        anchor_hidden = hidden_active[:, -1]

        prefix_hidden = hidden_active[:, -self.prefix_len :]
        prefix_valid = valid_active[:, -self.prefix_len :]
        if prefix_hidden.shape[1] < self.prefix_len:
            pad = self.prefix_len - prefix_hidden.shape[1]
            prefix_hidden = F.pad(prefix_hidden, (0, 0, pad, 0))
            prefix_valid = F.pad(prefix_valid, (pad, 0), value=False)
        return anchor_hidden, prefix_hidden, prefix_valid


class NormalizedNoisyFutureChunkEncoder(nn.Module):

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
        step_tokens = step_tokens.view(batch_size, self.num_chunks, self.chunk_size, self.flow_dim)
        chunk_tokens = self.chunk_pool(step_tokens.mean(dim=2))
        return step_tokens, chunk_tokens, tau_emb


# Backward-compatible alias name used in earlier new_2 design notes.
NormalizedNoisyFutureEncoder = NormalizedNoisyFutureChunkEncoder


class ChunkMemoryCrossBlock(nn.Module):
    """Shared chunk-level block that mixes future chunks and prefix memory.

    The same block is reused multiple times so the model can strengthen memory
    re-querying without growing parameter count much.
    """

    def __init__(
        self,
        flow_dim: int,
        prefix_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.chunk_attn_norm = nn.LayerNorm(flow_dim)
        self.chunk_attn = nn.MultiheadAttention(
            embed_dim=flow_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.prefix_norm = nn.LayerNorm(prefix_dim)
        self.prefix_proj = nn.Linear(prefix_dim, flow_dim)
        self.cross_attn_norm = nn.LayerNorm(flow_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=flow_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_gate_mlp = nn.Sequential(
            nn.Linear(flow_dim * 2, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim),
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

    def forward(
        self,
        chunk_tokens: torch.Tensor,
        prefix_memory: torch.Tensor,
        prefix_memory_mask: torch.Tensor,
        context: torch.Tensor,
        tau_emb: torch.Tensor,
    ) -> torch.Tensor:
        attn_in = self.chunk_attn_norm(chunk_tokens)
        attn_out, _ = self.chunk_attn(attn_in, attn_in, attn_in, need_weights=False)
        chunk_tokens = chunk_tokens + attn_out

        prefix_tokens = self.prefix_proj(self.prefix_norm(prefix_memory))
        cross_in = self.cross_attn_norm(chunk_tokens)
        cross_out, _ = self.cross_attn(
            cross_in,
            prefix_tokens,
            prefix_tokens,
            key_padding_mask=~prefix_memory_mask,
            need_weights=False,
        )
        cross_gate = torch.sigmoid(self.cross_gate_mlp(torch.cat([context, tau_emb], dim=-1))).unsqueeze(1)
        chunk_tokens = chunk_tokens + cross_gate * cross_out

        cond = self.cond_mlp(torch.cat([context, tau_emb], dim=-1))
        scale, bias, gate = cond.chunk(3, dim=-1)
        mlp_in = self.mlp_norm(chunk_tokens)
        mlp_in = mlp_in * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        chunk_tokens = chunk_tokens + torch.sigmoid(gate).unsqueeze(1) * self.mlp(mlp_in)
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
    """Hybrid local flow decoder.

    The decoder preserves new_2's normalized 4x5 future representation while
    adding a lightweight chunk-to-prefix-memory re-query path. The cross block
    weights are shared across repeats so total parameter count stays close to
    the original 7M-class model budget.
    """

    def __init__(
        self,
        context_dim: int,
        flow_dim: int,
        num_chunk_heads: int = 4,
        num_chunk_layers: int = 2,
        prefix_dim: int = 128,
    ) -> None:
        super().__init__()
        self.context_projector = AnchorContextProjector(context_dim, flow_dim)
        self.noisy_future_encoder = NormalizedNoisyFutureChunkEncoder(flow_dim=flow_dim)
        self.chunk_memory_block = ChunkMemoryCrossBlock(
            flow_dim=flow_dim,
            prefix_dim=prefix_dim,
            num_heads=num_chunk_heads,
        )
        self.num_chunk_layers = num_chunk_layers
        self.step_refiner = ChunkStepRefiner(flow_dim=flow_dim, num_heads=num_chunk_heads)
        self.velocity_head = FlowVelocityHead(flow_dim=flow_dim)

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        prefix_memory: torch.Tensor,
        prefix_memory_mask: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_projector(anchor_hidden)
        step_tokens, chunk_tokens, tau_emb = self.noisy_future_encoder(x_t_norm, tau)
        for _ in range(self.num_chunk_layers):
            chunk_tokens = self.chunk_memory_block(
                chunk_tokens=chunk_tokens,
                prefix_memory=prefix_memory,
                prefix_memory_mask=prefix_memory_mask,
                context=context,
                tau_emb=tau_emb,
            )
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

        cos_sin = F.normalize(first_chunk[..., 2:4], dim=-1, eps=1e-6)
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
        token_traj_all: torch.Tensor,
        token_agent_shape: torch.Tensor,
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
        dist = torch.norm(token_traj_all - contour_local.unsqueeze(1), dim=-1).mean(dim=(-1, -2))
        return torch.argmin(dist, dim=-1)
