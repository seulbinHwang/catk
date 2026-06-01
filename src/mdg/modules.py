from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.mdg.geometry import (
    global_to_local_xy,
    heading_vector,
    relation_features,
    rotate_points,
    wrap_angle,
)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, dim: int, token_dim: int, channel_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.token_norm(x).transpose(-1, -2)
        y = self.token_mlp(y).transpose(-1, -2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class MLPMixerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MixerBlock(
                    seq_len=seq_len,
                    dim=hidden_dim,
                    token_dim=max(32, seq_len * 4),
                    channel_dim=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        x = self.input(x)
        x = x * valid.unsqueeze(-1).to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x)
            x = x * valid.unsqueeze(-1).to(dtype=x.dtype)
        x = self.output_norm(x)
        x = x.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        pooled = x.max(dim=-2).values
        return torch.where(valid.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))


def _fourier_relation_features(rel: Tensor, num_bands: int) -> Tensor:
    if num_bands <= 0:
        return rel
    frequencies = 2.0 ** torch.arange(num_bands, device=rel.device, dtype=rel.dtype)
    angles = rel.unsqueeze(-1) * frequencies * torch.pi
    sin = torch.sin(angles).flatten(-2)
    cos = torch.cos(angles).flatten(-2)
    return torch.cat((rel, sin, cos), dim=-1)


class RelativeMHA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_relation_freq_bands: int,
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_relation_freq_bands = int(num_relation_freq_bands)
        self.use_relation_bias = bool(use_relation_bias)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.hidden_dim = int(hidden_dim)
        self.head_dim = int(hidden_dim) // int(num_heads)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.use_relation_bias:
            relation_dim = 3 * (1 + 2 * self.num_relation_freq_bands)
            self.rel_emb = nn.Sequential(
                nn.Linear(relation_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.rel_emb = None

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        rel: Optional[Tensor] = None,
        key_valid: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, num_query, _ = query.shape
        num_key = key_value.shape[1]
        q = self.q_proj(query).view(bsz, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(bsz, num_key, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(bsz, num_key, self.num_heads, self.head_dim).transpose(1, 2)

        rel_heads = None
        if rel is not None and self.rel_emb is not None:
            rel_emb = self.rel_emb(_fourier_relation_features(rel, self.num_relation_freq_bands))
            rel_heads = rel_emb.view(bsz, num_query, num_key, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            logits = (q.unsqueeze(3) * (k.unsqueeze(2) + rel_heads)).sum(dim=-1)
        else:
            attn_mask = None
            if key_valid is not None:
                attn_mask = key_valid[:, None, None, :]
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, num_query, self.hidden_dim)
            return self.out_proj(out)

        logits = logits / (self.head_dim ** 0.5)
        if key_valid is not None:
            logits = logits.masked_fill(~key_valid[:, None, None, :], -1.0e4)
        attn = self.dropout(torch.softmax(logits, dim=-1))

        out = (attn.unsqueeze(-1) * (v.unsqueeze(2) + rel_heads)).sum(dim=3)
        out = out.transpose(1, 2).contiguous().view(bsz, num_query, self.hidden_dim)
        return self.out_proj(out)

    def _shared_relation_heads(self, rel: Tensor) -> Tensor:
        if self.rel_emb is None:
            raise RuntimeError("Time-shared relation attention requires relation bias parameters.")
        bsz, num_query, num_key = rel.shape[:3]
        rel_emb = self.rel_emb(_fourier_relation_features(rel, self.num_relation_freq_bands))
        return rel_emb.view(bsz, num_query, num_key, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

    def forward_time_shared_self(
        self,
        query: Tensor,
        key_value: Tensor,
        rel: Tensor,
        key_valid: Optional[Tensor] = None,
    ) -> Tensor:
        """Relation-aware attention over agents with a separate time axis.

        ``query`` and ``key_value`` have shape ``[B, T, N, D]`` while ``rel`` has
        shape ``[B, N, N, 3]``. This is mathematically equivalent to flattening
        ``[B, T]`` and repeating ``rel`` for each timestep, but it embeds the
        relation tensor only once per scene.
        """
        bsz, num_time, num_query, _ = query.shape
        num_key = int(key_value.shape[2])
        q = self.q_proj(query).view(
            bsz,
            num_time,
            num_query,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        k = self.k_proj(key_value).view(
            bsz,
            num_time,
            num_key,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        v = self.v_proj(key_value).view(
            bsz,
            num_time,
            num_key,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        rel_heads = self._shared_relation_heads(rel)

        logits = torch.matmul(q, k.transpose(-2, -1))
        logits = logits + torch.einsum("bthqd,bhqkd->bthqk", q, rel_heads)
        logits = logits / (self.head_dim ** 0.5)
        if key_valid is not None:
            logits = logits.masked_fill(~key_valid[:, None, None, None, :], -1.0e4)
        attn = self.dropout(torch.softmax(logits, dim=-1))

        out = torch.matmul(attn, v)
        out = out + torch.einsum("bthqk,bhqkd->bthqd", attn, rel_heads)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(
            bsz,
            num_time,
            num_query,
            self.hidden_dim,
        )
        return self.out_proj(out)

    def forward_time_shared_scene(
        self,
        query: Tensor,
        key_value: Tensor,
        rel: Tensor,
        key_valid: Optional[Tensor] = None,
    ) -> Tensor:
        """Relation-aware attention from agent-time queries to scene tokens.

        ``query`` has shape ``[B, N, T, D]`` and ``rel`` has shape
        ``[B, N, S, 3]``. Relation embeddings are shared across the ``T`` action
        timesteps instead of materialized as ``[B, N*T, S, D]``.
        """
        bsz, num_agents, num_time, _ = query.shape
        num_key = int(key_value.shape[1])
        q = self.q_proj(query).view(
            bsz,
            num_agents,
            num_time,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4)
        k = self.k_proj(key_value).view(
            bsz,
            num_key,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).view(
            bsz,
            num_key,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        rel_heads = self._shared_relation_heads(rel)

        logits = torch.einsum("bthnd,bhsd->bthns", q, k)
        logits = logits + torch.einsum("bthnd,bhnsd->bthns", q, rel_heads)
        logits = logits / (self.head_dim ** 0.5)
        if key_valid is not None:
            logits = logits.masked_fill(~key_valid[:, None, None, None, :], -1.0e4)
        attn = self.dropout(torch.softmax(logits, dim=-1))

        out = torch.einsum("bthns,bhsd->bthnd", attn, v)
        out = out + torch.einsum("bthns,bhnsd->bthnd", attn, rel_heads)
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(
            bsz,
            num_agents,
            num_time,
            self.hidden_dim,
        )
        return self.out_proj(out)


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_relation_freq_bands: int,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = RelativeMHA(hidden_dim, num_heads, dropout, num_relation_freq_bands)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(self, x: Tensor, rel: Optional[Tensor], valid: Tensor) -> Tensor:
        y = self.attn(self.attn_norm(x), self.attn_norm(x), rel=rel, key_valid=valid)
        x = x + self.attn_drop(y)
        x = x + self.ffn(self.ffn_norm(x))
        return x * valid.unsqueeze(-1).to(dtype=x.dtype)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_relation_freq_bands: int,
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = RelativeMHA(
            hidden_dim,
            num_heads,
            dropout,
            num_relation_freq_bands,
            use_relation_bias=use_relation_bias,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        rel: Optional[Tensor],
        key_valid: Optional[Tensor],
        query_valid: Optional[Tensor] = None,
    ) -> Tensor:
        y = self.attn(
            self.q_norm(query),
            self.kv_norm(key_value),
            rel=rel,
            key_valid=key_valid,
        )
        query = query + self.attn_drop(y)
        query = query + self.ffn(self.ffn_norm(query))
        if query_valid is not None:
            query = query * query_valid.unsqueeze(-1).to(dtype=query.dtype)
        return query

    def forward_time_shared_self(
        self,
        query: Tensor,
        rel: Tensor,
        key_valid: Tensor,
    ) -> Tensor:
        y = self.attn.forward_time_shared_self(
            self.q_norm(query),
            self.kv_norm(query),
            rel=rel,
            key_valid=key_valid,
        )
        query = query + self.attn_drop(y)
        query = query + self.ffn(self.ffn_norm(query))
        return query * key_valid[:, None, :, None].to(dtype=query.dtype)

    def forward_time_shared_scene(
        self,
        query: Tensor,
        key_value: Tensor,
        rel: Tensor,
        key_valid: Tensor,
        query_valid: Tensor,
    ) -> Tensor:
        y = self.attn.forward_time_shared_scene(
            self.q_norm(query),
            self.kv_norm(key_value),
            rel=rel,
            key_valid=key_valid,
        )
        query = query + self.attn_drop(y)
        query = query + self.ffn(self.ffn_norm(query))
        return query * query_valid[:, :, None, None].to(dtype=query.dtype)


@dataclass
class SceneEncoding:
    context: Tensor
    valid: Tensor
    agent_context: Tensor
    agent_valid: Tensor
    agent_anchor_pos: Tensor
    agent_anchor_heading: Tensor
    scene_anchor_pos: Tensor
    scene_anchor_heading: Tensor


class SceneEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_encoder_layers: int,
        num_mixer_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        history_steps: int,
        map_waypoints: int,
        num_relation_freq_bands: int,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.map_waypoints = int(map_waypoints)
        self.agent_encoder = MLPMixerEncoder(10, history_steps, hidden_dim, num_mixer_layers, dropout)
        self.map_encoder = MLPMixerEncoder(4, map_waypoints, hidden_dim, num_mixer_layers, dropout)
        self.signal_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agent_type_emb = nn.Embedding(3, hidden_dim)
        self.map_type_emb = nn.Embedding(16, hidden_dim)
        self.map_light_emb = nn.Embedding(9, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(
                    hidden_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    num_relation_freq_bands,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def encode_agents(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pos = batch["agent_position"][:, :, : self.history_steps, :2]
        heading = batch["agent_heading"][:, :, : self.history_steps]
        velocity = batch["agent_velocity"][:, :, : self.history_steps, :2]
        valid = batch["agent_valid_mask"][:, :, : self.history_steps] & batch["agent_valid"].unsqueeze(-1)
        shape = batch["agent_shape"]

        anchor_pos = pos[:, :, -1]
        anchor_heading = heading[:, :, -1]
        local_pos = global_to_local_xy(pos, anchor_pos, anchor_heading)
        local_vel = rotate_points(velocity, -anchor_heading.unsqueeze(-1))
        local_heading = wrap_angle(heading - anchor_heading.unsqueeze(-1))
        shape_feat = shape.unsqueeze(2).expand(-1, -1, self.history_steps, -1)
        features = torch.cat(
            (
                local_pos,
                torch.cos(local_heading).unsqueeze(-1),
                torch.sin(local_heading).unsqueeze(-1),
                local_vel,
                shape_feat,
                valid.unsqueeze(-1).to(dtype=pos.dtype),
            ),
            dim=-1,
        )
        bsz, num_agents = pos.shape[:2]
        encoded = self.agent_encoder(
            features.reshape(bsz * num_agents, self.history_steps, -1),
            valid.reshape(bsz * num_agents, self.history_steps),
        ).reshape(bsz, num_agents, -1)
        encoded = encoded + self.agent_type_emb(batch["agent_type"].clamp(0, 2))
        return encoded, batch["agent_valid"], anchor_pos, anchor_heading

    def encode_map(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pos = batch["map_position"]
        heading = batch["map_heading"]
        valid = batch["map_valid"]
        anchor_pos = pos[:, :, 0]
        anchor_heading = heading[:, :, 0]
        local_pos = global_to_local_xy(pos, anchor_pos, anchor_heading)
        local_heading = wrap_angle(heading - anchor_heading.unsqueeze(-1))
        features = torch.cat(
            (
                local_pos,
                torch.cos(local_heading).unsqueeze(-1),
                torch.sin(local_heading).unsqueeze(-1),
            ),
            dim=-1,
        )
        point_valid = valid.unsqueeze(-1).expand(-1, -1, self.map_waypoints)
        bsz, num_polylines = pos.shape[:2]
        encoded = self.map_encoder(
            features.reshape(bsz * num_polylines, self.map_waypoints, -1),
            point_valid.reshape(bsz * num_polylines, self.map_waypoints),
        ).reshape(bsz, num_polylines, -1)
        encoded = (
            encoded
            + self.map_type_emb(batch["map_type"].clamp(0, 15))
            + self.map_light_emb(batch["map_light_type"].clamp(0, 8))
        )
        return encoded, valid, anchor_pos, anchor_heading

    def encode_signals(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = batch["signal_state"].clamp(0, 8)
        valid = batch["signal_valid"]
        pos = batch["signal_position"]
        heading = batch["signal_heading"]
        phase = F.one_hot(state, num_classes=9).to(dtype=pos.dtype)
        encoded = self.signal_mlp(phase)
        return encoded, valid, pos, heading

    def forward(self, batch: dict[str, Tensor]) -> SceneEncoding:
        agent_x, agent_valid, agent_pos, agent_heading = self.encode_agents(batch)
        map_x, map_valid, map_pos, map_heading = self.encode_map(batch)
        signal_x, signal_valid, signal_pos, signal_heading = self.encode_signals(batch)

        context = torch.cat((agent_x, map_x, signal_x), dim=1)
        valid = torch.cat((agent_valid, map_valid, signal_valid), dim=1)
        anchor_pos = torch.cat((agent_pos, map_pos, signal_pos), dim=1)
        anchor_heading = torch.cat((agent_heading, map_heading, signal_heading), dim=1)
        rel = relation_features(anchor_pos, anchor_heading, anchor_pos, anchor_heading)

        context = context * valid.unsqueeze(-1).to(dtype=context.dtype)
        for block in self.blocks:
            context = block(context, rel=rel, valid=valid)

        num_agents = int(agent_x.shape[1])
        return SceneEncoding(
            context=context,
            valid=valid,
            agent_context=context[:, :num_agents],
            agent_valid=agent_valid,
            agent_anchor_pos=agent_pos,
            agent_anchor_heading=agent_heading,
            scene_anchor_pos=anchor_pos,
            scene_anchor_heading=anchor_heading,
        )


class KinematicDynamics(nn.Module):
    def __init__(self, action_chunk: int, dt: float, action_mean: tuple[float, float], action_std: tuple[float, float]) -> None:
        super().__init__()
        self.action_chunk = int(action_chunk)
        self.dt = float(dt)
        self.register_buffer("action_mean", torch.tensor(action_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("action_std", torch.tensor(action_std, dtype=torch.float32), persistent=False)

    def denormalize(self, action: Tensor) -> Tensor:
        return action * self.action_std.to(dtype=action.dtype, device=action.device) + self.action_mean.to(
            dtype=action.dtype, device=action.device
        )

    def normalize(self, action: Tensor) -> Tensor:
        return (action - self.action_mean.to(dtype=action.dtype, device=action.device)) / self.action_std.to(
            dtype=action.dtype, device=action.device
        )

    def trajectory_to_actions(
        self,
        current_pos: Tensor,
        current_heading: Tensor,
        current_speed: Tensor,
        future_pos: Tensor,
        future_heading: Tensor,
        future_velocity: Tensor,
    ) -> Tensor:
        _, _, future_steps = future_heading.shape
        if future_steps % self.action_chunk != 0:
            raise ValueError(
                "future_steps must be divisible by action_chunk, "
                f"got future_steps={future_steps}, action_chunk={self.action_chunk}."
            )
        action_steps = future_steps // self.action_chunk
        speed = torch.linalg.norm(future_velocity[:, :, :, :2], dim=-1)
        prev_speed = torch.cat((current_speed.unsqueeze(-1), speed[:, :, :-1]), dim=-1)
        prev_heading = torch.cat((current_heading.unsqueeze(-1), future_heading[:, :, :-1]), dim=-1)
        per_step_acc = (speed - prev_speed) / self.dt
        per_step_yaw_rate = wrap_angle(future_heading - prev_heading) / self.dt
        per_step_action = torch.stack((per_step_acc, per_step_yaw_rate), dim=-1)
        chunk_action = per_step_action.reshape(
            *per_step_action.shape[:2],
            action_steps,
            self.action_chunk,
            2,
        ).mean(dim=3)
        return self.normalize(chunk_action)

    def forward(
        self,
        action: Tensor,
        current_pos: Tensor,
        current_heading: Tensor,
        current_speed: Tensor,
        current_velocity: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        action = self.denormalize(action)
        acc = action[..., 0]
        yaw_rate = action[..., 1]
        initial_velocity = (
            current_velocity
            if current_velocity is not None
            else heading_vector(current_heading) * current_speed.unsqueeze(-1)
        )
        full_acc = acc.repeat_interleave(self.action_chunk, dim=2)
        full_yaw_rate = yaw_rate.repeat_interleave(self.action_chunk, dim=2)

        full_heading = wrap_angle(
            current_heading.unsqueeze(-1) + torch.cumsum(full_yaw_rate * self.dt, dim=2)
        )
        full_speed = current_speed.unsqueeze(-1) + torch.cumsum(full_acc * self.dt, dim=2)

        updated_velocity = heading_vector(full_heading) * full_speed.unsqueeze(-1)
        velocity_for_position = torch.cat((initial_velocity.unsqueeze(2), updated_velocity[:, :, :-1]), dim=2)
        full_pos = current_pos.unsqueeze(2) + torch.cumsum(velocity_for_position * self.dt, dim=2)

        chunk_end_idx = torch.arange(
            self.action_chunk - 1,
            full_pos.shape[2],
            self.action_chunk,
            device=full_pos.device,
        )
        chunk_pos = full_pos.index_select(2, chunk_end_idx)
        chunk_heading = full_heading.index_select(2, chunk_end_idx)
        chunk_speed = full_speed.index_select(2, chunk_end_idx)
        rel_pos = chunk_pos - current_pos.unsqueeze(2)
        chunk_state = torch.cat(
            (
                rel_pos,
                torch.cos(chunk_heading).unsqueeze(-1),
                torch.sin(chunk_heading).unsqueeze(-1),
                chunk_speed.unsqueeze(-1),
            ),
            dim=-1,
        )
        return (
            full_pos,
            full_heading,
            full_speed,
            chunk_state,
            action,
        )


class DenoiserBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_relation_freq_bands: int,
    ) -> None:
        super().__init__()
        self.temporal = CrossAttentionBlock(
            hidden_dim,
            num_heads,
            ffn_dim,
            dropout,
            0,
            use_relation_bias=False,
        )
        self.inter_agent = CrossAttentionBlock(
            hidden_dim,
            num_heads,
            ffn_dim,
            dropout,
            num_relation_freq_bands,
        )
        self.scene = CrossAttentionBlock(
            hidden_dim,
            num_heads,
            ffn_dim,
            dropout,
            num_relation_freq_bands,
        )

    def forward(
        self,
        x: Tensor,
        agent_valid: Tensor,
        scene: SceneEncoding,
    ) -> Tensor:
        bsz, num_agents, action_steps, hidden_dim = x.shape
        temporal = x.reshape(bsz * num_agents, action_steps, hidden_dim)
        temporal = self.temporal(temporal, temporal, rel=None, key_valid=None)
        x = temporal.reshape(bsz, num_agents, action_steps, hidden_dim)

        inter = x.permute(0, 2, 1, 3)
        rel_aa = relation_features(
            scene.agent_anchor_pos,
            scene.agent_anchor_heading,
            scene.agent_anchor_pos,
            scene.agent_anchor_heading,
        )
        inter = self.inter_agent.forward_time_shared_self(
            inter,
            rel=rel_aa,
            key_valid=agent_valid,
        )
        x = inter.permute(0, 2, 1, 3)

        rel_as = relation_features(
            scene.agent_anchor_pos,
            scene.agent_anchor_heading,
            scene.scene_anchor_pos,
            scene.scene_anchor_heading,
            self_relation_value=None,
        )
        diag = torch.arange(num_agents, device=rel_as.device)
        rel_as[:, diag, diag] = 1.0e-4
        x = self.scene.forward_time_shared_scene(
            x,
            scene.context,
            rel=rel_as,
            key_valid=scene.valid,
            query_valid=agent_valid,
        )
        return x * agent_valid[:, :, None, None].to(dtype=x.dtype)


class MDGDenoiser(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_steps: int,
        num_noise_levels: int,
        num_blocks: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_relation_freq_bands: int,
    ) -> None:
        super().__init__()
        self.action_steps = int(action_steps)
        self.state_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mask_emb = nn.Embedding(num_noise_levels, hidden_dim)
        self.time_emb = nn.Embedding(action_steps, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(
                    hidden_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    num_relation_freq_bands,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, noised_state: Tensor, mask_level: Tensor, scene: SceneEncoding) -> Tensor:
        bsz, num_agents, action_steps = mask_level.shape
        time_idx = torch.arange(action_steps, device=mask_level.device)
        x = self.state_mlp(noised_state)
        x = x + self.mask_emb(mask_level)
        x = x + self.time_emb(time_idx).view(1, 1, action_steps, -1)
        x = x * scene.agent_valid[:, :, None, None].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, scene.agent_valid, scene)
        return self.out(x) * scene.agent_valid[:, :, None, None].to(dtype=x.dtype)


class AuxiliaryPredictor(nn.Module):
    def __init__(self, hidden_dim: int, modes: int, future_steps: int) -> None:
        super().__init__()
        self.modes = int(modes)
        self.future_steps = int(future_steps)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, modes * future_steps * 3),
        )

    def forward(self, agent_context: Tensor) -> Tensor:
        out = self.head(agent_context)
        return out.view(*agent_context.shape[:2], self.modes, self.future_steps, 3)


class MDGBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        history_steps: int = 11,
        future_steps: int = 80,
        action_chunk: int = 2,
        map_waypoints: int = 16,
        num_noise_levels: int = 5,
        num_mixer_layers: int = 2,
        num_encoder_layers: int = 6,
        num_denoiser_blocks: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        predictor_modes: int = 6,
        num_relation_freq_bands: int = 4,
        action_mean: tuple[float, float] = (0.0, 0.0),
        action_std: tuple[float, float] = (1.0, 0.5),
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.action_chunk = int(action_chunk)
        self.action_steps = self.future_steps // self.action_chunk
        self.num_noise_levels = int(num_noise_levels)
        self.scene_encoder = SceneEncoder(
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_mixer_layers=num_mixer_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            history_steps=history_steps,
            map_waypoints=map_waypoints,
            num_relation_freq_bands=num_relation_freq_bands,
        )
        self.dynamics = KinematicDynamics(action_chunk, dt=0.1, action_mean=action_mean, action_std=action_std)
        self.denoiser = MDGDenoiser(
            hidden_dim=hidden_dim,
            action_steps=self.action_steps,
            num_noise_levels=num_noise_levels,
            num_blocks=num_denoiser_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_relation_freq_bands=num_relation_freq_bands,
        )
        self.aux_predictor = AuxiliaryPredictor(hidden_dim, predictor_modes, future_steps)

    def current_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pos = batch["agent_position"][:, :, self.history_steps - 1, :2]
        heading = batch["agent_heading"][:, :, self.history_steps - 1]
        velocity = batch["agent_velocity"][:, :, self.history_steps - 1, :2]
        speed = torch.linalg.norm(velocity, dim=-1)
        return pos, heading, speed, velocity

    def denoise_actions(
        self,
        batch: dict[str, Tensor],
        noised_action: Tensor,
        mask_level: Tensor,
        scene: Optional[SceneEncoding] = None,
        compute_aux: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SceneEncoding, Optional[Tensor]]:
        if scene is None:
            scene = self.scene_encoder(batch)
        current_pos, current_heading, current_speed, current_velocity = self.current_state(batch)
        _, _, _, noised_state, _ = self.dynamics(
            noised_action,
            current_pos,
            current_heading,
            current_speed,
            current_velocity=current_velocity,
        )
        pred_action = self.denoiser(noised_state, mask_level, scene)
        pred_pos, pred_heading, pred_speed, pred_chunk_state, denorm_action = self.dynamics(
            pred_action,
            current_pos,
            current_heading,
            current_speed,
            current_velocity=current_velocity,
        )
        aux = self.aux_predictor(scene.agent_context) if compute_aux else None
        return pred_action, pred_pos, pred_heading, pred_speed, pred_chunk_state, scene, aux

    def clean_actions_from_batch(self, batch: dict[str, Tensor]) -> Tensor:
        current_pos, current_heading, current_speed, _ = self.current_state(batch)
        future_pos = batch["agent_position"][:, :, self.history_steps :, :2]
        future_heading = batch["agent_heading"][:, :, self.history_steps :]
        future_velocity = batch["agent_velocity"][:, :, self.history_steps :, :2]
        return self.dynamics.trajectory_to_actions(
            current_pos=current_pos,
            current_heading=current_heading,
            current_speed=current_speed,
            future_pos=future_pos,
            future_heading=future_heading,
            future_velocity=future_velocity,
        )

    def clean_actions_and_chunk_state_from_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        current_pos, current_heading, current_speed, current_velocity = self.current_state(batch)
        future_pos = batch["agent_position"][:, :, self.history_steps :, :2]
        future_heading = batch["agent_heading"][:, :, self.history_steps :]
        future_velocity = batch["agent_velocity"][:, :, self.history_steps :, :2]
        clean_action = self.dynamics.trajectory_to_actions(
            current_pos=current_pos,
            current_heading=current_heading,
            current_speed=current_speed,
            future_pos=future_pos,
            future_heading=future_heading,
            future_velocity=future_velocity,
        )
        _, _, _, clean_chunk_state, _ = self.dynamics(
            clean_action,
            current_pos,
            current_heading,
            current_speed,
            current_velocity=current_velocity,
        )
        return clean_action, clean_chunk_state

    def full_noise_sample(
        self,
        batch: dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        shape = (
            batch["agent_position"].shape[0],
            batch["agent_position"].shape[1],
            self.action_steps,
            2,
        )
        noise = torch.randn(
            shape,
            device=batch["agent_position"].device,
            dtype=batch["agent_position"].dtype,
            generator=generator,
        )
        mask = torch.full(shape[:-1], self.num_noise_levels - 1, dtype=torch.long, device=noise.device)
        return noise, mask
