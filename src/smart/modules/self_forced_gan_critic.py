from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RolloutSetCriticShapes:
    """Discriminator 입력/출력 shape 요약입니다.

    Attributes:
        rollout_pose: rollout set 입력 shape입니다. ``[B, K, T, N, 4]`` 입니다.
        current_pose: 현재 pose shape입니다. ``[B, N, 4]`` 입니다.
        agent_context: frozen scene encoder에서 나온 agent 조건 token shape입니다.
            ``[B, N, H]`` 입니다.
        map_context: frozen scene encoder에서 나온 map 조건 token shape입니다.
            ``[B, M, H]`` 입니다.
        output_logit: 최종 score shape입니다. ``[B, 1]`` 입니다.
    """

    rollout_pose: tuple[int, int, int, int, int]
    current_pose: tuple[int, int, int]
    agent_context: tuple[int, int, int]
    map_context: tuple[int, int, int]
    output_logit: tuple[int, int]


class SmallMLP(nn.Module):
    """작은 MLP 블록입니다."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PoolingProjection(nn.Module):
    """mean/max pooling concat 결과를 작게 투영하는 1-layer projection입니다."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TemporalResidualBlock(nn.Module):
    """20-step trajectory를 가볍게 섞는 temporal residual block입니다."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """temporal block을 적용합니다.

        Args:
            x: trajectory token입니다. shape은 ``[B, K, T, N, H]`` 입니다.

        Returns:
            Tensor: 같은 shape의 trajectory token입니다.
        """
        bsz, n_rollout, n_step, n_agent, hidden_dim = x.shape
        residual = x
        y = self.temporal_norm(x)
        y = y.permute(0, 1, 3, 4, 2).reshape(-1, hidden_dim, n_step)
        y = self.temporal_conv(y)
        y = y.reshape(bsz, n_rollout, n_agent, hidden_dim, n_step)
        y = y.permute(0, 1, 4, 2, 3).contiguous()
        x = residual + y
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _safe_masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """mask가 있는 평균을 계산합니다."""
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return (x * mask_f).sum(dim=dim) / denom


def _safe_masked_max(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """mask가 있는 max를 계산합니다. 전부 invalid면 0으로 둡니다."""
    neg_large = torch.finfo(x.dtype).min
    masked = x.masked_fill(~mask, neg_large)
    value = masked.max(dim=dim).values
    has_any = mask.any(dim=dim)
    return torch.where(has_any, value, torch.zeros_like(value))


def _safe_masked_min(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """mask가 있는 min을 계산합니다. 전부 invalid면 0으로 둡니다."""
    pos_large = torch.finfo(x.dtype).max
    masked = x.masked_fill(~mask, pos_large)
    value = masked.min(dim=dim).values
    has_any = mask.any(dim=dim)
    return torch.where(has_any, value, torch.zeros_like(value))


def _safe_masked_std(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """mask가 있는 표준편차를 계산합니다. 전부 invalid면 0으로 둡니다."""
    mean = _safe_masked_mean(x, mask, dim=dim)
    centered = x - mean.unsqueeze(dim)
    var = _safe_masked_mean(centered.square(), mask, dim=dim)
    return torch.sqrt(var.clamp_min(1.0e-12))


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_from_pose(pose: Tensor) -> Tensor:
    return torch.atan2(pose[..., 3], pose[..., 2])


def _build_relation_features(
    *,
    delta_xy: Tensor,
    receiver_yaw: Tensor,
    sender_yaw: Tensor,
    radius_m: float,
) -> tuple[Tensor, Tensor]:
    """distance, bearing, relative heading relation feature를 만듭니다."""
    distance = torch.sqrt(delta_xy.square().sum(dim=-1).clamp_min(1.0e-12))
    bearing_global = torch.atan2(delta_xy[..., 1], delta_xy[..., 0] + 1.0e-6)
    bearing = _wrap_angle(bearing_global - receiver_yaw)
    relative_heading = _wrap_angle(sender_yaw - receiver_yaw)
    relation = torch.stack(
        [
            distance / max(float(radius_m), 1.0e-6),
            bearing / math.pi,
            relative_heading / math.pi,
        ],
        dim=-1,
    )
    return relation, distance


class RadiusAttentionLayer(nn.Module):
    """radius mask와 기하 relation bias를 쓰는 작은 multi-head attention입니다."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        relation_dim: int = 3,
        sender_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sender_chunk_size = max(0, int(sender_chunk_size))

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _streaming_context(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        relation: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """sender chunk 단위의 stable softmax attention context를 계산합니다."""
        bsz, n_rollout, n_endpoint, n_query, n_head, head_dim = q.shape
        n_sender = k.shape[3]
        chunk_size = max(1, self.sender_chunk_size)
        device = q.device
        max_logit = torch.full(
            (bsz, n_rollout, n_endpoint, n_query, n_head),
            -torch.inf,
            device=device,
            dtype=torch.float32,
        )
        normalizer = torch.zeros_like(max_logit)
        accumulator = torch.zeros(
            bsz,
            n_rollout,
            n_endpoint,
            n_query,
            n_head,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        for start in range(0, n_sender, chunk_size):
            end = min(start + chunk_size, n_sender)
            logits = torch.einsum("bkeqhd,bkeshd->bkeqsh", q, k[..., start:end, :, :]).float()
            logits = logits * self.scale
            logits = logits + self.relation_bias(relation[..., start:end, :]).float()
            mask = attention_mask[..., start:end].unsqueeze(-1)
            masked_logits = logits.masked_fill(~mask, -torch.inf)

            chunk_max = masked_logits.amax(dim=4)
            safe_chunk_max = torch.where(torch.isfinite(chunk_max), chunk_max, torch.zeros_like(chunk_max))
            exp_logits = torch.exp(masked_logits - safe_chunk_max.unsqueeze(4))
            exp_logits = exp_logits * mask.to(dtype=exp_logits.dtype)
            chunk_sum = exp_logits.sum(dim=4)
            chunk_context = torch.einsum(
                "bkeqsh,bkeshd->bkeqhd",
                exp_logits.to(dtype=v.dtype),
                v[..., start:end, :, :],
            ).float()

            next_max = torch.maximum(max_logit, chunk_max)
            safe_next_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
            old_scale = torch.where(
                torch.isfinite(max_logit),
                torch.exp(max_logit - safe_next_max),
                torch.zeros_like(max_logit),
            )
            chunk_scale = torch.where(
                torch.isfinite(chunk_max),
                torch.exp(chunk_max - safe_next_max),
                torch.zeros_like(chunk_max),
            )
            accumulator = accumulator * old_scale.unsqueeze(-1) + chunk_context * chunk_scale.unsqueeze(-1)
            normalizer = normalizer * old_scale + chunk_sum * chunk_scale
            max_logit = next_max

        return accumulator / normalizer.clamp_min(1.0e-6).unsqueeze(-1)

    def forward(
        self,
        *,
        query_token: Tensor,
        key_token: Tensor,
        value_token: Tensor,
        relation: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """attention을 적용합니다.

        Args:
            query_token: receiver token입니다. shape은 ``[B, K, E, Nq, H]`` 입니다.
            key_token: sender key token입니다. shape은 ``[B, K, E, Ns, H]`` 입니다.
            value_token: sender value token입니다. shape은 ``[B, K, E, Ns, H]`` 입니다.
            relation: sender-receiver relation입니다. shape은 ``[B, K, E, Nq, Ns, 3]`` 입니다.
            attention_mask: valid sender-receiver mask입니다. shape은 ``[B, K, E, Nq, Ns]`` 입니다.

        Returns:
            Tensor: attention update를 더한 receiver token입니다. shape은 ``[B, K, E, Nq, H]`` 입니다.
        """
        bsz, n_rollout, n_endpoint, n_query, hidden_dim = query_token.shape
        n_sender = key_token.shape[3]
        q = self.query(query_token).reshape(
            bsz, n_rollout, n_endpoint, n_query, self.num_heads, self.head_dim
        )
        k = self.key(key_token).reshape(
            bsz, n_rollout, n_endpoint, n_sender, self.num_heads, self.head_dim
        )
        v = self.value(value_token).reshape(
            bsz, n_rollout, n_endpoint, n_sender, self.num_heads, self.head_dim
        )

        if self.sender_chunk_size > 0 and n_sender > self.sender_chunk_size:
            context = self._streaming_context(
                q=q,
                k=k,
                v=v,
                relation=relation,
                attention_mask=attention_mask,
            ).to(dtype=query_token.dtype)
        else:
            logits = torch.einsum("bkeqhd,bkeshd->bkeqsh", q, k) * self.scale
            logits = logits + self.relation_bias(relation)
            mask = attention_mask.unsqueeze(-1)
            masked_logits = logits.masked_fill(~mask, -1.0e4)
            weights = torch.softmax(masked_logits, dim=4)
            weights = weights * mask.to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=4, keepdim=True).clamp_min(1.0e-6)
            context = torch.einsum("bkeqsh,bkeshd->bkeqhd", weights, v)
        context = context.reshape(bsz, n_rollout, n_endpoint, n_query, hidden_dim)
        return self.norm(query_token + self.output(context))


class EndpointPool(nn.Module):
    """4개 endpoint token을 mean+max pooling으로 agent token 하나로 줄입니다."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = PoolingProjection(hidden_dim * 2, hidden_dim)

    def forward(self, endpoint_token: Tensor) -> Tensor:
        endpoint_mean = endpoint_token.mean(dim=2)
        endpoint_max = endpoint_token.amax(dim=2)
        return self.projection(torch.cat([endpoint_mean, endpoint_max], dim=-1))


class SceneConditionFusion(nn.Module):
    """trajectory token에 frozen agent scene context를 붙입니다."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fusion = SmallMLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, agent_token: Tensor, agent_context: Tensor) -> Tensor:
        agent_context_rollout = agent_context.unsqueeze(1).expand(-1, agent_token.shape[1], -1, -1)
        return self.fusion(torch.cat([agent_token, agent_context_rollout], dim=-1))


class MapComplianceEncoder(nn.Module):
    """4개 endpoint에서 map token과 radius cross-attention을 수행합니다."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        radius_m: float = 30.0,
        num_heads: int = 4,
        query_chunk_size: int = 16,
        sender_chunk_size: int = 0,
        rollout_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.radius_m = float(radius_m)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.rollout_chunk_size = max(0, int(rollout_chunk_size))
        self.attention = RadiusAttentionLayer(
            hidden_dim,
            num_heads=num_heads,
            relation_dim=3,
            sender_chunk_size=sender_chunk_size,
        )
        self.endpoint_pool = EndpointPool(hidden_dim)

    def forward(
        self,
        *,
        endpoint_token: Tensor,
        endpoint_pose: Tensor,
        map_context: Tensor,
        map_position: Tensor,
        map_orientation: Tensor,
        map_valid_mask: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """map-aware endpoint token과 pooled agent token을 반환합니다."""
        bsz, n_rollout, n_endpoint, n_agent, hidden_dim = endpoint_token.shape
        n_map = map_context.shape[1]
        if n_map == 0:
            return endpoint_token, self.endpoint_pool(endpoint_token)

        map_xy = map_position[:, None, None, None, :, :]
        sender_yaw = map_orientation[:, None, None, None, :]
        rollout_step = self.rollout_chunk_size if self.rollout_chunk_size > 0 else n_rollout
        rollout_chunks: list[Tensor] = []
        for rollout_start in range(0, n_rollout, rollout_step):
            rollout_end = min(rollout_start + rollout_step, n_rollout)
            rollout_width = rollout_end - rollout_start
            map_token = map_context[:, None, None, :, :].expand(
                bsz, rollout_width, n_endpoint, n_map, hidden_dim
            )
            agent_chunks: list[Tensor] = []
            for start in range(0, n_agent, self.query_chunk_size):
                end = min(start + self.query_chunk_size, n_agent)
                endpoint_pose_chunk = endpoint_pose[:, rollout_start:rollout_end, :, start:end, :]
                endpoint_xy = endpoint_pose_chunk[..., :2]
                delta_xy = map_xy - endpoint_xy.unsqueeze(4)
                receiver_yaw = _yaw_from_pose(endpoint_pose_chunk).unsqueeze(4)
                relation, distance = _build_relation_features(
                    delta_xy=delta_xy,
                    receiver_yaw=receiver_yaw,
                    sender_yaw=sender_yaw,
                    radius_m=self.radius_m,
                )
                attention_mask = (
                    valid_mask[:, None, None, start:end, None]
                    & map_valid_mask[:, None, None, None, :]
                    & (distance <= self.radius_m)
                )
                agent_chunks.append(
                    self.attention(
                        query_token=endpoint_token[:, rollout_start:rollout_end, :, start:end, :],
                        key_token=map_token,
                        value_token=map_token,
                        relation=relation,
                        attention_mask=attention_mask,
                    )
                )
            rollout_chunks.append(torch.cat(agent_chunks, dim=3))
        map_aware_endpoint = torch.cat(rollout_chunks, dim=1)
        map_agent_token = self.endpoint_pool(map_aware_endpoint)
        return map_aware_endpoint, map_agent_token


class InteractionEncoder(nn.Module):
    """4개 endpoint에서 agent-agent radius attention을 수행합니다."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        radius_m: float = 60.0,
        num_heads: int = 4,
        query_chunk_size: int = 16,
        sender_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.radius_m = float(radius_m)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.attention = RadiusAttentionLayer(
            hidden_dim,
            num_heads=num_heads,
            relation_dim=3,
            sender_chunk_size=sender_chunk_size,
        )
        self.endpoint_pool = EndpointPool(hidden_dim)

    def forward(
        self,
        *,
        endpoint_token: Tensor,
        endpoint_pose: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """interaction-aware endpoint token과 pooled agent token을 반환합니다."""
        bsz, n_rollout, n_endpoint, n_agent, _ = endpoint_token.shape
        endpoint_xy = endpoint_pose[..., :2]
        sender_xy = endpoint_xy.unsqueeze(3)
        yaw = _yaw_from_pose(endpoint_pose)
        sender_yaw = yaw.unsqueeze(3)
        not_self = ~torch.eye(n_agent, device=endpoint_token.device, dtype=torch.bool)
        output_chunks: list[Tensor] = []
        for start in range(0, n_agent, self.query_chunk_size):
            end = min(start + self.query_chunk_size, n_agent)
            receiver_xy = endpoint_xy[..., start:end, :].unsqueeze(4)
            delta_xy = sender_xy - receiver_xy
            receiver_yaw = yaw[..., start:end].unsqueeze(4)
            relation, distance = _build_relation_features(
                delta_xy=delta_xy,
                receiver_yaw=receiver_yaw,
                sender_yaw=sender_yaw,
                radius_m=self.radius_m,
            )
            attention_mask = (
                valid_mask[:, None, None, start:end, None]
                & valid_mask[:, None, None, None, :]
                & not_self[start:end, :].view(1, 1, 1, end - start, n_agent)
                & (distance <= self.radius_m)
            )
            output_chunks.append(
                self.attention(
                    query_token=endpoint_token[..., start:end, :],
                    key_token=endpoint_token,
                    value_token=endpoint_token,
                    relation=relation,
                    attention_mask=attention_mask,
                )
            )
        interaction_endpoint = torch.cat(output_chunks, dim=3)
        interaction_agent_token = self.endpoint_pool(interaction_endpoint)
        return interaction_endpoint, interaction_agent_token


class SelfForcedGANDiscriminator(nn.Module):
    """Set-level teacher-student GAN discriminator입니다.

    설명:
        입력 rollout set은 ``[B, K, 20, N, 4]`` 입니다. 4개 채널은
        ``x, y, cos(yaw), sin(yaw)`` 입니다. Frozen pretrained scene encoder에서 나온
        agent token ``[B, N, 128]`` 과 map token/geometry ``[B, M, 128/2/1]`` 를 조건으로
        사용하고, 새로 학습하는 critic head는 1M 미만으로 유지합니다.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        n_rollout: int = 16,
        n_step: int = 20,
        position_type_scale: tuple[float, float, float] = (22.3461620418, 4.5793447978, 18.5374388830),
        map_radius_m: float = 30.0,
        interaction_radius_m: float = 60.0,
        num_attention_heads: int = 4,
        map_query_chunk_size: int = 16,
        interaction_query_chunk_size: int = 16,
        map_sender_chunk_size: int = 0,
        interaction_sender_chunk_size: int = 0,
        map_rollout_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_rollout = int(n_rollout)
        self.n_step = int(n_step)
        self.register_buffer(
            "position_type_scale",
            torch.tensor(position_type_scale, dtype=torch.float32),
            persistent=False,
        )
        self.endpoint_indices = (4, 9, 14, 19)

        self.input_projection = nn.Linear(4, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(n_step, hidden_dim))
        self.temporal_blocks = nn.ModuleList(
            [TemporalResidualBlock(hidden_dim) for _ in range(2)]
        )
        self.temporal_pool = PoolingProjection(hidden_dim * 2, hidden_dim)
        self.scene_fusion = SceneConditionFusion(hidden_dim)
        self.map_compliance_encoder = MapComplianceEncoder(
            hidden_dim=hidden_dim,
            radius_m=map_radius_m,
            num_heads=num_attention_heads,
            query_chunk_size=map_query_chunk_size,
            sender_chunk_size=map_sender_chunk_size,
            rollout_chunk_size=map_rollout_chunk_size,
        )
        self.interaction_encoder = InteractionEncoder(
            hidden_dim=hidden_dim,
            radius_m=interaction_radius_m,
            num_heads=num_attention_heads,
            query_chunk_size=interaction_query_chunk_size,
            sender_chunk_size=interaction_sender_chunk_size,
        )
        self.agent_pool = PoolingProjection(hidden_dim * 4, hidden_dim)
        self.scalar_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.time_embedding, std=0.02)

    def _normalize_pose(
        self,
        rollout_pose: Tensor,
        current_pose: Tensor,
        agent_type: Tensor,
    ) -> Tensor:
        """x/y만 현재 pose 기준 local coordinate로 바꾸고 yaw cos/sin은 그대로 둡니다."""
        current_xy = current_pose[:, None, None, :, :2]
        current_cos = current_pose[:, None, None, :, 2]
        current_sin = current_pose[:, None, None, :, 3]

        delta = rollout_pose[..., :2] - current_xy
        local_x = delta[..., 0] * current_cos + delta[..., 1] * current_sin
        local_y = -delta[..., 0] * current_sin + delta[..., 1] * current_cos

        type_index = agent_type.long().clamp(min=0, max=2)
        scale = self.position_type_scale.to(device=rollout_pose.device, dtype=rollout_pose.dtype)
        scale = scale[type_index][:, None, None, :, None].clamp_min(1.0e-6)
        local_xy = torch.stack([local_x, local_y], dim=-1) / scale
        return torch.cat([local_xy, rollout_pose[..., 2:]], dim=-1)

    def _trajectory_encode(
        self,
        rollout_pose: Tensor,
        current_pose: Tensor,
        agent_type: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Stage 2 trajectory encoder를 실행합니다."""
        x = self._normalize_pose(rollout_pose, current_pose, agent_type)
        x = self.input_projection(x)
        x = x + self.time_embedding.view(1, 1, self.n_step, 1, self.hidden_dim)
        for block in self.temporal_blocks:
            x = block(x)
        endpoint_token = x[:, :, list(self.endpoint_indices), :, :]
        temporal_mean = x.mean(dim=2)
        temporal_max = x.amax(dim=2)
        agent_token = self.temporal_pool(torch.cat([temporal_mean, temporal_max], dim=-1))
        return x, endpoint_token, agent_token

    def _pool_agents(self, agent_token: Tensor, valid_mask: Tensor) -> Tensor:
        """Stage 7: agent token을 mean/std/max/min pooling으로 rollout token으로 줄입니다."""
        mask = valid_mask[:, None, :, None].to(dtype=torch.bool)
        mean = _safe_masked_mean(agent_token, mask, dim=2)
        std = _safe_masked_std(agent_token, mask, dim=2)
        maximum = _safe_masked_max(agent_token, mask, dim=2)
        minimum = _safe_masked_min(agent_token, mask, dim=2)
        return self.agent_pool(torch.cat([mean, std, maximum, minimum], dim=-1))

    @staticmethod
    def _pool_set(rollout_token: Tensor) -> Tensor:
        """Stage 8: rollout token 16개를 mean/std/max/min set token으로 줄입니다."""
        mean = rollout_token.mean(dim=1)
        std = rollout_token.std(dim=1, unbiased=False)
        maximum = rollout_token.amax(dim=1)
        minimum = rollout_token.amin(dim=1)
        return torch.cat([mean, std, maximum, minimum], dim=-1)

    def _validate_forward_shapes(
        self,
        *,
        rollout_pose: Tensor,
        current_pose: Tensor,
        agent_type: Tensor,
        valid_mask: Tensor,
        agent_context: Tensor,
        map_context: Tensor,
        map_position: Tensor,
        map_orientation: Tensor,
        map_valid_mask: Tensor,
    ) -> None:
        if rollout_pose.dim() != 5 or rollout_pose.shape[-1] != 4:
            raise ValueError(
                "rollout_pose must have shape [B, K, T, N, 4], "
                f"got {tuple(rollout_pose.shape)}."
            )
        bsz, n_rollout, n_step, n_agent, _ = rollout_pose.shape
        if n_rollout != self.n_rollout:
            raise ValueError(f"expected K={self.n_rollout}, got {n_rollout}.")
        if n_step != self.n_step:
            raise ValueError(f"expected T={self.n_step}, got {n_step}.")
        expected_agent_shape = (bsz, n_agent)
        if tuple(current_pose.shape) != (bsz, n_agent, 4):
            raise ValueError(f"current_pose must be [B, N, 4], got {tuple(current_pose.shape)}.")
        if tuple(agent_type.shape) != expected_agent_shape:
            raise ValueError(f"agent_type must be [B, N], got {tuple(agent_type.shape)}.")
        if tuple(valid_mask.shape) != expected_agent_shape:
            raise ValueError(f"valid_mask must be [B, N], got {tuple(valid_mask.shape)}.")
        if tuple(agent_context.shape) != (bsz, n_agent, self.hidden_dim):
            raise ValueError(
                f"agent_context must be [B, N, {self.hidden_dim}], got {tuple(agent_context.shape)}."
            )
        if map_context.dim() != 3 or map_context.shape[0] != bsz or map_context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"map_context must be [B, M, {self.hidden_dim}], got {tuple(map_context.shape)}."
            )
        n_map = map_context.shape[1]
        if tuple(map_position.shape) != (bsz, n_map, 2):
            raise ValueError(f"map_position must be [B, M, 2], got {tuple(map_position.shape)}.")
        if tuple(map_orientation.shape) != (bsz, n_map):
            raise ValueError(f"map_orientation must be [B, M], got {tuple(map_orientation.shape)}.")
        if tuple(map_valid_mask.shape) != (bsz, n_map):
            raise ValueError(f"map_valid_mask must be [B, M], got {tuple(map_valid_mask.shape)}.")

    def forward(
        self,
        rollout_pose: Tensor,
        *,
        current_pose: Tensor,
        agent_type: Tensor,
        valid_mask: Tensor,
        agent_context: Tensor,
        map_context: Tensor,
        map_position: Tensor,
        map_orientation: Tensor,
        map_valid_mask: Tensor | None = None,
    ) -> Tensor:
        """teacher-like scalar logit을 계산합니다.

        Args:
            rollout_pose: rollout set입니다. shape은 ``[B, K, 20, N, 4]`` 입니다.
            current_pose: 현재 pose입니다. shape은 ``[B, N, 4]`` 입니다.
            agent_type: agent type입니다. shape은 ``[B, N]`` 입니다.
            valid_mask: agent valid mask입니다. shape은 ``[B, N]`` 입니다.
            agent_context: frozen scene encoder의 agent context입니다. shape은 ``[B, N, 128]`` 입니다.
            map_context: frozen scene encoder의 map context입니다. shape은 ``[B, M, 128]`` 입니다.
            map_position: map token position입니다. shape은 ``[B, M, 2]`` 입니다.
            map_orientation: map token orientation입니다. shape은 ``[B, M]`` 입니다.
            map_valid_mask: map token valid mask입니다. shape은 ``[B, M]`` 입니다.

        Returns:
            Tensor: scene별 discriminator logit입니다. shape은 ``[B, 1]`` 입니다.
        """
        if map_valid_mask is None:
            map_valid_mask = torch.ones(
                map_context.shape[:2],
                device=map_context.device,
                dtype=torch.bool,
            )
        self._validate_forward_shapes(
            rollout_pose=rollout_pose,
            current_pose=current_pose,
            agent_type=agent_type,
            valid_mask=valid_mask,
            agent_context=agent_context,
            map_context=map_context,
            map_position=map_position,
            map_orientation=map_orientation,
            map_valid_mask=map_valid_mask,
        )

        _, endpoint_token, agent_token = self._trajectory_encode(
            rollout_pose=rollout_pose,
            current_pose=current_pose,
            agent_type=agent_type,
        )
        scene_agent_token = self.scene_fusion(agent_token, agent_context)
        endpoint_pose = rollout_pose[:, :, list(self.endpoint_indices), :, :]
        map_endpoint_token, map_agent_token = self.map_compliance_encoder(
            endpoint_token=endpoint_token,
            endpoint_pose=endpoint_pose,
            map_context=map_context,
            map_position=map_position,
            map_orientation=map_orientation,
            map_valid_mask=map_valid_mask.bool(),
            valid_mask=valid_mask.bool(),
        )
        _, interaction_agent_token = self.interaction_encoder(
            endpoint_token=map_endpoint_token,
            endpoint_pose=endpoint_pose,
            valid_mask=valid_mask.bool(),
        )
        final_agent_token = scene_agent_token + map_agent_token + interaction_agent_token
        rollout_token = self._pool_agents(final_agent_token, valid_mask=valid_mask.bool())
        set_token = self._pool_set(rollout_token)
        return self.scalar_head(set_token)

    def count_trainable_parameters(self) -> int:
        """학습 가능한 parameter 수를 반환합니다."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def add_rollout_pose_perturbation(
    rollout_pose: Tensor,
    *,
    current_pose: Tensor,
    agent_type: Tensor,
    position_type_scale: Tensor,
    position_sigma: float,
    yaw_sigma: float,
) -> Tensor:
    """R1/R2용 작은 pose perturbation을 추가합니다.

    Args:
        rollout_pose: 원본 rollout pose입니다. shape은 ``[B, K, T, N, 4]`` 입니다.
        current_pose: 현재 pose입니다. shape은 ``[B, N, 4]`` 입니다.
        agent_type: agent type입니다. shape은 ``[B, N]`` 입니다.
        position_type_scale: type별 위치 scale입니다. shape은 ``[3]`` 입니다.
        position_sigma: local normalized coordinate에서의 위치 noise 표준편차입니다.
        yaw_sigma: yaw angle noise 표준편차입니다. 단위는 radian입니다.

    Returns:
        Tensor: perturbation이 들어간 rollout pose입니다. shape은 입력과 같습니다.
    """

    type_index = agent_type.long().clamp(min=0, max=2)
    scale = position_type_scale.to(device=rollout_pose.device, dtype=rollout_pose.dtype)[type_index]
    local_noise = torch.randn_like(rollout_pose[..., :2])
    local_noise = local_noise * scale[:, None, None, :, None] * float(position_sigma)

    cur_cos = current_pose[:, None, None, :, 2]
    cur_sin = current_pose[:, None, None, :, 3]
    global_dx = local_noise[..., 0] * cur_cos - local_noise[..., 1] * cur_sin
    global_dy = local_noise[..., 0] * cur_sin + local_noise[..., 1] * cur_cos
    perturbed_xy = rollout_pose[..., :2] + torch.stack([global_dx, global_dy], dim=-1)

    yaw = torch.atan2(rollout_pose[..., 3], rollout_pose[..., 2])
    yaw = yaw + torch.randn_like(yaw) * float(yaw_sigma)
    perturbed_heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return torch.cat([perturbed_xy, perturbed_heading], dim=-1)


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """모듈 parameter를 잠깐 freeze합니다.

    Args:
        module: freeze할 모듈입니다.

    Yields:
        None: context 안에서는 parameter gradient가 꺼집니다.

    설명:
        generator update 때 discriminator parameter는 업데이트하지 않지만,
        discriminator 연산을 통해 fake rollout input으로 gradient는 흘려야 합니다.
    """
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), states):
            parameter.requires_grad_(requires_grad)


def relativistic_discriminator_loss(real_logit: Tensor, fake_logit: Tensor) -> Tensor:
    """R3GAN-style relativistic discriminator loss를 계산합니다.

    Args:
        real_logit: teacher set logit입니다. shape은 ``[B, 1]`` 입니다.
        fake_logit: student set logit입니다. shape은 ``[B, 1]`` 입니다.

    Returns:
        Tensor: scalar discriminator adversarial loss입니다.
    """
    return F.softplus(-(real_logit - fake_logit)).mean()


def relativistic_generator_loss(real_logit: Tensor, fake_logit: Tensor) -> Tensor:
    """R3GAN-style relativistic generator loss를 계산합니다.

    Args:
        real_logit: teacher set logit입니다. shape은 ``[B, 1]`` 입니다.
        fake_logit: student set logit입니다. shape은 ``[B, 1]`` 입니다.

    Returns:
        Tensor: scalar generator adversarial loss입니다.
    """
    return F.softplus(-(fake_logit - real_logit.detach())).mean()
