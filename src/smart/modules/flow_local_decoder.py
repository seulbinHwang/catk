from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.utils import subgraph

from src.smart.layers.attention_layer import AttentionLayer

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.utils import (
    angle_between_2d_vectors,
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
        if x_t_norm.shape[1] != self.num_steps:
            raise ValueError(
                "NormalizedNoisyFutureEncoder expected "
                f"{self.num_steps} future steps, got {x_t_norm.shape[1]}."
            )

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

        # chunk_a2a_mixers가 agent 간 미래 정렬을 담당하므로, 기존 chunk 내부
        # 조건 주입부는 같은 기능을 유지하되 hidden 폭만 줄여 파라미터 예산을 맞춥니다.
        self.cond_mlp = nn.Sequential(
            nn.Linear(flow_dim * 2, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim * 3),
        )

        self.mlp_norm = nn.LayerNorm(flow_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim),
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


class ChunkAgentInteractionMixer(nn.Module):
    def __init__(
        self,
        flow_dim: int,
        num_heads: int,
        a2a_radius: float,
        dropout: float = 0.0,
        max_num_neighbors: int = 300,
    ) -> None:
        """같은 미래 chunk 위치에 있는 agent 후보끼리 정보를 교환합니다.

        Args:
            flow_dim: chunk token의 폭입니다.
            num_heads: attention head 개수입니다.
            a2a_radius: 서로 영향을 줄 수 있는 최대 거리입니다.
            dropout: attention 내부 dropout 확률입니다.
            max_num_neighbors: 한 노드가 볼 수 있는 최대 이웃 수입니다.
        """
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        self.a2a_radius = float(a2a_radius)
        self.max_num_neighbors = int(max_num_neighbors)
        self.relation_mlp = nn.Sequential(
            nn.Linear(5, flow_dim),
            nn.LayerNorm(flow_dim),
            nn.SiLU(),
            nn.Linear(flow_dim, flow_dim),
        )
        self.attn = AttentionLayer(
            hidden_dim=flow_dim,
            num_heads=int(num_heads),
            head_dim=max(1, int(flow_dim) // int(num_heads)),
            dropout=dropout,
            bipartite=False,
            has_pos_emb=True,
        )

    def _build_chunk_batch(
        self,
        agent_batch: torch.Tensor,
        num_chunks: int,
        anchor_step_id: torch.Tensor | None,
    ) -> torch.Tensor:
        """chunk, 장면, anchor 시점을 하나의 graph 번호로 묶습니다.

        Args:
            agent_batch: 각 후보가 속한 장면 번호입니다. shape은 ``[n_candidate]`` 입니다.
            num_chunks: 미래를 나눈 chunk 개수입니다.
            anchor_step_id: 학습용 anchor 시점 번호입니다. shape은 ``[n_candidate]`` 입니다.
                ``None``이면 모든 후보가 같은 anchor 시점에 있다고 봅니다.

        Returns:
            torch.Tensor: 펼친 chunk별 graph 번호입니다.
                shape은 ``[num_chunks * n_candidate]`` 입니다.
        """
        # agent_batch: [n_candidate]
        if agent_batch.numel() == 0:
            return agent_batch.new_zeros((0,))

        num_scene_graphs = int(agent_batch.max().item()) + 1
        if anchor_step_id is None:
            anchor_step_id = agent_batch.new_zeros(agent_batch.shape)
        else:
            anchor_step_id = anchor_step_id.to(device=agent_batch.device, dtype=agent_batch.dtype)

        num_anchor_steps = int(anchor_step_id.max().item()) + 1 if anchor_step_id.numel() > 0 else 1
        base_batch = agent_batch + anchor_step_id * num_scene_graphs
        num_base_batches = max(1, num_scene_graphs * num_anchor_steps)
        chunk_offsets = (
            torch.arange(num_chunks, device=agent_batch.device, dtype=agent_batch.dtype)
            .repeat_interleave(agent_batch.shape[0])
            * num_base_batches
        )
        return base_batch.repeat(num_chunks) + chunk_offsets

    def _build_relation_embedding(
        self,
        pos_flat: torch.Tensor,
        head_flat: torch.Tensor,
        motion_flat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """edge마다 상대 위치, 방향, 이동 차이를 embedding으로 바꿉니다.

        Args:
            pos_flat: 펼친 chunk 중심 위치입니다. shape은 ``[num_chunks * n_candidate, 2]`` 입니다.
            head_flat: 펼친 chunk 방향입니다. shape은 ``[num_chunks * n_candidate]`` 입니다.
            motion_flat: 펼친 chunk 이동량입니다. shape은 ``[num_chunks * n_candidate, 2]`` 입니다.
            edge_index: source와 target 노드 번호입니다. shape은 ``[2, n_edge]`` 입니다.

        Returns:
            torch.Tensor: attention에 더할 edge 특징입니다. shape은 ``[n_edge, flow_dim]`` 입니다.
        """
        # edge_index[0]: source chunk, edge_index[1]: receiver chunk
        rel_pos = pos_flat[edge_index[0]] - pos_flat[edge_index[1]]
        rel_head = wrap_angle(head_flat[edge_index[0]] - head_flat[edge_index[1]])
        rel_motion = motion_flat[edge_index[0]] - motion_flat[edge_index[1]]

        recv_head = head_flat[edge_index[1]]
        recv_cos = recv_head.cos()
        recv_sin = recv_head.sin()
        rel_motion_long = rel_motion[:, 0] * recv_cos + rel_motion[:, 1] * recv_sin
        rel_motion_lat = -rel_motion[:, 0] * recv_sin + rel_motion[:, 1] * recv_cos
        recv_head_vector = torch.stack([recv_cos, recv_sin], dim=-1)

        relation_inputs = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=recv_head_vector,
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_head,
                rel_motion_long,
                rel_motion_lat,
            ],
            dim=-1,
        )
        return self.relation_mlp(relation_inputs)

    def forward(
        self,
        chunk_tokens: torch.Tensor,
        chunk_pos: torch.Tensor,
        chunk_head: torch.Tensor,
        chunk_motion: torch.Tensor,
        agent_batch: torch.Tensor,
        anchor_step_id: torch.Tensor | None = None,
        chunk_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """같은 상대 미래 chunk끼리만 agent-to-agent attention을 수행합니다.

        Args:
            chunk_tokens: chunk별 feature입니다. shape은 ``[n_candidate, num_chunks, flow_dim]`` 입니다.
            chunk_pos: chunk별 전역 중심 위치입니다. shape은 ``[n_candidate, num_chunks, 2]`` 입니다.
            chunk_head: chunk별 전역 방향입니다. shape은 ``[n_candidate, num_chunks]`` 입니다.
            chunk_motion: 이전 chunk 중심에서 현재 chunk 중심까지의 이동량입니다.
                shape은 ``[n_candidate, num_chunks, 2]`` 입니다.
            agent_batch: 각 후보가 속한 장면 번호입니다. shape은 ``[n_candidate]`` 입니다.
            anchor_step_id: 학습용 anchor 시점 번호입니다. shape은 ``[n_candidate]`` 입니다.
            chunk_valid: chunk 유효 여부입니다. shape은 ``[n_candidate, num_chunks]`` 입니다.
                ``None``이면 모든 chunk를 유효하다고 봅니다.

        Returns:
            torch.Tensor: agent 간 정렬 정보가 반영된 chunk feature입니다.
                shape은 ``[n_candidate, num_chunks, flow_dim]`` 입니다.
        """
        # chunk_tokens: [n_candidate, num_chunks, flow_dim]
        n_candidate, num_chunks, flow_dim = chunk_tokens.shape
        if n_candidate <= 1 or num_chunks == 0 or self.a2a_radius <= 0.0:
            return chunk_tokens
        if agent_batch.shape[0] != n_candidate:
            raise ValueError(
                "agent_batch must have one item per candidate, "
                f"got {agent_batch.shape[0]} and {n_candidate}."
            )
        if chunk_pos.shape != (n_candidate, num_chunks, 2):
            raise ValueError(
                "chunk_pos shape must be [n_candidate, num_chunks, 2], "
                f"got {tuple(chunk_pos.shape)}."
            )
        if chunk_head.shape != (n_candidate, num_chunks):
            raise ValueError(
                "chunk_head shape must be [n_candidate, num_chunks], "
                f"got {tuple(chunk_head.shape)}."
            )
        if chunk_motion.shape != (n_candidate, num_chunks, 2):
            raise ValueError(
                "chunk_motion shape must be [n_candidate, num_chunks, 2], "
                f"got {tuple(chunk_motion.shape)}."
            )

        if chunk_valid is None:
            chunk_valid = torch.ones(
                (n_candidate, num_chunks),
                device=chunk_tokens.device,
                dtype=torch.bool,
            )
        else:
            if chunk_valid.shape != (n_candidate, num_chunks):
                raise ValueError(
                    "chunk_valid shape must be [n_candidate, num_chunks], "
                    f"got {tuple(chunk_valid.shape)}."
                )
            chunk_valid = chunk_valid.to(device=chunk_tokens.device, dtype=torch.bool)

        # 아래 flatten 순서는 [chunk0의 모든 후보, chunk1의 모든 후보, ...] 입니다.
        token_flat = chunk_tokens.transpose(0, 1).reshape(num_chunks * n_candidate, flow_dim)
        pos_flat = chunk_pos.transpose(0, 1).reshape(num_chunks * n_candidate, 2)
        head_flat = chunk_head.transpose(0, 1).reshape(num_chunks * n_candidate)
        motion_flat = chunk_motion.transpose(0, 1).reshape(num_chunks * n_candidate, 2)
        valid_flat = chunk_valid.transpose(0, 1).reshape(num_chunks * n_candidate)
        chunk_batch = self._build_chunk_batch(
            agent_batch=agent_batch.to(device=chunk_tokens.device),
            num_chunks=num_chunks,
            anchor_step_id=anchor_step_id,
        )

        edge_index = radius_graph(
            x=pos_flat[:, :2],
            r=self.a2a_radius,
            batch=chunk_batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = subgraph(subset=valid_flat, edge_index=edge_index)[0]
        if edge_index.numel() == 0:
            return chunk_tokens

        relation = self._build_relation_embedding(
            pos_flat=pos_flat,
            head_flat=head_flat,
            motion_flat=motion_flat,
            edge_index=edge_index,
        )
        mixed_flat = self.attn(token_flat, relation, edge_index)
        return mixed_flat.view(num_chunks, n_candidate, flow_dim).transpose(0, 1)


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
        num_future_steps: int = 20,
        num_chunk_heads: int = 4,
        num_chunk_layers: int = 2,
        chunk_size: int = 5,
        a2a_radius: float = 60.0,
        num_chunk_a2a_layers: int = 1,
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
        self.num_chunks = num_chunks
        self.chunk_size = int(chunk_size)
        self.pos_scale_m = 20.0
        self.context_projector = AnchorContextProjector(context_dim, flow_dim)
        self.noisy_future_encoder = NormalizedNoisyFutureEncoder(
            flow_dim=flow_dim,
            num_chunks=num_chunks,
            chunk_size=int(chunk_size),
        )
        self.chunk_mixers = nn.ModuleList(
            [
                HalfSecondChunkMixerBlock(flow_dim=flow_dim, num_heads=num_chunk_heads)
                for _ in range(num_chunk_layers)
            ]
        )
        self.chunk_a2a_mixers = nn.ModuleList(
            [
                ChunkAgentInteractionMixer(
                    flow_dim=flow_dim,
                    num_heads=num_chunk_heads,
                    a2a_radius=a2a_radius,
                )
                for _ in range(int(num_chunk_a2a_layers))
            ]
        )
        self.step_refiner = ChunkStepRefiner(
            flow_dim=flow_dim,
            num_heads=num_chunk_heads,
        )
        self.velocity_head = FlowVelocityHead(flow_dim=flow_dim)

    def _build_chunk_global_kinematics(
        self,
        x_t_norm: torch.Tensor,
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """정규화된 미래 상태에서 chunk별 전역 위치와 이동량을 만듭니다.

        Args:
            x_t_norm: 현재 flow 상태입니다. shape은 ``[n_candidate, num_future_steps, 4]`` 입니다.
            current_pos: 각 후보의 현재 중심 위치입니다. shape은 ``[n_candidate, 2]`` 입니다.
            current_head: 각 후보의 현재 방향입니다. shape은 ``[n_candidate]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - chunk_pos: chunk별 전역 중심 위치입니다. shape은 ``[n_candidate, num_chunks, 2]`` 입니다.
                - chunk_head: chunk별 전역 방향입니다. shape은 ``[n_candidate, num_chunks]`` 입니다.
                - chunk_motion: 이전 중심에서 현재 chunk 중심까지의 이동량입니다.
                  shape은 ``[n_candidate, num_chunks, 2]`` 입니다.
        """
        # x_t_chunks: [n_candidate, num_chunks, chunk_size, 4]
        n_candidate = x_t_norm.shape[0]
        if current_pos.shape != (n_candidate, 2):
            raise ValueError(
                "current_pos shape must be [n_candidate, 2], "
                f"got {tuple(current_pos.shape)}."
            )
        if current_head.shape != (n_candidate,):
            raise ValueError(
                "current_head shape must be [n_candidate], "
                f"got {tuple(current_head.shape)}."
            )

        x_t_chunks = x_t_norm.reshape(n_candidate, self.num_chunks, self.chunk_size, 4)
        chunk_pos_local = x_t_chunks[..., :2].mean(dim=2) * self.pos_scale_m

        step_head_vec = F.normalize(x_t_chunks[..., 2:4], dim=-1, eps=1.0e-6)
        chunk_head_vec = F.normalize(step_head_vec.mean(dim=2), dim=-1, eps=1.0e-6)
        chunk_head_local = torch.atan2(chunk_head_vec[..., 1], chunk_head_vec[..., 0])

        chunk_pos, chunk_head = transform_to_global(
            pos_local=chunk_pos_local,
            head_local=chunk_head_local,
            pos_now=current_pos.to(device=x_t_norm.device, dtype=x_t_norm.dtype),
            head_now=current_head.to(device=x_t_norm.device, dtype=x_t_norm.dtype),
        )
        chunk_head = wrap_angle(chunk_head)

        previous_pos = torch.cat(
            [current_pos.to(device=x_t_norm.device, dtype=x_t_norm.dtype).unsqueeze(1), chunk_pos[:, :-1]],
            dim=1,
        )
        chunk_motion = chunk_pos - previous_pos
        return chunk_pos, chunk_head, chunk_motion

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        x_t_norm: torch.Tensor,
        tau: torch.Tensor,
        current_pos: torch.Tensor | None = None,
        current_head: torch.Tensor | None = None,
        agent_batch: torch.Tensor | None = None,
        anchor_step_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        anchor_hidden : (A, 13, H) -> (N=A*13, H) -> context : (N, D)
        """
        context = self.context_projector(anchor_hidden)
        """
        x_t_norm : [B, 20, 4]
        tau : [B]
        
        중간
            tau_emb : (B, D) # MLP
            step_tokens : (B, 20, 4) -> (B, 20, D)
                - step_ids : "각 토큰에 “이게 미래 몇 번째 step인지” 정보를 step_tokens 에 더함
            step_tokens = step_tokens + tau_emb.unsqueeze(1) : (B, 20, D)
            step_tokens = step_tokens.view(B, 4, 5, D) [B, 20, D] -> [B, 4, 5, D]
            chunk_tokens : [B, 4, D]
        """
        step_tokens, chunk_tokens, tau_emb = self.noisy_future_encoder(x_t_norm, tau)
        """
        4개 half-second chunk ( chunk_tokens ) 끼리 서로 정보 교환
        
        anchor 문맥 + 현재 diffusion 시간(tau)을 조건으로 주입
            input: context : (N, D) / tau_emb : (B, D)
            둘이 합침 : (B, 2D) # "과거~현재 + 지도 + agent끼리 상호작용한 정보" + "미래 noising 정도"
            (B, 2D) -> (B, 3D) -> scale, bias, gate = cond.chunk(3, dim=-1): 각각 [B, D]
            
            chunk_tokens 에 scale, bias, gate 적용 (각각 chunk에 균일 적용)
            chunk_tokens : (B, 4, D)
            
            
        """
        for block in self.chunk_mixers:
            chunk_tokens = block(chunk_tokens, context, tau_emb)

        if (
            current_pos is not None
            and current_head is not None
            and agent_batch is not None
            and len(self.chunk_a2a_mixers) > 0
        ):
            chunk_pos, chunk_head, chunk_motion = self._build_chunk_global_kinematics(
                x_t_norm=x_t_norm,
                current_pos=current_pos,
                current_head=current_head,
            )
            for block in self.chunk_a2a_mixers:
                chunk_tokens = block(
                    chunk_tokens=chunk_tokens,
                    chunk_pos=chunk_pos,
                    chunk_head=chunk_head,
                    chunk_motion=chunk_motion,
                    agent_batch=agent_batch,
                    anchor_step_id=anchor_step_id,
                )
        """
        input
            step_tokens : (B, 20, D)
            chunk_tokens : (B, 4, D)
            context : (B, D)
        로직
            chunk_tokens 을 step_tokens 에 더함
            context 을 step_tokens 에 더함
            
            chunk별 로컬 self-attention (각 구간에서 5개 step끼리만 보여 attention)
        
        output
            step_tokens : (b, 20, D)
        """
        step_tokens = self.step_refiner(step_tokens, chunk_tokens, context)
        """
        output : (B, 20, 4)
        """
        return self.velocity_head(step_tokens)


class ContinuousCommitBridge:
    """Bridge continuous flow output back to SMART coarse rollout state."""

    def __init__(self, commit_steps: int = 5, pos_scale_m: float = 20.0) -> None:
        self.commit_steps = int(commit_steps)
        self.pos_scale_m = float(pos_scale_m)

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
        first_chunk = y_hat_norm[:, : self.commit_steps].clone()
        first_chunk[..., :2] = first_chunk[..., :2] * self.pos_scale_m

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
