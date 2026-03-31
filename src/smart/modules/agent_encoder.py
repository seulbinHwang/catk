# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling
# or otherwise documented as NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c)  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.modules.continuous_motion_history import (
    build_motion_point_sequence_features,
    build_motion_summary_features,
)
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle


class LightweightContinuousMotionEncoder(nn.Module):
    """연속 5개 점을 가볍게 압축하는 경량 인코더입니다.

    이 인코더는 무거운 attention이나 transformer 없이, 점별 특징을 짧은 MLP로 읽고
    평균값 / 최댓값 / 마지막 점 요약을 합쳐 구간 임베딩 하나를 만듭니다.
    """

    def __init__(
        self,
        hidden_dim: int,
        point_hidden_dim: int = 96,
        pooled_hidden_dim: int = 256,
    ) -> None:
        """경량 5-point 인코더를 만듭니다.

        Args:
            hidden_dim: 최종 구간 임베딩 크기입니다.
            point_hidden_dim: 각 점을 읽을 때 내부에서 쓰는 특징 크기입니다.
            pooled_hidden_dim: 여러 점을 합친 뒤 한 번 더 섞을 때 쓰는 특징 크기입니다.
        """
        super().__init__()
        self.point_proj = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, point_hidden_dim),
            nn.GELU(),
        )
        self.curve_mlp = nn.Sequential(
            nn.Linear(point_hidden_dim * 3, pooled_hidden_dim),
            nn.GELU(),
            nn.Linear(pooled_hidden_dim, hidden_dim),
        )

    def forward(self, motion_points_local: Tensor) -> Tensor:
        """5개 local 점을 하나의 구간 임베딩으로 압축합니다.

        Args:
            motion_points_local: 구간 시작 시점 기준 local 5개 점입니다.
                shape은 ``[n_item, 5, 2]`` 입니다.

        Returns:
            Tensor: 구간 전체를 대표하는 임베딩입니다.
            shape은 ``[n_item, hidden_dim]`` 입니다.
        """
        point_feature = build_motion_point_sequence_features(motion_points_local)
        point_token = self.point_proj(point_feature)
        pooled_mean = point_token.mean(dim=1)
        pooled_max = point_token.amax(dim=1)
        pooled_last = point_token[:, -1]
        pooled = torch.cat([pooled_mean, pooled_max, pooled_last], dim=-1)
        return self.curve_mlp(pooled)


class SmallShapeEncoder(nn.Module):
    """차량 크기처럼 작은 정적 값을 가볍게 읽는 MLP입니다."""

    def __init__(self, out_dim: int) -> None:
        """정적 값 인코더를 만듭니다.

        Args:
            out_dim: 출력 특징 크기입니다.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, agent_shape: Tensor) -> Tensor:
        """shape 값을 경량 특징으로 바꿉니다.

        Args:
            agent_shape: 에이전트 크기 값입니다. shape은 ``[n_item, 3]`` 입니다.

        Returns:
            Tensor: shape에서 만든 특징입니다. shape은 ``[n_item, out_dim]`` 입니다.
        """
        return self.net(agent_shape)


class SMARTAgentEncoder(nn.Module):
    """Shared agent-context encoder used by the flow-matching model."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob
        # 연속 표현만 쓰므로 실제 agent vocab 크기는 더 이상 의미가 없습니다.
        self.n_token_agent = 0 if n_token_agent is None else int(n_token_agent)

        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_motion_summary = 8

        self.motion_segment_encoder = LightweightContinuousMotionEncoder(hidden_dim=hidden_dim)
        self.motion_summary_mlp = nn.Sequential(
            nn.Linear(input_dim_motion_summary, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.type_context_emb = nn.Embedding(3, 16)
        self.shape_context_mlp = SmallShapeEncoder(out_dim=32)
        self.motion_fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16 + 32, 320),
            nn.GELU(),
            nn.Linear(320, hidden_dim),
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index: Tensor,
        trajectory_token_veh: Tensor | None,
        trajectory_token_ped: Tensor | None,
        trajectory_token_cyc: Tensor | None,
        pos_a: Tensor,
        head_vector_a: Tensor,
        agent_type: Tensor,
        agent_shape: Tensor,
        mask: Tensor | None = None,
        inference: bool = False,
    ):
        """연속 5개 점을 경량 네트워크로 직접 읽어 agent embedding을 만듭니다.

        Args:
            agent_token_index: local 5개 점 구간입니다.
                shape은 ``[n_agent, n_step, 5, 2]`` 입니다.
            trajectory_token_veh: 더 이상 쓰지 않는 기존 호출부 인자입니다.
            trajectory_token_ped: 더 이상 쓰지 않는 기존 호출부 인자입니다.
            trajectory_token_cyc: 더 이상 쓰지 않는 기존 호출부 인자입니다.
            pos_a: 기존 호출부 호환용 인자입니다.
            head_vector_a: 기존 호출부 호환용 인자입니다.
            agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_shape: agent 크기 정보입니다. shape은 ``[n_agent, 3]`` 입니다.
            mask: 각 coarse slot의 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            inference: 추론용 부가 반환값을 함께 낼지 정합니다.

        Returns:
            학습 시에는 ``feat_a`` 하나를, 추론 시에는 기존 호출부가 기대하는 묶음을
            그대로 돌려줍니다.
        """
        del trajectory_token_veh, trajectory_token_ped, trajectory_token_cyc, pos_a, head_vector_a

        motion_points_local = agent_token_index
        if motion_points_local.ndim != 4 or motion_points_local.size(-2) != 5 or motion_points_local.size(-1) != 2:
            raise ValueError(
                "agent_token_index must carry local 5-point motion segments with shape "
                f"[n_agent, n_step, 5, 2], got {tuple(motion_points_local.shape)}"
            )

        n_agent, n_step, _, _ = motion_points_local.shape
        device = motion_points_local.device
        flat_motion = motion_points_local.reshape(n_agent * n_step, 5, 2)
        flat_type = agent_type.unsqueeze(1).expand(-1, n_step).reshape(-1)
        flat_shape = agent_shape.unsqueeze(1).expand(-1, n_step, -1).reshape(-1, agent_shape.shape[-1])

        segment_emb_flat = self.motion_segment_encoder(flat_motion)
        summary_feat_flat = build_motion_summary_features(flat_motion)
        summary_emb_flat = self.motion_summary_mlp(summary_feat_flat)
        type_context_flat = self.type_context_emb(flat_type.long())
        shape_context_flat = self.shape_context_mlp(flat_shape)

        fused_flat = self.motion_fusion_mlp(
            torch.cat(
                [segment_emb_flat, summary_emb_flat, type_context_flat, shape_context_flat],
                dim=-1,
            )
        )
        feat_a = fused_flat.view(n_agent, n_step, self.hidden_dim)
        agent_token_emb = segment_emb_flat.view(n_agent, n_step, self.hidden_dim)

        if mask is not None:
            feat_a = feat_a * mask.unsqueeze(-1).to(feat_a.dtype)
            agent_token_emb = agent_token_emb * mask.unsqueeze(-1).to(agent_token_emb.dtype)

        if inference:
            dummy_bank = torch.zeros(
                (1, self.hidden_dim),
                device=device,
                dtype=agent_token_emb.dtype,
            )
            veh_mask = agent_type == 0
            ped_mask = agent_type == 1
            cyc_mask = agent_type == 2
            categorical_embs = [
                self.type_context_emb(agent_type.long()),
                self.shape_context_mlp(agent_shape),
            ]
            return (
                feat_a,
                agent_token_emb,
                dummy_bank,
                dummy_bank.clone(),
                dummy_bank.clone(),
                veh_mask,
                ped_mask,
                cyc_mask,
                categorical_embs,
            )
        return feat_a

    def build_temporal_edge(
        self,
        pos_a,
        head_a,
        head_vector_a,
        mask,
        inference_mask=None,
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)
        if self.hist_drop_prob > 0 and self.training:
            mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & mask_keep
        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :,
            edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift,
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]],
                    nbr_vector=rel_pos_t,
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a,
        head_a,
        head_vector_a,
        batch_s,
        mask,
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl,
        orient_pl,
        pos_a,
        head_a,
        head_vector_a,
        mask,
        batch_s,
        batch_pl,
    ):
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a
