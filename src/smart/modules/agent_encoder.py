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

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle


def build_motion_point_sequence_features(motion_points_local: Tensor) -> Tensor:
    """0.5초 5개 점을 순서가 있는 점 특징으로 바꿉니다.

    Args:
        motion_points_local: 구간 시작 시점 기준 local 5개 점입니다.
            shape은 ``[n_item, 5, 2]`` 입니다.

    Returns:
        Tensor: 각 점마다 위치, 직전 점 대비 이동량, 이동 길이, 누적 이동 길이,
        시간 진행률을 붙인 값입니다. shape은 ``[n_item, 5, 7]`` 입니다.
    """
    if motion_points_local.ndim != 3 or motion_points_local.size(-2) != 5 or motion_points_local.size(-1) != 2:
        raise ValueError(
            "motion_points_local must have shape [n_item, 5, 2], "
            f"got {tuple(motion_points_local.shape)}"
        )

    origin = motion_points_local.new_zeros((motion_points_local.shape[0], 1, 2))
    path_points = torch.cat([origin, motion_points_local], dim=1)
    delta = path_points[:, 1:] - path_points[:, :-1]
    step_length = torch.norm(delta, p=2, dim=-1, keepdim=True)
    cumulative_length = torch.cumsum(step_length, dim=1)
    progress = torch.linspace(
        0.2,
        1.0,
        steps=5,
        device=motion_points_local.device,
        dtype=motion_points_local.dtype,
    ).view(1, 5, 1)
    progress = progress.expand(motion_points_local.shape[0], -1, -1)
    return torch.cat(
        [motion_points_local, delta, step_length, cumulative_length, progress],
        dim=-1,
    )



def build_motion_global_features(motion_points_local: Tensor) -> Tensor:
    """0.5초 5개 점을 구간 요약값으로 바꿉니다.

    Args:
        motion_points_local: 구간 시작 시점 기준 local 5개 점입니다.
            shape은 ``[n_item, 5, 2]`` 입니다.

    Returns:
        Tensor: 구간 끝점, 전체 이동 길이, 마지막 이동 길이, 옆방향 흔들림,
        직진성, 좌우 회전 방향을 담은 값입니다. shape은 ``[n_item, 8]`` 입니다.
    """
    if motion_points_local.ndim != 3 or motion_points_local.size(-2) != 5 or motion_points_local.size(-1) != 2:
        raise ValueError(
            "motion_points_local must have shape [n_item, 5, 2], "
            f"got {tuple(motion_points_local.shape)}"
        )

    origin = motion_points_local.new_zeros((motion_points_local.shape[0], 1, 2))
    path_points = torch.cat([origin, motion_points_local], dim=1)
    delta = path_points[:, 1:] - path_points[:, :-1]
    step_length = torch.norm(delta, p=2, dim=-1)
    total_length = step_length.sum(dim=-1)
    mean_step_length = step_length.mean(dim=-1)
    tail_length = step_length[:, -1]
    end_point = motion_points_local[:, -1]
    end_disp = torch.norm(end_point, p=2, dim=-1)
    straightness = end_disp / total_length.clamp_min(1e-3)
    max_abs_lat = motion_points_local[..., 1].abs().amax(dim=-1)
    poly_cross = (
        path_points[:, :-1, 0] * path_points[:, 1:, 1]
        - path_points[:, :-1, 1] * path_points[:, 1:, 0]
    )
    signed_area = 0.5 * poly_cross.sum(dim=-1)
    return torch.stack(
        [
            end_point[:, 0],
            end_point[:, 1],
            total_length,
            mean_step_length,
            tail_length,
            max_abs_lat,
            straightness,
            signed_area,
        ],
        dim=-1,
    )


class ContinuousMotionSegmentEncoder(nn.Module):
    """5개 연속 점을 하나의 구간 표현으로 압축합니다."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """짧은 점 시퀀스를 읽는 작은 인코더를 만듭니다.

        Args:
            hidden_dim: 출력 특징 크기입니다.
            num_heads: self-attention head 수입니다.
            dropout: dropout 비율입니다.
        """
        super().__init__()
        point_feature_dim = 7
        motion_num_heads = math.gcd(hidden_dim, num_heads)
        motion_num_heads = max(1, motion_num_heads)

        self.point_proj = MLPEmbedding(
            input_dim=point_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.step_emb = nn.Embedding(5, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=motion_num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def reset_parameters(self) -> None:
        """학습 가능한 토큰을 다시 초기화합니다."""
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        motion_points_local: Tensor,
        static_context: Tensor,
    ) -> Tensor:
        """5개 점 시퀀스를 읽어 구간 임베딩 하나를 만듭니다.

        Args:
            motion_points_local: local 5개 점입니다. shape은 ``[n_item, 5, 2]`` 입니다.
            static_context: 차종과 크기에서 만든 고정 문맥입니다.
                shape은 ``[n_item, hidden_dim]`` 입니다.

        Returns:
            Tensor: 구간 전체를 대표하는 임베딩입니다.
            shape은 ``[n_item, hidden_dim]`` 입니다.
        """
        point_feature = build_motion_point_sequence_features(motion_points_local)
        n_item, n_point, _ = point_feature.shape
        point_token = self.point_proj(point_feature.reshape(-1, point_feature.shape[-1]))
        point_token = point_token.view(n_item, n_point, -1)

        step_index = torch.arange(n_point, device=motion_points_local.device)
        step_token = self.step_emb(step_index).unsqueeze(0).expand(n_item, -1, -1)
        point_token = point_token + step_token + static_context.unsqueeze(1)

        cls_token = self.cls_token.expand(n_item, -1, -1) + static_context.unsqueeze(1)
        encoded = self.encoder(torch.cat([cls_token, point_token], dim=1))
        return self.out_norm(encoded[:, 0])


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
        self.n_token_agent = n_token_agent

        input_dim_global_motion = 8
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)
        self.motion_segment_encoder = ContinuousMotionSegmentEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.motion_global_emb = FourierEmbedding(
            input_dim=input_dim_global_motion,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
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
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2,
            hidden_dim=self.hidden_dim,
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
        self.motion_segment_encoder.reset_parameters()

    def agent_token_embedding(
        self,
        agent_token_index,
        trajectory_token_veh,
        trajectory_token_ped,
        trajectory_token_cyc,
        pos_a,
        head_vector_a,
        agent_type,
        agent_shape,
        inference: bool = False,
    ):
        """연속 5개 점을 직접 읽어 agent embedding을 만듭니다.

        Args:
            agent_token_index: local 5개 점 구간입니다.
                shape은 ``[n_agent, n_step, 5, 2]`` 입니다.
            trajectory_token_veh: 기존 호출부 호환용 인자입니다.
            trajectory_token_ped: 기존 호출부 호환용 인자입니다.
            trajectory_token_cyc: 기존 호출부 호환용 인자입니다.
            pos_a: 기존 호출부 호환용 인자입니다.
            head_vector_a: 기존 호출부 호환용 인자입니다.
            agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_shape: agent 크기 정보입니다. shape은 ``[n_agent, 3]`` 입니다.
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

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2

        motion_points_flat = motion_points_local.reshape(n_agent * n_step, 5, 2)
        flat_agent_type = agent_type.unsqueeze(1).expand(-1, n_step).reshape(-1)
        flat_agent_shape = agent_shape.unsqueeze(1).expand(-1, n_step, -1).reshape(-1, agent_shape.shape[-1])

        type_emb_flat = self.type_a_emb(flat_agent_type.long())
        shape_emb_flat = self.shape_emb(flat_agent_shape)
        static_context_flat = type_emb_flat + shape_emb_flat

        motion_segment_emb_flat = self.motion_segment_encoder(
            motion_points_local=motion_points_flat,
            static_context=static_context_flat,
        )
        motion_global_feature = build_motion_global_features(motion_points_flat)
        motion_global_emb_flat = self.motion_global_emb(
            continuous_inputs=motion_global_feature,
            categorical_embs=[type_emb_flat, shape_emb_flat],
        )

        feat_a_flat = self.fusion_emb(
            torch.cat([motion_segment_emb_flat, motion_global_emb_flat], dim=-1)
        )
        feat_a = feat_a_flat.view(n_agent, n_step, self.hidden_dim)
        agent_token_emb = motion_segment_emb_flat.view(n_agent, n_step, self.hidden_dim)

        if inference:
            dummy_bank = torch.zeros(
                (1, self.hidden_dim),
                device=device,
                dtype=agent_token_emb.dtype,
            )
            categorical_embs = [
                self.type_a_emb(agent_type.long()),
                self.shape_emb(agent_shape),
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
