# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.dynamic_light_time import (
    NO_LANE_STATE_LIGHT_TYPE,
    resolve_light_time_delta_norm,
)
from src.smart.utils import angle_between_2d_vectors, safe_norm_2d, weight_init, wrap_angle


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

        input_dim_x_a = 3
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        token_num_steps = 6
        token_num_vertices = 4
        token_xy_dim = 2
        input_dim_token = token_num_steps * token_num_vertices * token_xy_dim

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)
        self.light_pl2a_emb = nn.Embedding(5, hidden_dim)
        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
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
        self.light_time_pl2a_emb = FourierEmbedding(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
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
        valid_mask: Optional[torch.Tensor] = None,
        inference=False,
    ):
        if valid_mask is None:
            raise ValueError(
                "valid_mask is required for agent motion features. Missing motion must "
                "not be treated as valid stationary motion."
            )
        n_agent, n_step, traj_dim = pos_a.shape
        device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2

        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim),
            device=device,
            dtype=agent_token_emb_veh.dtype,
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        feature_a = self._build_motion_feature(
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            valid_mask=valid_mask,
        )
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                emb.repeat_interleave(repeats=n_step, dim=0)
                for emb in categorical_embs
            ],
        )
        x_a = x_a.view(-1, n_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,
                agent_token_emb,
                agent_token_emb_veh,
                agent_token_emb_ped,
                agent_token_emb_cyc,
                veh_mask,
                ped_mask,
                cyc_mask,
                categorical_embs,
            )
        return feat_a

    @staticmethod
    def _build_motion_valid_mask(
        pos_a: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_agent, n_step, _ = pos_a.shape
        motion_valid_a = torch.zeros(
            n_agent,
            n_step,
            device=pos_a.device,
            dtype=torch.bool,
        )
        if n_step <= 1:
            return motion_valid_a

        if valid_mask is None:
            raise ValueError(
                "valid_mask is required to build motion_valid. Missing motion must "
                "not be treated as valid stationary motion."
            )

        if tuple(valid_mask.shape) != (n_agent, n_step):
            raise ValueError(
                "valid_mask shape must match the first two dimensions of pos_a, "
                f"got {tuple(valid_mask.shape)} and {(n_agent, n_step)}."
            )
        motion_valid_a[:, 1:] = valid_mask[:, 1:].bool() & valid_mask[:, :-1].bool()
        return motion_valid_a

    @staticmethod
    def _build_motion_vector(
        pos_a: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_agent, n_step, traj_dim = pos_a.shape
        motion_vector_a = pos_a.new_zeros(n_agent, n_step, traj_dim)
        if n_step <= 1:
            return motion_vector_a

        step_delta = pos_a[:, 1:] - pos_a[:, :-1]
        motion_valid_a = SMARTAgentEncoder._build_motion_valid_mask(pos_a, valid_mask)
        # Invalid samples are stored at the origin; keep the value at zero and
        # expose missingness through the separate motion_valid feature.
        step_delta = step_delta.masked_fill(~motion_valid_a[:, 1:].unsqueeze(-1), 0.0)
        motion_vector_a[:, 1:] = step_delta
        return motion_vector_a

    @staticmethod
    def _build_motion_feature_from_vector(
        motion_vector_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        motion_valid_a: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                safe_norm_2d(motion_vector_a[..., :2]),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a,
                    nbr_vector=motion_vector_a[..., :2],
                ),
                motion_valid_a.to(dtype=motion_vector_a.dtype),
            ],
            dim=-1,
        )

    @staticmethod
    def _build_motion_feature(
        pos_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        motion_vector_a = SMARTAgentEncoder._build_motion_vector(pos_a, valid_mask)
        motion_valid_a = SMARTAgentEncoder._build_motion_valid_mask(pos_a, valid_mask)
        return SMARTAgentEncoder._build_motion_feature_from_vector(
            motion_vector_a=motion_vector_a,
            head_vector_a=head_vector_a,
            motion_valid_a=motion_valid_a,
        )

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
                safe_norm_2d(rel_pos_t),
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

        valid_node_idx = torch.nonzero(mask, as_tuple=False).flatten()
        if valid_node_idx.numel() == 0:
            edge_index_a2a = torch.empty(2, 0, device=pos_a.device, dtype=torch.long)
            r_a2a = self.r_a2a_emb(
                continuous_inputs=pos_a.new_zeros((0, self.r_a2a_emb.input_dim)),
                categorical_embs=None,
            )
            return edge_index_a2a, r_a2a

        pos_valid = pos_s[valid_node_idx]
        batch_valid = batch_s[valid_node_idx]
        edge_index_a2a = radius_graph(
            x=pos_valid[:, :2],
            r=self.a2a_radius,
            batch=batch_valid,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = valid_node_idx[edge_index_a2a]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                safe_norm_2d(rel_pos_a2a[:, :2]),
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
        light_type: Optional[torch.Tensor] = None,
        light_time_delta_norm: Optional[torch.Tensor] = None,
    ):
        n_agent, n_step = pos_a.shape[:2]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        has_light_type = light_type is not None
        if has_light_type:
            light_type = light_type.to(device=pos_pl.device, dtype=torch.long)
        # ``torch_cluster.radius`` 도 ``radius_graph`` 와 마찬가지로 batch index 가
        # 단조 비감소 순서로 들어와야 그룹 분리가 silent 가정으로 동작합니다.
        # caller 는 ``batch_s`` 를 ``tokenized_agent["batch"].repeat(n_step)`` 으로
        # 만들기 때문에 step 사이마다 큰 scene 번호에서 작은 값으로 떨어집니다.
        # 그대로 호출하면 같은 scene 안의 map-agent edge 가 silent 하게 일부
        # 누락돼 일부 agent 가 자기 scene 의 지도 정보를 받지 못하고 학습됩니다.
        # 호출 직전에 양쪽 batch 를 정렬해 ``radius`` 를 부르고, 받은 edge index
        # 를 원래 순서로 되돌려 downstream feature 계산이 그대로 작동하도록 합니다.
        sort_order_x = torch.argsort(batch_s, stable=True)
        sort_order_y = torch.argsort(batch_pl, stable=True)
        edge_index_pl2a_sorted = radius(
            x=pos_s[sort_order_x, :2],
            y=pos_pl[sort_order_y, :2],
            r=self.pl2a_radius,
            batch_x=batch_s[sort_order_x],
            batch_y=batch_pl[sort_order_y],
            max_num_neighbors=300,
        )
        edge_index_pl2a = torch.stack(
            [
                sort_order_y[edge_index_pl2a_sorted[0]],
                sort_order_x[edge_index_pl2a_sorted[1]],
            ],
            dim=0,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                safe_norm_2d(rel_pos_pl2a[:, :2]),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        if has_light_type and edge_index_pl2a.numel() > 0:
            edge_light_type = light_type[edge_index_pl2a[0]]
            signal_edge_mask = edge_light_type != NO_LANE_STATE_LIGHT_TYPE
            if signal_edge_mask.any():
                light_time_delta_norm = resolve_light_time_delta_norm(
                    light_time_delta_norm=light_time_delta_norm,
                    num_agents=n_agent,
                    num_steps=n_step,
                    device=pos_pl.device,
                    dtype=pos_pl.dtype,
                    shift_steps=self.shift,
                )
                light_time_delta_flat = light_time_delta_norm.transpose(0, 1).reshape(-1)
                signal_edge_index = signal_edge_mask.nonzero(as_tuple=False).flatten()
                signal_light_type = edge_light_type[signal_edge_index]
                signal_light_time = light_time_delta_flat[
                    edge_index_pl2a[1, signal_edge_index]
                ].unsqueeze(-1)
                light_bias = self.light_time_pl2a_emb(
                    continuous_inputs=signal_light_time,
                    categorical_embs=[self.light_pl2a_emb(signal_light_type)],
                )
                r_pl2a = r_pl2a.index_add(0, signal_edge_index, light_bias)
        return edge_index_pl2a, r_pl2a
