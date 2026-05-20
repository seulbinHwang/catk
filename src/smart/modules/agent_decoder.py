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

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.dynamic_light_time import (
    NO_LANE_STATE_LIGHT_TYPE,
    normalize_light_time_delta_seconds,
    resolve_step_light_time_delta_norm,
)
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)


class SMARTAgentDecoder(nn.Module):

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
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 3
        input_dim_r_t = 4
        input_dim_r_pt2a = 4
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)
        self.light_relation_type_emb = nn.Embedding(5, hidden_dim)

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
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
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
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token_veh,  # [n_token, 8]
        trajectory_token_ped,  # [n_token, 8]
        trajectory_token_cyc,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        valid_mask: Optional[torch.Tensor] = None,  # [n_agent, n_step]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim),
            device=_device,
            dtype=agent_token_emb_veh.dtype,
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        feature_a = self._build_motion_feature(
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            valid_mask=valid_mask,
        )  # [n_agent, n_step, 3]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    @staticmethod
    def _build_motion_valid_mask(
        pos_a: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_agent, n_step, _ = pos_a.shape
        if valid_mask is None:
            raise ValueError(
                "valid_mask is required for SMART motion features. Missing motion "
                "must not be treated as valid stationary motion."
            )
        if tuple(valid_mask.shape) != (n_agent, n_step):
            raise ValueError(
                "valid_mask shape must match the first two dimensions of pos_a, "
                f"got {tuple(valid_mask.shape)} and {(n_agent, n_step)}."
            )

        motion_valid_a = torch.zeros(
            n_agent,
            n_step,
            device=pos_a.device,
            dtype=torch.bool,
        )
        if n_step > 1:
            motion_valid_a[:, 1:] = valid_mask[:, 1:].bool() & valid_mask[:, :-1].bool()
        return motion_valid_a

    @staticmethod
    def _build_motion_vector(
        pos_a: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_agent, n_step, traj_dim = pos_a.shape
        motion_vector_a = pos_a.new_zeros(n_agent, n_step, traj_dim)
        if n_step <= 1:
            return motion_vector_a

        motion_valid_a = SMARTAgentDecoder._build_motion_valid_mask(pos_a, valid_mask)
        step_delta = pos_a[:, 1:] - pos_a[:, :-1]
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
                torch.norm(motion_vector_a[..., :2], p=2, dim=-1),
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
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        motion_vector_a = SMARTAgentDecoder._build_motion_vector(pos_a, valid_mask)
        motion_valid_a = SMARTAgentDecoder._build_motion_valid_mask(pos_a, valid_mask)
        return SMARTAgentDecoder._build_motion_feature_from_vector(
            motion_vector_a=motion_vector_a,
            head_vector_a=head_vector_a,
            motion_valid_a=motion_valid_a,
        )

    def build_temporal_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2],
        mask,  # [n_agent, n_step]
        inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
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
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        batch_s,  # [n_agent*n_step]
        mask,  # [n_agent, n_step]
    ):
        mask_flat = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        valid_node_idx = torch.nonzero(mask_flat, as_tuple=False).flatten()
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
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl]
        light_type: Optional[torch.Tensor] = None,
        light_time_delta_norm: Optional[torch.Tensor] = None,
    ):
        n_agent, n_step = pos_a.shape[:2]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        # torch_cluster.radius assumes grouped batch ids. With static map tokens,
        # agent ids arrive as [scene0..N, scene0..N, ...] across time steps, so
        # sort before radius and restore the original indices afterward.
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
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self._build_map2agent_relation_embedding(
            geometry_relation=r_pl2a,
            edge_index_pl2a=edge_index_pl2a,
            light_type=light_type,
            light_time_delta_norm=light_time_delta_norm,
            num_map=pos_pl.shape[0],
            num_agents=n_agent,
            num_steps=n_step,
            device=pos_pl.device,
            dtype=pos_pl.dtype,
        )
        return edge_index_pl2a, r_pl2a

    @staticmethod
    def _build_selected_fourier_pre_embedding(
        embedding: FourierEmbedding,
        continuous_inputs: torch.Tensor,
        *,
        input_dim_offset: int,
    ) -> torch.Tensor:
        """FourierEmbedding의 최종 projection 직전 표현을 선택 차원만 계산한다.

        Args:
            embedding: 사용할 FourierEmbedding 모듈이다.
            continuous_inputs: 선택한 연속값 입력이다. Shape은 ``[N, D]``이다.
            input_dim_offset: ``continuous_inputs``의 첫 번째 열이 원래 embedding의 몇 번째
                입력 차원인지 나타낸다.

        Returns:
            최종 ``to_out``을 지나기 전의 합산 표현이다. Shape은 ``[N, hidden_dim]``이다.

        Raises:
            ValueError: 입력 차원 수가 맞지 않거나 embedding 범위를 벗어나면 발생한다.
        """
        if continuous_inputs.ndim != 2:
            raise ValueError(
                "continuous_inputs must have shape [num_items, num_dims], "
                f"got {tuple(continuous_inputs.shape)}."
            )
        num_dims = int(continuous_inputs.shape[1])
        if num_dims <= 0:
            raise ValueError("continuous_inputs must contain at least one dimension.")
        input_dim_offset = int(input_dim_offset)
        if input_dim_offset < 0 or input_dim_offset + num_dims > embedding.input_dim:
            raise ValueError(
                "selected Fourier input dimensions exceed embedding.input_dim, "
                f"offset={input_dim_offset}, num_dims={num_dims}, "
                f"input_dim={embedding.input_dim}."
            )
        if embedding.freqs is None:
            raise ValueError("FourierEmbedding with input_dim=0 cannot embed continuous inputs.")

        # continuous_inputs: [N, D]
        # freqs: [D, F]
        # encoded: [N, D, 2F + 1]
        freqs = embedding.freqs.weight[
            input_dim_offset : input_dim_offset + num_dims
        ]
        encoded = continuous_inputs.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        encoded = torch.cat(
            [encoded.cos(), encoded.sin(), continuous_inputs.unsqueeze(-1)],
            dim=-1,
        )
        pre_embeddings = [
            embedding.mlps[input_dim_offset + dim_idx](encoded[:, dim_idx])
            for dim_idx in range(num_dims)
        ]
        return torch.stack(pre_embeddings).sum(dim=0)

    def _build_stale_time_pre_embedding_by_step(
        self,
        step_light_time_delta_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """신호등 stale-time 표현을 time step 수만큼만 계산한다.

        Args:
            step_light_time_delta_norm: 정규화된 stale-time 값이다. Shape은 ``[num_steps]``이다.

        Returns:
            첫 번째 값은 step별 stale-time pre-embedding이며 shape은
            ``[num_steps, hidden_dim]``이다. 두 번째 값은 stale-time scalar가 0일 때의
            pre-embedding이며 shape은 ``[hidden_dim]``이다.

        Raises:
            ValueError: 입력이 1차원이 아니면 발생한다.
        """
        if step_light_time_delta_norm.ndim != 1:
            raise ValueError(
                "step_light_time_delta_norm must have shape [num_steps], "
                f"got {tuple(step_light_time_delta_norm.shape)}."
            )
        step_stale_pre = self._build_selected_fourier_pre_embedding(
            embedding=self.r_pt2a_emb,
            continuous_inputs=step_light_time_delta_norm.view(-1, 1),
            input_dim_offset=3,
        )
        zero_stale = step_light_time_delta_norm.new_zeros((1, 1))
        zero_stale_pre = self._build_selected_fourier_pre_embedding(
            embedding=self.r_pt2a_emb,
            continuous_inputs=zero_stale,
            input_dim_offset=3,
        )[0]
        return step_stale_pre, zero_stale_pre

    @staticmethod
    def _build_scalar_light_time_delta_norm(
        *,
        delta_seconds: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """closed-loop 한 block의 stale-time 값을 scalar tensor로 만든다.

        Args:
            delta_seconds: 현재 예측 block에서 관측 시점 이후 지난 시간이다.
            device: 결과 tensor를 둘 장치이다.
            dtype: 결과 tensor의 dtype이다.

        Returns:
            정규화된 stale-time scalar이다. Shape은 ``[]``이다.
        """
        delta = torch.as_tensor(float(delta_seconds), device=device, dtype=dtype)
        return normalize_light_time_delta_seconds(delta)

    def _build_map2agent_relation_embedding(
        self,
        geometry_relation: torch.Tensor,
        edge_index_pl2a: torch.Tensor,
        light_type: Optional[torch.Tensor],
        light_time_delta_norm: Optional[torch.Tensor],
        *,
        num_map: int,
        num_agents: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """map-to-agent relation을 f6e96cf8 의미 그대로 더 싸게 만든다.

        Args:
            geometry_relation: edge별 거리, 방향, 상대 heading이다. Shape은 ``[E, 3]``이다.
            edge_index_pl2a: map token에서 agent/time token으로 가는 edge index이다.
                Shape은 ``[2, E]``이다.
            light_type: map token별 현재 관측 신호등 상태이다. Shape은 ``[num_map]``이다.
                값이 없으면 모든 map token을 ``NO_LANE_STATE``로 본다.
            light_time_delta_norm: 정규화된 stale-time 입력이다. ``None``, scalar,
                ``[num_steps]``, 또는 ``[num_agents, num_steps]``를 받을 수 있다.
            num_map: map token 수이다.
            num_agents: agent 수이다.
            num_steps: 현재 relation이 포함하는 model time step 수이다.
            device: 계산 장치이다.
            dtype: 연속값 계산 dtype이다.

        Returns:
            AttentionLayer에 넣을 relation embedding이다. Shape은 ``[E, hidden_dim]``이다.

        Raises:
            ValueError: 입력 shape이 relation 구성과 맞지 않으면 발생한다.
        """
        if geometry_relation.ndim != 2 or geometry_relation.shape[1] != 3:
            raise ValueError(
                "geometry_relation must have shape [num_edges, 3], "
                f"got {tuple(geometry_relation.shape)}."
            )
        if edge_index_pl2a.ndim != 2 or edge_index_pl2a.shape[0] != 2:
            raise ValueError(
                "edge_index_pl2a must have shape [2, num_edges], "
                f"got {tuple(edge_index_pl2a.shape)}."
            )
        if geometry_relation.shape[0] != edge_index_pl2a.shape[1]:
            raise ValueError(
                "geometry_relation and edge_index_pl2a must contain the same number "
                f"of edges, got {geometry_relation.shape[0]} and {edge_index_pl2a.shape[1]}."
            )

        geometry_pre = self._build_selected_fourier_pre_embedding(
            embedding=self.r_pt2a_emb,
            continuous_inputs=geometry_relation.to(device=device, dtype=dtype),
            input_dim_offset=0,
        )
        if edge_index_pl2a.numel() == 0:
            return self.r_pt2a_emb.to_out(geometry_pre)
        if num_map <= 0 or num_agents <= 0 or num_steps <= 0:
            raise ValueError(
                "num_map, num_agents, and num_steps must be positive when edges exist, "
                f"got {num_map}, {num_agents}, {num_steps}."
            )

        if light_type is None:
            light_type_for_map = torch.zeros(num_map, device=device, dtype=torch.long)
        else:
            light_type_for_map = light_type.to(device=device, dtype=torch.long)
            if light_type_for_map.shape[0] != num_map:
                raise ValueError(
                    "light_type must have shape [num_map], "
                    f"got {tuple(light_type_for_map.shape)} and num_map={num_map}."
                )

        step_light_time = resolve_step_light_time_delta_norm(
            light_time_delta_norm=light_time_delta_norm,
            num_steps=num_steps,
            num_agents=num_agents,
            device=device,
            dtype=dtype,
            shift_steps=self.shift,
        )
        stale_pre_by_step, zero_stale_pre = self._build_stale_time_pre_embedding_by_step(
            step_light_time_delta_norm=step_light_time,
        )

        num_light_types = self.light_relation_type_emb.num_embeddings
        edge_light_type_raw = light_type_for_map[edge_index_pl2a[0]]
        edge_light_type = edge_light_type_raw.clamp(min=0, max=num_light_types - 1)
        edge_step = torch.div(edge_index_pl2a[1], num_agents, rounding_mode="floor")
        edge_step = edge_step.clamp(min=0, max=num_steps - 1)

        # edge_light_stale_pre: [E, hidden_dim]
        # NO_LANE_STATE는 f6e96cf8과 같이 stale scalar만 0으로 둔다.
        # light type 0 embedding은 그대로 들어가므로 "완전한 zero bias"와 다르다.
        edge_stale_pre = stale_pre_by_step[edge_step]
        has_observed_light = edge_light_type_raw != NO_LANE_STATE_LIGHT_TYPE
        edge_stale_pre = torch.where(
            has_observed_light.unsqueeze(-1),
            edge_stale_pre,
            zero_stale_pre.view(1, -1),
        )
        relation_pre = (
            geometry_pre
            + edge_stale_pre
            + self.light_relation_type_emb(edge_light_type)
        )
        return self.r_pt2a_emb.to_out(relation_pre)

    @staticmethod
    def _build_sampling_generators_by_batch(
        sampling_seeds: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[list[torch.Generator]]:
        if sampling_seeds is None:
            return None

        generators = []
        for seed in sampling_seeds.detach().cpu().tolist():
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
            valid_mask=mask,  # [n_agent, n_step]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]

        batch_s_a2a = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)  # [n_agent*n_step]
        batch_pl = map_feature["batch"]  # [n_pl]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s_a2a,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],  # [n_pl, 2]
            orient_pl=map_feature["orientation"],  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s_pl2a,  # [n_agent*n_step]
            batch_pl=batch_pl,  # [n_pl]
            light_type=map_feature.get("light_type"),
        )

        # ! attention layers
        feat_map = map_feature["pt_token"]  # [n_pl, hidden_dim]

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a)

        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        scenario_sampling_seeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()
        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            valid_mask=tokenized_agent["valid_mask"][:, :step_current_2hz],
            inference=True,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )

        pred_valid = tokenized_agent["valid_mask"].clone()
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}
        sampling_generators_by_batch = self._build_sampling_generators_by_batch(
            sampling_seeds=scenario_sampling_seeds,
            device=pos_a.device,
        )
        sampling_batch = (
            tokenized_agent["batch"]
            if sampling_generators_by_batch is not None
            else None
        )
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s_a2a = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                batch_s_pl2a = tokenized_agent["batch"].repeat(hist_step)
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s_a2a = tokenized_agent["batch"]
                batch_s_pl2a = tokenized_agent["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            light_time_delta_norm = (
                None
                if t == 0
                else self._build_scalar_light_time_delta_norm(
                    delta_seconds=float(t * self.shift) * 0.1,
                    device=pos_a.device,
                    dtype=pos_a.dtype,
                )
            )
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                batch_s=batch_s_pl2a,  # [n_agent*hist_step]
                batch_pl=map_feature["batch"],  # [n_pl]
                light_type=map_feature.get("light_type"),
                light_time_delta_norm=light_time_delta_norm,
            )
            feat_map = map_feature["pt_token"]
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s_a2a,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    _feat_temporal = self.pt2a_attn_layers[i](
                        (feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                    )
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )
                    # * give same results as below, but more efficient
                    # feat_a_now = self.t_attn_layers[i](
                    #     _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    # ).view(n_agent, n_step, -1)[:, -1]

                    feat_a_now = self.pt2a_attn_layers[i](
                        (feat_map, feat_a_now), r_pl2a, edge_index_pl2a
                    )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            next_token_idx, next_token_traj_all = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
                sampling_generators_by_batch=sampling_generators_by_batch,
                sampling_batch=sampling_batch,
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            motion_valid_a = pred_valid[:, n_step] & pred_valid[:, t_now]
            motion_vector_a = motion_vector_a.masked_fill(
                ~motion_valid_a.unsqueeze(-1),
                0.0,
            )
            x_a = self._build_motion_feature_from_vector(
                motion_vector_a=motion_vector_a,
                head_vector_a=head_vector_a[:, -1],
                motion_valid_a=motion_valid_a,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
        }

        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])

        return out_dict
