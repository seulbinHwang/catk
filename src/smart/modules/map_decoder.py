# Not a contribution
#
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

from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle


class SMARTMapDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pl2pl_radius = pl2pl_radius
        self.poly2road_radius = pl2pl_radius
        self.num_layers = num_layers

        self.type_pt_emb = nn.Embedding(12, hidden_dim)
        self.polygon_type_emb = nn.Embedding(6, hidden_dim)
        self.light_pl_emb = nn.Embedding(5, hidden_dim)

        input_dim_r_pt2pt = 3
        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.pt2pt_layers = nn.ModuleList(
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

        # map_token_traj_src: [n_token, 11, 2].flatten(0, 1)
        self.token_emb = MLPEmbedding(input_dim=22, hidden_dim=hidden_dim)

        # polygon semantic branch
        self.polygon_point_dim = hidden_dim // 2
        self.polygon_point_emb = MLPEmbedding(input_dim=4, hidden_dim=self.polygon_point_dim)
        self.polygon_point_attn = nn.MultiheadAttention(
            embed_dim=self.polygon_point_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.polygon_point_post_norm = nn.LayerNorm(self.polygon_point_dim)
        self.polygon_size_emb = MLPEmbedding(input_dim=2, hidden_dim=32)
        self.polygon_semantic_type_emb = nn.Embedding(3, 32)
        self.polygon_fusion_emb = MLPEmbedding(input_dim=hidden_dim + 32 + 32, hidden_dim=hidden_dim)

        self.r_road2poly_emb = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.road2poly_attn = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=True,
            has_pos_emb=True,
        )
        self.map_source_emb = nn.Embedding(2, hidden_dim)

        self.apply(weight_init)

    def _encode_polygon_tokens(
        self,
        boundary_local: torch.Tensor,
        polygon_size: torch.Tensor,
        polygon_type: torch.Tensor,
    ) -> torch.Tensor:
        """polygon 경계 입력을 초기 의미 토큰으로 바꿉니다.

        Args:
            boundary_local: 중심 기준 local 경계점입니다.
                shape은 ``[n_poly, k_boundary, 2]`` 입니다.
            polygon_size: 짧은 축과 긴 축 길이입니다.
                shape은 ``[n_poly, 2]`` 입니다.
            polygon_type: polygon 종류 번호입니다.
                shape은 ``[n_poly]`` 입니다.
                0은 crosswalk, 1은 speed bump, 2는 driveway 입니다.

        Returns:
            torch.Tensor: 자기 모양과 종류만 담은 초기 polygon token 입니다.
                shape은 ``[n_poly, hidden_dim]`` 입니다.
        """
        if boundary_local.shape[0] == 0:
            return boundary_local.new_zeros((0, self.hidden_dim))

        boundary_next = torch.roll(boundary_local, shifts=-1, dims=1)
        boundary_edge = boundary_next - boundary_local
        boundary_input = torch.cat([boundary_local, boundary_edge], dim=-1)

        n_poly, k_boundary, _ = boundary_input.shape
        point_hidden = self.polygon_point_emb(boundary_input.view(-1, 4))
        point_hidden = point_hidden.view(n_poly, k_boundary, self.polygon_point_dim)

        point_hidden_attn, _ = self.polygon_point_attn(
            point_hidden,
            point_hidden,
            point_hidden,
            need_weights=False,
        )
        point_hidden = self.polygon_point_post_norm(point_hidden + point_hidden_attn)

        shape_summary = torch.cat(
            [point_hidden.mean(dim=1), point_hidden.max(dim=1).values],
            dim=-1,
        )
        size_feature = self.polygon_size_emb(polygon_size)
        type_feature = self.polygon_semantic_type_emb(polygon_type)
        fused_feature = torch.cat([shape_summary, size_feature, type_feature], dim=-1)
        return self.polygon_fusion_emb(fused_feature)

    def _refine_polygon_tokens(
        self,
        x_poly: torch.Tensor,
        pos_poly: torch.Tensor,
        orient_poly: torch.Tensor,
        batch_poly: torch.Tensor,
        x_road: torch.Tensor,
        pos_road: torch.Tensor,
        orient_road: torch.Tensor,
        batch_road: torch.Tensor,
    ) -> torch.Tensor:
        """주변 road token을 한 번만 읽어 polygon token을 다듬습니다.

        Args:
            x_poly: 초기 polygon token 입니다. shape은 ``[n_poly, hidden_dim]`` 입니다.
            pos_poly: polygon 중심점입니다. shape은 ``[n_poly, 2]`` 입니다.
            orient_poly: polygon 방향입니다. shape은 ``[n_poly]`` 입니다.
            batch_poly: polygon 배치 번호입니다. shape은 ``[n_poly]`` 입니다.
            x_road: road token 입니다. shape은 ``[n_road, hidden_dim]`` 입니다.
            pos_road: road 중심점입니다. shape은 ``[n_road, 2]`` 입니다.
            orient_road: road 방향입니다. shape은 ``[n_road]`` 입니다.
            batch_road: road 배치 번호입니다. shape은 ``[n_road]`` 입니다.

        Returns:
            torch.Tensor: 주변 도로 문맥이 반영된 polygon token 입니다.
                shape은 ``[n_poly, hidden_dim]`` 입니다.
        """
        if x_poly.shape[0] == 0 or x_road.shape[0] == 0:
            return x_poly

        orient_vector_poly = torch.stack([orient_poly.cos(), orient_poly.sin()], dim=-1)
        edge_index_road2poly = radius(
            x=pos_poly[:, :2],
            y=pos_road[:, :2],
            r=self.poly2road_radius,
            batch_x=batch_poly,
            batch_y=batch_road,
            max_num_neighbors=100,
        )
        if edge_index_road2poly.numel() == 0:
            return x_poly

        rel_pos_road2poly = pos_road[edge_index_road2poly[0]] - pos_poly[edge_index_road2poly[1]]
        rel_orient_road2poly = wrap_angle(
            orient_road[edge_index_road2poly[0]] - orient_poly[edge_index_road2poly[1]]
        )
        r_road2poly = torch.stack(
            [
                torch.norm(rel_pos_road2poly[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector_poly[edge_index_road2poly[1]],
                    nbr_vector=rel_pos_road2poly[:, :2],
                ),
                rel_orient_road2poly,
            ],
            dim=-1,
        )
        r_road2poly = self.r_road2poly_emb(
            continuous_inputs=r_road2poly,
            categorical_embs=None,
        )
        return self.road2poly_attn((x_road, x_poly), r_road2poly, edge_index_road2poly)

    def forward(self, tokenized_map: Dict) -> Dict[str, torch.Tensor]:
        pos_pt = tokenized_map["position"]
        orient_pt = tokenized_map["orientation"]

        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        pt_token_emb_src = self.token_emb(tokenized_map["token_traj_src"])
        x_pt = pt_token_emb_src[tokenized_map["token_idx"]]
        x_pt_categorical_embs = [
            self.type_pt_emb(tokenized_map["type"]),
            self.polygon_type_emb(tokenized_map["pl_type"]),
            self.light_pl_emb(tokenized_map["light_type"]),
        ]
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)

        edge_index_pt2pt = radius_graph(
            x=pos_pt,
            r=self.pl2pl_radius,
            batch=tokenized_map["batch"],
            loop=False,
            max_num_neighbors=100,
        )
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
        )
        r_pt2pt = torch.stack(
            [
                torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                    nbr_vector=rel_pos_pt2pt[:, :2],
                ),
                rel_orient_pt2pt,
            ],
            dim=-1,
        )
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        has_polygon = (
            "polygon_position" in tokenized_map
            and tokenized_map["polygon_position"].numel() > 0
        )
        if not has_polygon:
            return {
                "pt_token": x_pt,
                "position": pos_pt,
                "orientation": orient_pt,
                "batch": tokenized_map["batch"],
            }

        pos_poly = tokenized_map["polygon_position"]
        orient_poly = tokenized_map["polygon_orientation"]
        boundary_local = tokenized_map["polygon_boundary"]
        size_poly = tokenized_map["polygon_size"]
        type_poly = tokenized_map["polygon_type"]
        batch_poly = tokenized_map["polygon_batch"]

        x_poly = self._encode_polygon_tokens(
            boundary_local=boundary_local,
            polygon_size=size_poly,
            polygon_type=type_poly,
        )
        x_poly = self._refine_polygon_tokens(
            x_poly=x_poly,
            pos_poly=pos_poly,
            orient_poly=orient_poly,
            batch_poly=batch_poly,
            x_road=x_pt,
            pos_road=pos_pt,
            orient_road=orient_pt,
            batch_road=tokenized_map["batch"],
        )

        road_source = self.map_source_emb(
            torch.zeros(x_pt.shape[0], dtype=torch.long, device=x_pt.device)
        )
        poly_source = self.map_source_emb(
            torch.ones(x_poly.shape[0], dtype=torch.long, device=x_poly.device)
        )

        return {
            "pt_token": torch.cat([x_pt + road_source, x_poly + poly_source], dim=0),
            "position": torch.cat([pos_pt, pos_poly], dim=0),
            "orientation": torch.cat([orient_pt, orient_poly], dim=0),
            "batch": torch.cat([tokenized_map["batch"], batch_poly], dim=0),
        }
