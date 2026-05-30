from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.utils import angle_between_2d_vectors, transform_to_local, weight_init, wrap_angle


def _masked_attention_pool(x: Tensor, valid: Tensor | None, scorer: nn.Module) -> Tensor:
    scores = scorer(x).squeeze(-1)
    if valid is None:
        weights = scores.softmax(dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=-2)

    valid = valid.bool()
    any_valid = valid.any(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    masked_scores = torch.where(any_valid, masked_scores, torch.zeros_like(masked_scores))
    weights = masked_scores.softmax(dim=-1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    return (x * weights.unsqueeze(-1)).sum(dim=-2)


def _fold_legacy_surface_categories(
    point_type: Tensor,
    polygon_type: Tensor,
) -> tuple[Tensor, Tensor]:
    point_type = torch.where(point_type >= 10, point_type.new_full((), 9), point_type)
    polygon_type = torch.where(polygon_type >= 4, polygon_type.new_full((), 3), polygon_type)
    return point_type, polygon_type


class UniMMMapEncoder(nn.Module):
    """Continuous map polyline encoder for UniMM.

    The repository cache already splits map features into short local polylines.
    This encoder embeds those local coordinates directly instead of using SMART's
    discrete map token vocabulary.
    """

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
        super().__init__()
        self.pl2pl_radius = float(pl2pl_radius)
        self.num_layers = int(num_layers)

        self.map_point_emb = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.map_point_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.type_pt_emb = nn.Embedding(10, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(5, hidden_dim)
        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=3,
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
                for _ in range(self.num_layers)
            ]
        )
        self.apply(weight_init)

    def forward(self, tokenized_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pos_pt = tokenized_map["position"]
        orient_pt = tokenized_map["orientation"]
        traj_pos = tokenized_map["traj_pos"]
        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,
            head_global=None,
            pos_now=pos_pt,
            head_now=orient_pt,
        )
        n_point = traj_pos_local.shape[1]
        point_offset = torch.linspace(
            -1.0,
            1.0,
            steps=n_point,
            device=traj_pos_local.device,
            dtype=traj_pos_local.dtype,
        ).view(1, n_point, 1)
        point_feature = torch.cat(
            [traj_pos_local, point_offset.expand(traj_pos_local.shape[0], -1, -1)],
            dim=-1,
        )
        x_point = self.map_point_emb(
            continuous_inputs=point_feature.reshape(-1, point_feature.shape[-1]),
            categorical_embs=None,
        ).view(traj_pos_local.shape[0], n_point, -1)
        x_pt = _masked_attention_pool(x_point, valid=None, scorer=self.map_point_score)

        point_type, polygon_type = _fold_legacy_surface_categories(
            tokenized_map["type"],
            tokenized_map["pl_type"],
        )
        x_pt = x_pt + self.type_pt_emb(point_type)
        x_pt = x_pt + self.polygon_type_emb(polygon_type)
        x_pt = x_pt + self.light_pl_emb(tokenized_map["light_type"])

        edge_index = radius_graph(
            x=pos_pt,
            r=self.pl2pl_radius,
            batch=tokenized_map["batch"],
            loop=False,
            max_num_neighbors=100,
        )
        if edge_index.numel() == 0:
            return {
                "pt_token": x_pt,
                "position": pos_pt,
                "orientation": orient_pt,
                "batch": tokenized_map["batch"],
                "light_type": tokenized_map["light_type"],
            }

        orient_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        rel_pos = pos_pt[edge_index[0]] - pos_pt[edge_index[1]]
        rel_orient = wrap_angle(orient_pt[edge_index[0]] - orient_pt[edge_index[1]])
        r = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_orient,
            ],
            dim=-1,
        )
        r = self.r_pt2pt_emb(continuous_inputs=r, categorical_embs=None)
        for layer in self.pt2pt_layers:
            x_pt = layer(x_pt, r, edge_index)

        return {
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": tokenized_map["batch"],
            "light_type": tokenized_map["light_type"],
        }


class UniMMAgentEncoder(nn.Module):
    """Factorized agent/map encoder for anchor-based UniMM."""

    def __init__(
        self,
        hidden_dim: int,
        time_span: int,
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.time_span = int(time_span)
        self.pl2a_radius = float(pl2a_radius)
        self.a2a_radius = float(a2a_radius)
        self.num_layers = int(num_layers)

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)
        self.tracklet_emb = FourierEmbedding(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.tracklet_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=3,
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
                for _ in range(self.num_layers)
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
                for _ in range(self.num_layers)
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
                for _ in range(self.num_layers)
            ]
        )
        self.apply(weight_init)

    def _agent_state_embedding(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        valid: Tensor,
        agent_type: Tensor,
        agent_shape: Tensor,
        tracklet_pos: Tensor | None = None,
        tracklet_head: Tensor | None = None,
        tracklet_valid: Tensor | None = None,
    ) -> Tensor:
        n_agent, n_step, _ = pos_a.shape
        categorical = [self.type_a_emb(agent_type.long()), self.shape_emb(agent_shape)]
        if tracklet_pos is None or tracklet_head is None or tracklet_valid is None:
            tracklet_pos = pos_a.unsqueeze(2)
            tracklet_head = head_a.unsqueeze(2)
            tracklet_valid = valid.unsqueeze(-1)

        n_tracklet = tracklet_pos.shape[2]
        local_pos, local_head = transform_to_local(
            pos_global=tracklet_pos.reshape(n_agent * n_step, n_tracklet, 2),
            head_global=tracklet_head.reshape(n_agent * n_step, n_tracklet),
            pos_now=pos_a.reshape(n_agent * n_step, 2),
            head_now=head_a.reshape(n_agent * n_step),
        )
        local_pos = local_pos.view(n_agent, n_step, n_tracklet, 2)
        local_head = wrap_angle(local_head).view(n_agent, n_step, n_tracklet)
        rel_time = torch.linspace(
            -1.0,
            0.0,
            steps=n_tracklet,
            device=pos_a.device,
            dtype=pos_a.dtype,
        ).view(1, 1, n_tracklet, 1)
        tracklet_feature = torch.cat(
            [
                local_pos,
                local_head.unsqueeze(-1),
                rel_time.expand(n_agent, n_step, -1, -1),
            ],
            dim=-1,
        )
        feat_point = self.tracklet_emb(
            continuous_inputs=tracklet_feature.reshape(-1, tracklet_feature.shape[-1]),
            categorical_embs=[
                emb.repeat_interleave(n_step * n_tracklet, dim=0)
                for emb in categorical
            ],
        ).view(n_agent, n_step, n_tracklet, self.hidden_dim)
        valid_point = tracklet_valid & valid.unsqueeze(-1)
        feat = _masked_attention_pool(feat_point, valid=valid_point, scorer=self.tracklet_score)
        return feat.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _temporal_edge(self, pos_a: Tensor, head_a: Tensor, head_vector_a: Tensor, valid: Tensor):
        n_agent, n_step = valid.shape
        mask_t = valid.unsqueeze(2) & valid.unsqueeze(1)
        edge_index = dense_to_sparse(mask_t)[0]
        edge_index = edge_index[:, edge_index[1] > edge_index[0]]
        edge_index = edge_index[:, edge_index[1] - edge_index[0] <= self.time_span]
        if edge_index.numel() == 0:
            r = self.r_t_emb(
                continuous_inputs=pos_a.new_zeros((0, 4)),
                categorical_embs=None,
            )
            return edge_index, r

        pos_flat = pos_a.flatten(0, 1)
        head_flat = head_a.flatten(0, 1)
        head_vector_flat = head_vector_a.flatten(0, 1)
        rel_pos = pos_flat[edge_index[0]] - pos_flat[edge_index[1]]
        rel_head = wrap_angle(head_flat[edge_index[0]] - head_flat[edge_index[1]])
        r = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_flat[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_head,
                (edge_index[0] - edge_index[1]).to(dtype=pos_a.dtype),
            ],
            dim=-1,
        )
        return edge_index, self.r_t_emb(continuous_inputs=r, categorical_embs=None)

    def _agent_agent_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        batch_s: Tensor,
        valid: Tensor,
    ):
        mask_flat = valid.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        valid_node_idx = torch.nonzero(mask_flat, as_tuple=False).flatten()
        if valid_node_idx.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=pos_a.device)
            r = self.r_a2a_emb(
                continuous_inputs=pos_a.new_zeros((0, 3)),
                categorical_embs=None,
            )
            return edge_index, r

        edge_index = radius_graph(
            x=pos_s[valid_node_idx, :2],
            r=self.a2a_radius,
            batch=batch_s[valid_node_idx],
            loop=False,
            max_num_neighbors=300,
        )
        edge_index = valid_node_idx[edge_index]
        if edge_index.numel() == 0:
            r = self.r_a2a_emb(
                continuous_inputs=pos_a.new_zeros((0, 3)),
                categorical_embs=None,
            )
            return edge_index, r

        rel_pos = pos_s[edge_index[0]] - pos_s[edge_index[1]]
        rel_head = wrap_angle(head_s[edge_index[0]] - head_s[edge_index[1]])
        r = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_head,
            ],
            dim=-1,
        )
        return edge_index, self.r_a2a_emb(continuous_inputs=r, categorical_embs=None)

    def _map_agent_edge(
        self,
        map_feature: Dict[str, Tensor],
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        batch_s: Tensor,
        valid: Tensor,
    ):
        mask_flat = valid.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        sort_order_x = torch.argsort(batch_s, stable=True)
        sort_order_y = torch.argsort(map_feature["batch"], stable=True)
        edge_sorted = radius(
            x=pos_s[sort_order_x, :2],
            y=map_feature["position"][sort_order_y, :2],
            r=self.pl2a_radius,
            batch_x=batch_s[sort_order_x],
            batch_y=map_feature["batch"][sort_order_y],
            max_num_neighbors=300,
        )
        edge_index = torch.stack(
            [sort_order_y[edge_sorted[0]], sort_order_x[edge_sorted[1]]],
            dim=0,
        )
        edge_index = edge_index[:, mask_flat[edge_index[1]]]
        if edge_index.numel() == 0:
            r = self.r_pt2a_emb(
                continuous_inputs=pos_a.new_zeros((0, 3)),
                categorical_embs=None,
            )
            return edge_index, r

        rel_pos = map_feature["position"][edge_index[0]] - pos_s[edge_index[1]]
        rel_orient = wrap_angle(map_feature["orientation"][edge_index[0]] - head_s[edge_index[1]])
        r = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_orient,
            ],
            dim=-1,
        )
        return edge_index, self.r_pt2a_emb(continuous_inputs=r, categorical_embs=None)

    def forward(self, tokenized_agent: Dict[str, Tensor], map_feature: Dict[str, Tensor]) -> Tensor:
        pos_a = tokenized_agent["state_pos"]
        head_a = tokenized_agent["state_head"]
        valid = tokenized_agent["state_valid"]
        n_agent, n_step, _ = pos_a.shape
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        feat_a = self._agent_state_embedding(
            pos_a=pos_a,
            head_a=head_a,
            valid=valid,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            tracklet_pos=tokenized_agent.get("tracklet_pos"),
            tracklet_head=tokenized_agent.get("tracklet_head"),
            tracklet_valid=tokenized_agent.get("tracklet_valid"),
        )
        edge_index_t, r_t = self._temporal_edge(pos_a, head_a, head_vector_a, valid)

        batch_s_a2a = torch.cat(
            [tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t for t in range(n_step)],
            dim=0,
        )
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)
        edge_index_a2a, r_a2a = self._agent_agent_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            batch_s=batch_s_a2a,
            valid=valid,
        )
        edge_index_pl2a, r_pl2a = self._map_agent_edge(
            map_feature=map_feature,
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            batch_s=batch_s_pl2a,
            valid=valid,
        )

        feat_map = map_feature["pt_token"]
        for layer_idx in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[layer_idx](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[layer_idx](
                (feat_map, feat_a),
                r_pl2a,
                edge_index_pl2a,
            )
            feat_a = self.a2a_attn_layers[layer_idx](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
        return feat_a


class UniMMMotionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_anchors: int,
        num_prediction_steps: int,
        min_laplace_scale: float = 0.05,
        min_von_mises_concentration: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.num_prediction_steps = int(num_prediction_steps)
        self.min_laplace_scale = float(min_laplace_scale)
        self.min_von_mises_concentration = float(min_von_mises_concentration)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_prediction_steps * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_prediction_steps * 6),
        )
        self.apply(weight_init)

    def decode_selected(self, agent_embedding: Tensor, selected_anchor: Tensor) -> Dict[str, Tensor]:
        anchor_feat = self.anchor_encoder(selected_anchor.flatten(-2, -1))
        raw = self.regressor(torch.cat([agent_embedding, anchor_feat], dim=-1))
        raw = raw.view(*agent_embedding.shape[:-1], self.num_prediction_steps, 6)

        mean_pos = selected_anchor[..., :2] + raw[..., :2]
        mean_head = wrap_angle(selected_anchor[..., 2] + raw[..., 2])
        pos_scale = torch.nn.functional.softplus(raw[..., 3:5]) + self.min_laplace_scale
        concentration = (
            torch.nn.functional.softplus(raw[..., 5]) + self.min_von_mises_concentration
        )
        return {
            "mean_pos": mean_pos,
            "mean_head": mean_head,
            "pos_scale": pos_scale,
            "head_concentration": concentration,
        }

    def forward(self, agent_embedding: Tensor, selected_anchor: Tensor) -> Dict[str, Tensor]:
        pred = self.decode_selected(agent_embedding, selected_anchor)
        pred["logits"] = self.scorer(agent_embedding)
        return pred


class UniMMAnchorBasedNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_anchors: int,
        num_prediction_steps: int,
        pl2pl_radius: float,
        pl2a_radius: float,
        a2a_radius: float,
        time_span: int,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        min_laplace_scale: float,
        min_von_mises_concentration: float,
    ) -> None:
        super().__init__()
        self.map_encoder = UniMMMapEncoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = UniMMAgentEncoder(
            hidden_dim=hidden_dim,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.motion_decoder = UniMMMotionDecoder(
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            num_prediction_steps=num_prediction_steps,
            min_laplace_scale=min_laplace_scale,
            min_von_mises_concentration=min_von_mises_concentration,
        )

    def encode(self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]) -> Tensor:
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder(tokenized_agent, map_feature)
