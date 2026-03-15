from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.flow import FlowODE
from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.flow_conditioner import FutureConditioner, StructuredFlowHead
from src.smart.utils import (
    angle_between_2d_vectors,
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    weight_init,
    wrap_angle,
)

class ContinuousCommitBridge:
    """연속 flow 출력을 SMART coarse rollout 상태로 잇는 보조 모듈이다.

    이 모듈은 학습 파라미터를 늘리지 않는다. 첫 0.5초 구간을 global 좌표로
    복원하고, 현재 contour + commit된 5개 미래 contour를 함께 비교해서
    다음 coarse token을 고른다.
    """

    def __init__(self, flow_position_scale: float) -> None:
        self.flow_position_scale = flow_position_scale

    def commit(
        self,
        future_local_norm: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """정규화된 2초 미래에서 첫 0.5초만 실제 좌표로 복원한다.

        Args:
            future_local_norm: [n_active, 20, 4] 모양의 정규화된 local 미래이다.
            current_pos: [n_active, 2] 모양의 현재 global 위치이다.
            current_head: [n_active] 모양의 현재 global heading이다.

        Returns:
            commit_pos: [n_active, 5, 2] 모양의 commit 위치이다.
            commit_head: [n_active, 5] 모양의 commit heading이다.
            next_pos: [n_active, 2] 모양의 다음 coarse 위치이다.
            next_head: [n_active] 모양의 다음 coarse heading이다.
        """
        local_commit = future_local_norm[:, :5].clone()
        local_commit[..., :2] = local_commit[..., :2] * self.flow_position_scale
        commit_pos, _ = transform_to_global(
            pos_local=local_commit[..., :2],
            head_local=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        delta_head = torch.atan2(local_commit[..., 3], local_commit[..., 2])
        commit_head = wrap_angle(current_head.unsqueeze(1) + delta_head)
        next_pos = commit_pos[:, -1]
        next_head = commit_head[:, -1]
        return commit_pos, commit_head, next_pos, next_head

    def retokenize(
        self,
        current_pos: Tensor,
        current_head: Tensor,
        commit_pos: Tensor,
        commit_head: Tensor,
        token_traj_all: Tensor,
        token_agent_shape: Tensor,
    ) -> Tensor:
        """현재 contour와 commit된 5개 contour를 함께 써서 다음 token을 고른다.

        Args:
            current_pos: [n_active, 2] 모양의 현재 global 위치이다.
            current_head: [n_active] 모양의 현재 global heading이다.
            commit_pos: [n_active, 5, 2] 모양의 commit 위치이다.
            commit_head: [n_active, 5] 모양의 commit heading이다.
            token_traj_all: [n_active, n_token, 6, 4, 2] 모양의 contour token 사전이다.
            token_agent_shape: [n_active, 2] 모양의 agent 폭/길이이다.

        Returns:
            [n_active] 모양의 다음 coarse token index이다.
        """
        current_contour = cal_polygon_contour(current_pos, current_head, token_agent_shape)
        future_contours = [
            cal_polygon_contour(commit_pos[:, step_idx], commit_head[:, step_idx], token_agent_shape)
            for step_idx in range(commit_pos.shape[1])
        ]
        contour_global = torch.stack([current_contour] + future_contours, dim=1)
        contour_local, _ = transform_to_local(
            pos_global=contour_global.flatten(1, 2),
            head_global=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        contour_local = contour_local.view_as(contour_global)
        dist = torch.norm(token_traj_all - contour_local.unsqueeze(1), dim=-1).mean(dim=(-1, -2))
        return torch.argmin(dist, dim=-1)


class SMARTAgentDecoder(nn.Module):
    """SMART backbone 위에 flow matching head를 올린 agent decoder이다.

    기존 SMART의 coarse token 기반 장면 인코더는 유지하고,
    마지막 token classification head만 flow matching용 구조로 교체한다.
    """

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
        flow_num_future_steps: int,
        flow_num_anchors: int,
        flow_anchor_stride: int,
        commit_num_future_steps: int,
        flow_tau_eps: float,
        flow_position_scale: float = 20.0,
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
        self.flow_num_future_steps = flow_num_future_steps
        self.flow_num_anchors = flow_num_anchors
        self.flow_anchor_stride = flow_anchor_stride
        self.commit_num_future_steps = commit_num_future_steps
        self.query_time_span_token = max(1, self.time_span // self.shift)
        self.step_current_token = num_historical_steps // self.shift
        self.flow_ode = FlowODE(tau_eps=flow_tau_eps)
        self.flow_position_scale = flow_position_scale
        self.commit_bridge = ContinuousCommitBridge(flow_position_scale=flow_position_scale)

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 5
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

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

        self.future_conditioner = FutureConditioner(
            future_dim=4,
            hidden_dim=hidden_dim,
            num_blocks=2,
        )
        self.query_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_query_attn = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=True,
            has_pos_emb=True,
        )
        self.flow_head = StructuredFlowHead(
            hidden_dim=hidden_dim,
            num_future_steps=flow_num_future_steps,
            output_dim=4,
        )
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index: Tensor,
        trajectory_token_veh: Tensor,
        trajectory_token_ped: Tensor,
        trajectory_token_cyc: Tensor,
        pos_a: Tensor,
        head_vector_a: Tensor,
        agent_type: Tensor,
        agent_shape: Tensor,
    ) -> Tensor:
        """coarse token과 실제 motion을 합쳐 agent 입력 특징을 만든다.

        Args:
            agent_token_index: [n_agent, n_step] coarse token id이다.
            trajectory_token_veh: [n_token, 8] 차량용 coarse token 사전이다.
            trajectory_token_ped: [n_token, 8] 보행자용 coarse token 사전이다.
            trajectory_token_cyc: [n_token, 8] 자전거용 coarse token 사전이다.
            pos_a: [n_agent, n_step, 2] 실제 위치이다.
            head_vector_a: [n_agent, n_step, 2] heading unit vector이다.
            agent_type: [n_agent] agent 종류이다.
            agent_shape: [n_agent, 3] agent 크기이다.

        Returns:
            [n_agent, n_step, hidden_dim] 모양의 입력 특징이다.
        """
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

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a,
                    nbr_vector=motion_vector_a[:, :, :2],
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]
        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                value.repeat_interleave(repeats=n_step, dim=0)
                for value in categorical_embs
            ],
        )
        x_a = x_a.view(-1, n_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)
        return feat_a

    def build_temporal_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """장면 인코더용 same-agent 시간 edge를 만든다.

        Args:
            pos_a: [n_agent, n_step, 2] 실제 위치이다.
            head_a: [n_agent, n_step] heading이다.
            head_vector_a: [n_agent, n_step, 2] heading unit vector이다.
            mask: [n_agent, n_step] coarse step 유효 마스크이다.

        Returns:
            edge index와 relation embedding을 돌려준다.
        """
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            keep_mask = torch.bernoulli(
                torch.ones_like(mask, dtype=pos_a.dtype) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & keep_mask

        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]],
                    nbr_vector=rel_pos_t[:, :2],
                ),
                rel_head_t,
                (edge_index_t[0] - edge_index_t[1]).to(pos_a.dtype),
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        batch_s: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """현재 anchor 상태만으로 a2a edge와 relation을 만든다.

        relation에는 기존 거리/방향/상대 heading에 더해,
        target anchor local frame 기준 상대 이동량 2개를 넣는다.
        """
        # 이 backbone은 다른 continuous edge feature를 별도 정규화하지 않는다.
        # 그래서 a2a motion feature도 m/s 대신 coarse 0.5초 step 이동량[m]으로 맞춘다.
        motion_valid = torch.cat(
            [
                mask.new_zeros(mask.shape[0], 1),
                mask[:, 1:] & mask[:, :-1],
            ],
            dim=1,
        )
        motion_a = torch.cat(
            [
                pos_a.new_zeros(pos_a.shape[0], 1, 2),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2], coarse 0.5초 step displacement in meters
        motion_a = motion_a.masked_fill(~motion_valid.unsqueeze(-1), 0.0)
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        motion_s = motion_a.transpose(0, 1).reshape(-1, 2)

        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(
            head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]]
        )

        rel_motion = motion_s[edge_index_a2a[0]] - motion_s[edge_index_a2a[1]]
        target_heading = head_vector_s[edge_index_a2a[1]]
        target_left = torch.stack([-target_heading[:, 1], target_heading[:, 0]], dim=-1)
        rel_motion_long = (rel_motion * target_heading).sum(dim=-1)
        rel_motion_lat = (rel_motion * target_left).sum(dim=-1)

        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
                rel_motion_long,
                rel_motion_lat,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl: Tensor,
        orient_pl: Tensor,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        batch_s: Tensor,
        batch_pl: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """현재 anchor 상태 기준 map-to-agent edge를 만든다."""
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
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

    def encode_scene(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        pos_a: Tensor,
        head_a: Tensor,
        mask: Tensor,
        agent_token_index: Tensor,
    ) -> Tensor:
        """scene-shared coarse token backbone를 그대로 통과시킨다.

        Args:
            tokenized_agent: agent 관련 입력 딕셔너리이다.
            map_feature: map encoder 출력이다.
            pos_a: [n_agent, n_step, 2] 실제 위치이다.
            head_a: [n_agent, n_step] heading이다.
            mask: [n_agent, n_step] coarse step 유효 마스크이다.
            agent_token_index: [n_agent, n_step] coarse token id이다.

        Returns:
            [n_agent, n_step, hidden_dim] 모양의 장면 문맥 특징이다.
        """
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        feat_a = self.agent_token_embedding(
            agent_token_index=agent_token_index,
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
        )

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
        )
        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * step_idx
                for step_idx in range(n_step)
            ],
            dim=0,
        )
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * step_idx
                for step_idx in range(n_step)
            ],
            dim=0,
        )
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            batch_s=batch_s,
            mask=mask,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
            batch_s=batch_s,
            batch_pl=batch_pl,
        )

        feat_map = map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        for layer_idx in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[layer_idx](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[layer_idx]((feat_map, feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[layer_idx](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
        return feat_a

    def build_anchor_temporal_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        anchor_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """anchor query가 history coarse token을 보도록 edge를 만든다.

        Args:
            pos_a: [n_agent, n_step, 2] 실제 위치이다.
            head_a: [n_agent, n_step] heading이다.
            head_vector_a: [n_agent, n_step, 2] heading unit vector이다.
            mask: [n_agent, n_step] coarse step 유효 마스크이다.
            anchor_indices: [n_anchor] anchor가 가리키는 coarse step index이다.

        Returns:
            anchor query attention용 edge와 relation embedding이다.
        """
        device = pos_a.device
        n_agent, n_step, _ = pos_a.shape
        n_anchor = anchor_indices.shape[0]

        src_steps = torch.arange(n_step, device=device).view(1, n_step, 1)
        dst_steps = anchor_indices.view(1, 1, n_anchor)
        src_valid = mask.unsqueeze(-1)
        dst_valid = mask[:, anchor_indices].unsqueeze(1)
        edge_mask = src_valid & dst_valid & (src_steps <= dst_steps)
        edge_mask = edge_mask & ((dst_steps - src_steps) <= self.query_time_span_token)

        edge_triplets = edge_mask.nonzero(as_tuple=False)
        if edge_triplets.numel() == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            empty_r = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_r

        agent_idx = edge_triplets[:, 0]
        src_step_idx = edge_triplets[:, 1]
        anchor_local_idx = edge_triplets[:, 2]
        dst_step_idx = anchor_indices[anchor_local_idx]

        src_index = agent_idx * n_step + src_step_idx
        dst_index = agent_idx * n_anchor + anchor_local_idx
        edge_index = torch.stack([src_index, dst_index], dim=0)

        rel_pos = pos_a[agent_idx, src_step_idx] - pos_a[agent_idx, dst_step_idx]
        rel_head = wrap_angle(head_a[agent_idx, src_step_idx] - head_a[agent_idx, dst_step_idx])
        rel_time = (src_step_idx - dst_step_idx).to(pos_a.dtype)
        anchor_head_vec = head_vector_a[agent_idx, dst_step_idx]
        r_t = torch.stack(
            [
                torch.norm(rel_pos, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=anchor_head_vec,
                    nbr_vector=rel_pos,
                ),
                rel_head,
                rel_time,
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index, r_t

    def build_sparse_anchor_temporal_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        agent_indices: Tensor,
        anchor_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """유효한 anchor query만 대상으로 sparse temporal edge를 만든다.

        Args:
            pos_a: [n_agent, n_step, 2] 모양의 coarse 위치이다.
            head_a: [n_agent, n_step] 모양의 coarse heading이다.
            head_vector_a: [n_agent, n_step, 2] 모양의 heading unit vector이다.
            mask: [n_agent, n_step] 모양의 coarse step 유효 마스크이다.
            agent_indices: [n_query] 모양의 query별 agent index이다.
            anchor_indices: [n_query] 모양의 query별 coarse anchor index이다.

        Returns:
            sparse query attention용 edge index와 relation embedding을 돌려준다.
        """
        device = pos_a.device
        _, n_step, _ = pos_a.shape
        n_query = agent_indices.shape[0]
        if n_query == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            empty_r = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_r

        src_steps = torch.arange(n_step, device=device).view(1, n_step)
        src_valid = mask[agent_indices]
        dst_valid = mask[agent_indices, anchor_indices].unsqueeze(-1)
        edge_mask = src_valid & dst_valid
        edge_mask = edge_mask & (src_steps <= anchor_indices.unsqueeze(-1))
        edge_mask = edge_mask & ((anchor_indices.unsqueeze(-1) - src_steps) <= self.query_time_span_token)

        query_idx, src_step_idx = edge_mask.nonzero(as_tuple=True)
        if query_idx.numel() == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            empty_r = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_r

        query_agent_idx = agent_indices[query_idx]
        dst_step_idx = anchor_indices[query_idx]
        src_index = query_agent_idx * n_step + src_step_idx
        dst_index = query_idx
        edge_index = torch.stack([src_index, dst_index], dim=0)

        rel_pos = pos_a[query_agent_idx, src_step_idx] - pos_a[query_agent_idx, dst_step_idx]
        rel_head = wrap_angle(head_a[query_agent_idx, src_step_idx] - head_a[query_agent_idx, dst_step_idx])
        rel_time = (src_step_idx - dst_step_idx).to(pos_a.dtype)
        anchor_head_vec = head_vector_a[query_agent_idx, dst_step_idx]
        r_t = torch.stack(
            [
                torch.norm(rel_pos, p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=anchor_head_vec, nbr_vector=rel_pos),
                rel_head,
                rel_time,
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index, r_t

    def predict_flow_sparse(
        self,
        scene_feature: Tensor,
        pos_a: Tensor,
        head_a: Tensor,
        mask: Tensor,
        agent_indices: Tensor,
        anchor_indices: Tensor,
        noised_future: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """유효한 agent-anchor query만 골라 flow velocity를 예측한다.

        Args:
            scene_feature: [n_agent, n_step, hidden_dim] 모양의 장면 문맥 특징이다.
            pos_a: [n_agent, n_step, 2] 모양의 coarse 위치이다.
            head_a: [n_agent, n_step] 모양의 coarse heading이다.
            mask: [n_agent, n_step] 모양의 coarse 유효 마스크이다.
            agent_indices: [n_query] 모양의 query별 agent index이다.
            anchor_indices: [n_query] 모양의 query별 coarse anchor index이다.
            noised_future: [n_query, 20, 4] 모양의 정규화된 noised future이다.
            tau: [n_query] 모양의 시간 값이다.

        Returns:
            [n_query, 20, 4] 모양의 step별 flow velocity이다.
        """
        if noised_future.numel() == 0:
            return noised_future.new_zeros((0, self.flow_num_future_steps, 4))

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        condition_vec = self.future_conditioner(noised_future, tau)
        anchor_feature = scene_feature[agent_indices, anchor_indices]
        anchor_query = anchor_feature + self.query_adapter(
            torch.cat([anchor_feature, condition_vec], dim=-1)
        )
        edge_index_q, r_q = self.build_sparse_anchor_temporal_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
            agent_indices=agent_indices,
            anchor_indices=anchor_indices,
        )
        anchor_query = self.anchor_query_attn(
            (scene_feature.flatten(0, 1), anchor_query),
            r_q,
            edge_index_q,
        )
        return self.flow_head(anchor_query)

    def denormalize_future_local(self, future_local_norm: Tensor) -> Tensor:
        """정규화된 local 미래를 meter 단위 local 미래로 되돌린다.

        Args:
            future_local_norm: [*, 20, 4] 모양의 정규화된 local 미래이다.

        Returns:
            [*, 20, 4] 모양의 meter 단위 local 미래이다.
        """
        future_local = future_local_norm.clone()
        future_local[..., :2] = future_local[..., :2] * self.flow_position_scale
        return future_local

    @staticmethod
    def decode_future(
        future_local: Tensor,
        anchor_pos: Tensor,
        anchor_head: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """local 미래를 global 위치와 heading으로 바꾼다.

        Args:
            future_local: [*, 20, 4] 모양의 local 미래이다.
            anchor_pos: [*, 2] 모양의 anchor 위치이다.
            anchor_head: [*] 모양의 anchor heading이다.

        Returns:
            global 위치와 global heading을 순서대로 돌려준다.
        """
        prefix_shape = tuple(future_local.shape[:-2])
        num_future_steps = future_local.shape[-2]
        pos_local = future_local[..., :2].reshape(-1, num_future_steps, 2)
        anchor_pos = anchor_pos.reshape(-1, 2)
        anchor_head = anchor_head.reshape(-1)
        future_pos_global, _ = transform_to_global(
            pos_local=pos_local,
            head_local=None,
            pos_now=anchor_pos,
            head_now=anchor_head,
        )
        delta_head = torch.atan2(future_local[..., 3], future_local[..., 2])
        future_head_global = wrap_angle(delta_head + anchor_head.view(*prefix_shape, 1))
        future_pos_global = future_pos_global.view(*prefix_shape, num_future_steps, 2)
        return future_pos_global, future_head_global

    @staticmethod
    def match_token_index(
        token_traj: Tensor,
        token_agent_shape: Tensor,
        pos_now: Tensor,
        head_now: Tensor,
        pos_next: Tensor,
        head_next: Tensor,
    ) -> Tensor:
        """다음 0.5초 상태를 가장 가까운 coarse token id로 바꾼다.

        Args:
            token_traj: [n_agent, n_token, 4, 2] coarse token contour 사전이다.
            token_agent_shape: [n_agent, 2] agent 폭과 길이이다.
            pos_now: [n_agent, 2] 현재 위치이다.
            head_now: [n_agent] 현재 heading이다.
            pos_next: [n_agent, 2] 다음 0.5초 위치이다.
            head_next: [n_agent] 다음 0.5초 heading이다.

        Returns:
            [n_agent] 모양의 coarse token index이다.
        """
        range_a = torch.arange(pos_now.shape[0], device=pos_now.device)
        gt_contour = cal_polygon_contour(pos_next, head_next, token_agent_shape)
        gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]
        token_world = transform_to_global(
            pos_local=token_traj.flatten(1, 2),
            head_local=None,
            pos_now=pos_now,
            head_now=head_now,
        )[0].view(*token_traj.shape)
        token_idx = torch.argmin(
            torch.norm(token_world - gt_contour, dim=-1).sum(-1),
            dim=-1,
        )
        return token_idx[range_a]

    def forward(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습용 flow prediction을 만든다.

        정규화된 flow state에서 ODE 샘플을 만들고, 실제로 유효한 anchor만 골라
        decoder를 통과시킨 뒤 다시 dense 형태로 되돌린다.
        """
        pos_a = tokenized_agent["coarse_pos"]
        head_a = tokenized_agent["coarse_head"]
        mask = tokenized_agent["valid_mask"]
        scene_feature = self.encode_scene(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            pos_a=pos_a,
            head_a=head_a,
            mask=mask,
            agent_token_index=tokenized_agent["gt_idx"],
        )

        anchor_mask = tokenized_agent["flow_train_mask"] if self.training else tokenized_agent["flow_eval_mask"]
        flow_clean_norm = tokenized_agent["flow_clean_norm"]
        pred_flow = flow_clean_norm.new_zeros(flow_clean_norm.shape)
        target_flow = flow_clean_norm.new_zeros(flow_clean_norm.shape)
        noised_future_norm = flow_clean_norm.new_zeros(flow_clean_norm.shape)
        pred_future_local_norm = flow_clean_norm.new_zeros(flow_clean_norm.shape)
        tau = flow_clean_norm.new_zeros(flow_clean_norm.shape[:2])

        if anchor_mask.any():
            agent_idx, anchor_local_idx = anchor_mask.nonzero(as_tuple=True)
            query_anchor_idx = tokenized_agent["flow_anchor_token_idx"][anchor_local_idx]
            flow_sample = self.flow_ode.sample(flow_clean_norm[anchor_mask])
            pred_flow_valid = self.predict_flow_sparse(
                scene_feature=scene_feature,
                pos_a=pos_a,
                head_a=head_a,
                mask=mask,
                agent_indices=agent_idx,
                anchor_indices=query_anchor_idx,
                noised_future=flow_sample.noised,
                tau=flow_sample.tau,
            )
            pred_future_local_norm_valid = self.flow_ode.reconstruct_start(
                flow_sample.noised,
                pred_flow_valid,
                flow_sample.tau,
            )
            pred_flow[anchor_mask] = pred_flow_valid
            target_flow[anchor_mask] = flow_sample.target
            noised_future_norm[anchor_mask] = flow_sample.noised
            pred_future_local_norm[anchor_mask] = pred_future_local_norm_valid
            tau[anchor_mask] = flow_sample.tau

        pred_future_local = self.denormalize_future_local(pred_future_local_norm)
        gt_future_local = tokenized_agent["flow_future_local"]
        pred_future_pos, pred_future_head = self.decode_future(
            future_local=pred_future_local,
            anchor_pos=tokenized_agent["flow_anchor_pos"],
            anchor_head=tokenized_agent["flow_anchor_head"],
        )
        return {
            "pred_flow": pred_flow,
            "target_flow": target_flow,
            "pred_flow_norm": pred_flow,
            "target_flow_norm": target_flow,
            "noised_future": self.denormalize_future_local(noised_future_norm),
            "noised_future_norm": noised_future_norm,
            "tau": tau,
            "pred_future_local": pred_future_local,
            "pred_future_local_norm": pred_future_local_norm,
            "gt_future_local": gt_future_local,
            "gt_future_local_norm": flow_clean_norm,
            "pred_future_pos": pred_future_pos,
            "pred_future_head": pred_future_head,
            "gt_future_pos": tokenized_agent["flow_future_pos"],
            "gt_future_head": tokenized_agent["flow_future_head"],
            "flow_anchor_valid": tokenized_agent["flow_anchor_valid"],
            "flow_train_mask": tokenized_agent["flow_train_mask"],
            "flow_eval_mask": tokenized_agent["flow_eval_mask"],
            "flow_future_valid": tokenized_agent["flow_future_valid"],
            "flow_state_normalized": True,
        }

    def inference(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        """random noise에서 시작해 2초 미래를 만들고 0.5초씩 commit한다.

        구조는 그대로 두되, 실제로 현재 살아 있는 agent만 flow 샘플을 만든다.
        retokenization은 마지막 한 점이 아니라 현재 contour + commit된 5개 contour를
        함께 써서 다음 coarse token을 고른다.
        """
        n_agent = tokenized_agent["valid_mask"].shape[0]
        rollout_segments = self.num_future_steps // self.commit_num_future_steps

        token_idx_seq = tokenized_agent["gt_idx"][:, : self.step_current_token].clone()
        pos_seq = tokenized_agent["coarse_pos"][:, : self.step_current_token].clone()
        head_seq = tokenized_agent["coarse_head"][:, : self.step_current_token].clone()
        valid_seq = tokenized_agent["valid_mask"][:, : self.step_current_token].clone()

        pred_traj_10hz = pos_seq.new_zeros(n_agent, self.num_future_steps, 2)
        pred_head_10hz = head_seq.new_zeros(n_agent, self.num_future_steps)

        for segment_idx in range(rollout_segments):
            scene_feature = self.encode_scene(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                pos_a=pos_seq,
                head_a=head_seq,
                mask=valid_seq,
                agent_token_index=token_idx_seq,
            )
            active_mask = valid_seq[:, -1]
            step_start = segment_idx * self.commit_num_future_steps
            step_end = step_start + self.commit_num_future_steps

            next_pos = pos_seq[:, -1].clone()
            next_head = head_seq[:, -1].clone()
            next_idx = token_idx_seq[:, -1].clone()
            commit_pos = pred_traj_10hz.new_zeros((n_agent, self.commit_num_future_steps, 2))
            commit_head = pred_head_10hz.new_zeros((n_agent, self.commit_num_future_steps))

            if active_mask.any():
                pos_seq_active = pos_seq[active_mask]
                head_seq_active = head_seq[active_mask]
                valid_seq_active = valid_seq[active_mask]
                scene_feature_active = scene_feature[active_mask]
                n_active = pos_seq_active.shape[0]
                agent_indices = torch.arange(n_active, device=pos_seq.device, dtype=torch.long)
                anchor_indices = torch.full(
                    (n_active,),
                    fill_value=pos_seq_active.shape[1] - 1,
                    device=pos_seq.device,
                    dtype=torch.long,
                )

                def model_fn(x_t: Tensor, tau: Tensor) -> Tensor:
                    return self.predict_flow_sparse(
                        scene_feature=scene_feature_active,
                        pos_a=pos_seq_active,
                        head_a=head_seq_active,
                        mask=valid_seq_active,
                        agent_indices=agent_indices,
                        anchor_indices=anchor_indices,
                        noised_future=x_t,
                        tau=tau,
                    )

                x_init = torch.randn(
                    n_active,
                    self.flow_num_future_steps,
                    4,
                    device=pos_seq.device,
                    dtype=pos_seq.dtype,
                )
                future_local_norm = self.flow_ode.generate(
                    x_init=x_init,
                    model_fn=model_fn,
                    sample_steps=sampling_scheme.sample_steps,
                    sample_temperature=sampling_scheme.sample_temperature,
                    sample_method=sampling_scheme.sample_method,
                )
                commit_pos_active, commit_head_active, next_pos_active, next_head_active = self.commit_bridge.commit(
                    future_local_norm=future_local_norm,
                    current_pos=pos_seq_active[:, -1],
                    current_head=head_seq_active[:, -1],
                )
                next_idx_active = self.commit_bridge.retokenize(
                    current_pos=pos_seq_active[:, -1],
                    current_head=head_seq_active[:, -1],
                    commit_pos=commit_pos_active,
                    commit_head=commit_head_active,
                    token_traj_all=tokenized_agent["token_traj_all"][active_mask],
                    token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                )
                commit_pos[active_mask] = commit_pos_active
                commit_head[active_mask] = commit_head_active
                next_pos[active_mask] = next_pos_active
                next_head[active_mask] = next_head_active
                next_idx[active_mask] = next_idx_active

            pred_traj_10hz[:, step_start:step_end] = commit_pos
            pred_head_10hz[:, step_start:step_end] = commit_head

            next_valid = active_mask.clone()
            token_idx_seq = torch.cat([token_idx_seq, next_idx.unsqueeze(1)], dim=1)
            pos_seq = torch.cat([pos_seq, next_pos.unsqueeze(1)], dim=1)
            head_seq = torch.cat([head_seq, next_head.unsqueeze(1)], dim=1)
            valid_seq = torch.cat([valid_seq, next_valid.unsqueeze(1)], dim=1)

        pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)
        return {
            "pred_idx": token_idx_seq,
            "pred_pos": pos_seq,
            "pred_head": head_seq,
            "pred_valid": valid_seq,
            "pred_traj_10hz": pred_traj_10hz,
            "pred_head_10hz": pred_head_10hz,
            "pred_z_10hz": pred_z.expand(-1, pred_traj_10hz.shape[1]),
        }
