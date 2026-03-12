# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from src.smart.utils.flow_traj import (
    assemble_4x6_to_21,
    build_flow_path,
    build_local_future_target,
    chunk_future_21_to_4x6,
    executed_chunk_to_rollout_update,
    midpoint_ode_integrate,
    nearest_agent_token_idx,
    segment_end_pose_global,
)


class SMARTAgentDecoder(nn.Module):
    """SMART agent NTP head를 대체하는 sparse factorized flow decoder.

    구조 요약:
        1. 기존 SMART token state space에서 과거 6개 slot을 읽는다.
        2. 현재 연속 상태 anchor token을 하나 더 붙인다.
        3. 미래 2초를 4개의 0.5초 segment로 쪼갠다.
        4. future temporal -> history cross -> map cross -> future a2a 순서로
           sparse attention을 반복한다.
        5. conditional flow matching velocity field를 예측한다.
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
        history_steps: int = 6,
        future_window_steps: int = 20,
        future_num_segments: int = 4,
        future_segment_points: int = 6,
        ode_steps: int = 4,
        hist2f_radius: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.hist2f_radius = a2a_radius if hist2f_radius is None else hist2f_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob
        self.history_steps = history_steps
        self.future_window_steps = future_window_steps
        self.future_num_segments = future_num_segments
        self.future_segment_points = future_segment_points
        self.ode_steps = ode_steps
        self.current_step = num_historical_steps - 1

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.segment_idx_emb = nn.Embedding(future_num_segments, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.flow_time_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_hist2f_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.token_emb_veh = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        self.current_anchor_emb = MLPEmbedding(input_dim=5, hidden_dim=hidden_dim)
        self.future_segment_emb = MLPEmbedding(input_dim=24, hidden_dim=hidden_dim)
        self.segment_out_head = MLPLayer(hidden_dim, hidden_dim, 24)

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
        self.hist2f_attn_layers = nn.ModuleList(
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


    def _embed_discrete_history_tokens(
        self,
        agent_token_index: Tensor,
        trajectory_token_veh: Tensor,
        trajectory_token_ped: Tensor,
        trajectory_token_cyc: Tensor,
        pos_a: Tensor,
        head_vector_a: Tensor,
        agent_type: Tensor,
        agent_type_emb: Tensor,
        agent_shape_emb: Tensor,
    ) -> Tensor:
        """과거 discrete token 6개를 history feature 로 바꾼다.

        Args:
            agent_token_index: ``[N, H]``. agent 별 과거 token id.
            trajectory_token_veh: ``[V_veh, 8]``. 차량 token 사전.
            trajectory_token_ped: ``[V_ped, 8]``. 보행자 token 사전.
            trajectory_token_cyc: ``[V_cyc, 8]``. 자전거 token 사전.
            pos_a: ``[N, H, 2]``. 과거 위치.
            head_vector_a: ``[N, H, 2]``. 과거 heading 단위 벡터.
            agent_type: ``[N]``. agent 종류 id.
            agent_type_emb: ``[N, D]``. agent 종류 임베딩.
            agent_shape_emb: ``[N, D]``. agent 크기 임베딩.

        Returns:
            ``[N, H, D]`` history token feature.
        """
        n_agent, n_step, traj_dim = pos_a.shape
        veh_mask = (agent_type == 0).unsqueeze(1).unsqueeze(2)
        ped_mask = (agent_type == 1).unsqueeze(1).unsqueeze(2)
        cyc_mask = (agent_type == 2).unsqueeze(1).unsqueeze(2)

        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)

        # Keep all type-specific embedding branches in the autograd graph every step.
        # Otherwise, rank-local batches missing a type can trigger DDP unused-parameter
        # errors when find_unused_parameters=False.
        idx_veh = agent_token_index.clamp(max=agent_token_emb_veh.size(0) - 1)
        idx_ped = agent_token_index.clamp(max=agent_token_emb_ped.size(0) - 1)
        idx_cyc = agent_token_index.clamp(max=agent_token_emb_cyc.size(0) - 1)
        emb_veh = agent_token_emb_veh[idx_veh]
        emb_ped = agent_token_emb_ped[idx_ped]
        emb_cyc = agent_token_emb_cyc[idx_cyc]
        agent_token_emb = (
            emb_veh * veh_mask.to(emb_veh.dtype)
            + emb_ped * ped_mask.to(emb_ped.dtype)
            + emb_cyc * cyc_mask.to(emb_cyc.dtype)
        )

        motion_vector = torch.cat(
            [pos_a.new_zeros(n_agent, 1, traj_dim), pos_a[:, 1:] - pos_a[:, :-1]],
            dim=1,
        )
        feature_a = torch.stack(
            [
                torch.norm(motion_vector[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a,
                    nbr_vector=motion_vector[:, :, :2],
                ),
            ],
            dim=-1,
        )
        categorical_embs = [agent_type_emb, agent_shape_emb]
        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs],
        ).view(n_agent, n_step, self.hidden_dim)
        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)
        return feat_a


    def _build_current_anchor_feat(
        self,
        current_vel: Tensor,
        current_head: Tensor,
        current_yaw_rate: Tensor,
        agent_type_emb: Tensor,
        agent_shape_emb: Tensor,
        ego_mask: Tensor,
    ) -> Tensor:
        """현재 시각의 연속 상태를 0.1초 anchor token 으로 바꾼다.

        Args:
            current_vel: ``[N, 2]``. 현재 속도.
            current_head: ``[N]``. 현재 heading.
            current_yaw_rate: ``[N]``. 현재 yaw rate.
            agent_type_emb: ``[N, D]``. agent 종류 임베딩.
            agent_shape_emb: ``[N, D]``. agent 크기 임베딩.
            ego_mask: ``[N]``. ego 여부.

        Returns:
            ``[N, D]`` 현재 시각 anchor feature.
        """
        dt = 0.1
        cos_h = current_head.cos()
        sin_h = current_head.sin()
        vx_local = current_vel[:, 0] * cos_h + current_vel[:, 1] * sin_h
        vy_local = -current_vel[:, 0] * sin_h + current_vel[:, 1] * cos_h
        delta_head = wrap_angle(current_yaw_rate * dt)
        anchor_cont = torch.stack(
            [
                vx_local * dt,
                vy_local * dt,
                delta_head.sin(),
                delta_head.cos(),
                ego_mask.float(),
            ],
            dim=-1,
        )
        feat = self.current_anchor_emb(anchor_cont)
        feat = feat + agent_type_emb + agent_shape_emb
        return feat

    def _build_gt_state(
        self,
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_step: int,
    ) -> Dict[str, Tensor]:
        """GT 기준 anchor 시각에서 history state를 만든다.

        Args:
            tokenized_agent: tokenized agent dict.
            agent_raw: raw ``data['agent']`` dict.
            anchor_step: raw 10Hz anchor step.

        Returns:
            history/current state dict.
        """
        n_agent = tokenized_agent["gt_idx"].shape[0]
        device = tokenized_agent["gt_idx"].device
        current_token_idx = anchor_step // self.shift - 1
        hist_start = max(0, current_token_idx - self.history_steps + 1)
        src_len = current_token_idx - hist_start + 1

        hist_idx = torch.zeros(n_agent, self.history_steps, dtype=torch.long, device=device)
        hist_pos = torch.zeros(n_agent, self.history_steps, 2, dtype=tokenized_agent["gt_pos"].dtype, device=device)
        hist_head = torch.zeros(n_agent, self.history_steps, dtype=tokenized_agent["gt_heading"].dtype, device=device)
        hist_valid = torch.zeros(n_agent, self.history_steps, dtype=torch.bool, device=device)
        if src_len > 0:
            hist_idx[:, -src_len:] = tokenized_agent["gt_idx"][:, hist_start : current_token_idx + 1]
            hist_pos[:, -src_len:] = tokenized_agent["gt_pos"][:, hist_start : current_token_idx + 1]
            hist_head[:, -src_len:] = tokenized_agent["gt_heading"][:, hist_start : current_token_idx + 1]
            hist_valid[:, -src_len:] = tokenized_agent["valid_mask"][:, hist_start : current_token_idx + 1]

        current_pos = agent_raw["position"][:, anchor_step, :2].contiguous()
        current_head = agent_raw["heading"][:, anchor_step].contiguous()
        current_vel = agent_raw["velocity"][:, anchor_step, :2].contiguous()
        prev_idx = max(anchor_step - 1, 0)
        current_yaw_rate = wrap_angle(current_head - agent_raw["heading"][:, prev_idx]) / 0.1
        current_valid = agent_raw["valid_mask"][:, anchor_step].contiguous()
        return {
            "hist_idx": hist_idx,
            "hist_pos": hist_pos,
            "hist_head": hist_head,
            "hist_valid": hist_valid,
            "current_pos": current_pos,
            "current_head": current_head,
            "current_vel": current_vel,
            "current_yaw_rate": current_yaw_rate,
            "current_valid": current_valid,
        }

    def _init_rollout_state(
        self,
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """8초 반복 생성용 초기 state를 만든다.

        Args:
            tokenized_agent: tokenized agent dict.
            agent_raw: raw ``data['agent']`` dict.

        Returns:
            rollout state dict.
        """
        state = self._build_gt_state(tokenized_agent, agent_raw, self.current_step)
        state["current_z"] = agent_raw["position"][:, self.current_step, 2].contiguous()
        return state

    def _advance_rollout_state(
        self,
        state: Dict[str, Tensor],
        new_token_idx: Tensor,
        rollout_update: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """실행한 첫 0.5초를 반영해 state를 갱신한다.

        Args:
            state: rollout state.
            new_token_idx: ``[N]`` nearest SMART token id.
            rollout_update: ``executed_chunk_to_rollout_update`` 결과.

        Returns:
            갱신된 rollout state.
        """
        hist_idx = torch.cat([state["hist_idx"][:, 1:], new_token_idx.unsqueeze(1)], dim=1)
        hist_pos = torch.cat([state["hist_pos"][:, 1:], rollout_update["next_pos"].unsqueeze(1)], dim=1)
        hist_head = torch.cat([state["hist_head"][:, 1:], rollout_update["next_head"].unsqueeze(1)], dim=1)
        hist_valid = torch.cat([state["hist_valid"][:, 1:], state["current_valid"].unsqueeze(1)], dim=1)
        return {
            **state,
            "hist_idx": hist_idx,
            "hist_pos": hist_pos,
            "hist_head": hist_head,
            "hist_valid": hist_valid,
            "current_pos": rollout_update["next_pos"],
            "current_head": rollout_update["next_head"],
            "current_vel": rollout_update["next_vel"],
            "current_yaw_rate": rollout_update["next_yaw_rate"],
        }


    def _build_history_memory(
        self,
        tokenized_agent: Dict[str, Tensor],
        state: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """history token 6개와 현재 anchor 1개를 memory 로 합친다.

        Args:
            tokenized_agent: tokenized agent dict.
            state: history/current state dict.

        Returns:
            다음 key 를 가진 memory dict.
            - ``feat``: ``[N, 7, D]``
            - ``pos``: ``[N, 7, 2]``
            - ``head``: ``[N, 7]``
            - ``valid``: ``[N, 7]``
        """
        n_agent = state["current_pos"].shape[0]
        static_inputs = self._resolve_agent_static_inputs(tokenized_agent, n_agent)

        hist_head_vec = torch.stack([state["hist_head"].cos(), state["hist_head"].sin()], dim=-1)
        hist_feat = self._embed_discrete_history_tokens(
            agent_token_index=state["hist_idx"],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=state["hist_pos"],
            head_vector_a=hist_head_vec,
            agent_type=static_inputs["type"],
            agent_type_emb=static_inputs["type_emb"],
            agent_shape_emb=static_inputs["shape_emb"],
        )
        anchor_feat = self._build_current_anchor_feat(
            current_vel=state["current_vel"],
            current_head=state["current_head"],
            current_yaw_rate=state["current_yaw_rate"],
            agent_type_emb=static_inputs["type_emb"],
            agent_shape_emb=static_inputs["shape_emb"],
            ego_mask=static_inputs["ego_mask"],
        )
        ctx_feat = torch.cat([hist_feat, anchor_feat.unsqueeze(1)], dim=1)
        ctx_pos = torch.cat([state["hist_pos"], state["current_pos"].unsqueeze(1)], dim=1)
        ctx_head = torch.cat([state["hist_head"], state["current_head"].unsqueeze(1)], dim=1)
        ctx_valid = torch.cat([state["hist_valid"], state["current_valid"].unsqueeze(1)], dim=1)
        return {"feat": ctx_feat, "pos": ctx_pos, "head": ctx_head, "valid": ctx_valid}

    def _expand_agent_mask_to_future_segments(self, agent_mask: Tensor) -> Tensor:
        """agent 단위 mask를 future segment 단위 mask로 늘린다.

        Args:
            agent_mask: ``[N]``. agent 단위 bool mask.

        Returns:
            ``[N, 4]``. 각 agent mask를 모든 future segment에 복사한 mask.
        """
        return agent_mask.bool().unsqueeze(1).expand(-1, self.future_num_segments)


    def _build_future_query(
        self,
        x_t: Tensor,
        tau: Tensor,
        tokenized_agent: Dict[str, Tensor],
    ) -> Tensor:
        """noisy future segment 를 query token 으로 바꾼다.

        Args:
            x_t: ``[N, 4, 6, 4]``. noisy future segment.
            tau: ``[N, 1]``. flow time.
            tokenized_agent: tokenized agent dict.

        Returns:
            ``[N, 4, D]`` future query feature.
        """
        n_agent = x_t.shape[0]
        static_inputs = self._resolve_agent_static_inputs(tokenized_agent, n_agent)

        seg_feat = self.future_segment_emb(x_t.flatten(-2, -1).reshape(-1, 24)).view(
            n_agent,
            self.future_num_segments,
            -1,
        )
        seg_idx = torch.arange(self.future_num_segments, device=x_t.device)
        seg_feat = seg_feat + self.segment_idx_emb(seg_idx).unsqueeze(0)
        seg_feat = seg_feat + static_inputs["type_emb"].unsqueeze(1)
        seg_feat = seg_feat + static_inputs["shape_emb"].unsqueeze(1)
        flow_t = self.flow_time_emb(continuous_inputs=tau)
        seg_feat = seg_feat + flow_t.unsqueeze(1)
        return seg_feat

    def _build_reference_future_pose(self, state: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """현재 상태만으로 map lookup용 deterministic 2초 reference pose를 만든다.

        Query feature와 dynamic future graph는 noisy ``x_t``를 그대로 사용하고,
        정적인 map cross-attention만 현재 상태 기반 reference를 사용해 graph support를
        안정화한다.
        """
        dt = state["current_pos"].new_tensor([0.5, 1.0, 1.5, 2.0])
        head_now = state["current_head"]
        vel_now = state["current_vel"]
        yaw_rate = state["current_yaw_rate"]

        cos_h = head_now.cos()
        sin_h = head_now.sin()
        vel_x_local = vel_now[:, 0] * cos_h + vel_now[:, 1] * sin_h
        vel_y_local = -vel_now[:, 0] * sin_h + vel_now[:, 1] * cos_h

        omega = yaw_rate.unsqueeze(1)
        omega_dt = omega * dt.unsqueeze(0)
        small_turn = omega.abs() < 1e-4
        safe_omega = torch.where(small_turn, torch.ones_like(omega), omega)

        dx_local_linear = vel_x_local.unsqueeze(1) * dt.unsqueeze(0)
        dy_local_linear = vel_y_local.unsqueeze(1) * dt.unsqueeze(0)
        dx_local_turn = (
            omega_dt.sin() * vel_x_local.unsqueeze(1)
            - (1.0 - omega_dt.cos()) * vel_y_local.unsqueeze(1)
        ) / safe_omega
        dy_local_turn = (
            (1.0 - omega_dt.cos()) * vel_x_local.unsqueeze(1)
            + omega_dt.sin() * vel_y_local.unsqueeze(1)
        ) / safe_omega
        dx_local = torch.where(small_turn, dx_local_linear, dx_local_turn)
        dy_local = torch.where(small_turn, dy_local_linear, dy_local_turn)

        dx_global = dx_local * cos_h.unsqueeze(1) - dy_local * sin_h.unsqueeze(1)
        dy_global = dx_local * sin_h.unsqueeze(1) + dy_local * cos_h.unsqueeze(1)
        future_pos = state["current_pos"].unsqueeze(1) + torch.stack([dx_global, dy_global], dim=-1)
        future_head = wrap_angle(head_now.unsqueeze(1) + omega_dt)
        return future_pos, future_head

    def build_temporal_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        inference_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """같은 agent 안의 causal temporal edge를 만든다."""
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)
        if self.hist_drop_prob > 0 and self.training and inference_mask is None:
            keep = torch.bernoulli(torch.ones_like(mask, dtype=pos_a.dtype) * (1 - self.hist_drop_prob)).bool()
            mask = mask & keep
        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def _empty_edge_and_relation(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """빈 edge_index와 relation embedding을 만든다.

        Args:
            device: 텐서를 만들 장치.
            dtype: relation embedding에 사용할 자료형.

        Returns:
            tuple:
                - edge_index: ``[2, 0]``
                - rel: ``[0, D]``
        """
        empty_edge = torch.empty((2, 0), device=device, dtype=torch.long)
        empty_rel = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
        return empty_edge, empty_rel

    @staticmethod
    def _sort_by_batch_if_needed(
        values: Sequence[Tensor],
        batch: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        """batch가 비내림차순이 아닐 때만 같은 permutation으로 정렬한다.

        ``torch_cluster``의 batch-aware radius 연산은 batch index가 정렬돼 있을 때를
        가정하므로, 필요한 경우에만 입력 텐서를 함께 재배열한다.
        """
        if batch.numel() <= 1 or bool(torch.all(batch[:-1] <= batch[1:])):
            return [value for value in values], batch

        perm = torch.argsort(batch)
        sorted_values = [value.index_select(0, perm) for value in values]
        return sorted_values, batch.index_select(0, perm)

    def build_interaction_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        batch_s: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """같은 future index끼리, 유효한 query만 사용해 agent-agent edge를 만든다.

        Args:
            pos_a: ``[N, 4, 2]``. future segment 끝점 위치.
            head_a: ``[N, 4]``. future segment 끝점 heading.
            head_vector_a: ``[N, 4, 2]``. future segment 끝점 heading 단위 벡터.
            batch_s: ``[4 * N]``. step-major 기준 future query batch id.
            mask: ``[N, 4]``. 유효한 future query mask.

        Returns:
            tuple:
                - edge_index: ``[2, E]``. 기존 step-major future index 기준.
                - r_a2a: ``[E, D]``. agent-agent relation embedding.
        """
        mask_flat = mask.transpose(0, 1).reshape(-1).bool()
        valid_index = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if valid_index.numel() == 0:
            return self._empty_edge_and_relation(pos_a.device, pos_a.dtype)

        pos_step_major = pos_a.transpose(0, 1).reshape(-1, pos_a.size(-1))
        head_step_major = head_a.transpose(0, 1).reshape(-1)
        head_vec_step_major = head_vector_a.transpose(0, 1).reshape(-1, head_vector_a.size(-1))

        pos_valid = pos_step_major.index_select(0, valid_index)
        head_valid = head_step_major.index_select(0, valid_index)
        head_vec_valid = head_vec_step_major.index_select(0, valid_index)
        batch_valid = batch_s.index_select(0, valid_index)

        edge_local = radius_graph(
            x=pos_valid[:, :2],
            r=self.a2a_radius,
            batch=batch_valid,
            loop=False,
            max_num_neighbors=300,
        )
        if edge_local.numel() == 0:
            return self._empty_edge_and_relation(pos_a.device, pos_a.dtype)

        src_local = edge_local[0]
        dst_local = edge_local[1]

        rel_pos_a2a = pos_valid.index_select(0, src_local) - pos_valid.index_select(0, dst_local)
        rel_head_a2a = wrap_angle(
            head_valid.index_select(0, src_local) - head_valid.index_select(0, dst_local)
        )
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vec_valid.index_select(0, dst_local),
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        edge_index = torch.stack(
            [
                valid_index.index_select(0, src_local),
                valid_index.index_select(0, dst_local),
            ],
            dim=0,
        )
        return edge_index, r_a2a

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
        token_index_pl: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """future segment와 road token 사이 sparse edge를 만든다.

        Args:
            pos_pl: ``[M, 2]``. map geometry 중심점.
            orient_pl: ``[M]``. map geometry heading.
            pos_a: ``[N, 4, 2]``. map lookup용 reference future 위치.
            head_a: ``[N, 4]``. reference future heading.
            head_vector_a: ``[N, 4, 2]``. reference future heading vector.
            mask: ``[N, 4]``. 유효한 future query mask.
            batch_s: ``[4*N]``. future query를 편 batch id.
            batch_pl: ``[4*M]``. future segment별로 늘린 map geometry batch id.
            token_index_pl: ``[M]``. 각 map geometry row가 실제로 읽어야 하는
                ``pt_token`` row index. ``None`` 이면 ``[0, 1, ..., M-1]`` 로 본다.

        Returns:
            tuple:
                - ``edge_index_pl2a``: ``[2, E]``. src는 반복된 map geometry row index,
                  dst는 future query index.
                - ``r_pl2a``: ``[E, 128]``. map-to-agent relation embedding.
                - ``src_token_index``: ``[E]``. 각 edge의 source가 최종 ``pt_token`` 의
                  어느 row를 읽어야 하는지 나타내는 index.
        """
        num_map_geom = pos_pl.shape[0]
        if token_index_pl is None:
            token_index_pl = torch.arange(num_map_geom, device=pos_pl.device, dtype=torch.long)
        else:
            token_index_pl = token_index_pl.long()

        if num_map_geom == 0:
            empty_edge_index = torch.empty((2, 0), device=pos_a.device, dtype=torch.long)
            empty_rel = pos_a.new_zeros((0, self.hidden_dim))
            empty_src_token_index = torch.empty((0,), device=pos_a.device, dtype=torch.long)
            return empty_edge_index, empty_rel, empty_src_token_index

        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl_rep = pos_pl.repeat(n_step, 1)
        orient_pl_rep = orient_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl_rep[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl_rep[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl_rep[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
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
        src_token_index = token_index_pl[edge_index_pl2a[0] % num_map_geom]
        return edge_index_pl2a, r_pl2a, src_token_index

    def _get_map_token_index(self, map_feature: Dict[str, Tensor]) -> Tensor:
        """map geometry row마다 어떤 encoded token을 읽을지 알려주는 index를 만든다.

        Args:
            map_feature: map dict.
                - ``pt_token``: ``[M_token, D]``.
                - ``position``: ``[M_geom, 2]``.
                - ``orientation``: ``[M_geom]``.
                - ``batch``: ``[M_geom]``.
                - optional ``token_index``: ``[M_geom]``.

        Returns:
            ``[M_geom]`` long tensor. 각 geometry row가 실제로 읽어야 하는
            ``pt_token`` row index.
        """
        token_index = map_feature.get("token_index")
        if token_index is not None:
            return token_index.long()

        num_token = map_feature["pt_token"].shape[0]
        num_geom = map_feature["position"].shape[0]
        if num_geom != num_token:
            raise ValueError(
                "map_feature geometry length differs from pt_token length, but token_index is missing."
            )
        return torch.arange(num_token, device=map_feature["pt_token"].device, dtype=torch.long)


    def _build_map2agent_edge_without_geometry_repeat(
        self,
        pos_pl: Tensor,
        orient_pl: Tensor,
        batch_pl: Tensor,
        token_index_pl: Tensor,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        batch_agent: Tensor,
        base_num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """map geometry를 복제하지 않고, 유효한 future query만으로 map-to-agent edge를 만든다.

        Args:
            pos_pl: ``[M, 2]``. map geometry 중심점.
            orient_pl: ``[M]``. map geometry heading.
            batch_pl: ``[M]``. 원본 scene 기준 map batch id.
            token_index_pl: ``[M]``. 각 geometry row가 읽을 token row index.
            pos_a: ``[N, 4, 2]``. future reference 위치.
            head_a: ``[N, 4]``. future reference heading.
            head_vector_a: ``[N, 4, 2]``. future heading 단위 벡터.
            mask: ``[N, 4]``. 유효한 future query mask.
            batch_agent: ``[N]``. anchor batch까지 반영된 agent batch id.
            base_num_graphs: 원래 scene 개수 ``B``.

        Returns:
            tuple:
                - src_token_index: ``[E]``. source token row index.
                - dst_index: ``[E]``. 기존 step-major future query index.
                - r_pl2a: ``[E, D]``. map-to-agent relation embedding.
        """
        num_map_geom = pos_pl.shape[0]
        if num_map_geom == 0:
            empty_index = torch.empty((0,), device=pos_a.device, dtype=torch.long)
            empty_rel = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_index, empty_rel

        _, n_step = pos_a.shape[:2]
        mask_flat = mask.transpose(0, 1).reshape(-1).bool()
        valid_index = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if valid_index.numel() == 0:
            empty_index = torch.empty((0,), device=pos_a.device, dtype=torch.long)
            empty_rel = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_index, empty_rel

        pos_step_major = pos_a.transpose(0, 1).reshape(-1, pos_a.size(-1))
        head_step_major = head_a.transpose(0, 1).reshape(-1)
        head_vec_step_major = head_vector_a.transpose(0, 1).reshape(-1, head_vector_a.size(-1))
        base_batch_step_major = torch.remainder(batch_agent, base_num_graphs).repeat(n_step)

        pos_valid = pos_step_major.index_select(0, valid_index)
        head_valid = head_step_major.index_select(0, valid_index)
        head_vec_valid = head_vec_step_major.index_select(0, valid_index)
        batch_valid = base_batch_step_major.index_select(0, valid_index)

        [pos_valid, head_valid, head_vec_valid, valid_index], batch_valid = self._sort_by_batch_if_needed(
            [pos_valid, head_valid, head_vec_valid, valid_index],
            batch_valid,
        )
        [pos_pl_sorted, orient_pl_sorted, token_index_pl_sorted], batch_pl_sorted = self._sort_by_batch_if_needed(
            [pos_pl, orient_pl, token_index_pl],
            batch_pl,
        )

        edge_local = radius(
            x=pos_valid[:, :2],
            y=pos_pl_sorted[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_valid,
            batch_y=batch_pl_sorted,
            max_num_neighbors=300,
        )
        if edge_local.numel() == 0:
            empty_index = torch.empty((0,), device=pos_a.device, dtype=torch.long)
            empty_rel = pos_a.new_zeros((0, self.hidden_dim))
            return empty_index, empty_index, empty_rel

        src_geom_index = edge_local[0]
        dst_valid_index = edge_local[1]

        rel_pos_pl2a = (
            pos_pl_sorted.index_select(0, src_geom_index)
            - pos_valid.index_select(0, dst_valid_index)
        )
        rel_orient_pl2a = wrap_angle(
            orient_pl_sorted.index_select(0, src_geom_index)
            - head_valid.index_select(0, dst_valid_index)
        )
        rel_cont = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vec_valid.index_select(0, dst_valid_index),
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=rel_cont, categorical_embs=None)

        src_token_index = token_index_pl_sorted.index_select(0, src_geom_index)
        dst_index = valid_index.index_select(0, dst_valid_index)
        return src_token_index, dst_index, r_pl2a

    def _build_compact_map_attention_inputs(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        state: Dict[str, Tensor],
        future_mask: Tensor,
        batch_s: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """실제로 edge 가 생긴 map token 만 남겨 map cross-attention 입력을 만든다.

        Args:
            tokenized_agent: agent dict.
            map_feature: map dict.
            state: 현재 상태 dict.
            future_mask: ``[N, 4]``. map cross-attention 에 참여시킬 mask.
            batch_s: ``[4*N]``. 기존 호출부와의 호환을 위해 받지만,
                geometry 공유 경로에서는 직접 쓰지 않는다.

        Returns:
            tuple:
                - ``feat_map``: ``[M_keep, D]``
                - ``edge_index_pl2a``: ``[2, E]``
                - ``r_pl2a``: ``[E, D]``
        """
        del batch_s
        base_num_graphs = int(tokenized_agent.get("base_num_graphs", int(tokenized_agent["num_graphs"])))
        map_ref_pos, map_ref_head = self._build_reference_future_pose(state)
        map_ref_head_vec = torch.stack([map_ref_head.cos(), map_ref_head.sin()], dim=-1)

        src_token_index, dst_index, r_pl2a = self._build_map2agent_edge_without_geometry_repeat(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            batch_pl=map_feature["batch"],
            token_index_pl=self._get_map_token_index(map_feature),
            pos_a=map_ref_pos,
            head_a=map_ref_head,
            head_vector_a=map_ref_head_vec,
            mask=future_mask,
            batch_agent=tokenized_agent["batch"],
            base_num_graphs=base_num_graphs,
        )

        if src_token_index.numel() == 0:
            feat_map = map_feature["pt_token"].new_zeros((0, map_feature["pt_token"].shape[-1]))
            edge_index_pl2a = torch.empty((2, 0), device=map_feature["pt_token"].device, dtype=torch.long)
            return feat_map, edge_index_pl2a, r_pl2a

        unique_src_token_index, compact_src_index = torch.unique(
            src_token_index,
            sorted=True,
            return_inverse=True,
        )
        edge_index_pl2a = torch.stack([compact_src_index, dst_index], dim=0)
        feat_map = map_feature["pt_token"].index_select(0, unique_src_token_index)
        return feat_map, edge_index_pl2a, r_pl2a

    def build_hist2f_edge(
        self,
        ctx_pos: Tensor,
        ctx_head: Tensor,
        ctx_valid: Tensor,
        future_pos: Tensor,
        future_head: Tensor,
        future_valid: Tensor,
        batch_agent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """유효한 history slot과 유효한 future query만으로 history-to-future edge를 만든다.

        Args:
            ctx_pos: ``[N, 7, 2]``. history + current 위치.
            ctx_head: ``[N, 7]``. history + current heading.
            ctx_valid: ``[N, 7]``. history + current 유효 mask.
            future_pos: ``[N, 4, 2]``. future segment 끝점 위치.
            future_head: ``[N, 4]``. future segment 끝점 heading.
            future_valid: ``[N, 4]``. future query 유효 mask.
            batch_agent: ``[N]``. agent batch id.

        Returns:
            tuple:
                - edge_index: ``[2, E]``. src=history, dst=future, 기존 agent-major index 기준.
                - r_hist2f: ``[E, D]``. history-to-future relation embedding.
        """
        ctx_pos_flat = ctx_pos.reshape(-1, ctx_pos.size(-1))
        ctx_head_flat = ctx_head.reshape(-1)
        ctx_valid_flat = ctx_valid.reshape(-1).bool()

        future_pos_flat = future_pos.reshape(-1, future_pos.size(-1))
        future_head_flat = future_head.reshape(-1)
        future_valid_flat = future_valid.reshape(-1).bool()

        ctx_index = torch.nonzero(ctx_valid_flat, as_tuple=False).squeeze(1)
        future_index = torch.nonzero(future_valid_flat, as_tuple=False).squeeze(1)
        if ctx_index.numel() == 0 or future_index.numel() == 0:
            return self._empty_edge_and_relation(ctx_pos.device, ctx_pos.dtype)

        ctx_pos_valid = ctx_pos_flat.index_select(0, ctx_index)
        ctx_head_valid = ctx_head_flat.index_select(0, ctx_index)
        future_pos_valid = future_pos_flat.index_select(0, future_index)
        future_head_valid = future_head_flat.index_select(0, future_index)

        batch_ctx = batch_agent.repeat_interleave(ctx_pos.shape[1]).index_select(0, ctx_index)
        batch_future = batch_agent.repeat_interleave(future_pos.shape[1]).index_select(0, future_index)

        edge_local = radius(
            x=future_pos_valid[:, :2],
            y=ctx_pos_valid[:, :2],
            r=self.hist2f_radius,
            batch_x=batch_future,
            batch_y=batch_ctx,
            max_num_neighbors=300,
        )
        if edge_local.numel() == 0:
            return self._empty_edge_and_relation(ctx_pos.device, ctx_pos.dtype)

        ctx_head_vec = torch.stack([ctx_head_valid.cos(), ctx_head_valid.sin()], dim=-1)
        future_head_vec = torch.stack([future_head_valid.cos(), future_head_valid.sin()], dim=-1)

        ctx_time_lookup = ctx_pos.new_tensor([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0])
        future_time_lookup = ctx_pos.new_tensor([0.5, 1.0, 1.5, 2.0])
        ctx_slot_index = torch.remainder(ctx_index, ctx_pos.shape[1])
        future_slot_index = torch.remainder(future_index, future_pos.shape[1])
        ctx_time_valid = ctx_time_lookup.index_select(0, ctx_slot_index)
        future_time_valid = future_time_lookup.index_select(0, future_slot_index)

        src_local = edge_local[0]
        dst_local = edge_local[1]

        rel_pos = ctx_pos_valid.index_select(0, src_local) - future_pos_valid.index_select(0, dst_local)
        rel_head = wrap_angle(
            ctx_head_valid.index_select(0, src_local) - future_head_valid.index_select(0, dst_local)
        )
        rel_time = ctx_time_valid.index_select(0, src_local) - future_time_valid.index_select(0, dst_local)
        r_hist2f = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=future_head_vec.index_select(0, dst_local),
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_head,
                rel_time,
            ],
            dim=-1,
        )
        r_hist2f = self.r_hist2f_emb(continuous_inputs=r_hist2f, categorical_embs=None)
        edge_index = torch.stack(
            [
                ctx_index.index_select(0, src_local),
                future_index.index_select(0, dst_local),
            ],
            dim=0,
        )
        return edge_index, r_hist2f

    def _predict_velocity_field(
        self,
        x_t: Tensor,
        tau: Tensor,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        state: Dict[str, Tensor],
        future_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """현재 context에서 conditional flow velocity field를 예측한다.

        Args:
            x_t: ``[N, 4, 6, 4]`` noisy future segment.
            tau: ``[N, 1]`` flow time.
            tokenized_agent: tokenized agent dict.
            map_feature: encoded map dict.
                open-loop anchor batch에서는 geometry row 수와 ``pt_token`` row 수가 다를 수 있고,
                이때는 ``token_index`` 로 geometry row가 어떤 token feature를 읽는지 연결한다.
            state: history/current state.
            future_mask: ``[N, 4]``. supervised future query에 실제로 참여시킬
                agent-segment mask. ``None`` 이면 ``current_valid`` 를 사용한다.

        Returns:
            ``[N, 4, 6, 4]`` predicted velocity field.
        """
        n_agent = x_t.shape[0]
        if future_mask is None:
            future_mask = self._expand_agent_mask_to_future_segments(state["current_valid"])
        future_mask = future_mask.bool()
        ctx = self._build_history_memory(tokenized_agent, state)
        future_feat = self._build_future_query(x_t, tau, tokenized_agent)

        future_pos, future_head = segment_end_pose_global(x_t, state["current_pos"], state["current_head"])
        future_head_vec = torch.stack([future_head.cos(), future_head.sin()], dim=-1)
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=future_pos,
            head_a=future_head,
            head_vector_a=future_head_vec,
            mask=future_mask,
        )

        ctx_flat = ctx["feat"].flatten(0, 1)
        edge_index_hist, r_hist = self.build_hist2f_edge(
            ctx_pos=ctx["pos"],
            ctx_head=ctx["head"],
            ctx_valid=ctx["valid"],
            future_pos=future_pos,
            future_head=future_head,
            future_valid=future_mask,
            batch_agent=tokenized_agent["batch"],
        )

        num_graphs = int(tokenized_agent["num_graphs"])
        batch_s = torch.cat(
            [tokenized_agent["batch"] + num_graphs * t for t in range(self.future_num_segments)],
            dim=0,
        )
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=future_pos,
            head_a=future_head,
            head_vector_a=future_head_vec,
            batch_s=batch_s,
            mask=future_mask,
        )
        feat_map, edge_index_pl2a, r_pl2a = self._build_compact_map_attention_inputs(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            state=state,
            future_mask=future_mask,
            batch_s=batch_s,
        )

        for i in range(self.num_layers):
            future_feat = self.t_attn_layers[i](future_feat.flatten(0, 1), r_t, edge_index_t).view(
                n_agent,
                self.future_num_segments,
                -1,
            )
            future_feat = self.hist2f_attn_layers[i]((ctx_flat, future_feat.flatten(0, 1)), r_hist, edge_index_hist).view(
                n_agent,
                self.future_num_segments,
                -1,
            )
            future_tm = future_feat.transpose(0, 1).flatten(0, 1)
            future_tm = self.pt2a_attn_layers[i]((feat_map, future_tm), r_pl2a, edge_index_pl2a)
            future_tm = self.a2a_attn_layers[i](future_tm, r_a2a, edge_index_a2a)
            future_feat = future_tm.view(self.future_num_segments, n_agent, -1).transpose(0, 1)

        flow_pred = self.segment_out_head(future_feat).view(
            n_agent,
            self.future_num_segments,
            self.future_segment_points,
            4,
        )
        return flow_pred

    def _stack_anchor_state_list(self, state_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """anchor별 state dict를 한 번에 쌓는다.

        Args:
            state_list: 길이 ``K`` 인 state dict 목록.
                각 dict의 값 shape은 아래와 같다.
                - ``hist_idx``: ``[N, H]``
                - ``hist_pos``: ``[N, H, 2]``
                - ``hist_head``: ``[N, H]``
                - ``hist_valid``: ``[N, H]``
                - ``current_pos``: ``[N, 2]``
                - ``current_head``: ``[N]``
                - ``current_vel``: ``[N, 2]``
                - ``current_yaw_rate``: ``[N]``
                - ``current_valid``: ``[N]``

        Returns:
            anchor 축이 앞에 추가된 state dict.
            각 값 shape은 ``[K, N, ...]`` 이다.
        """
        if len(state_list) == 0:
            raise ValueError("state_list must not be empty.")
        return {key: torch.stack([state[key] for state in state_list], dim=0) for key in state_list[0].keys()}

    @staticmethod
    def _flatten_anchor_tensor(tensor: Tensor) -> Tensor:
        """anchor 축과 agent 축을 하나로 합친다.

        Args:
            tensor: ``[K, N, ...]`` 텐서.

        Returns:
            ``[K*N, ...]`` 텐서.
        """
        if tensor.dim() < 2:
            raise ValueError(f"Expected tensor with at least 2 dims, got shape {tuple(tensor.shape)}.")
        return tensor.reshape(-1, *tensor.shape[2:])

    def _flatten_anchor_state(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """anchor batch state dict를 decoder 입력 shape로 평탄화한다.

        Args:
            state: 값 shape이 ``[K, N, ...]`` 인 state dict.

        Returns:
            값 shape이 ``[K*N, ...]`` 인 state dict.
        """
        return {key: self._flatten_anchor_tensor(value) for key, value in state.items()}


    def _resolve_agent_static_inputs(
        self,
        tokenized_agent: Dict[str, Tensor],
        n_agent: int,
    ) -> Dict[str, Tensor]:
        """현재 decoder 입력 크기에 맞는 agent 정적 정보를 준비한다.

        Args:
            tokenized_agent: tokenized agent dict.
                open-loop anchor batch 경로에서는 아래 key 를 추가로 받을 수 있다.
                - ``static_index``: ``[N_cur]``
                - ``type_emb_base``: ``[N_base, D]``
                - ``shape_emb_base``: ``[N_base, D]``
            n_agent: 현재 decoder 가 처리하는 agent 개수 ``N_cur``.

        Returns:
            dict:
                - ``type``: ``[N_cur]``
                - ``type_emb``: ``[N_cur, D]``
                - ``shape_emb``: ``[N_cur, D]``
                - ``ego_mask``: ``[N_cur]``
        """
        static_index = tokenized_agent.get("static_index")
        type_base = tokenized_agent["type"]
        ego_base = tokenized_agent["ego_mask"]
        type_emb_base = tokenized_agent.get("type_emb_base")
        if type_emb_base is None:
            type_emb_base = self.type_a_emb(type_base.long())
        shape_emb_base = tokenized_agent.get("shape_emb_base")
        if shape_emb_base is None:
            shape_emb_base = self.shape_emb(tokenized_agent["shape"])

        if static_index is None:
            if type_base.shape[0] != n_agent:
                raise ValueError(
                    f"Expected {n_agent} agent static rows, got {type_base.shape[0]} without static_index."
                )
            return {
                "type": type_base,
                "type_emb": type_emb_base,
                "shape_emb": shape_emb_base,
                "ego_mask": ego_base,
            }

        static_index = static_index.long()
        if static_index.shape[0] != n_agent:
            raise ValueError(
                f"Expected static_index with {n_agent} rows, got {static_index.shape[0]}."
            )
        return {
            "type": type_base.index_select(0, static_index),
            "type_emb": type_emb_base.index_select(0, static_index),
            "shape_emb": shape_emb_base.index_select(0, static_index),
            "ego_mask": ego_base.index_select(0, static_index),
        }

    def _expand_open_loop_agent_context(
        self,
        tokenized_agent: Dict[str, Tensor],
        n_anchor: int,
    ) -> Dict[str, Tensor]:
        """open-loop anchor batch 에서 agent 정적 정보는 공유하고 batch 표식만 늘린다.

        Args:
            tokenized_agent: 원본 tokenized agent dict.
            n_anchor: anchor 개수 ``K``.

        Returns:
            anchor batch 용 agent dict.
            - ``type``: ``[N]``
            - ``shape``: ``[N, 3]``
            - ``ego_mask``: ``[N]``
            - ``static_index``: ``[K*N]``
            - ``batch``: ``[K*N]``
            - ``num_graphs``: ``B*K``
        """
        base_num_graphs = int(tokenized_agent["num_graphs"])
        batch = tokenized_agent["batch"]
        n_agent = batch.shape[0]
        batch_offsets = (
            torch.arange(n_anchor, device=batch.device, dtype=batch.dtype).unsqueeze(1)
            * base_num_graphs
        )
        expanded_batch = (batch.unsqueeze(0) + batch_offsets).reshape(-1)
        static_index = (
            torch.arange(n_agent, device=batch.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(n_anchor, -1)
            .reshape(-1)
        )
        return {
            "num_graphs": base_num_graphs * n_anchor,
            "base_num_graphs": base_num_graphs,
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "ego_mask": tokenized_agent["ego_mask"],
            "batch": expanded_batch,
            "static_index": static_index,
            "type_emb_base": self.type_a_emb(tokenized_agent["type"].long()),
            "shape_emb_base": self.shape_emb(tokenized_agent["shape"]),
            "trajectory_token_veh": tokenized_agent["trajectory_token_veh"],
            "trajectory_token_ped": tokenized_agent["trajectory_token_ped"],
            "trajectory_token_cyc": tokenized_agent["trajectory_token_cyc"],
        }


    def _expand_map_feature_for_anchor_batch(
        self,
        map_feature: Dict[str, Tensor],
        n_anchor: int,
        num_graphs: int,
    ) -> Dict[str, Tensor]:
        """anchor batch 에서 map geometry 를 복제하지 않고 공유 dict 만 만든다.

        Args:
            map_feature: 인코딩된 map dict.
            n_anchor: anchor 개수 ``K``. 호환을 위해 받지만 복제에는 쓰지 않는다.
            num_graphs: 원래 scene 개수 ``B``. 호환을 위해 받는다.

        Returns:
            geometry 복제 없는 map dict.
            - ``pt_token``: ``[M_token, D]``
            - ``position``: ``[M_geom, 2]``
            - ``orientation``: ``[M_geom]``
            - ``batch``: ``[M_geom]``
            - ``token_index``: ``[M_geom]``
        """
        del n_anchor, num_graphs
        return {
            "pt_token": map_feature["pt_token"],
            "position": map_feature["position"],
            "orientation": map_feature["orientation"],
            "batch": map_feature["batch"],
            "token_index": self._get_map_token_index(map_feature),
        }

    def forward_anchor_batch(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_steps: Sequence[int] | Tensor,
        return_full_outputs: bool = True,
    ) -> Dict[str, Tensor]:
        """여러 anchor를 한 번의 decoder forward로 처리한다.

        무거운 graph 생성과 attention 계산은 한 번만 수행하고,
        loss 집계는 바깥에서 anchor별로 그대로 평균낼 수 있도록
        출력만 ``[K, N, ...]`` shape로 돌려준다.

        Args:
            tokenized_agent: 원본 tokenized agent dict.
            map_feature: 인코딩된 map dict.
            agent_raw: raw ``data['agent']`` dict.
            anchor_steps: 길이 ``K`` 인 raw 10Hz anchor step 목록.
            return_full_outputs: ``True`` 이면 validation/분석용 auxiliary 출력까지
                모두 반환하고, ``False`` 이면 train loss에 필요한 텐서만 반환한다.

        Returns:
            open-loop flow loss 계산에 필요한 batched dict.
            항상 아래 key는 포함된다.
            - ``flow_pred``: ``[K, N, 4, 6, 4]``
            - ``flow_target``: ``[K, N, 4, 6, 4]``
            - ``pred_segments``: ``[K, N, 4, 6, 4]``
            - ``future_valid``: ``[K, N]``
            그리고 ``return_full_outputs=True`` 일 때만 아래 key를 추가로 포함한다.
            - ``gt_segments``: ``[K, N, 4, 6, 4]``
            - ``pred_future_local``: ``[K, N, 21, 4]``
            - ``gt_future_local``: ``[K, N, 21, 4]``
        """
        anchor_steps_tensor = torch.as_tensor(
            anchor_steps,
            device=agent_raw["position"].device,
            dtype=torch.long,
        ).flatten()
        anchor_steps_list = [int(step) for step in anchor_steps_tensor.tolist()]
        if len(anchor_steps_list) == 0:
            raise ValueError("anchor_steps must not be empty.")

        state_list = [
            self._build_gt_state(tokenized_agent=tokenized_agent, agent_raw=agent_raw, anchor_step=anchor_step)
            for anchor_step in anchor_steps_list
        ]
        state_batch = self._stack_anchor_state_list(state_list)

        gt_future_local_list: Optional[List[Tensor]] = [] if return_full_outputs else None
        gt_segments_list: List[Tensor] = []
        future_valid_list: List[Tensor] = []
        for anchor_step in anchor_steps_list:
            gt_future_local, _, _, _ = build_local_future_target(
                pos_global=agent_raw["position"][..., :2],
                head_global=agent_raw["heading"],
                anchor_step=anchor_step,
                future_window_steps=self.future_window_steps,
            )
            gt_segments_list.append(chunk_future_21_to_4x6(gt_future_local))
            if gt_future_local_list is not None:
                gt_future_local_list.append(gt_future_local)
            future_valid_list.append(
                agent_raw["valid_mask"][:, anchor_step : anchor_step + self.future_window_steps + 1].all(dim=1)
            )

        gt_segments = torch.stack(gt_segments_list, dim=0)
        future_valid = torch.stack(future_valid_list, dim=0)
        gt_future_local = torch.stack(gt_future_local_list, dim=0) if gt_future_local_list is not None else None
        del gt_segments_list
        del future_valid_list
        if gt_future_local_list is not None:
            del gt_future_local_list

        x0 = torch.randn_like(gt_segments)
        tau = torch.rand(
            gt_segments.shape[0],
            gt_segments.shape[1],
            1,
            1,
            1,
            device=gt_segments.device,
            dtype=gt_segments.dtype,
        )
        x_t, flow_target = build_flow_path(x0=x0, x1=gt_segments, tau=tau)

        state_flat = self._flatten_anchor_state(state_batch)
        x_t_flat = self._flatten_anchor_tensor(x_t)
        tau_flat = self._flatten_anchor_tensor(tau)
        future_valid_flat = self._flatten_anchor_tensor(future_valid)
        future_mask_flat = self._expand_agent_mask_to_future_segments(future_valid_flat)

        batched_tokenized_agent = self._expand_open_loop_agent_context(
            tokenized_agent=tokenized_agent,
            n_anchor=len(anchor_steps_list),
        )
        batched_map_feature = self._expand_map_feature_for_anchor_batch(
            map_feature=map_feature,
            n_anchor=len(anchor_steps_list),
            num_graphs=int(tokenized_agent["num_graphs"]),
        )
        flow_pred_flat = self._predict_velocity_field(
            x_t=x_t_flat,
            tau=tau_flat.reshape(-1, 1),
            tokenized_agent=batched_tokenized_agent,
            map_feature=batched_map_feature,
            state=state_flat,
            future_mask=future_mask_flat,
        )
        pred_segments_flat = x_t_flat + (1.0 - tau_flat) * flow_pred_flat

        n_anchor, n_agent = gt_segments.shape[:2]
        pred_batch = {
            "flow_pred": flow_pred_flat.reshape(n_anchor, n_agent, *flow_pred_flat.shape[1:]),
            "flow_target": flow_target,
            "pred_segments": pred_segments_flat.reshape(n_anchor, n_agent, *pred_segments_flat.shape[1:]),
            "future_valid": future_valid,
        }
        if return_full_outputs:
            pred_future_local_flat = assemble_4x6_to_21(pred_segments_flat)
            pred_batch["gt_segments"] = gt_segments
            pred_batch["pred_future_local"] = pred_future_local_flat.reshape(
                n_anchor,
                n_agent,
                *pred_future_local_flat.shape[1:],
            )
            pred_batch["gt_future_local"] = gt_future_local
        else:
            del gt_segments
        return pred_batch

    def forward(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_step: int,
    ) -> Dict[str, Tensor]:
        """한 anchor 시각에 대한 open-loop flow matching 출력을 만든다.

        Args:
            tokenized_agent: tokenized agent dict.
            map_feature: encoded map dict.
            agent_raw: raw ``data['agent']`` dict.
            anchor_step: raw 10Hz anchor step.

        Returns:
            flow loss 계산에 필요한 dict.
        """
        state = self._build_gt_state(tokenized_agent, agent_raw, anchor_step)
        gt_future_local, _, _, _ = build_local_future_target(
            pos_global=agent_raw["position"][..., :2],
            head_global=agent_raw["heading"],
            anchor_step=anchor_step,
            future_window_steps=self.future_window_steps,
        )
        gt_segments = chunk_future_21_to_4x6(gt_future_local)
        future_valid = agent_raw["valid_mask"][
            :,
            anchor_step : anchor_step + self.future_window_steps + 1,
        ].all(dim=1)
        future_mask = self._expand_agent_mask_to_future_segments(future_valid)
        x0 = torch.randn_like(gt_segments)
        tau = torch.rand(gt_segments.shape[0], 1, 1, 1, device=gt_segments.device, dtype=gt_segments.dtype)
        x_t, flow_target = build_flow_path(x0, gt_segments, tau)
        flow_pred = self._predict_velocity_field(
            x_t=x_t,
            tau=tau.view(gt_segments.shape[0], 1),
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            state=state,
            future_mask=future_mask,
        )
        pred_segments = x_t + (1.0 - tau) * flow_pred
        pred_future_local = assemble_4x6_to_21(pred_segments)
        return {
            "flow_pred": flow_pred,
            "flow_target": flow_target,
            "pred_segments": pred_segments,
            "gt_segments": gt_segments,
            "pred_future_local": pred_future_local,
            "gt_future_local": gt_future_local,
            "future_valid": future_valid,
        }

    def closed_loop_train(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        unroll_steps: int,
    ) -> list[Dict[str, Tensor]]:
        """짧은 closed-loop fine-tuning용 0.5초 반복 unroll을 수행한다.

        Args:
            tokenized_agent: tokenized agent dict.
            map_feature: encoded map dict.
            agent_raw: raw ``data['agent']`` dict.
            unroll_steps: 몇 번 0.5초 전진할지.

        Returns:
            각 unroll step의 open-loop style 출력 dict list.
        """
        state = self._init_rollout_state(tokenized_agent, agent_raw)
        outputs = []
        for step in range(unroll_steps):
            anchor_step = self.current_step + step * self.shift
            gt_future_local, _, _, _ = build_local_future_target(
                pos_global=agent_raw["position"][..., :2],
                head_global=agent_raw["heading"],
                anchor_step=anchor_step,
                future_window_steps=self.future_window_steps,
            )
            gt_segments = chunk_future_21_to_4x6(gt_future_local)
            future_valid = agent_raw["valid_mask"][
                :,
                anchor_step : anchor_step + self.future_window_steps + 1,
            ].all(dim=1)
            future_mask = self._expand_agent_mask_to_future_segments(future_valid)
            x0 = torch.randn_like(gt_segments)
            tau = torch.rand(gt_segments.shape[0], 1, 1, 1, device=gt_segments.device, dtype=gt_segments.dtype)
            x_t, flow_target = build_flow_path(x0, gt_segments, tau)
            flow_pred = self._predict_velocity_field(
                x_t=x_t,
                tau=tau.view(gt_segments.shape[0], 1),
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                state=state,
                future_mask=future_mask,
            )
            pred_segments = x_t + (1.0 - tau) * flow_pred
            pred_future_local = assemble_4x6_to_21(pred_segments)
            outputs.append(
                {
                    "flow_pred": flow_pred,
                    "flow_target": flow_target,
                    "pred_segments": pred_segments,
                    "gt_segments": gt_segments,
                    "pred_future_local": pred_future_local,
                    "gt_future_local": gt_future_local,
                    "future_valid": future_valid,
                }
            )
            rollout = executed_chunk_to_rollout_update(
                future_local_21=pred_future_local.detach(),
                pos_now=state["current_pos"],
                head_now=state["current_head"],
            )
            nearest_idx = nearest_agent_token_idx(
                local_chunk_6=rollout["exec_local_6"],
                agent_shape=tokenized_agent["token_agent_shape"],
                token_traj_all=tokenized_agent["token_traj_all"],
            )
            state = self._advance_rollout_state(state, nearest_idx, rollout)
        return outputs

    @torch.no_grad()
    def inference(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        """8초 closed-loop rollout을 수행한다.

        Args:
            tokenized_agent: tokenized agent dict.
            map_feature: encoded map dict.
            agent_raw: raw ``data['agent']`` dict.
            sampling_scheme: Hydra config. flow 버전에서는 노이즈 seed만 달라지는 용도로만 쓴다.

        Returns:
            WOSAC 제출과 평가에 필요한 dict.
        """
        del sampling_scheme  # flow head는 별도 top-k sampling을 쓰지 않는다.
        n_agent = tokenized_agent["gt_idx"].shape[0]
        pred_traj_10hz = torch.zeros(
            n_agent,
            self.num_future_steps,
            2,
            device=tokenized_agent["gt_pos"].device,
            dtype=tokenized_agent["gt_pos"].dtype,
        )
        pred_head_10hz = torch.zeros(
            n_agent,
            self.num_future_steps,
            device=tokenized_agent["gt_heading"].device,
            dtype=tokenized_agent["gt_heading"].dtype,
        )
        state = self._init_rollout_state(tokenized_agent, agent_raw)

        for step in range(self.num_future_steps // self.shift):
            x0 = torch.randn(
                n_agent,
                self.future_num_segments,
                self.future_segment_points,
                4,
                device=pred_traj_10hz.device,
                dtype=pred_traj_10hz.dtype,
            )

            def _velocity_fn(x: Tensor, t: Tensor) -> Tensor:
                return self._predict_velocity_field(
                    x_t=x,
                    tau=t,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    state=state,
                )

            pred_segments = midpoint_ode_integrate(x0=x0, ode_steps=self.ode_steps, velocity_fn=_velocity_fn)
            pred_future_local = assemble_4x6_to_21(pred_segments)
            rollout = executed_chunk_to_rollout_update(
                future_local_21=pred_future_local,
                pos_now=state["current_pos"],
                head_now=state["current_head"],
            )
            pred_traj_10hz[:, step * self.shift : (step + 1) * self.shift] = rollout["exec_global_pos_6"][:, 1:]
            pred_head_10hz[:, step * self.shift : (step + 1) * self.shift] = rollout["exec_global_head_6"][:, 1:]
            nearest_idx = nearest_agent_token_idx(
                local_chunk_6=rollout["exec_local_6"],
                agent_shape=tokenized_agent["token_agent_shape"],
                token_traj_all=tokenized_agent["token_traj_all"],
            )
            state = self._advance_rollout_state(state, nearest_idx, rollout)

        pred_z = state["current_z"].unsqueeze(1).expand(-1, self.num_future_steps)
        return {
            "pred_traj_10hz": pred_traj_10hz,
            "pred_head_10hz": pred_head_10hz,
            "pred_z_10hz": pred_z,
        }
