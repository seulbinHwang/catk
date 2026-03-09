from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.smart.layers import FourierEmbedding, MLPLayer, MLPEmbedding
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.utils.flow_traj import (
    assemble_4x6_to_21,
    build_current_anchor_feature,
    build_local_future_target,
    build_ot_flow_path,
    chunk_future_21_to_4x6,
    match_first_segment_token,
    normalize_sincos,
    segment_endpoint_pose_global,
    segment_local_to_global,
)
from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import transform_to_global


class SMARTAgentFlowDecoder(SMARTAgentDecoder):
    """SMART backbone 위에 2초 flow head를 붙인 agent decoder입니다.

    기존 SMARTAgentDecoder가 이미 가지고 있는
    - token 임베딩
    - temporal / map / a2a sparse attention
    - relation 임베딩

    을 그대로 재사용하고, 마지막 next-token 분류 head 대신
    조각 미래를 복원하는 flow head만 추가합니다.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        future_window_steps: int,
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
        history_slots: int = 6,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )
        # Flow pretraining/fine-tuning never uses the legacy next-token classifier head.
        # Leaving it trainable creates DDP-ununsed parameters for the flow path.
        for p in self.token_predict_head.parameters():
            p.requires_grad = False
        self.future_window_steps = future_window_steps
        self.history_slots = history_slots
        self.n_future_segments = 4
        self.segment_points = 6
        self.state_dim = 4

        self.segment_index_emb = nn.Embedding(self.n_future_segments, hidden_dim)
        self.flow_time_emb = FourierEmbedding(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.current_anchor_emb = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.future_segment_emb = MLPEmbedding(input_dim=24, hidden_dim=hidden_dim)
        self.context_fusion = MLPEmbedding(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        self.segment_out_head = MLPLayer(hidden_dim, hidden_dim, 24)

    def build_initial_state(
        self,
        tokenized_agent: Dict[str, Tensor],
        data,
        anchor_10hz: int,
    ) -> Dict[str, Tensor]:
        """주어진 anchor에서 rollout 시작 상태를 만듭니다.

        Args:
            tokenized_agent: token processor 출력입니다.
            data: 원본 scene batch입니다.
            anchor_10hz: 현재 anchor 시각입니다.

        Returns:
            내부 history token 상태와 현재 연속 상태를 담은 사전입니다.
        """
        anchor_token_idx = anchor_10hz // self.shift
        start_token_idx = max(0, anchor_token_idx - self.history_slots)

        current_head = data["agent"]["heading"][:, anchor_10hz].clone()  # [n_agent]
        current_vel = data["agent"]["velocity"][:, anchor_10hz].clone()  # [n_agent, 2]
        current_valid = data["agent"]["valid_mask"][:, anchor_10hz].clone()  # [n_agent]
        if anchor_10hz > 0:
            prev_head = data["agent"]["heading"][:, anchor_10hz - 1]
            current_yaw_rate = wrap_angle(current_head - prev_head) / 0.1  # [n_agent]
        else:
            current_yaw_rate = current_head.new_zeros(current_head.shape)

        return {
            "token_idx": tokenized_agent["gt_idx"][:, start_token_idx:anchor_token_idx].clone(),
            "pos_token": tokenized_agent["gt_pos"][:, start_token_idx:anchor_token_idx].clone(),
            "head_token": tokenized_agent["gt_heading"][:, start_token_idx:anchor_token_idx].clone(),
            "valid_token": tokenized_agent["valid_mask"][:, start_token_idx:anchor_token_idx].clone(),
            "current_pos": data["agent"]["position"][:, anchor_10hz, :2].clone(),
            "current_head": current_head,
            "current_vel_global": current_vel,
            "current_yaw_rate": current_yaw_rate,
            "current_valid": current_valid,
        }

    def encode_history(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        state: Dict[str, Tensor],
    ) -> Tensor:
        """현재 history token들을 SMART sparse attention으로 요약합니다.

        Args:
            tokenized_agent: token processor 출력입니다.
            map_feature: map encoder 출력입니다.
            state: 현재 rollout 상태입니다.

        Returns:
            agent별 요약 특징입니다.
            shape: `[n_agent, hidden_dim]`
        """
        pos_a = state["pos_token"]  # [n_agent, n_hist, 2]
        head_a = state["head_token"]  # [n_agent, n_hist]
        mask = state["valid_token"]  # [n_agent, n_hist]

        if pos_a.shape[1] == 0:
            return state["current_pos"].new_zeros((state["current_pos"].shape[0], self.hidden_dim))

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)  # [n_agent, n_hist, 2]
        feat_a = self.agent_token_embedding(
            agent_token_index=state["token_idx"],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
        )  # [n_agent, n_hist, hidden_dim]

        n_agent, n_hist = head_a.shape
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
        )
        batch_s = torch.cat(
            [tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t for t in range(n_hist)],
            dim=0,
        )  # [n_agent * n_hist]
        batch_pl = torch.cat(
            [map_feature["batch"] + tokenized_agent["num_graphs"] * t for t in range(n_hist)],
            dim=0,
        )  # [n_pl * n_hist]
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
        feat_map = map_feature["pt_token"].unsqueeze(0).expand(n_hist, -1, -1).flatten(0, 1)

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_hist, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_hist, n_agent, -1).transpose(0, 1)

        last_idx = mask.long().sum(dim=1).sub(1).clamp(min=0)  # [n_agent]
        hist_feat = feat_a[torch.arange(n_agent, device=feat_a.device), last_idx]  # [n_agent, hidden_dim]
        hist_feat = hist_feat * mask.any(dim=1, keepdim=True).to(hist_feat.dtype)
        return hist_feat

    def prepare_context(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        state: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """현재 history와 anchor 연속 상태를 합쳐 flow 조건을 만듭니다."""
        hist_feat = self.encode_history(tokenized_agent, map_feature, state)  # [n_agent, hidden_dim]
        anchor_feature = build_current_anchor_feature(
            current_head=state["current_head"],
            current_vel_global=state["current_vel_global"],
            current_yaw_rate=state["current_yaw_rate"],
            agent_shape=tokenized_agent["shape"],
            agent_type=tokenized_agent["type"],
        )  # [n_agent, 8]
        anchor_emb = self.current_anchor_emb(anchor_feature)  # [n_agent, hidden_dim]
        ctx_feat = self.context_fusion(torch.cat([hist_feat, anchor_emb], dim=-1))  # [n_agent, hidden_dim]
        return {
            "ctx_feat": ctx_feat,
            "current_pos": state["current_pos"],
            "current_head": state["current_head"],
            "active_mask": state["current_valid"],
        }

    def build_future_queries(self, context: Dict[str, Tensor], z: Tensor, tau: Tensor) -> Tensor:
        """현재 noisy future 조각을 query token으로 바꿉니다."""
        n_agent = z.shape[0]
        seg_feat = self.future_segment_emb(z.reshape(n_agent * self.n_future_segments, -1)).view(
            n_agent, self.n_future_segments, -1
        )  # [n_agent, 4, hidden_dim]
        tau_feat = self.flow_time_emb(
            continuous_inputs=tau.view(-1, 1, 1).expand(-1, self.n_future_segments, -1).reshape(-1, 1),
            categorical_embs=None,
        ).view(n_agent, self.n_future_segments, -1)
        seg_index = torch.arange(self.n_future_segments, device=z.device)
        seg_index_feat = self.segment_index_emb(seg_index).unsqueeze(0)  # [1, 4, hidden_dim]
        return seg_feat + tau_feat + seg_index_feat + context["ctx_feat"].unsqueeze(1)

    def flow_field(
            self,
            tokenized_agent: Dict[str, Tensor],
            map_feature: Dict[str, Tensor],
            context: Dict[str, Tensor],
            z: Tensor,
            tau: Tensor,
    ) -> Tensor:
        """현재 noisy future에서 velocity field를 예측합니다."""
        # IMPORTANT:
        # flow matching에서는 velocity target이 "정규화 전 선형 OT 경로"를 기준으로
        # 정의됩니다. 따라서 query conditioning도 같은 raw `z`를 봐야 합니다.
        # 여기서 미리 normalize를 해 버리면 모델은 다른 경로의 상태를 입력으로 받고,
        # target velocity는 원래 경로를 향해 있어 학습이 일관되지 않습니다.
        query = self.build_future_queries(context, z,
                                          tau)  # [n_agent, 4, hidden_dim]

        # 다만 sparse relation을 만들 때는 heading이 안정적이어야 하므로,
        # geometry를 뽑아낼 때만 별도의 정규화 사본을 사용합니다.
        geom_z = normalize_sincos(z)
        end_pos, end_head = segment_endpoint_pose_global(
            segments_local=geom_z,
            current_pos=context["current_pos"],
            current_head=context["current_head"],
        )
        head_vector = torch.stack([end_head.cos(), end_head.sin()],
                                  dim=-1)  # [n_agent, 4, 2]
        mask = context["active_mask"].unsqueeze(1).expand(-1,
                                                          self.n_future_segments)  # [n_agent, 4]
        n_agent = z.shape[0]

        batch_s = torch.cat(
            [tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t for t
             in range(self.n_future_segments)],
            dim=0,
        )  # [n_agent * 4]
        batch_pl = torch.cat(
            [map_feature["batch"] + tokenized_agent["num_graphs"] * t for t in
             range(self.n_future_segments)],
            dim=0,
        )  # [n_pl * 4]

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=end_pos,
            head_a=end_head,
            head_vector_a=head_vector,
            mask=mask,
        )
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=end_pos,
            head_a=end_head,
            head_vector_a=head_vector,
            batch_s=batch_s,
            mask=mask,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=end_pos,
            head_a=end_head,
            head_vector_a=head_vector,
            mask=mask,
            batch_s=batch_s,
            batch_pl=batch_pl,
        )
        feat_map = map_feature["pt_token"].unsqueeze(0).expand(
            self.n_future_segments, -1, -1).flatten(0, 1)
        feat = query
        for i in range(self.num_layers):
            feat = feat.flatten(0, 1)
            feat = self.t_attn_layers[i](feat, r_t, edge_index_t)
            feat = feat.view(n_agent, self.n_future_segments, -1).transpose(0,
                                                                            1).flatten(
                0, 1)
            feat = self.pt2a_attn_layers[i]((feat_map, feat), r_pl2a,
                                            edge_index_pl2a)
            feat = self.a2a_attn_layers[i](feat, r_a2a, edge_index_a2a)
            feat = feat.view(self.n_future_segments, n_agent, -1).transpose(0,
                                                                            1)

        velocity = self.segment_out_head(feat).view(
            n_agent, self.n_future_segments, self.segment_points, self.state_dim
        )
        velocity = velocity * context["active_mask"].to(velocity.dtype).view(-1,
                                                                             1,
                                                                             1,
                                                                             1)
        velocity[:, 0, 0, 0] = 0.0
        velocity[:, 0, 0, 1] = 0.0
        velocity[:, 0, 0, 2] = 0.0
        velocity[:, 0, 0, 3] = 0.0
        return velocity

    def forward(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        data,
        anchor_10hz: int,
        sampling_cfg: DictConfig,
    ) -> Dict[str, Tensor]:
        """open-loop anchor 하나에 대한 flow 학습 입력을 만듭니다."""
        state = self.build_initial_state(tokenized_agent, data, anchor_10hz)
        context = self.prepare_context(tokenized_agent, map_feature, state)
        target_future, target_valid = build_local_future_target(
            pos_global=data["agent"]["position"][..., :2],
            head_global=data["agent"]["heading"],
            valid_mask=data["agent"]["valid_mask"],
            anchor_10hz=anchor_10hz,
            anchor_pos=state["current_pos"],
            anchor_head=state["current_head"],
            future_window_steps=self.future_window_steps,
        )
        target_segments = chunk_future_21_to_4x6(target_future)  # [n_agent, 4, 6, 4]
        _, z_tau, tau, target_velocity = build_ot_flow_path(target_segments, sampling_cfg.noise_scale)
        pred_velocity = self.flow_field(tokenized_agent, map_feature, context, z_tau, tau)
        pred_segments = normalize_sincos(z_tau + (1.0 - tau.view(-1, 1, 1, 1)) * pred_velocity)
        pred_future = assemble_4x6_to_21(pred_segments)
        return {
            "pred_velocity": pred_velocity,
            "target_velocity": target_velocity,
            "pred_segments": pred_segments,
            "target_segments": target_segments,
            "pred_future": pred_future,
            "target_future": target_future,
            "target_valid": target_valid,
            "active_mask": context["active_mask"],
        }

    def sample_future(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        context: Dict[str, Tensor],
        sampling_cfg: DictConfig,
    ) -> Dict[str, Tensor]:
        """4-step midpoint ODE로 2초 미래를 샘플링합니다."""
        n_agent = context["ctx_feat"].shape[0]
        z = context["ctx_feat"].new_zeros((n_agent, self.n_future_segments, self.segment_points, self.state_dim))
        z = z + torch.randn_like(z) * sampling_cfg.noise_scale
        z[:, 0, 0, 0] = 0.0
        z[:, 0, 0, 1] = 0.0
        z[:, 0, 0, 2] = 0.0
        z[:, 0, 0, 3] = 1.0
        dt = 1.0 / float(sampling_cfg.ode_steps)
        for step in range(sampling_cfg.ode_steps):
            t0 = step * dt
            tau0 = z.new_full((n_agent,), t0)
            k1 = self.flow_field(tokenized_agent, map_feature, context, z, tau0)
            z_mid = normalize_sincos(z + 0.5 * dt * k1)
            z_mid[:, 0, 0, 0] = 0.0
            z_mid[:, 0, 0, 1] = 0.0
            z_mid[:, 0, 0, 2] = 0.0
            z_mid[:, 0, 0, 3] = 1.0
            tau_mid = z.new_full((n_agent,), t0 + 0.5 * dt)
            k2 = self.flow_field(tokenized_agent, map_feature, context, z_mid, tau_mid)
            z = normalize_sincos(z + dt * k2)
            z[:, 0, 0, 0] = 0.0
            z[:, 0, 0, 1] = 0.0
            z[:, 0, 0, 2] = 0.0
            z[:, 0, 0, 3] = 1.0
        future = assemble_4x6_to_21(z)
        return {"pred_segments": z, "pred_future": future}

    def append_next_state(
        self,
        tokenized_agent: Dict[str, Tensor],
        state: Dict[str, Tensor],
        first_segment_local: Tensor,
    ) -> Dict[str, Tensor]:
        """첫 0.5초를 사용해 내부 history state를 다음 step으로 넘깁니다."""
        active_mask = state["current_valid"]  # [n_agent]
        n_agent = first_segment_local.shape[0]
        device = first_segment_local.device
        token_idx_next = match_first_segment_token(
            first_segment_local=first_segment_local,
            token_traj_all=tokenized_agent["token_traj_all"],
            token_agent_shape=tokenized_agent["token_agent_shape"],
        )  # [n_agent]

        token_local = tokenized_agent["token_traj_all"][torch.arange(n_agent, device=device), token_idx_next]
        token_global = transform_to_global(
            pos_local=token_local.flatten(1, 2),
            head_local=None,
            pos_now=state["current_pos"],
            head_now=state["current_head"],
        )[0].view_as(token_local)
        token_pos_next = token_global[:, -1].mean(dim=1)  # [n_agent, 2]
        token_diff = token_global[:, -1, 0] - token_global[:, -1, 3]
        token_head_next = torch.atan2(token_diff[:, 1], token_diff[:, 0])  # [n_agent]

        pos_global, head_global = segment_local_to_global(
            segment_local=first_segment_local,
            current_pos=state["current_pos"],
            current_head=state["current_head"],
        )
        current_pos_next = pos_global[:, -1]
        current_head_next = head_global[:, -1]
        current_vel_next = (pos_global[:, -1] - pos_global[:, -2]) / 0.1
        current_yaw_rate_next = wrap_angle(head_global[:, -1] - head_global[:, -2]) / 0.1

        token_idx_append = torch.where(active_mask, token_idx_next, state["token_idx"][:, -1])
        token_pos_append = torch.where(active_mask.unsqueeze(-1), token_pos_next, state["pos_token"][:, -1])
        token_head_append = torch.where(active_mask, token_head_next, state["head_token"][:, -1])
        valid_append = active_mask

        token_idx = torch.cat([state["token_idx"], token_idx_append.unsqueeze(1)], dim=1)
        pos_token = torch.cat([state["pos_token"], token_pos_append.unsqueeze(1)], dim=1)
        head_token = torch.cat([state["head_token"], token_head_append.unsqueeze(1)], dim=1)
        valid_token = torch.cat([state["valid_token"], valid_append.unsqueeze(1)], dim=1)

        if token_idx.shape[1] > self.history_slots:
            token_idx = token_idx[:, -self.history_slots :]
            pos_token = pos_token[:, -self.history_slots :]
            head_token = head_token[:, -self.history_slots :]
            valid_token = valid_token[:, -self.history_slots :]

        return {
            "token_idx": token_idx,
            "pos_token": pos_token,
            "head_token": head_token,
            "valid_token": valid_token,
            "current_pos": torch.where(active_mask.unsqueeze(-1), current_pos_next, state["current_pos"]),
            "current_head": torch.where(active_mask, current_head_next, state["current_head"]),
            "current_vel_global": torch.where(active_mask.unsqueeze(-1), current_vel_next, state["current_vel_global"]),
            "current_yaw_rate": torch.where(active_mask, current_yaw_rate_next, state["current_yaw_rate"]),
            "current_valid": active_mask,
        }

    def rollout(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_cfg: DictConfig,
        data=None,
        rollout_steps: Optional[int] = None,
        return_targets: bool = False,
    ) -> Dict[str, Tensor]:
        """0.5초씩 반복해서 closed-loop rollout을 수행합니다.

        Args:
            tokenized_agent: token processor 출력입니다.
            map_feature: map encoder 출력입니다.
            sampling_cfg: flow 샘플링 설정입니다.
            data: 원본 scene batch입니다.
            rollout_steps: 0.5초 단위 rollout step 수입니다.
            return_targets: 학습용으로 각 step의 GT 2초 미래도 함께 돌려줄지 여부입니다.

        Returns:
            WOSAC 제출과 local val에서 바로 쓸 수 있는 결과 dict입니다.
        """
        n_agent = tokenized_agent["gt_idx"].shape[0]
        rollout_steps = rollout_steps or (self.num_future_steps // self.shift)
        state = self.build_initial_state(tokenized_agent, data, self.num_historical_steps - 1)
        pred_traj_10hz = state["current_pos"].new_zeros((n_agent, rollout_steps * self.shift, 2))
        pred_head_10hz = state["current_head"].new_zeros((n_agent, rollout_steps * self.shift))
        pred_local_futures: List[Tensor] = []
        target_local_futures: List[Tensor] = []
        target_valids: List[Tensor] = []

        for step in range(rollout_steps):
            anchor_10hz = self.num_historical_steps - 1 + step * self.shift
            context = self.prepare_context(tokenized_agent, map_feature, state)
            pred_dict = self.sample_future(tokenized_agent, map_feature, context, sampling_cfg)
            pred_local_futures.append(pred_dict["pred_future"])
            first_segment_local = pred_dict["pred_segments"][:, 0]  # [n_agent, 6, 4]
            pos_global, head_global = segment_local_to_global(
                segment_local=first_segment_local,
                current_pos=state["current_pos"],
                current_head=state["current_head"],
            )
            active_traj = context["active_mask"].unsqueeze(-1).unsqueeze(-1)
            active_head = context["active_mask"].unsqueeze(-1)
            pred_traj_10hz[:, step * self.shift : (step + 1) * self.shift] = torch.where(
                active_traj,
                pos_global[:, 1:],
                pred_traj_10hz[:, step * self.shift : (step + 1) * self.shift],
            )
            pred_head_10hz[:, step * self.shift : (step + 1) * self.shift] = torch.where(
                active_head,
                head_global[:, 1:],
                pred_head_10hz[:, step * self.shift : (step + 1) * self.shift],
            )
            if return_targets and data is not None:
                target_future, target_valid = build_local_future_target(
                    pos_global=data["agent"]["position"][..., :2],
                    head_global=data["agent"]["heading"],
                    valid_mask=data["agent"]["valid_mask"],
                    anchor_10hz=anchor_10hz,
                    anchor_pos=state["current_pos"],
                    anchor_head=state["current_head"],
                    future_window_steps=self.future_window_steps,
                )
                target_local_futures.append(target_future)
                target_valids.append(target_valid)
            state = self.append_next_state(tokenized_agent, state, first_segment_local)

        if "gt_z_raw" in tokenized_agent:
            pred_z_10hz = tokenized_agent["gt_z_raw"].unsqueeze(1).expand(-1, pred_traj_10hz.shape[1])
        else:
            pred_z_10hz = pred_traj_10hz.new_zeros((n_agent, pred_traj_10hz.shape[1]))

        out = {
            "pred_traj_10hz": pred_traj_10hz,
            "pred_head_10hz": pred_head_10hz,
            "pred_z_10hz": pred_z_10hz,
        }
        if return_targets:
            out["pred_local_futures"] = pred_local_futures
            out["target_local_futures"] = target_local_futures
            out["target_valids"] = target_valids
        return out
