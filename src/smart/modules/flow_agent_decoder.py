from __future__ import annotations

from typing import Dict

import torch
from omegaconf import DictConfig
from torch_cluster import radius_graph
from torch_geometric.utils import subgraph

from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.flow_local_decoder import (
    ContinuousCommitBridge,
    FlowODE,
    HierarchicalFlowDecoder,
)
from src.smart.utils import angle_between_2d_vectors, wrap_angle


class SMARTFlowAgentDecoder(SMARTAgentEncoder):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: int | None,
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        flow_dim: int,
        flow_num_chunk_heads: int,
        flow_num_chunk_layers: int,
        flow_solver_steps: int,
        flow_solver_method: str,
        flow_solver_eps: float,
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
        self.r_a2a_emb = FourierEmbedding(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.flow_decoder = HierarchicalFlowDecoder(
            context_dim=hidden_dim,
            flow_dim=flow_dim,
            num_chunk_heads=flow_num_chunk_heads,
            num_chunk_layers=flow_num_chunk_layers,
        )
        self.flow_ode = FlowODE(
            eps=flow_solver_eps,
            solver_steps=flow_solver_steps,
            solver_method=flow_solver_method,
        )
        self.commit_bridge = ContinuousCommitBridge()

    def build_interaction_edge(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        batch_s: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask_flat = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        motion_a = torch.cat(
            [
                pos_a.new_zeros(pos_a.shape[0], 1, pos_a.shape[-1]),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )
        motion_s = motion_a.transpose(0, 1).reshape(-1, 2)

        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask_flat, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])

        # Use coarse-step relative displacement instead of raw m/s velocity so the
        # added relation channels stay on a meter-scale comparable to the existing
        # distance feature without introducing another global normalization rule.
        rel_motion = motion_s[edge_index_a2a[0]] - motion_s[edge_index_a2a[1]]
        recv_head = head_s[edge_index_a2a[1]]
        recv_cos = recv_head.cos()
        recv_sin = recv_head.sin()
        rel_motion_long = rel_motion[:, 0] * recv_cos + rel_motion[:, 1] * recv_sin
        rel_motion_lat = -rel_motion[:, 0] * recv_sin + rel_motion[:, 1] * recv_cos

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

    def _build_step_offset_batch(
        self,
        batch: torch.Tensor,
        num_steps: int,
        num_graphs: int,
    ) -> torch.Tensor:
        """시간축이 다른 agent 노드가 서로 섞이지 않도록 batch 번호를 벌립니다.

        Args:
            batch: 장면 번호입니다. shape은 ``[n_agent]`` 입니다.
            num_steps: 펼칠 coarse step 개수입니다.
            num_graphs: 한 배치 안의 장면 개수입니다.

        Returns:
            torch.Tensor:
                step마다 다른 영역으로 밀어낸 batch 번호입니다.
                shape은 ``[num_steps * n_agent]`` 입니다.
        """
        step_offsets = (
            torch.arange(num_steps, device=batch.device, dtype=batch.dtype)
            .repeat_interleave(batch.shape[0])
            * num_graphs
        )
        return batch.repeat(num_steps) + step_offsets

    def _pack_anchor_hidden(
        self,
        anchor_hidden: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """유효한 anchor hidden만 anchor 순서대로 압축합니다.

        Args:
            anchor_hidden: context encoder 출력입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 유효 anchor 여부입니다. shape은 ``[n_agent, 13]`` 입니다.

        Returns:
            torch.Tensor:
                유효한 anchor만 모은 hidden입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
        """
        packed_hidden = [
            anchor_hidden[:, anchor_idx][anchor_mask[:, anchor_idx]]
            for anchor_idx in range(anchor_hidden.shape[1])
            if anchor_mask[:, anchor_idx].any()
        ]
        if len(packed_hidden) == 0:
            return anchor_hidden.new_zeros((0, anchor_hidden.shape[-1]))
        return torch.cat(packed_hidden, dim=0)

    def _encode_context(
        self,
        agent_token_index: torch.Tensor,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        mask: torch.Tensor,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
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
        batch_s_a2a = self._build_step_offset_batch(
            batch=tokenized_agent["batch"],
            num_steps=n_step,
            num_graphs=tokenized_agent["num_graphs"],
        )
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            batch_s=batch_s_a2a,
            mask=mask,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
            batch_s=batch_s_pl2a,
            batch_pl=map_feature["batch"],
        )

        feat_map = map_feature["pt_token"]
        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
        return feat_a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        anchor_mask: torch.Tensor,
        flow_clean_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ctx_hidden_pack = self._encode_context(
            agent_token_index=tokenized_agent["ctx_sampled_idx"],
            pos_a=tokenized_agent["ctx_sampled_pos"],
            head_a=tokenized_agent["ctx_sampled_heading"],
            mask=tokenized_agent["ctx_valid"],
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        anchor_hidden = ctx_hidden_pack[:, 1:, :]
        anchor_hidden_valid = self._pack_anchor_hidden(anchor_hidden, anchor_mask)

        if flow_clean_norm.numel() == 0:
            empty = flow_clean_norm.new_zeros((0, 20, 4))
            return {
                "flow_pred_norm": empty,
                "flow_target_norm": empty,
                "flow_pred_clean_norm": empty,
                "flow_clean_norm": empty,
                "ctx_hidden_pack": ctx_hidden_pack,
                "anchor_hidden": anchor_hidden,
                "anchor_mask": anchor_mask,
            }

        flow_sample = self.flow_ode.sample(flow_clean_norm, target_type="velocity")
        flow_pred_norm = self.flow_decoder(anchor_hidden_valid, flow_sample.x_t, flow_sample.tau)
        flow_pred_clean_norm = self.flow_ode.predict_clean_from_velocity(
            flow_sample.x_t,
            flow_pred_norm,
            flow_sample.tau,
        )
        return {
            "flow_pred_norm": flow_pred_norm,
            "flow_target_norm": flow_sample.target,
            "flow_pred_clean_norm": flow_pred_clean_norm,
            "flow_clean_norm": flow_clean_norm,
            "ctx_hidden_pack": ctx_hidden_pack,
            "anchor_hidden": anchor_hidden,
            "anchor_mask": anchor_mask,
        }

    @torch.no_grad()
    def prepare_inference_cache(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """여러 rollout이 공통으로 쓰는 초기 문맥을 한 번만 만듭니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징 사전입니다.

        Returns:
            Dict[str, object]:
                첫 rollout 직전 상태를 담은 캐시입니다.
                창 상태 텐서는 ``[n_agent, n_hist, ...]`` 꼴이고,
                layer별 시계열 캐시는 ``feat_a_t_dict[layer]`` 형태로 저장됩니다.
        """
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps
        n_step_future_2hz = n_step_future_10hz // self.shift
        step_current_10hz = self.num_historical_steps - 1
        step_current_2hz = step_current_10hz // self.shift
        max_context_steps = 14

        pos_window = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_window = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_window = torch.stack([head_window.cos(), head_window.sin()], dim=-1)
        valid_window = tokenized_agent["valid_mask"][:, :step_current_2hz].clone()
        pred_idx_window = tokenized_agent["gt_idx"][:, :step_current_2hz].clone()

        (
            feat_a,
            agent_token_emb,
            agent_token_emb_veh,
            agent_token_emb_ped,
            agent_token_emb_cyc,
            veh_mask,
            ped_mask,
            cyc_mask,
            categorical_embs,
        ) = self.agent_token_embedding(
            agent_token_index=pred_idx_window,
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_window,
            head_vector_a=head_vector_window,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        n_step = pos_window.shape[1]
        batch_s_a2a = self._build_step_offset_batch(
            batch=tokenized_agent["batch"],
            num_steps=n_step,
            num_graphs=tokenized_agent["num_graphs"],
        )
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_window,
            head_a=head_window,
            head_vector_a=head_vector_window,
            mask=valid_window,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_window,
            head_a=head_window,
            head_vector_a=head_vector_window,
            mask=valid_window,
            batch_s=batch_s_pl2a,
            batch_pl=map_feature["batch"],
        )
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_window,
            head_a=head_window,
            head_vector_a=head_vector_window,
            batch_s=batch_s_a2a,
            mask=valid_window,
        )

        feat_map = map_feature["pt_token"]
        feat_a_t_dict: Dict[int, torch.Tensor] = {}
        feat_a_now = feat_a[:, -1].clone()
        for i in range(self.num_layers):
            temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
            temporal_feat = self.t_attn_layers[i](
                temporal_feat.flatten(0, 1),
                r_t,
                edge_index_t,
            ).view(n_agent, n_step, -1)
            temporal_feat = temporal_feat.transpose(0, 1).flatten(0, 1)
            temporal_feat = self.pt2a_attn_layers[i]((feat_map, temporal_feat), r_pl2a, edge_index_pl2a)
            temporal_feat = self.a2a_attn_layers[i](temporal_feat, r_a2a, edge_index_a2a)
            temporal_feat = temporal_feat.view(n_step, n_agent, -1).transpose(0, 1)
            feat_a_now = temporal_feat[:, -1]
            if i + 1 < self.num_layers:
                feat_a_t_dict[i + 1] = temporal_feat

        return {
            "n_agent": n_agent,
            "n_step_future_10hz": n_step_future_10hz,
            "n_step_future_2hz": n_step_future_2hz,
            "max_context_steps": max_context_steps,
            "pos_window": pos_window,
            "head_window": head_window,
            "head_vector_window": head_vector_window,
            "valid_window": valid_window,
            "pred_idx_window": pred_idx_window,
            "feat_a": feat_a,
            "agent_token_emb": agent_token_emb,
            "agent_token_emb_veh": agent_token_emb_veh,
            "agent_token_emb_ped": agent_token_emb_ped,
            "agent_token_emb_cyc": agent_token_emb_cyc,
            "veh_mask": veh_mask,
            "ped_mask": ped_mask,
            "cyc_mask": cyc_mask,
            "categorical_embs": categorical_embs,
            "feat_a_now": feat_a_now,
            "feat_a_t_dict": feat_a_t_dict,
        }

    def _clone_rollout_cache(self, rollout_cache: Dict[str, object]) -> Dict[str, object]:
        """rollout마다 달라지는 상태만 안전하게 복사합니다.

        Args:
            rollout_cache: ``prepare_inference_cache`` 가 만든 원본 캐시입니다.

        Returns:
            Dict[str, object]:
                현재 rollout에서만 쓸 복사본입니다.
        """
        cloned_cache = dict(rollout_cache)
        for key in [
            "pos_window",
            "head_window",
            "head_vector_window",
            "valid_window",
            "pred_idx_window",
            "feat_a",
            "agent_token_emb",
            "feat_a_now",
        ]:
            value = rollout_cache[key]
            if torch.is_tensor(value):
                cloned_cache[key] = value.clone()
        feat_a_t_dict = rollout_cache["feat_a_t_dict"]
        if isinstance(feat_a_t_dict, dict):
            cloned_cache["feat_a_t_dict"] = {
                layer_idx: layer_value.clone()
                for layer_idx, layer_value in feat_a_t_dict.items()
            }
        return cloned_cache

    @torch.no_grad()
    def rollout_from_cache(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, torch.Tensor]:
        """공통 캐시를 복사해 한 번의 closed-loop rollout만 수행합니다.

        Args:
            rollout_cache: ``prepare_inference_cache`` 가 만든 원본 캐시입니다.
            tokenized_agent: 평가용 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징 사전입니다.
            sampling_scheme: 샘플링 설정입니다.

        Returns:
            Dict[str, torch.Tensor]:
                한 번의 rollout 결과입니다. 기존 inference 반환과 같은 키를 가집니다.
        """
        state = self._clone_rollout_cache(rollout_cache)

        n_agent = int(state["n_agent"])
        n_step_future_10hz = int(state["n_step_future_10hz"])
        n_step_future_2hz = int(state["n_step_future_2hz"])
        max_context_steps = int(state["max_context_steps"])
        pos_window = state["pos_window"]
        head_window = state["head_window"]
        head_vector_window = state["head_vector_window"]
        valid_window = state["valid_window"]
        pred_idx_window = state["pred_idx_window"]
        feat_a = state["feat_a"]
        agent_token_emb = state["agent_token_emb"]
        agent_token_emb_veh = state["agent_token_emb_veh"]
        agent_token_emb_ped = state["agent_token_emb_ped"]
        agent_token_emb_cyc = state["agent_token_emb_cyc"]
        veh_mask = state["veh_mask"]
        ped_mask = state["ped_mask"]
        cyc_mask = state["cyc_mask"]
        categorical_embs = state["categorical_embs"]
        feat_a_now = state["feat_a_now"]
        feat_a_t_dict = state["feat_a_t_dict"]

        coarse_pos_list = [pos_window[:, i].clone() for i in range(pos_window.shape[1])]
        coarse_head_list = [head_window[:, i].clone() for i in range(head_window.shape[1])]
        coarse_valid_list = [valid_window[:, i].clone() for i in range(valid_window.shape[1])]
        coarse_idx_list = [pred_idx_window[:, i].clone() for i in range(pred_idx_window.shape[1])]

        pred_traj_10hz = torch.zeros(
            (n_agent, n_step_future_10hz, 2),
            dtype=pos_window.dtype,
            device=pos_window.device,
        )
        pred_head_10hz = torch.zeros(
            (n_agent, n_step_future_10hz),
            dtype=head_window.dtype,
            device=head_window.device,
        )

        for t in range(n_step_future_2hz):
            n_step = pos_window.shape[1]
            if t == 0:
                current_hidden = feat_a_now
            else:
                inference_mask = valid_window.clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_window,
                    head_a=head_window,
                    head_vector_a=head_vector_window,
                    mask=valid_window,
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

                edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                    pos_pl=map_feature["position"],
                    orient_pl=map_feature["orientation"],
                    pos_a=pos_window[:, -1:],
                    head_a=head_window[:, -1:],
                    head_vector_a=head_vector_window[:, -1:],
                    mask=inference_mask[:, -1:],
                    batch_s=tokenized_agent["batch"],
                    batch_pl=map_feature["batch"],
                )
                edge_index_a2a, r_a2a = self.build_interaction_edge(
                    pos_a=pos_window[:, -1:],
                    head_a=head_window[:, -1:],
                    head_vector_a=head_vector_window[:, -1:],
                    batch_s=tokenized_agent["batch"],
                    mask=inference_mask[:, -1:],
                )

                for i in range(self.num_layers):
                    temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
                    current_hidden = self.t_attn_layers[i](
                        (temporal_feat.flatten(0, 1), temporal_feat[:, -1]),
                        r_t,
                        edge_index_t,
                    )
                    current_hidden = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], current_hidden),
                        r_pl2a,
                        edge_index_pl2a,
                    )
                    current_hidden = self.a2a_attn_layers[i](current_hidden, r_a2a, edge_index_a2a)
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            [feat_a_t_dict[i + 1], current_hidden.unsqueeze(1)],
                            dim=1,
                        )

            active_mask = valid_window[:, -1]
            next_pos = pos_window[:, -1].clone()
            next_head = head_window[:, -1].clone()
            next_token_idx = pred_idx_window[:, -1].clone()
            commit_traj_step = pred_traj_10hz.new_zeros((n_agent, 5, 2))
            commit_head_step = pred_head_10hz.new_zeros((n_agent, 5))

            if active_mask.any():
                active_hidden = current_hidden[active_mask]
                x_init_norm = torch.randn(
                    active_hidden.shape[0],
                    20,
                    4,
                    device=active_hidden.device,
                    dtype=active_hidden.dtype,
                ) * getattr(sampling_scheme, "noise_scale", 1.0)
                flow_sample_steps = getattr(
                    sampling_scheme,
                    "sample_steps",
                    self.flow_ode.solver_steps,
                )
                flow_sample_method = getattr(
                    sampling_scheme,
                    "sample_method",
                    self.flow_ode.solver_method,
                )
                y_hat_norm = self.flow_ode.generate(
                    x_init=x_init_norm,
                    model_fn=lambda x_t, tau: self.flow_decoder(active_hidden, x_t, tau),
                    steps=flow_sample_steps,
                    method=flow_sample_method,
                )
                commit_pos_act, commit_head_act, next_pos_act, next_head_act = self.commit_bridge.commit(
                    y_hat_norm=y_hat_norm,
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                )
                next_token_idx_act = self.commit_bridge.retokenize(
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                    commit_pos=commit_pos_act,
                    commit_head=commit_head_act,
                    agent_type=tokenized_agent["type"][active_mask],
                    token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                    token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                    token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                    token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                )
                commit_traj_step[active_mask] = commit_pos_act
                commit_head_step[active_mask] = commit_head_act
                next_pos[active_mask] = next_pos_act
                next_head[active_mask] = next_head_act
                next_token_idx[active_mask] = next_token_idx_act

            pred_traj_10hz[:, t * 5 : (t + 1) * 5] = commit_traj_step
            pred_head_10hz[:, t * 5 : (t + 1) * 5] = commit_head_step

            next_valid = active_mask.clone()
            coarse_pos_list.append(next_pos.clone())
            coarse_head_list.append(next_head.clone())
            coarse_valid_list.append(next_valid.clone())
            coarse_idx_list.append(next_token_idx.clone())

            pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)
            valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
            pos_window = torch.cat([pos_window, next_pos.unsqueeze(1)], dim=1)
            head_window = torch.cat([head_window, next_head.unsqueeze(1)], dim=1)
            head_vector_next = torch.stack([next_head.cos(), next_head.sin()], dim=-1)
            head_vector_window = torch.cat([head_vector_window, head_vector_next.unsqueeze(1)], dim=1)

            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
            agent_token_emb = torch.cat([agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1)

            motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_window[:, -1],
                        nbr_vector=motion_vector_a,
                    ),
                ],
                dim=-1,
            )
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            feat_a_next = self.fusion_emb(torch.cat([agent_token_emb_next, x_a], dim=-1).unsqueeze(1))
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

            if pos_window.shape[1] > max_context_steps:
                pos_window = pos_window[:, -max_context_steps:]
                head_window = head_window[:, -max_context_steps:]
                head_vector_window = head_vector_window[:, -max_context_steps:]
                valid_window = valid_window[:, -max_context_steps:]
                pred_idx_window = pred_idx_window[:, -max_context_steps:]
                agent_token_emb = agent_token_emb[:, -max_context_steps:]
                feat_a = feat_a[:, -max_context_steps:]
                for key in feat_a_t_dict:
                    feat_a_t_dict[key] = feat_a_t_dict[key][:, -max_context_steps:]

        pred_pos = torch.stack(coarse_pos_list, dim=1)
        pred_head = torch.stack(coarse_head_list, dim=1)
        pred_valid = torch.stack(coarse_valid_list, dim=1)
        pred_idx = torch.stack(coarse_idx_list, dim=1)
        out_dict = {
            "pred_pos": pred_pos,
            "pred_head": pred_head,
            "pred_valid": pred_valid,
            "pred_idx": pred_idx,
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],
            "gt_head_raw": tokenized_agent["gt_head_raw"],
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],
            "gt_pos": tokenized_agent["gt_pos"],
            "gt_head": tokenized_agent["gt_heading"],
            "gt_valid": tokenized_agent["valid_mask"],
            "pred_traj_10hz": pred_traj_10hz,
            "pred_head_10hz": pred_head_10hz,
        }
        pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)
        out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
        return out_dict

    @torch.no_grad()
    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, torch.Tensor]:
        rollout_cache = self.prepare_inference_cache(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        return self.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
        )
