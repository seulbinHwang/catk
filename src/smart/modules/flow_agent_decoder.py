from __future__ import annotations

from typing import Dict

import torch
from omegaconf import DictConfig
from torch_cluster import radius_graph
from torch_geometric.utils import subgraph

from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.dynamics_feasible_commit_bridge import (
    DynamicsAwareFeasibleCommitBridge,
)
from src.smart.modules.flow_local_decoder import (
    ContinuousCommitBridge,
    FlowODE,
    HierarchicalFlowDecoder,
)
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    validate_flow_window_steps,
    wrap_angle,
)


class SMARTFlowAgentDecoder(SMARTAgentEncoder):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        flow_window_steps: int,
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
        closed_loop_rollout_mode: str = "raw_fm",
        use_dynamics_feasible_commit_bridge: bool = False,
        use_stationary_refinement_in_dynamics_bridge: bool = False,
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
        self.flow_window_steps = validate_flow_window_steps(
            flow_window_steps=flow_window_steps,
            commit_steps=self.shift,
            num_future_steps=num_future_steps,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.flow_decoder = HierarchicalFlowDecoder(
            context_dim=hidden_dim,
            flow_dim=flow_dim,
            num_future_steps=self.flow_window_steps,
            num_chunk_heads=flow_num_chunk_heads,
            num_chunk_layers=flow_num_chunk_layers,
            chunk_size=self.shift,
            a2a_radius=a2a_radius,
        )
        self.flow_ode = FlowODE(
            eps=flow_solver_eps,
            solver_steps=flow_solver_steps,
            solver_method=flow_solver_method,
        )
        if closed_loop_rollout_mode not in {"raw_fm", "matched_token_chunk"}:
            raise ValueError(
                "closed_loop_rollout_mode must be one of {'raw_fm', 'matched_token_chunk'}, "
                f"got {closed_loop_rollout_mode!r}."
            )
        self.closed_loop_rollout_mode = closed_loop_rollout_mode
        self.use_dynamics_feasible_commit_bridge = bool(use_dynamics_feasible_commit_bridge)
        self.use_stationary_refinement_in_dynamics_bridge = bool(
            use_stationary_refinement_in_dynamics_bridge
        )
        self.commit_bridge = ContinuousCommitBridge(commit_steps=self.shift)
        self.dynamics_commit_bridge = (
            DynamicsAwareFeasibleCommitBridge(
                preview_steps=self.flow_window_steps,
                commit_steps=self.shift,
                use_stationary_refinement=self.use_stationary_refinement_in_dynamics_bridge
            )
            if self.use_dynamics_feasible_commit_bridge
            else None
        )

    def _get_map2agent_radius(self) -> float:
        """flow 생성 길이에 맞춰 지도 검색 반경을 계산합니다.

        ``pl2a_radius``는 2초 flow window의 기준 반경으로 둡니다.
        현재 모델은 10Hz 미래를 만들기 때문에 2초는 20 step입니다.
        따라서 ``flow_window_steps``가 20, 40, 60, 80이면 각각
        config 반경의 1, 2, 3, 4배를 씁니다. 기본 config가 30m일 때
        2초 30m, 4초 60m, 6초 90m, 8초 120m가 됩니다.

        Returns:
            float: 현재 flow window에 맞춘 map-to-agent 검색 반경입니다.
        """
        base_window_steps = 20
        return (
            float(self.pl2a_radius)
            * float(self.flow_window_steps)
            / float(base_window_steps)
        )

    def build_interaction_edge(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        batch_s: torch.Tensor,
        mask: torch.Tensor,
        motion_a: torch.Tensor | None = None,
    ):
        mask_flat = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        if motion_a is None:
            motion_a = torch.cat(
                [
                    pos_a.new_zeros(pos_a.shape[0], 1, pos_a.shape[-1]),
                    pos_a[:, 1:] - pos_a[:, :-1],
                ],
                dim=1,
            )
        else:
            if motion_a.shape != pos_a.shape:
                raise ValueError(
                    "motion_a shape must match pos_a shape, "
                    f"got {tuple(motion_a.shape)} and {tuple(pos_a.shape)}"
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

    def _build_recent_coarse_motion(
        self,
        pos_window: torch.Tensor,
        valid_window: torch.Tensor,
    ) -> torch.Tensor:
        """마지막 두 coarse 상태 차이로 최근 이동량을 만듭니다.

        Args:
            pos_window: 최근 coarse 중심점 창입니다.
                shape은 ``[n_agent, n_step, 2]`` 입니다.
            valid_window: 같은 창의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            torch.Tensor:
                각 agent의 최근 coarse 이동량입니다.
                shape은 ``[n_agent, 2]`` 입니다.
                마지막 두 상태가 모두 유효하지 않으면 0으로 둡니다.
        """
        recent_motion = pos_window.new_zeros((pos_window.shape[0], pos_window.shape[-1]))
        if pos_window.shape[1] < 2:
            return recent_motion

        recent_motion_valid = valid_window[:, -1] & valid_window[:, -2]
        recent_motion[recent_motion_valid] = (
            pos_window[recent_motion_valid, -1] - pos_window[recent_motion_valid, -2]
        )
        return recent_motion


    def _build_initial_exec_state_pair(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """closed-loop 첫 block에서 쓸 최근 fine 실행 상태 2개를 준비합니다.

        우선 10Hz 실제 history 마지막 두 점을 그대로 쓰고,
        그 정보가 없으면 현재 coarse 창의 마지막 두 상태를 fallback으로 씁니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - exec_pos_pair: 최근 fine 중심점 2개입니다.
                  shape은 ``[n_agent, 2, 2]`` 입니다.
                - exec_head_pair: 최근 fine 방향 2개입니다.
                  shape은 ``[n_agent, 2]`` 입니다.
                - exec_valid_pair: 최근 fine 상태 유효 여부입니다.
                  shape은 ``[n_agent, 2]`` 입니다.
        """
        if all(
            key in tokenized_agent
            for key in [
                "rollout_init_fine_pos_pair",
                "rollout_init_fine_head_pair",
                "rollout_init_fine_valid_pair",
            ]
        ):
            return (
                tokenized_agent["rollout_init_fine_pos_pair"].clone(),
                tokenized_agent["rollout_init_fine_head_pair"].clone(),
                tokenized_agent["rollout_init_fine_valid_pair"].clone(),
            )

        coarse_pos = tokenized_agent["gt_pos"]
        coarse_head = tokenized_agent["gt_heading"]
        coarse_valid = tokenized_agent["valid_mask"]
        if coarse_pos.shape[1] >= 2:
            return (
                coarse_pos[:, -2:].clone(),
                coarse_head[:, -2:].clone(),
                coarse_valid[:, -2:].clone(),
            )

        exec_pos_pair = torch.cat([coarse_pos[:, -1:], coarse_pos[:, -1:]], dim=1)
        exec_head_pair = torch.cat([coarse_head[:, -1:], coarse_head[:, -1:]], dim=1)
        exec_valid_pair = torch.cat([coarse_valid[:, -1:], coarse_valid[:, -1:]], dim=1)
        return exec_pos_pair, exec_head_pair, exec_valid_pair

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


    def _pack_anchor_sequence_tensor(
        self,
        anchor_tensor: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """anchor별 tensor를 flow target 순서와 같은 순서로 압축합니다.

        Args:
            anchor_tensor: anchor 축이 있는 tensor입니다.
                shape은 ``[n_agent, n_anchor, ...]`` 입니다.
            anchor_mask: 사용할 anchor 여부입니다. shape은 ``[n_agent, n_anchor]`` 입니다.

        Returns:
            torch.Tensor: 유효 anchor만 모은 tensor입니다.
                shape은 ``[n_valid_anchor, ...]`` 입니다.
        """
        if anchor_tensor.shape[:2] != anchor_mask.shape:
            raise ValueError(
                "anchor_tensor first two dims must match anchor_mask, "
                f"got {tuple(anchor_tensor.shape[:2])} and {tuple(anchor_mask.shape)}."
            )
        packed = [
            anchor_tensor[:, anchor_idx][anchor_mask[:, anchor_idx]]
            for anchor_idx in range(anchor_mask.shape[1])
            if anchor_mask[:, anchor_idx].any()
        ]
        if len(packed) == 0:
            return anchor_tensor.new_zeros((0, *anchor_tensor.shape[2:]))
        return torch.cat(packed, dim=0)

    def _pack_agent_tensor_by_anchor_mask(
        self,
        agent_tensor: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """agent별 tensor를 anchor mask 순서에 맞춰 반복 압축합니다.

        Args:
            agent_tensor: agent별 tensor입니다. shape은 ``[n_agent, ...]`` 입니다.
            anchor_mask: 사용할 anchor 여부입니다. shape은 ``[n_agent, n_anchor]`` 입니다.

        Returns:
            torch.Tensor: 유효 anchor 순서에 맞춘 tensor입니다.
                shape은 ``[n_valid_anchor, ...]`` 입니다.
        """
        if agent_tensor.shape[0] != anchor_mask.shape[0]:
            raise ValueError(
                "agent_tensor first dim must match anchor_mask agents, "
                f"got {agent_tensor.shape[0]} and {anchor_mask.shape[0]}."
            )
        packed = [
            agent_tensor[anchor_mask[:, anchor_idx]]
            for anchor_idx in range(anchor_mask.shape[1])
            if anchor_mask[:, anchor_idx].any()
        ]
        if len(packed) == 0:
            return agent_tensor.new_zeros((0, *agent_tensor.shape[1:]))
        return torch.cat(packed, dim=0)

    def _pack_anchor_step_id(self, anchor_mask: torch.Tensor) -> torch.Tensor:
        """각 packed 후보가 어느 anchor 시점에서 나왔는지 번호를 만듭니다.

        Args:
            anchor_mask: 사용할 anchor 여부입니다. shape은 ``[n_agent, n_anchor]`` 입니다.

        Returns:
            torch.Tensor: packed 후보별 anchor 시점 번호입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
        """
        packed_step_ids = [
            torch.full(
                (int(anchor_mask[:, anchor_idx].sum().item()),),
                anchor_idx,
                device=anchor_mask.device,
                dtype=torch.long,
            )
            for anchor_idx in range(anchor_mask.shape[1])
            if anchor_mask[:, anchor_idx].any()
        ]
        if len(packed_step_ids) == 0:
            return torch.zeros((0,), device=anchor_mask.device, dtype=torch.long)
        return torch.cat(packed_step_ids, dim=0)

    def _pack_flow_decoder_context(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        anchor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """flow decoder의 agent-to-agent chunk attention에 필요한 정보를 압축합니다.

        Args:
            tokenized_agent: 에이전트 토큰 사전입니다.
            anchor_mask: 사용할 anchor 여부입니다. shape은 ``[n_agent, n_anchor]`` 입니다.

        Returns:
            Dict[str, torch.Tensor]: 현재 위치, 방향, 장면 번호, anchor 시점 번호입니다.
                각 tensor의 첫 차원은 ``n_valid_anchor`` 입니다.
        """
        n_anchor = anchor_mask.shape[1]
        anchor_pos = tokenized_agent["ctx_sampled_pos"][:, 1 : 1 + n_anchor]
        anchor_head = tokenized_agent["ctx_sampled_heading"][:, 1 : 1 + n_anchor]
        if anchor_pos.shape[1] != n_anchor or anchor_head.shape[1] != n_anchor:
            raise ValueError(
                "ctx_sampled_pos/heading must contain one context slot plus all flow anchors."
            )
        return {
            "current_pos": self._pack_anchor_sequence_tensor(anchor_pos, anchor_mask),
            "current_head": self._pack_anchor_sequence_tensor(anchor_head, anchor_mask),
            "agent_batch": self._pack_agent_tensor_by_anchor_mask(
                tokenized_agent["batch"],
                anchor_mask,
            ).long(),
            "anchor_step_id": self._pack_anchor_step_id(anchor_mask),
        }

    def _sample_open_loop_future_from_hidden(
        self,
        anchor_hidden_valid: torch.Tensor,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        backprop_last_k: int | None = None,
        current_pos: torch.Tensor | None = None,
        current_head: torch.Tensor | None = None,
        agent_batch: torch.Tensor | None = None,
        anchor_step_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """유효 anchor 문맥만 받아 실제 생성 경로로 2초 미래를 만듭니다.

        Args:
            anchor_hidden_valid: 유효 anchor만 모은 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: validation마다 같은 출발 잡음을 만들기 위한 seed입니다.
            backprop_last_k: 마지막 몇 step만 역전파할지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다.
            current_pos: 유효 anchor별 현재 중심 위치입니다. shape은 ``[n_valid_anchor, 2]`` 입니다.
            current_head: 유효 anchor별 현재 방향입니다. shape은 ``[n_valid_anchor]`` 입니다.
            agent_batch: 유효 anchor별 장면 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            anchor_step_id: 유효 anchor별 anchor 시점 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            torch.Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            return anchor_hidden_valid.new_zeros((0, self.flow_window_steps, 4))

        generator = None
        if sampling_seed is not None:
            generator = torch.Generator(device=anchor_hidden_valid.device)
            generator.manual_seed(int(sampling_seed))

        x_init_norm = torch.randn(
            anchor_hidden_valid.shape[0],
            self.flow_window_steps,
            4,
            device=anchor_hidden_valid.device,
            dtype=anchor_hidden_valid.dtype,
            generator=generator,
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
        if backprop_last_k is None:
            backprop_last_k = getattr(sampling_scheme, "backprop_last_k", None)

        return self.flow_ode.generate(
            x_init=x_init_norm,
            model_fn=lambda x_t, tau: self.flow_decoder(
                anchor_hidden_valid,
                x_t,
                tau,
                current_pos=current_pos,
                current_head=current_head,
                agent_batch=agent_batch,
                anchor_step_id=anchor_step_id,
            ),
            steps=flow_sample_steps,
            method=flow_sample_method,
            backprop_last_k=backprop_last_k,
        )

    def sample_open_loop_future(
        self,
        anchor_hidden: torch.Tensor,
        anchor_mask: torch.Tensor,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        backprop_last_k: int | None = None,
        current_pos: torch.Tensor | None = None,
        current_head: torch.Tensor | None = None,
        agent_batch: torch.Tensor | None = None,
        anchor_step_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """모든 anchor 문맥에서 유효한 것만 골라 실제 생성 경로를 수행합니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, 13]`` 입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: validation마다 같은 출발 잡음을 만들기 위한 seed입니다.
            backprop_last_k: 마지막 몇 step만 역전파할지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다.
            current_pos: 유효 anchor별 현재 중심 위치입니다. shape은 ``[n_valid_anchor, 2]`` 입니다.
            current_head: 유효 anchor별 현재 방향입니다. shape은 ``[n_valid_anchor]`` 입니다.
            agent_batch: 유효 anchor별 장면 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.
            anchor_step_id: 유효 anchor별 anchor 시점 번호입니다. shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            torch.Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        anchor_hidden_valid = self._pack_anchor_hidden(anchor_hidden, anchor_mask)
        return self._sample_open_loop_future_from_hidden(
            anchor_hidden_valid=anchor_hidden_valid,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            backprop_last_k=backprop_last_k,
            current_pos=current_pos,
            current_head=current_head,
            agent_batch=agent_batch,
            anchor_step_id=anchor_step_id,
        )

    def _build_rollout_noise_tape(
        self,
        num_agent: int,
        tape_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: torch.Tensor | None = None,
        agent_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """closed-loop 전체에서 재사용할 긴 잡음 테이프를 한 번만 만듭니다.

        Args:
            num_agent: 현재 batch 안 전체 agent 수입니다.
            tape_steps: 긴 잡음 테이프의 시간 길이입니다.
            device: 잡음 테이프를 만들 장치입니다.
            dtype: 잡음 테이프 자료형입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: batch 전체를 하나의 seed로 만들 때 쓰는 seed입니다.
            scenario_sampling_seeds: 시나리오별 고정 seed입니다.
                shape은 ``[n_scenario]`` 입니다.
            agent_batch: 각 agent가 어느 시나리오에 속하는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            torch.Tensor:
                각 agent가 rollout 전체에서 공유할 긴 Gaussian 잡음입니다.
                shape은 ``[n_agent, tape_steps, 4]`` 입니다.
        """
        noise_scale = float(getattr(sampling_scheme, "noise_scale", 1.0))
        if num_agent == 0:
            return torch.zeros((0, tape_steps, 4), device=device, dtype=dtype)

        if scenario_sampling_seeds is not None:
            if agent_batch is None:
                raise ValueError("scenario별 잡음 테이프를 만들려면 agent_batch가 필요합니다.")
            noise_tape = torch.empty((num_agent, tape_steps, 4), device=device, dtype=dtype)
            scenario_seed_list = scenario_sampling_seeds.detach().cpu().tolist()
            for scenario_idx, scenario_seed in enumerate(scenario_seed_list):
                scenario_mask = agent_batch == scenario_idx
                if not bool(scenario_mask.any()):
                    continue
                generator = torch.Generator(device=device)
                generator.manual_seed(int(scenario_seed))
                noise_tape[scenario_mask] = torch.randn(
                    int(scenario_mask.sum().item()),
                    tape_steps,
                    4,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            return noise_tape * noise_scale

        generator = None
        if sampling_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(sampling_seed))
        return torch.randn(
            num_agent,
            tape_steps,
            4,
            device=device,
            dtype=dtype,
            generator=generator,
        ) * noise_scale

    def _encode_context(
        self,
        agent_token_index: torch.Tensor,
        pos_a: torch.Tensor, # ctx_sampled_pos
        head_a: torch.Tensor, # ctx_sampled_heading
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
            pos_a=pos_a, # ctx_sampled_pos
            head_vector_a=head_vector_a, # ctx_sampled_heading
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
        )

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a, # ctx_sampled_pos
            head_a=head_a, # ctx_sampled_heading
            head_vector_a=head_vector_a, # ctx_sampled_heading
            mask=mask,
        )
        batch_s_a2a = self._build_step_offset_batch(
            batch=tokenized_agent["batch"],
            num_steps=n_step,
            num_graphs=tokenized_agent["num_graphs"],
        )
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a, # ctx_sampled_pos
            head_a=head_a, # ctx_sampled_heading
            head_vector_a=head_vector_a, # ctx_sampled_heading
            batch_s=batch_s_a2a,
            mask=mask,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_a, # ctx_sampled_pos
            head_a=head_a,  # ctx_sampled_heading
            head_vector_a=head_vector_a, # ctx_sampled_heading
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
        flow_loss_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """학습 또는 평가용 anchor를 골라 flow decoder 출력을 만듭니다.

        Args:
            tokenized_agent: agent 토큰 사전입니다.
            map_feature: map encoder가 만든 지도 특징 사전입니다.
            anchor_mask: 사용할 anchor 표시입니다. shape은 ``[n_agent, n_anchor]`` 입니다.
            flow_clean_norm: 정답 미래입니다.
                shape은 ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
            flow_loss_mask: loss에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 step을 사용합니다.

        Returns:
            Dict[str, torch.Tensor]:
                flow prediction, target, anchor 문맥, 현재 위치/방향, batch 정보를 담은 사전입니다.
        """
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
        flow_decoder_context = self._pack_flow_decoder_context(
            tokenized_agent=tokenized_agent,
            anchor_mask=anchor_mask,
        )

        if flow_loss_mask is not None:
            expected_shape = tuple(flow_clean_norm.shape[:2])
            if tuple(flow_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "flow_loss_mask shape must match flow_clean_norm first two dimensions: "
                    f"expected={expected_shape}, actual={tuple(flow_loss_mask.shape)}."
                )
            flow_loss_mask = flow_loss_mask.to(device=flow_clean_norm.device, dtype=torch.bool)

        if flow_clean_norm.numel() == 0:
            empty = flow_clean_norm.new_zeros((0, self.flow_window_steps, 4))
            output = {
                "flow_pred_norm": empty,
                "flow_target_norm": empty,
                "flow_pred_clean_norm": empty,
                "flow_clean_norm": empty,
                "ctx_hidden_pack": ctx_hidden_pack,
                "anchor_hidden": anchor_hidden,
                "anchor_mask": anchor_mask,
                "flow_current_pos": flow_decoder_context["current_pos"],
                "flow_current_head": flow_decoder_context["current_head"],
                "flow_agent_batch": flow_decoder_context["agent_batch"],
                "flow_anchor_step_id": flow_decoder_context["anchor_step_id"],
            }
            if flow_loss_mask is not None:
                output["flow_loss_mask"] = flow_loss_mask
            return output

        flow_sample = self.flow_ode.sample(flow_clean_norm, target_type="velocity")
        flow_pred_norm = self.flow_decoder(
            anchor_hidden_valid,
            flow_sample.x_t,
            flow_sample.tau,
            current_pos=flow_decoder_context["current_pos"],
            current_head=flow_decoder_context["current_head"],
            agent_batch=flow_decoder_context["agent_batch"],
            anchor_step_id=flow_decoder_context["anchor_step_id"],
        )
        flow_pred_clean_norm = self.flow_ode.predict_clean_from_velocity(
            flow_sample.x_t,
            flow_pred_norm,
            flow_sample.tau,
        )
        output = {
            "flow_pred_norm": flow_pred_norm,
            "flow_target_norm": flow_sample.target,
            "flow_pred_clean_norm": flow_pred_clean_norm,
            "flow_clean_norm": flow_clean_norm,
            "ctx_hidden_pack": ctx_hidden_pack,
            "anchor_hidden": anchor_hidden,
            "anchor_mask": anchor_mask,
            "flow_current_pos": flow_decoder_context["current_pos"],
            "flow_current_head": flow_decoder_context["current_head"],
            "flow_agent_batch": flow_decoder_context["agent_batch"],
            "flow_anchor_step_id": flow_decoder_context["anchor_step_id"],
        }
        if flow_loss_mask is not None:
            output["flow_loss_mask"] = flow_loss_mask
        return output

    def _prepare_rollout_cache_impl(
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
        exec_pos_pair_10hz, exec_head_pair_10hz, exec_valid_pair_10hz = (
            self._build_initial_exec_state_pair(tokenized_agent=tokenized_agent)
        )

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
            "exec_pos_pair_10hz": exec_pos_pair_10hz,
            "exec_head_pair_10hz": exec_head_pair_10hz,
            "exec_valid_pair_10hz": exec_valid_pair_10hz,
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

    @torch.no_grad()
    def prepare_inference_cache(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """평가와 제출에서 쓸 no-gradient rollout cache를 만듭니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다. agent 축 shape은 ``[n_agent, ...]`` 입니다.
            map_feature: 지도 인코더 출력입니다.

        Returns:
            Dict[str, object]: closed-loop rollout의 초기 상태 cache입니다.
        """
        return self._prepare_rollout_cache_impl(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )

    def prepare_training_rollout_cache(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """self-forced 학습에서 gradient를 유지한 rollout cache를 만듭니다.

        Args:
            tokenized_agent: 평가 모드 기준 토큰 사전입니다. agent 축 shape은 ``[n_agent, ...]`` 입니다.
            map_feature: 현재 Generator의 지도 인코더 출력입니다.

        Returns:
            Dict[str, object]: N초 self-rollout에 쓸 초기 cache입니다.
        """
        return self._prepare_rollout_cache_impl(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )

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
            "exec_pos_pair_10hz",
            "exec_head_pair_10hz",
            "exec_valid_pair_10hz",
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

    def _rollout_from_cache_impl(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: torch.Tensor | None = None,
        return_flow_2s_preview: bool = False,
        rollout_steps_2hz: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """공통 캐시를 복사해 한 번의 closed-loop rollout만 수행합니다.

        Args:
            rollout_cache: ``prepare_inference_cache`` 가 만든 원본 캐시입니다.
            tokenized_agent: 평가용 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징 사전입니다.
            sampling_scheme: 샘플링 설정입니다.
            sampling_seed: batch 전체를 하나의 seed로 만들 때 쓰는 고정 난수 seed입니다.
            scenario_sampling_seeds: 시나리오별 고정 seed입니다.
                shape은 ``[n_scenario]`` 입니다.

        Returns:
            Dict[str, torch.Tensor]:
                한 번의 rollout 결과입니다. 기존 inference 반환과 같은 키를 가집니다.
                ``return_flow_2s_preview=True`` 이면 step별 raw 2초 preview도
                함께 반환합니다.
        """
        state = self._clone_rollout_cache(rollout_cache)

        n_agent = int(state["n_agent"])
        total_step_future_2hz = int(state["n_step_future_2hz"])
        if rollout_steps_2hz is None:
            n_step_future_2hz = total_step_future_2hz
        else:
            n_step_future_2hz = int(rollout_steps_2hz)
            if n_step_future_2hz <= 0:
                raise ValueError("rollout_steps_2hz must be positive.")
            if n_step_future_2hz > total_step_future_2hz:
                raise ValueError(
                    "rollout_steps_2hz cannot exceed the full rollout length: "
                    f"got {n_step_future_2hz} and {total_step_future_2hz}."
                )
        n_step_future_10hz = n_step_future_2hz * self.shift
        max_context_steps = int(state["max_context_steps"])
        pos_window = state["pos_window"]
        head_window = state["head_window"]
        head_vector_window = state["head_vector_window"]
        valid_window = state["valid_window"]
        pred_idx_window = state["pred_idx_window"]
        exec_pos_pair_10hz = state["exec_pos_pair_10hz"]
        exec_head_pair_10hz = state["exec_head_pair_10hz"]
        exec_valid_pair_10hz = state["exec_valid_pair_10hz"]
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
        pred_flow_2s_traj = None
        pred_flow_2s_valid = None
        if return_flow_2s_preview:
            pred_flow_2s_traj = torch.zeros(
                (n_agent, n_step_future_2hz, self.flow_window_steps, 2),
                dtype=pos_window.dtype,
                device=pos_window.device,
            )
            pred_flow_2s_valid = torch.zeros(
                (n_agent, n_step_future_2hz),
                dtype=torch.bool,
                device=pos_window.device,
            )
        sample_window_steps = self.flow_window_steps
        rollout_noise_tape = self._build_rollout_noise_tape(
            num_agent=n_agent,
            tape_steps=n_step_future_10hz + sample_window_steps - self.shift,
            device=feat_a_now.device,
            dtype=feat_a_now.dtype,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            agent_batch=tokenized_agent["batch"],
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
                recent_motion = self._build_recent_coarse_motion(
                    pos_window=pos_window,
                    valid_window=valid_window,
                )
                edge_index_a2a, r_a2a = self.build_interaction_edge(
                    pos_a=pos_window[:, -1:],
                    head_a=head_window[:, -1:],
                    head_vector_a=head_vector_window[:, -1:],
                    batch_s=tokenized_agent["batch"],
                    mask=inference_mask[:, -1:],
                    motion_a=recent_motion.unsqueeze(1),
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
            commit_traj_step = pred_traj_10hz.new_zeros((n_agent, self.shift, 2))
            commit_head_step = pred_head_10hz.new_zeros((n_agent, self.shift))

            if active_mask.any():
                active_hidden = current_hidden[active_mask]
                noise_start = t * self.shift
                x_init_norm = rollout_noise_tape[
                    active_mask,
                    noise_start : noise_start + sample_window_steps,
                ].contiguous()
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
                current_pos_act = pos_window[active_mask, -1]
                current_head_act = head_window[active_mask, -1]
                active_agent_batch = tokenized_agent["batch"][active_mask]
                y_hat_norm = self.flow_ode.generate(
                    x_init=x_init_norm,
                    model_fn=lambda x_t, tau: self.flow_decoder(
                        active_hidden,
                        x_t,
                        tau,
                        current_pos=current_pos_act,
                        current_head=current_head_act,
                        agent_batch=active_agent_batch,
                    ),
                    steps=flow_sample_steps,
                    method=flow_sample_method,
                )
                active_agent_type = tokenized_agent["type"][active_mask]
                if return_flow_2s_preview:
                    preview_pos_local = y_hat_norm[..., :2] * 20.0
                    preview_pos_global, _ = transform_to_global(
                        pos_local=preview_pos_local,
                        head_local=None,
                        pos_now=current_pos_act,
                        head_now=current_head_act,
                    )
                    pred_flow_2s_traj[active_mask, t] = preview_pos_global
                    pred_flow_2s_valid[active_mask, t] = True
                if self.use_dynamics_feasible_commit_bridge and self.dynamics_commit_bridge is not None:
                    (
                        commit_pos_act,
                        commit_head_act,
                        next_pos_act,
                        next_head_act,
                    ) = self.dynamics_commit_bridge.commit(
                        y_hat_norm=y_hat_norm,
                        current_pos=current_pos_act,
                        current_head=current_head_act,
                        agent_type=active_agent_type,
                        agent_shape=tokenized_agent["shape"][active_mask],
                        exec_pos_pair=exec_pos_pair_10hz[active_mask],
                        exec_head_pair=exec_head_pair_10hz[active_mask],
                        exec_valid_pair=exec_valid_pair_10hz[active_mask],
                    )
                else:
                    (
                        commit_pos_act,
                        commit_head_act,
                        next_pos_act,
                        next_head_act,
                    ) = self.commit_bridge.commit(
                        y_hat_norm=y_hat_norm,
                        current_pos=current_pos_act,
                        current_head=current_head_act,
                    )
                next_token_idx_act = self.commit_bridge.retokenize(
                    current_pos=current_pos_act,
                    current_head=current_head_act,
                    commit_pos=commit_pos_act,
                    commit_head=commit_head_act,
                    agent_type=active_agent_type,
                    token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                    token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                    token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                    token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                )
                commit_pos_export_act = commit_pos_act
                commit_head_export_act = commit_head_act
                if self.closed_loop_rollout_mode == "matched_token_chunk":
                    if self.use_dynamics_feasible_commit_bridge:
                        ped_active_mask = active_agent_type == 1
                        if ped_active_mask.any():
                            (
                                ped_commit_pos_export,
                                ped_commit_head_export,
                                _,
                                _,
                            ) = self.commit_bridge.restore_token_chunk(
                                current_pos=current_pos_act[ped_active_mask],
                                current_head=current_head_act[ped_active_mask],
                                next_token_idx=next_token_idx_act[ped_active_mask],
                                agent_type=active_agent_type[ped_active_mask],
                                token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                                token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                                token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                            )
                            commit_pos_export_act[ped_active_mask] = ped_commit_pos_export
                            commit_head_export_act[ped_active_mask] = ped_commit_head_export
                    else:
                        (
                            commit_pos_export_act,
                            commit_head_export_act,
                            _,
                            _,
                        ) = self.commit_bridge.restore_token_chunk(
                            current_pos=current_pos_act,
                            current_head=current_head_act,
                            next_token_idx=next_token_idx_act,
                            agent_type=active_agent_type,
                            token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                            token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                            token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                        )
                commit_traj_step[active_mask] = commit_pos_export_act
                commit_head_step[active_mask] = commit_head_export_act
                next_pos[active_mask] = next_pos_act
                next_head[active_mask] = next_head_act
                next_token_idx[active_mask] = next_token_idx_act
                exec_pos_pair_10hz[active_mask, 0] = commit_pos_act[:, -2]
                exec_pos_pair_10hz[active_mask, 1] = commit_pos_act[:, -1]
                exec_head_pair_10hz[active_mask, 0] = commit_head_act[:, -2]
                exec_head_pair_10hz[active_mask, 1] = commit_head_act[:, -1]
                exec_valid_pair_10hz[active_mask] = True

            pred_traj_10hz[:, t * self.shift : (t + 1) * self.shift] = commit_traj_step
            pred_head_10hz[:, t * self.shift : (t + 1) * self.shift] = commit_head_step

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
        if return_flow_2s_preview:
            out_dict["pred_flow_preview_traj"] = pred_flow_2s_traj
            out_dict["pred_flow_preview_valid"] = pred_flow_2s_valid
            out_dict["pred_flow_2s_traj"] = pred_flow_2s_traj
            out_dict["pred_flow_2s_valid"] = pred_flow_2s_valid
        return out_dict

    @torch.no_grad()
    def rollout_from_cache(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: torch.Tensor | None = None,
        return_flow_2s_preview: bool = False,
        rollout_steps_2hz: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """평가와 제출에서 no-gradient closed-loop rollout을 실행합니다.

        Args:
            rollout_cache: ``prepare_inference_cache`` 가 만든 초기 상태입니다.
            tokenized_agent: 평가용 토큰 사전입니다.
            map_feature: 지도 인코더 출력입니다.
            sampling_scheme: flow sampling 설정입니다.
            sampling_seed: batch 공통 seed입니다.
            scenario_sampling_seeds: scenario별 seed입니다. shape은 ``[n_scenario]`` 입니다.
            return_flow_2s_preview: preview 저장 여부입니다.
            rollout_steps_2hz: 실행할 0.5초 block 수입니다. ``None`` 이면 전체 8초를 실행합니다.

        Returns:
            Dict[str, torch.Tensor]: closed-loop rollout 결과입니다.
        """
        return self._rollout_from_cache_impl(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            return_flow_2s_preview=return_flow_2s_preview,
            rollout_steps_2hz=rollout_steps_2hz,
        )

    def training_rollout_from_cache(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: torch.Tensor | None = None,
        rollout_steps_2hz: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """self-forced 학습에서 gradient를 유지한 closed-loop rollout을 실행합니다.

        Args:
            rollout_cache: ``prepare_training_rollout_cache`` 가 만든 초기 상태입니다.
            tokenized_agent: 평가 모드 기준 토큰 사전입니다.
            map_feature: 현재 Generator의 지도 인코더 출력입니다.
            sampling_scheme: flow sampling 설정입니다.
            sampling_seed: batch 공통 seed입니다.
            scenario_sampling_seeds: scenario별 seed입니다. shape은 ``[n_scenario]`` 입니다.
            rollout_steps_2hz: 실행할 0.5초 block 수입니다. 기본 self-forced 학습은
                ``flow_window_steps / 5`` 를 넘깁니다.

        Returns:
            Dict[str, torch.Tensor]: N초 committed self-rollout 결과입니다.
        """
        return self._rollout_from_cache_impl(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            return_flow_2s_preview=False,
            rollout_steps_2hz=rollout_steps_2hz,
        )

    def path_flow_velocity_for_anchor0(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        path_noisy_norm: torch.Tensor,
        tau: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """첫 flow anchor의 noisy path에 대한 flow velocity를 예측합니다.

        Args:
            tokenized_agent: 평가 모드 기준 토큰 사전입니다.
            map_feature: 이 decoder가 직접 만든 지도 특징입니다.
            path_noisy_norm: noisy N초 path입니다. shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
            tau: flow interpolation time입니다. shape은 ``[n_valid_agent]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent 마스크입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, torch.Tensor]: ``velocity`` 와 ``clean`` 을 담은 사전입니다. 두 텐서 shape은
            ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        """
        if path_noisy_norm.numel() == 0:
            empty = path_noisy_norm.new_zeros((0, self.flow_window_steps, 4))
            return {"velocity": empty, "clean": empty}
        if path_noisy_norm.shape[1:] != (self.flow_window_steps, 4):
            raise ValueError(
                "path_noisy_norm must have shape [n_valid_agent, flow_window_steps, 4], "
                f"got {tuple(path_noisy_norm.shape)}."
            )
        if int(anchor_mask.sum().item()) != int(path_noisy_norm.shape[0]):
            raise ValueError(
                "anchor_mask true count must match path_noisy_norm first dim, "
                f"got {int(anchor_mask.sum().item())} and {path_noisy_norm.shape[0]}."
            )

        single_anchor_mask = torch.zeros(
            anchor_mask.shape[0],
            13,
            device=anchor_mask.device,
            dtype=torch.bool,
        )
        single_anchor_mask[:, 0] = anchor_mask.bool()
        ctx_hidden_pack = self._encode_context(
            agent_token_index=tokenized_agent["ctx_sampled_idx"],
            pos_a=tokenized_agent["ctx_sampled_pos"],
            head_a=tokenized_agent["ctx_sampled_heading"],
            mask=tokenized_agent["ctx_valid"],
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        anchor_hidden = ctx_hidden_pack[:, 1:, :]
        anchor_hidden_valid = self._pack_anchor_hidden(anchor_hidden, single_anchor_mask)
        flow_decoder_context = self._pack_flow_decoder_context(
            tokenized_agent=tokenized_agent,
            anchor_mask=single_anchor_mask,
        )
        velocity = self.flow_decoder(
            anchor_hidden_valid,
            path_noisy_norm,
            tau,
            current_pos=flow_decoder_context["current_pos"],
            current_head=flow_decoder_context["current_head"],
            agent_batch=flow_decoder_context["agent_batch"],
            anchor_step_id=flow_decoder_context["anchor_step_id"],
        )
        clean = self.flow_ode.predict_clean_from_velocity(path_noisy_norm, velocity, tau)
        return {"velocity": velocity, "clean": clean}

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
