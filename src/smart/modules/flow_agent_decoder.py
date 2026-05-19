from __future__ import annotations

import os
from typing import Dict

import torch
from omegaconf import DictConfig
from torch_cluster import radius_graph
from torch_geometric.utils import subgraph

from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.layers.graph_flash_attention import build_graph_attention_metadata
from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.dynamic_light_time import (
    build_constant_light_time_delta_norm,
    normalize_light_time_delta_seconds,
)
from src.smart.modules.flow_local_decoder import (
    ContinuousCommitBridge,
    FlowODE,
    HierarchicalFlowDecoder,
    LQRCommitBridgeConfig,
)
from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    POSE_FLOW_DIM,
    control_norm_to_pose_norm,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)
from src.smart.modules.self_forced_rollout_detach import (
    detach_training_rollout_state,
)
from src.smart.utils import (
    angle_between_2d_vectors,
    safe_norm_2d,
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
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        control_pos_scale_m: float = 1.0,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
        control_vehicle_no_slip_point_ratio: float = 0.0,
        control_cyclist_no_slip_point_ratio: float = 0.0,
        closed_loop_rollout_mode: str = "raw_fm",
        use_lqr: bool = False,
        use_stop_motion: bool = False,
        lqr_commit: DictConfig | None = None,
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
        self.use_kinematic_control_flow = bool(use_kinematic_control_flow)
        self.use_holonomic_model_only = bool(use_holonomic_model_only)
        self.control_pos_scale_m = float(control_pos_scale_m)
        self.control_vehicle_yaw_scale_rad = control_vehicle_yaw_scale_rad
        self.control_pedestrian_yaw_scale_rad = control_pedestrian_yaw_scale_rad
        self.control_cyclist_yaw_scale_rad = control_cyclist_yaw_scale_rad
        (
            self.control_vehicle_no_slip_point_ratio,
            self.control_cyclist_no_slip_point_ratio,
        ) = validate_control_no_slip_ratio_config(
            vehicle_no_slip_point_ratio=control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=control_cyclist_no_slip_point_ratio,
        )
        if self.use_kinematic_control_flow:
            (
                self.control_vehicle_yaw_scale_rad,
                self.control_pedestrian_yaw_scale_rad,
                self.control_cyclist_yaw_scale_rad,
            ) = validate_control_yaw_scale_config(
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            )
        self.flow_state_dim = CONTROL_FLOW_DIM if self.use_kinematic_control_flow else POSE_FLOW_DIM
        self.r_a2a_emb = FourierEmbedding(
            input_dim=6,
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
            flow_state_dim=self.flow_state_dim,
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
        self.use_lqr = bool(use_lqr)
        self.use_stop_motion = bool(use_stop_motion)
        lqr_commit_cfg = LQRCommitBridgeConfig(
            dt=float(getattr(lqr_commit, "dt", 0.1)) if lqr_commit is not None else 0.1,
            history_steps=int(getattr(lqr_commit, "history_steps", 6)) if lqr_commit is not None else 6,
            horizon_steps=int(getattr(lqr_commit, "horizon_steps", 10)) if lqr_commit is not None else 10,
            velocity_smooth_lambda=float(getattr(lqr_commit, "velocity_smooth_lambda", 1.0e-4)) if lqr_commit is not None else 1.0e-4,
            curvature_smooth_lambda=float(getattr(lqr_commit, "curvature_smooth_lambda", 1.0e-2)) if lqr_commit is not None else 1.0e-2,
            curvature_init_reg=float(getattr(lqr_commit, "curvature_init_reg", 1.0e-10)) if lqr_commit is not None else 1.0e-10,
            stop_speed_mps=float(getattr(lqr_commit, "stop_speed_mps", 0.2)) if lqr_commit is not None else 0.2,
            stop_speed_kp=float(getattr(lqr_commit, "stop_speed_kp", 0.5)) if lqr_commit is not None else 0.5,
            longitudinal_q=float(getattr(lqr_commit, "longitudinal_q", 10.0)) if lqr_commit is not None else 10.0,
            longitudinal_r=float(getattr(lqr_commit, "longitudinal_r", 1.0)) if lqr_commit is not None else 1.0,
            lateral_q_lat=float(getattr(lqr_commit, "lateral_q_lat", 1.0)) if lqr_commit is not None else 1.0,
            lateral_q_head=float(getattr(lqr_commit, "lateral_q_head", 10.0)) if lqr_commit is not None else 10.0,
            lateral_q_kappa=float(getattr(lqr_commit, "lateral_q_kappa", 0.1)) if lqr_commit is not None else 0.1,
            lateral_r=float(getattr(lqr_commit, "lateral_r", 1.0)) if lqr_commit is not None else 1.0,
            accel_tau_s=float(getattr(lqr_commit, "accel_tau_s", 0.2)) if lqr_commit is not None else 0.2,
            curvature_tau_s=float(getattr(lqr_commit, "curvature_tau_s", 0.05)) if lqr_commit is not None else 0.05,
            min_speed_for_curvature_clip_mps=float(getattr(lqr_commit, "min_speed_for_curvature_clip_mps", 0.5)) if lqr_commit is not None else 0.5,
        )
        self.commit_bridge = ContinuousCommitBridge(
            commit_steps=self.shift,
            use_lqr=self.use_lqr,
            use_stop_motion=self.use_stop_motion,
            config=lqr_commit_cfg,
            use_kinematic_control_flow=self.use_kinematic_control_flow,
            use_holonomic_model_only=self.use_holonomic_model_only,
            control_pos_scale_m=self.control_pos_scale_m,
            control_vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
            control_pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
            control_cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            control_vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
            control_cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
        )

    def build_interaction_edge(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        batch_s: torch.Tensor,
        mask: torch.Tensor,
        motion_a: torch.Tensor | None = None,
        motion_valid_a: torch.Tensor | None = None,
    ):
        mask_flat = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        if motion_a is None:
            motion_a = self._build_motion_vector(pos_a, mask)
            motion_valid_a = self._build_motion_valid_mask(pos_a, mask)
        else:
            if motion_a.shape != pos_a.shape:
                raise ValueError(
                    "motion_a shape must match pos_a shape, "
                    f"got {tuple(motion_a.shape)} and {tuple(pos_a.shape)}"
                )
            if motion_valid_a is None:
                raise ValueError(
                    "motion_valid_a is required when motion_a is provided. "
                    "Missing motion must not be treated as valid zero motion."
                )
            if tuple(motion_valid_a.shape) != tuple(motion_a.shape[:2]):
                raise ValueError(
                    "motion_valid_a shape must match the first two dimensions of motion_a, "
                    f"got {tuple(motion_valid_a.shape)} and {tuple(motion_a.shape[:2])}"
                )
        motion_valid_a = motion_valid_a.bool()
        motion_s = motion_a.transpose(0, 1).reshape(-1, 2)
        motion_valid_s = motion_valid_a.transpose(0, 1).reshape(-1)

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
        rel_motion_valid = (
            motion_valid_s[edge_index_a2a[0]]
            & motion_valid_s[edge_index_a2a[1]]
        )
        recv_head = head_s[edge_index_a2a[1]]
        recv_cos = recv_head.cos()
        recv_sin = recv_head.sin()
        rel_motion_long = rel_motion[:, 0] * recv_cos + rel_motion[:, 1] * recv_sin
        rel_motion_lat = -rel_motion[:, 0] * recv_sin + rel_motion[:, 1] * recv_cos
        rel_motion_long = rel_motion_long.masked_fill(~rel_motion_valid, 0.0)
        rel_motion_lat = rel_motion_lat.masked_fill(~rel_motion_valid, 0.0)

        r_a2a = torch.stack(
            [
                safe_norm_2d(rel_pos_a2a[:, :2]),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
                rel_motion_long,
                rel_motion_lat,
                rel_motion_valid.to(dtype=rel_motion_long.dtype),
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

    def _build_rollout_light_time_delta_norm(
        self,
        *,
        num_agent: int,
        device: torch.device,
        dtype: torch.dtype,
        rollout_step_index: int,
        rollout_start_seconds: float = 0.0,
    ) -> torch.Tensor:
        """closed-loop rollout에서 현재 신호가 얼마나 오래된 정보인지 만듭니다.

        Args:
            num_agent: 현재 batch 안 agent 수입니다.
            device: 반환 tensor를 둘 장치입니다.
            dtype: 반환 tensor 자료형입니다.
            rollout_step_index: 0.5초 rollout block 번호입니다. 첫 block은 0입니다.
            rollout_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다. RoaD처럼
                매 block을 새 입력으로 만들 때 stale 시간이 0초로 reset되지 않도록 더합니다.

        Returns:
            torch.Tensor: 모든 agent에 대한 정규화된 신호 시간 차입니다.
                shape은 ``[num_agent, 1]`` 입니다.
        """
        delta_seconds = float(rollout_start_seconds) + (
            float(rollout_step_index) * float(self.shift) * 0.1
        )
        return build_constant_light_time_delta_norm(
            num_agents=num_agent,
            num_steps=1,
            delta_seconds=delta_seconds,
            device=device,
            dtype=dtype,
        )

    def _build_rollout_context_light_time_delta_norm(
        self,
        *,
        num_agent: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        rollout_start_seconds: float = 0.0,
    ) -> torch.Tensor:
        """초기 rollout cache의 context slot별 신호 stale 시간을 만듭니다.

        RoaD처럼 중간 block을 새 sample로 다시 만들 때, 마지막 context slot은
        ``rollout_start_seconds`` 만큼 오래된 신호를 봐야 하고 그 이전 slot들은
        0.5초 간격으로 더 과거 값을 봐야 합니다.
        """
        if num_agent < 0 or num_steps < 0:
            raise ValueError(
                f"num_agent and num_steps must be non-negative, got {num_agent}, {num_steps}."
            )
        if num_steps == 0:
            return torch.zeros((num_agent, 0), device=device, dtype=dtype)

        current_raw_step = self.num_historical_steps - 1
        raw_steps = torch.arange(1, num_steps + 1, device=device, dtype=dtype) * float(self.shift)
        delta_seconds = float(rollout_start_seconds) + (
            raw_steps - float(current_raw_step)
        ) * 0.1
        delta_norm = normalize_light_time_delta_seconds(delta_seconds)
        return delta_norm.view(1, num_steps).expand(num_agent, num_steps)

    @staticmethod
    def _build_recent_coarse_motion(
        pos_window: torch.Tensor,
        valid_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """마지막 두 coarse 상태 차이로 최근 이동량을 만듭니다.

        Args:
            pos_window: 최근 coarse 중심점 창입니다.
                shape은 ``[n_agent, n_step, 2]`` 입니다.
            valid_window: 같은 창의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                각 agent의 최근 coarse 이동량과 그 유효 여부입니다.
                shape은 각각 ``[n_agent, 2]`` 와 ``[n_agent]`` 입니다.
                마지막 두 상태가 모두 유효하지 않으면 이동량은 0, 유효 여부는
                ``False`` 로 둡니다.
        """
        recent_motion = pos_window.new_zeros((pos_window.shape[0], pos_window.shape[-1]))
        recent_motion_valid = torch.zeros(
            pos_window.shape[0],
            device=pos_window.device,
            dtype=torch.bool,
        )
        if pos_window.shape[1] < 2:
            return recent_motion, recent_motion_valid

        recent_motion_valid = valid_window[:, -1] & valid_window[:, -2]
        recent_motion[recent_motion_valid] = (
            pos_window[recent_motion_valid, -1] - pos_window[recent_motion_valid, -2]
        )
        return recent_motion, recent_motion_valid

    def _build_initial_exec_state_history(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """closed-loop LQR bridge가 쓸 최근 0.5초 실제 10Hz 상태 6개를 준비합니다.

        우선 token processor가 만든 실제 fine history를 그대로 쓰고,
        그 정보가 없으면 최근 pair 또는 coarse 상태를 반복해 길이를 6으로 맞춥니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - exec_pos_history: 최근 fine 중심점 6개입니다.
                  shape은 ``[n_agent, 6, 2]`` 입니다.
                - exec_head_history: 최근 fine 방향 6개입니다.
                  shape은 ``[n_agent, 6]`` 입니다.
                - exec_valid_history: 최근 fine 상태 유효 여부입니다.
                  shape은 ``[n_agent, 6]`` 입니다.
        """
        history_keys = [
            "rollout_init_fine_pos_history",
            "rollout_init_fine_head_history",
            "rollout_init_fine_valid_history",
        ]
        if all(key in tokenized_agent for key in history_keys):
            return (
                tokenized_agent[history_keys[0]].clone(),
                tokenized_agent[history_keys[1]].clone(),
                tokenized_agent[history_keys[2]].clone(),
            )

        exec_pos_pair, exec_head_pair, exec_valid_pair = self._build_initial_exec_state_pair(
            tokenized_agent=tokenized_agent,
        )
        history_steps = int(getattr(self.commit_bridge.config, "history_steps", 6))
        if exec_pos_pair.shape[1] >= history_steps:
            return (
                exec_pos_pair[:, -history_steps:].clone(),
                exec_head_pair[:, -history_steps:].clone(),
                exec_valid_pair[:, -history_steps:].clone(),
            )

        pad_len = history_steps - exec_pos_pair.shape[1]
        return (
            torch.cat([exec_pos_pair[:, :1].expand(-1, pad_len, -1), exec_pos_pair], dim=1),
            torch.cat([exec_head_pair[:, :1].expand(-1, pad_len), exec_head_pair], dim=1),
            torch.cat([exec_valid_pair[:, :1].expand(-1, pad_len), exec_valid_pair], dim=1),
        )

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
                "rollout_init_fine_pos_history",
                "rollout_init_fine_head_history",
                "rollout_init_fine_valid_history",
            ]
        ):
            return (
                tokenized_agent["rollout_init_fine_pos_history"][:, -2:].clone(),
                tokenized_agent["rollout_init_fine_head_history"][:, -2:].clone(),
                tokenized_agent["rollout_init_fine_valid_history"][:, -2:].clone(),
            )
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
                shape은 ``[n_agent, n_anchor, hidden_dim]`` 입니다.
            anchor_mask: 유효 anchor 여부입니다. shape은 ``[n_agent, n_anchor]`` 입니다.

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

    def build_anchor_context(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        anchor_mask: torch.Tensor,
        flow_clean_norm: torch.Tensor,
        flow_agent_type: torch.Tensor | None = None,
        flow_agent_length: torch.Tensor | None = None,
        flow_loss_mask: torch.Tensor | None = None,
        flow_clean_metric_norm: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Open-loop anchor sampling에 필요한 context hidden만 계산합니다."""
        if flow_clean_metric_norm is not None:
            expected_metric_shape = tuple(flow_clean_norm.shape[:2]) + (POSE_FLOW_DIM,)
            if tuple(flow_clean_metric_norm.shape) != expected_metric_shape:
                raise ValueError(
                    "flow_clean_metric_norm must be raw pose-space target with shape "
                    f"{expected_metric_shape}, got {tuple(flow_clean_metric_norm.shape)}."
                )
            flow_clean_metric_norm = flow_clean_metric_norm.to(
                device=flow_clean_norm.device,
                dtype=flow_clean_norm.dtype,
            )

        ctx_hidden_pack = self._encode_context(
            agent_token_index=tokenized_agent["ctx_sampled_idx"],
            pos_a=tokenized_agent["ctx_sampled_pos"],
            head_a=tokenized_agent["ctx_sampled_heading"],
            mask=tokenized_agent["ctx_valid"],
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        num_anchor = int(anchor_mask.shape[1])
        required_context_steps = num_anchor + 1
        if ctx_hidden_pack.shape[1] < required_context_steps:
            raise ValueError(
                "Flow anchor context requires one leading token plus all anchor tokens: "
                f"required={required_context_steps}, actual={ctx_hidden_pack.shape[1]}."
            )
        anchor_hidden = ctx_hidden_pack[:, 1:required_context_steps, :]
        output = {
            "flow_clean_norm": flow_clean_norm,
            "ctx_hidden_pack": ctx_hidden_pack,
            "anchor_hidden": anchor_hidden,
            "anchor_mask": anchor_mask,
        }
        if flow_agent_type is not None:
            output["flow_metric_agent_type"] = flow_agent_type
        if flow_agent_length is not None:
            output["flow_metric_agent_length"] = flow_agent_length
        if flow_loss_mask is not None:
            output["flow_loss_mask"] = flow_loss_mask
        if flow_clean_metric_norm is not None:
            output["flow_clean_metric_norm"] = flow_clean_metric_norm
        return output

    def _to_pose_metric_norm(
        self,
        value: torch.Tensor,
        agent_type: torch.Tensor | None,
        agent_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_kinematic_control_flow or value.shape[-1] != CONTROL_FLOW_DIM:
            return value
        if agent_type is None:
            raise ValueError(
                "agent_type is required to convert control-space flow output "
                "to pose-space metric representation."
            )
        return control_norm_to_pose_norm(
            control_norm=value,
            agent_type=agent_type.to(device=value.device),
            agent_length=(
                agent_length.to(device=value.device, dtype=value.dtype)
                if agent_length is not None
                else None
            ),
            pos_scale_m=self.control_pos_scale_m,
            vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            use_holonomic_model_only=getattr(self, "use_holonomic_model_only", False),
            vehicle_no_slip_point_ratio=getattr(self, "control_vehicle_no_slip_point_ratio", 0.0),
            cyclist_no_slip_point_ratio=getattr(self, "control_cyclist_no_slip_point_ratio", 0.0),
        )

    def flow_norm_to_pose_metric_norm(
        self,
        value: torch.Tensor,
        agent_type: torch.Tensor | None,
        agent_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Metric/시각화 경로가 쓰는 pose-space flow 표현으로 변환합니다."""
        return self._to_pose_metric_norm(
            value=value,
            agent_type=agent_type,
            agent_length=agent_length,
        )


    def _sample_open_loop_future_from_hidden(
        self,
        anchor_hidden_valid: torch.Tensor,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        backprop_last_k: int | None = None,
    ) -> torch.Tensor:
        """유효 anchor 문맥만 받아 실제 생성 경로로 2초 미래를 만듭니다.

        Args:
            anchor_hidden_valid: 유효 anchor만 모은 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: validation마다 같은 출발 잡음을 만들기 위한 seed입니다.
            backprop_last_k: 마지막 몇 step만 역전파할지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다.

        Returns:
            torch.Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            return anchor_hidden_valid.new_zeros((0, self.flow_window_steps, self.flow_state_dim))

        generator = None
        if sampling_seed is not None:
            generator = torch.Generator(device=anchor_hidden_valid.device)
            generator.manual_seed(int(sampling_seed))

        x_init_norm = torch.randn(
            anchor_hidden_valid.shape[0],
            self.flow_window_steps,
            self.flow_state_dim,
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
            model_fn=lambda x_t, tau: self.flow_decoder(anchor_hidden_valid, x_t, tau),
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
    ) -> torch.Tensor:
        """모든 anchor 문맥에서 유효한 것만 골라 실제 생성 경로를 수행합니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, n_anchor, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, n_anchor]`` 입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: validation마다 같은 출발 잡음을 만들기 위한 seed입니다.
            backprop_last_k: 마지막 몇 step만 역전파할지 정합니다.
                ``None`` 이면 전체 step을 역전파합니다.

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
            return torch.zeros((0, tape_steps, self.flow_state_dim), device=device, dtype=dtype)

        if scenario_sampling_seeds is not None:
            if agent_batch is None:
                raise ValueError("scenario별 잡음 테이프를 만들려면 agent_batch가 필요합니다.")
            noise_tape = torch.empty((num_agent, tape_steps, self.flow_state_dim), device=device, dtype=dtype)
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
                    self.flow_state_dim,
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
            self.flow_state_dim,
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
            valid_mask=mask,
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
            light_type=map_feature.get("light_type"),
        )

        t_metadata = build_graph_attention_metadata(
            edge_index=edge_index_t,
            num_dst_nodes=n_agent * n_step,
        )
        r_t = t_metadata.reorder_edge_features(r_t)
        edge_index_t = t_metadata.sorted_edge_index
        pl2a_metadata = build_graph_attention_metadata(
            edge_index=edge_index_pl2a,
            num_dst_nodes=n_agent * n_step,
        )
        r_pl2a = pl2a_metadata.reorder_edge_features(r_pl2a)
        edge_index_pl2a = pl2a_metadata.sorted_edge_index
        a2a_metadata = build_graph_attention_metadata(
            edge_index=edge_index_a2a,
            num_dst_nodes=n_agent * n_step,
        )
        r_a2a = a2a_metadata.reorder_edge_features(r_a2a)
        edge_index_a2a = a2a_metadata.sorted_edge_index

        feat_map = map_feature["pt_token"]
        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](
                feat_a,
                r_t,
                edge_index_t,
                attention_metadata=t_metadata,
                r_is_sorted=True,
            )
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a),
                r_pl2a,
                edge_index_pl2a,
                attention_metadata=pl2a_metadata,
                r_is_sorted=True,
            )
            feat_a = self.a2a_attn_layers[i](
                feat_a,
                r_a2a,
                edge_index_a2a,
                attention_metadata=a2a_metadata,
                r_is_sorted=True,
            )
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
        return feat_a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        anchor_mask: torch.Tensor,
        flow_clean_norm: torch.Tensor,
        flow_agent_type: torch.Tensor | None = None,
        flow_agent_length: torch.Tensor | None = None,
        flow_loss_mask: torch.Tensor | None = None,
        flow_clean_metric_norm: torch.Tensor | None = None,
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
            flow_clean_metric_norm: open-loop metric/시각화가 정답으로 쓸 raw GT pose-space
                표현입니다. control-space 학습에서는 clean control target과 분리됩니다.

        Returns:
            Dict[str, torch.Tensor]:
                flow prediction, target, anchor 문맥, 현재 위치/방향, batch 정보를 담은 사전입니다.
        """
        if flow_loss_mask is not None:
            expected_shape = tuple(flow_clean_norm.shape[:2])
            if tuple(flow_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "flow_loss_mask shape must match flow_clean_norm first two dimensions: "
                    f"expected={expected_shape}, actual={tuple(flow_loss_mask.shape)}."
                )
            flow_loss_mask = flow_loss_mask.to(device=flow_clean_norm.device, dtype=torch.bool)
        if flow_clean_metric_norm is not None:
            expected_metric_shape = tuple(flow_clean_norm.shape[:2]) + (POSE_FLOW_DIM,)
            if tuple(flow_clean_metric_norm.shape) != expected_metric_shape:
                raise ValueError(
                    "flow_clean_metric_norm must be raw pose-space target with shape "
                    f"{expected_metric_shape}, got {tuple(flow_clean_metric_norm.shape)}."
                )
            flow_clean_metric_norm = flow_clean_metric_norm.to(
                device=flow_clean_norm.device,
                dtype=flow_clean_norm.dtype,
            )

        anchor_context = self.build_anchor_context(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=anchor_mask,
            flow_clean_norm=flow_clean_norm,
            flow_agent_type=flow_agent_type,
            flow_agent_length=flow_agent_length,
            flow_loss_mask=flow_loss_mask,
            flow_clean_metric_norm=flow_clean_metric_norm,
        )
        ctx_hidden_pack = anchor_context["ctx_hidden_pack"]
        anchor_hidden = anchor_context["anchor_hidden"]
        anchor_hidden_valid = self._pack_anchor_hidden(anchor_hidden, anchor_mask)

        if flow_clean_norm.numel() == 0:
            empty = flow_clean_norm.new_zeros((0, self.flow_window_steps, self.flow_state_dim))
            output = {
                "flow_pred_norm": empty,
                "flow_target_norm": empty,
                "flow_pred_clean_norm": empty,
                "flow_clean_norm": empty,
                "ctx_hidden_pack": ctx_hidden_pack,
                "anchor_hidden": anchor_hidden,
                "anchor_mask": anchor_mask,
            }
            if flow_agent_type is not None:
                output["flow_metric_agent_type"] = flow_agent_type
                if flow_agent_length is not None:
                    output["flow_metric_agent_length"] = flow_agent_length
                output["flow_pred_clean_metric_norm"] = self._to_pose_metric_norm(
                    empty,
                    flow_agent_type,
                    flow_agent_length,
                )
                output["flow_clean_metric_norm"] = (
                    flow_clean_metric_norm
                    if flow_clean_metric_norm is not None
                    else self._to_pose_metric_norm(empty, flow_agent_type, flow_agent_length)
                )
            elif flow_clean_metric_norm is not None:
                output["flow_clean_metric_norm"] = flow_clean_metric_norm
            if flow_loss_mask is not None:
                output["flow_loss_mask"] = flow_loss_mask
            return output

        flow_sample = self.flow_ode.sample(flow_clean_norm, target_type="velocity")
        flow_pred_norm = self.flow_decoder(
            anchor_hidden_valid,
            flow_sample.x_t,
            flow_sample.tau,
            future_valid_mask=flow_loss_mask,
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
        }
        if flow_agent_type is not None:
            output["flow_metric_agent_type"] = flow_agent_type
            if flow_agent_length is not None:
                output["flow_metric_agent_length"] = flow_agent_length
            output["flow_pred_clean_metric_norm"] = self._to_pose_metric_norm(
                flow_pred_clean_norm,
                flow_agent_type,
                flow_agent_length,
            )
            output["flow_clean_metric_norm"] = (
                flow_clean_metric_norm
                if flow_clean_metric_norm is not None
                else self._to_pose_metric_norm(
                    flow_clean_norm,
                    flow_agent_type,
                    flow_agent_length,
                )
            )
        elif flow_clean_metric_norm is not None:
            output["flow_clean_metric_norm"] = flow_clean_metric_norm
        if flow_loss_mask is not None:
            output["flow_loss_mask"] = flow_loss_mask
        return output

    def _prepare_rollout_cache_impl(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        light_time_start_seconds: float = 0.0,
    ) -> Dict[str, object]:
        """여러 rollout이 공통으로 쓰는 초기 문맥을 한 번만 만듭니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징 사전입니다.
            light_time_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다.
                RoaD처럼 중간 block을 새 입력으로 만들 때 초기 context의 신호 stale
                시간이 0초로 reset되지 않도록 더합니다.

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
        exec_pos_history_10hz, exec_head_history_10hz, exec_valid_history_10hz = (
            self._build_initial_exec_state_history(tokenized_agent=tokenized_agent)
        )
        exec_pos_pair_10hz = exec_pos_history_10hz[:, -2:].clone()
        exec_head_pair_10hz = exec_head_history_10hz[:, -2:].clone()
        exec_valid_pair_10hz = exec_valid_history_10hz[:, -2:].clone()

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
            valid_mask=valid_window,
            inference=True,
        )

        n_step = pos_window.shape[1]
        batch_s_a2a = self._build_step_offset_batch(
            batch=tokenized_agent["batch"],
            num_steps=n_step,
            num_graphs=tokenized_agent["num_graphs"],
        )
        batch_s_pl2a = tokenized_agent["batch"].repeat(n_step)
        context_light_time_delta_norm = self._build_rollout_context_light_time_delta_norm(
            num_agent=n_agent,
            num_steps=n_step,
            device=pos_window.device,
            dtype=pos_window.dtype,
            rollout_start_seconds=light_time_start_seconds,
        )
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
            light_type=map_feature.get("light_type"),
            light_time_delta_norm=context_light_time_delta_norm,
        )
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_window,
            head_a=head_window,
            head_vector_a=head_vector_window,
            batch_s=batch_s_a2a,
            mask=valid_window,
        )

        t_metadata = build_graph_attention_metadata(
            edge_index=edge_index_t,
            num_dst_nodes=n_agent * n_step,
        )
        r_t = t_metadata.reorder_edge_features(r_t)
        edge_index_t = t_metadata.sorted_edge_index
        pl2a_metadata = build_graph_attention_metadata(
            edge_index=edge_index_pl2a,
            num_dst_nodes=n_agent * n_step,
        )
        r_pl2a = pl2a_metadata.reorder_edge_features(r_pl2a)
        edge_index_pl2a = pl2a_metadata.sorted_edge_index
        a2a_metadata = build_graph_attention_metadata(
            edge_index=edge_index_a2a,
            num_dst_nodes=n_agent * n_step,
        )
        r_a2a = a2a_metadata.reorder_edge_features(r_a2a)
        edge_index_a2a = a2a_metadata.sorted_edge_index

        feat_map = map_feature["pt_token"]
        feat_a_t_dict: Dict[int, torch.Tensor] = {}
        feat_a_now = feat_a[:, -1].clone()
        for i in range(self.num_layers):
            temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
            temporal_feat = self.t_attn_layers[i](
                temporal_feat.flatten(0, 1),
                r_t,
                edge_index_t,
                attention_metadata=t_metadata,
                r_is_sorted=True,
            ).view(n_agent, n_step, -1)
            temporal_feat = temporal_feat.transpose(0, 1).flatten(0, 1)
            temporal_feat = self.pt2a_attn_layers[i](
                (feat_map, temporal_feat),
                r_pl2a,
                edge_index_pl2a,
                attention_metadata=pl2a_metadata,
                r_is_sorted=True,
            )
            temporal_feat = self.a2a_attn_layers[i](
                temporal_feat,
                r_a2a,
                edge_index_a2a,
                attention_metadata=a2a_metadata,
                r_is_sorted=True,
            )
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
            "exec_pos_history_10hz": exec_pos_history_10hz,
            "exec_head_history_10hz": exec_head_history_10hz,
            "exec_valid_history_10hz": exec_valid_history_10hz,
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
        light_time_start_seconds: float = 0.0,
    ) -> Dict[str, object]:
        """평가와 제출에서 쓸 no-gradient rollout cache를 만듭니다.

        Args:
            tokenized_agent: 평가용 토큰 사전입니다. agent 축 shape은 ``[n_agent, ...]`` 입니다.
            map_feature: 지도 인코더 출력입니다.
            light_time_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다.

        Returns:
            Dict[str, object]: closed-loop rollout의 초기 상태 cache입니다.
        """
        return self._prepare_rollout_cache_impl(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            light_time_start_seconds=light_time_start_seconds,
        )

    def prepare_training_rollout_cache(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        light_time_start_seconds: float = 0.0,
    ) -> Dict[str, object]:
        """self-forced 학습에서 gradient를 유지한 rollout cache를 만듭니다.

        Args:
            tokenized_agent: 평가 모드 기준 토큰 사전입니다. agent 축 shape은 ``[n_agent, ...]`` 입니다.
            map_feature: 현재 Generator의 지도 인코더 출력입니다.
            light_time_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다.

        Returns:
            Dict[str, object]: N초 self-rollout에 쓸 초기 cache입니다.
        """
        return self._prepare_rollout_cache_impl(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            light_time_start_seconds=light_time_start_seconds,
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
            "exec_pos_history_10hz",
            "exec_head_history_10hz",
            "exec_valid_history_10hz",
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

    @staticmethod
    def _get_random_terminal_world_size() -> int:
        """random terminal 값을 맞춰야 하는 rank 수를 확인합니다.

        torch.distributed가 준비되지 않은 단일 실행에서는 1을 돌려줍니다.
        DDP 실행에서는 실제 rank 수를 돌려줍니다. 반환값이 2 이상일 때만
        rank0에서 뽑은 값을 다른 rank로 복사합니다.

        Returns:
            int: 현재 실행에서 값을 맞춰야 하는 rank 수입니다.
        """
        distributed = getattr(torch, "distributed", None)
        if distributed is None:
            return 1
        try:
            if not distributed.is_available():
                return 1
            if not distributed.is_initialized():
                return 1
            return int(distributed.get_world_size())
        except RuntimeError:
            return 1

    def _sync_random_terminal_s_one(self, terminal_s_one: torch.Tensor) -> torch.Tensor:
        """rank0에서 뽑은 random terminal s 하나를 모든 rank에 복사합니다.

        Args:
            terminal_s_one: 각 rank가 가진 terminal s 후보입니다.
                shape은 ``[1]`` 이고 dtype은 ``torch.long`` 입니다.

        Returns:
            torch.Tensor: 모든 rank가 같은 값을 갖는 terminal s입니다.
                shape은 ``[1]`` 입니다.
        """
        if tuple(terminal_s_one.shape) != (1,):
            raise ValueError(
                "terminal_s_one must have shape [1], "
                f"got {tuple(terminal_s_one.shape)}."
            )
        if self._get_random_terminal_world_size() <= 1:
            return terminal_s_one
        synced_terminal_s_one = terminal_s_one.clone()
        torch.distributed.broadcast(synced_terminal_s_one, src=0)
        return synced_terminal_s_one

    def _build_terminal_step_tensors_from_s_one(
        self,
        terminal_s_one: torch.Tensor,
        sample_steps: int,
        num_scenario: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """공유된 terminal s 하나를 scenario별 tensor로 바꿉니다.

        Args:
            terminal_s_one: 모든 rank가 공유하는 terminal s입니다. shape은 ``[1]`` 입니다.
            sample_steps: 전체 denoising step 수입니다.
            num_scenario: 현재 rank mini-batch 안 scenario 수입니다.
            device: 반환 tensor를 올릴 장치입니다.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: terminal step 수와 논문 표기 s입니다.
                두 tensor 모두 scenario 축 shape은 ``[num_scenario]`` 입니다.
        """
        if int(num_scenario) < 0:
            raise ValueError(f"num_scenario must be non-negative, got {num_scenario}.")
        if int(num_scenario) == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            return empty_long, empty_long

        terminal_s = terminal_s_one.to(device=device, dtype=torch.long).expand(
            int(num_scenario)
        ).clone()
        terminal_steps = int(sample_steps) + 1 - terminal_s
        return terminal_steps, terminal_s

    def _resolve_training_backprop_last_k(
        self,
        sampling_scheme: DictConfig,
    ) -> int | None:
        """self-forced 생성 중 gradient를 남길 마지막 denoising step 수를 정합니다.

        Args:
            sampling_scheme: self-forced rollout sampling 설정입니다.

        Returns:
            int | None:
                마지막 몇 denoising step에 gradient를 남길지 나타냅니다.
                값이 ``None`` 이면 전체 denoising step이 gradient 대상입니다.
                ``random_terminal_step.policy=all`` 이고 사용자가 값을 주지 않으면
                기본값 ``8`` 을 돌려줍니다.
        """
        configured_last_k = getattr(sampling_scheme, "backprop_last_k", None)
        if configured_last_k is not None:
            backprop_last_k = int(configured_last_k)
            if backprop_last_k < 1:
                raise ValueError(
                    "sampling.backprop_last_k must be positive when set, "
                    f"got {backprop_last_k}."
                )
            return backprop_last_k

        random_cfg = getattr(sampling_scheme, "random_terminal_step", None)
        if random_cfg is None or not bool(getattr(random_cfg, "enabled", False)):
            return None

        policy = str(getattr(random_cfg, "policy", "paper_uniform"))
        if policy == "all":
            return 8
        return None

    def _sample_training_terminal_step_for_batch(
        self,
        sampling_scheme: DictConfig,
        num_scenario: int,
        device: torch.device,
        self_forced_epoch: int | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """DDP 전체 rank가 공유할 high-K random terminal step 하나를 샘플링합니다.

        Args:
            sampling_scheme: self-forced rollout sampling 설정입니다.
            num_scenario: 현재 rank mini-batch 안 scenario 개수입니다.
            device: 반환 tensor를 둘 장치입니다.
            self_forced_epoch: 현재 self-forced epoch입니다. ``None``이면 random terminal step을 끕니다.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: terminal step 수 ``K``와 논문 표기 ``s``입니다.
                random terminal이 꺼져 있으면 두 값 모두 ``None`` 입니다.
                켜져 있으면 각 tensor의 scenario 축 shape은 ``[num_scenario]`` 입니다.

        Notes:
            ``policy=paper_uniform`` 은 기존처럼 실행할 denoising step 수를 균등 샘플링합니다.
            ``policy=all`` 은 random terminal step을 만들지 않고 전체 denoising을 실행합니다.
            이때 gradient는 ``sampling.backprop_last_k`` 로 지정한 마지막 step에만 남깁니다.
            ``sampling.backprop_last_k`` 를 생략하면 기본값은 ``8`` 입니다.
        """
        random_cfg = getattr(sampling_scheme, "random_terminal_step", None)
        if self_forced_epoch is None or random_cfg is None:
            return None, None
        if not bool(getattr(random_cfg, "enabled", False)):
            return None, None

        sample_steps = int(getattr(sampling_scheme, "sample_steps", self.flow_ode.solver_steps))
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}.")
        if int(num_scenario) < 0:
            raise ValueError(f"num_scenario must be non-negative, got {num_scenario}.")

        policy = str(getattr(random_cfg, "policy", "paper_uniform"))
        if policy == "all":
            return None, None
        if policy != "paper_uniform":
            raise ValueError(
                "random_terminal_step.policy only supports 'paper_uniform' or 'all'. "
                "Use 'all' to execute every denoising step with last-k backprop, "
                "or use 'paper_uniform' with min_executed_steps."
            )

        scope = str(getattr(random_cfg, "scope", "global_batch"))
        if scope != "global_batch":
            raise ValueError(
                "random_terminal_step.scope only supports 'global_batch' for self-forced training, "
                f"got {scope!r}."
            )

        min_executed_steps = int(getattr(random_cfg, "min_executed_steps", 24))
        if min_executed_steps < 1 or min_executed_steps > sample_steps:
            raise ValueError(
                "random_terminal_step.min_executed_steps must be in [1, sample_steps], "
                f"got min_executed_steps={min_executed_steps}, sample_steps={sample_steps}."
            )

        max_terminal_s = sample_steps + 1 - min_executed_steps
        distributed_enabled = self._get_random_terminal_world_size() > 1
        if distributed_enabled and torch.distributed.get_rank() != 0:
            terminal_s_one = torch.empty((1,), device=device, dtype=torch.long)
        else:
            terminal_s_one = torch.randint(
                low=1,
                high=max_terminal_s + 1,
                size=(1,),
                device=device,
                dtype=torch.long,
            )
        terminal_s_one = self._sync_random_terminal_s_one(terminal_s_one)

        return self._build_terminal_step_tensors_from_s_one(
            terminal_s_one=terminal_s_one,
            sample_steps=sample_steps,
            num_scenario=num_scenario,
            device=device,
        )

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
        self_forced_epoch: int | None = None,
        detach_block_transition: bool = False,
        use_stop_motion: bool | None = None,
        light_time_start_seconds: float = 0.0,
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
            self_forced_epoch: self-forced 학습 epoch입니다. ``None`` 이면 random terminal
                denoising step을 쓰지 않는 평가/추론 경로로 봅니다.
            light_time_start_seconds: RoaD처럼 중간 block에서 새 sample을 만들 때 이미 지난
                rollout 시간입니다. 일반 closed-loop는 0초를 씁니다.

        Returns:
            Dict[str, torch.Tensor]:
                한 번의 rollout 결과입니다. 기존 inference 반환과 같은 키를 가집니다.
                ``return_flow_2s_preview=True`` 이면 step별 raw 2초 preview도
                함께 반환합니다.
        """
        state = self._clone_rollout_cache(rollout_cache)
        rollout_use_stop_motion = (
            self.use_stop_motion
            if use_stop_motion is None
            else bool(use_stop_motion)
        )

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
        exec_pos_history_10hz = state["exec_pos_history_10hz"]
        exec_head_history_10hz = state["exec_head_history_10hz"]
        exec_valid_history_10hz = state["exec_valid_history_10hz"]
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
        # Derive scenario count from the always-present `batch` index instead of
        # `tokenized_agent["num_graphs"]`. The latter is only populated on the
        # training-side `tokenized_agent` (built by `_build_eval_tokenized_inputs`)
        # but is dropped by the validation/inference helper
        # `_build_parallel_rollout_tokenized_agent`, so any read of "num_graphs"
        # here would KeyError on the very first closed-loop validation step
        # even though `_sample_training_terminal_step_for_batch` is a no-op for
        # the eval path (`self_forced_epoch is None`). `batch` is required by
        # downstream PyG ops in this same function and is therefore guaranteed
        # to exist on every code path that reaches this point.
        agent_batch_index = tokenized_agent["batch"]
        num_scenario_for_random_s = (
            int(agent_batch_index.max().item()) + 1
            if agent_batch_index.numel() > 0
            else 0
        )
        (
            terminal_steps_by_scenario,
            terminal_s_by_scenario,
        ) = self._sample_training_terminal_step_for_batch(
            sampling_scheme=sampling_scheme,
            num_scenario=num_scenario_for_random_s,
            device=feat_a_now.device,
            self_forced_epoch=self_forced_epoch,
        )
        terminal_step_by_agent = (
            terminal_steps_by_scenario[tokenized_agent["batch"]]
            if terminal_steps_by_scenario is not None
            else None
        )
        terminal_step_for_rollout = (
            int(terminal_steps_by_scenario[0].item())
            if terminal_steps_by_scenario is not None and terminal_steps_by_scenario.numel() > 0
            else None
        )

        for t in range(n_step_future_2hz):
            if detach_block_transition and t > 0:
                detached_state = detach_training_rollout_state(
                    {
                        "pos_window": pos_window,
                        "head_window": head_window,
                        "head_vector_window": head_vector_window,
                        "valid_window": valid_window,
                        "pred_idx_window": pred_idx_window,
                        "exec_pos_history_10hz": exec_pos_history_10hz,
                        "exec_head_history_10hz": exec_head_history_10hz,
                        "exec_valid_history_10hz": exec_valid_history_10hz,
                        "exec_pos_pair_10hz": exec_pos_pair_10hz,
                        "exec_head_pair_10hz": exec_head_pair_10hz,
                        "exec_valid_pair_10hz": exec_valid_pair_10hz,
                        "feat_a": feat_a,
                        "agent_token_emb": agent_token_emb,
                        "feat_a_t_dict": feat_a_t_dict,
                    }
                )
                pos_window = detached_state["pos_window"]
                head_window = detached_state["head_window"]
                head_vector_window = detached_state["head_vector_window"]
                valid_window = detached_state["valid_window"]
                pred_idx_window = detached_state["pred_idx_window"]
                exec_pos_history_10hz = detached_state["exec_pos_history_10hz"]
                exec_head_history_10hz = detached_state["exec_head_history_10hz"]
                exec_valid_history_10hz = detached_state["exec_valid_history_10hz"]
                exec_pos_pair_10hz = detached_state["exec_pos_pair_10hz"]
                exec_head_pair_10hz = detached_state["exec_head_pair_10hz"]
                exec_valid_pair_10hz = detached_state["exec_valid_pair_10hz"]
                feat_a = detached_state["feat_a"]
                agent_token_emb = detached_state["agent_token_emb"]
                feat_a_t_dict = detached_state["feat_a_t_dict"]
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
                # r_t was built from the original edge_index_t, so keep it immutable for autograd.
                edge_index_t_current = torch.stack(
                    [
                        edge_index_t[0],
                        (edge_index_t[1] + 1) // n_step - 1,
                    ],
                    dim=0,
                )

                edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                    pos_pl=map_feature["position"],
                    orient_pl=map_feature["orientation"],
                    pos_a=pos_window[:, -1:],
                    head_a=head_window[:, -1:],
                    head_vector_a=head_vector_window[:, -1:],
                    mask=inference_mask[:, -1:],
                    batch_s=tokenized_agent["batch"],
                    batch_pl=map_feature["batch"],
                    light_type=map_feature.get("light_type"),
                    light_time_delta_norm=self._build_rollout_light_time_delta_norm(
                        num_agent=pos_window.shape[0],
                        device=pos_window.device,
                        dtype=pos_window.dtype,
                        rollout_step_index=t,
                        rollout_start_seconds=light_time_start_seconds,
                    ),
                )
                recent_motion, recent_motion_valid = self._build_recent_coarse_motion(
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
                    motion_valid_a=recent_motion_valid.unsqueeze(1),
                )

                for i in range(self.num_layers):
                    temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
                    current_hidden = self.t_attn_layers[i](
                        (temporal_feat.flatten(0, 1), temporal_feat[:, -1]),
                        r_t,
                        edge_index_t_current,
                    )
                    current_hidden = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], current_hidden),
                        r_pl2a,
                        edge_index_pl2a,
                    )
                    current_hidden = self.a2a_attn_layers[i](current_hidden, r_a2a, edge_index_a2a)
                    if i + 1 < self.num_layers:
                        current_hidden_for_cache = (
                            current_hidden.detach()
                            if terminal_step_by_agent is not None
                            else current_hidden
                        )
                        feat_a_t_dict[i + 1] = torch.cat(
                            [feat_a_t_dict[i + 1], current_hidden_for_cache.unsqueeze(1)],
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
                flow_sample_steps = int(getattr(
                    sampling_scheme,
                    "sample_steps",
                    self.flow_ode.solver_steps,
                ))
                flow_sample_method = getattr(
                    sampling_scheme,
                    "sample_method",
                    self.flow_ode.solver_method,
                )
                if terminal_step_by_agent is None:
                    flow_sample_backprop_last_k = self._resolve_training_backprop_last_k(
                        sampling_scheme=sampling_scheme,
                    )
                    y_hat_norm = self.flow_ode.generate(
                        x_init=x_init_norm,
                        model_fn=lambda x_t, tau: self.flow_decoder(active_hidden, x_t, tau),
                        steps=flow_sample_steps,
                        method=flow_sample_method,
                        backprop_last_k=flow_sample_backprop_last_k,
                    )
                else:
                    y_hat_norm = self.flow_ode.generate(
                        x_init=x_init_norm,
                        model_fn=lambda x_t, tau: self.flow_decoder(active_hidden, x_t, tau),
                        steps=flow_sample_steps,
                        method=flow_sample_method,
                        terminal_step=terminal_step_for_rollout,
                        return_terminal_clean=True,
                    )
                current_pos_act = pos_window[active_mask, -1]
                current_head_act = head_window[active_mask, -1]
                active_agent_type = tokenized_agent["type"][active_mask]
                active_agent_length = tokenized_agent["shape"][active_mask, 0]
                if return_flow_2s_preview:
                    y_hat_metric_norm = self._to_pose_metric_norm(
                        y_hat_norm,
                        active_agent_type,
                        active_agent_length,
                    )
                    preview_pos_local = y_hat_metric_norm[..., :2] * 20.0
                    preview_pos_global, _ = transform_to_global(
                        pos_local=preview_pos_local,
                        head_local=None,
                        pos_now=current_pos_act,
                        head_now=current_head_act,
                    )
                    pred_flow_2s_traj[active_mask, t] = preview_pos_global
                    pred_flow_2s_valid[active_mask, t] = True
                (
                    raw_commit_pos_act,
                    raw_commit_head_act,
                    _,
                    _,
                ) = self.commit_bridge.commit(
                    y_hat_norm=y_hat_norm,
                    current_pos=current_pos_act,
                    current_head=current_head_act,
                    agent_type=active_agent_type,
                    agent_length=active_agent_length,
                )
                exec_pos_history_act = exec_pos_history_10hz[active_mask].clone()
                exec_head_history_act = exec_head_history_10hz[active_mask].clone()
                exec_valid_history_act = exec_valid_history_10hz[active_mask].clone()

                commit_pos_act = raw_commit_pos_act.clone()
                commit_head_act = raw_commit_head_act.clone()
                next_pos_act = commit_pos_act[:, -1].clone()
                next_head_act = commit_head_act[:, -1].clone()

                stop_mask_act = torch.zeros(
                    active_agent_type.shape[0],
                    dtype=torch.bool,
                    device=active_agent_type.device,
                )
                if rollout_use_stop_motion:
                    _, stop_mask_act = self.commit_bridge.build_stop_motion_mask(
                        current_pos=current_pos_act,
                        current_head=current_head_act,
                        commit_pos=raw_commit_pos_act,
                        commit_head=raw_commit_head_act,
                        agent_type=active_agent_type,
                        token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                        token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                        token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                        token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                    )
                    if stop_mask_act.any():
                        (
                            stop_commit_pos_act,
                            stop_commit_head_act,
                            stop_next_pos_act,
                            stop_next_head_act,
                        ) = self.commit_bridge.freeze_commit_chunk(
                            current_pos=current_pos_act[stop_mask_act],
                            current_head=current_head_act[stop_mask_act],
                        )
                        commit_pos_act[stop_mask_act] = stop_commit_pos_act
                        commit_head_act[stop_mask_act] = stop_commit_head_act
                        next_pos_act[stop_mask_act] = stop_next_pos_act
                        next_head_act[stop_mask_act] = stop_next_head_act

                lqr_mask_act = ((active_agent_type == 0) | (active_agent_type == 2)) & (~stop_mask_act)
                if self.use_lqr and lqr_mask_act.any():
                    (
                        lqr_commit_pos_act,
                        lqr_commit_head_act,
                        lqr_next_pos_act,
                        lqr_next_head_act,
                    ) = self.commit_bridge.execute_lqr_commit(
                        y_hat_norm=y_hat_norm[lqr_mask_act],
                        current_pos=current_pos_act[lqr_mask_act],
                        current_head=current_head_act[lqr_mask_act],
                        exec_pos_history=exec_pos_history_act[lqr_mask_act],
                        exec_head_history=exec_head_history_act[lqr_mask_act],
                        exec_valid_history=exec_valid_history_act[lqr_mask_act],
                        agent_type=active_agent_type[lqr_mask_act],
                    )
                    commit_pos_act[lqr_mask_act] = lqr_commit_pos_act
                    commit_head_act[lqr_mask_act] = lqr_commit_head_act
                    next_pos_act[lqr_mask_act] = lqr_next_pos_act
                    next_head_act[lqr_mask_act] = lqr_next_head_act

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
                commit_pos_export_act = commit_pos_act.clone()
                commit_head_export_act = commit_head_act.clone()
                if self.closed_loop_rollout_mode == "matched_token_chunk":
                    restore_mask_act = ~stop_mask_act
                    if self.use_lqr:
                        restore_mask_act = restore_mask_act & (~((active_agent_type == 0) | (active_agent_type == 2)))
                    if restore_mask_act.any():
                        (
                            restored_commit_pos_act,
                            restored_commit_head_act,
                            _,
                            _,
                        ) = self.commit_bridge.restore_token_chunk(
                            current_pos=current_pos_act[restore_mask_act],
                            current_head=current_head_act[restore_mask_act],
                            next_token_idx=next_token_idx_act[restore_mask_act],
                            agent_type=active_agent_type[restore_mask_act],
                            token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                            token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                            token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                        )
                        commit_pos_export_act[restore_mask_act] = restored_commit_pos_act
                        commit_head_export_act[restore_mask_act] = restored_commit_head_act
                commit_traj_step[active_mask] = commit_pos_export_act
                commit_head_step[active_mask] = commit_head_export_act
                next_pos[active_mask] = next_pos_act
                next_head[active_mask] = next_head_act
                next_token_idx[active_mask] = next_token_idx_act
                exec_pos_history_act = torch.cat([current_pos_act.unsqueeze(1), commit_pos_act], dim=1)
                exec_head_history_act = torch.cat([current_head_act.unsqueeze(1), commit_head_act], dim=1)
                exec_valid_history_act = torch.ones_like(exec_head_history_act, dtype=torch.bool)
                if terminal_step_by_agent is not None:
                    exec_pos_history_act = exec_pos_history_act.detach()
                    exec_head_history_act = exec_head_history_act.detach()
                    exec_valid_history_act = exec_valid_history_act.detach()
                exec_pos_history_10hz[active_mask] = exec_pos_history_act
                exec_head_history_10hz[active_mask] = exec_head_history_act
                exec_valid_history_10hz[active_mask] = exec_valid_history_act
                exec_pos_pair_10hz[active_mask] = exec_pos_history_act[:, -2:]
                exec_head_pair_10hz[active_mask] = exec_head_history_act[:, -2:]
                exec_valid_pair_10hz[active_mask] = exec_valid_history_act[:, -2:]

            pred_traj_10hz[:, t * self.shift : (t + 1) * self.shift] = commit_traj_step
            pred_head_10hz[:, t * self.shift : (t + 1) * self.shift] = commit_head_step

            next_pos_for_context = (
                next_pos.detach() if terminal_step_by_agent is not None else next_pos
            )
            next_head_for_context = (
                next_head.detach() if terminal_step_by_agent is not None else next_head
            )
            next_valid = active_mask.clone()
            coarse_pos_list.append(next_pos_for_context.clone())
            coarse_head_list.append(next_head_for_context.clone())
            coarse_valid_list.append(next_valid.clone())
            coarse_idx_list.append(next_token_idx.clone())

            pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)
            valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
            pos_window = torch.cat([pos_window, next_pos_for_context.unsqueeze(1)], dim=1)
            head_window = torch.cat([head_window, next_head_for_context.unsqueeze(1)], dim=1)
            head_vector_next = torch.stack(
                [next_head_for_context.cos(), next_head_for_context.sin()], dim=-1
            )
            head_vector_window = torch.cat([head_vector_window, head_vector_next.unsqueeze(1)], dim=1)

            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
            agent_token_emb_next_for_context = (
                agent_token_emb_next.detach()
                if terminal_step_by_agent is not None
                else agent_token_emb_next
            )
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next_for_context.unsqueeze(1)], dim=1
            )

            motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
            motion_valid_a = valid_window[:, -1] & valid_window[:, -2]
            motion_vector_a = motion_vector_a.masked_fill(
                ~motion_valid_a.unsqueeze(-1),
                0.0,
            )
            x_a = self._build_motion_feature_from_vector(
                motion_vector_a=motion_vector_a,
                head_vector_a=head_vector_window[:, -1],
                motion_valid_a=motion_valid_a,
            )
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            feat_a_next = self.fusion_emb(
                torch.cat([agent_token_emb_next_for_context, x_a], dim=-1).unsqueeze(1)
            )
            feat_a_next_for_context = (
                feat_a_next.detach() if terminal_step_by_agent is not None else feat_a_next
            )
            feat_a = torch.cat([feat_a, feat_a_next_for_context], dim=1)

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
        if terminal_steps_by_scenario is not None:
            out_dict["sf_terminal_step_by_scenario"] = terminal_steps_by_scenario
            out_dict["sf_terminal_s_by_scenario"] = terminal_s_by_scenario
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
        light_time_start_seconds: float = 0.0,
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
            light_time_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다.

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
            light_time_start_seconds=light_time_start_seconds,
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
        self_forced_epoch: int | None = None,
        detach_block_transition: bool = False,
        use_stop_motion: bool | None = None,
        light_time_start_seconds: float = 0.0,
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
            self_forced_epoch: 현재 self-forced epoch입니다. ``None`` 이면 training
                random terminal denoising step을 끕니다.
            use_stop_motion: ``None``이면 decoder 기본 inference 설정을 사용합니다.
                self-forced 학습에서는 별도 config 값을 넘겨 inference 설정과 분리합니다.
            light_time_start_seconds: 외부 생성기가 이미 진행한 rollout 시간입니다.

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
            self_forced_epoch=self_forced_epoch,
            detach_block_transition=detach_block_transition,
            use_stop_motion=use_stop_motion,
            light_time_start_seconds=light_time_start_seconds,
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
            path_noisy_norm: noisy N초 flow state입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.
            tau: flow interpolation time입니다. shape은 ``[n_valid_agent]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent 마스크입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, torch.Tensor]: ``velocity`` 와 ``clean`` 을 담은 사전입니다. 두 텐서 shape은
            ``[n_valid_agent, flow_window_steps, flow_state_dim]`` 입니다.
        """
        if path_noisy_norm.numel() == 0:
            empty = path_noisy_norm.new_zeros((0, self.flow_window_steps, self.flow_state_dim))
            return {"velocity": empty, "clean": empty}
        if path_noisy_norm.shape[1:] != (self.flow_window_steps, self.flow_state_dim):
            raise ValueError(
                "path_noisy_norm must have shape [n_valid_agent, flow_window_steps, flow_state_dim], "
                f"got {tuple(path_noisy_norm.shape)}."
            )
        if int(anchor_mask.sum().item()) != int(path_noisy_norm.shape[0]):
            raise ValueError(
                "anchor_mask true count must match path_noisy_norm first dim, "
                f"got {int(anchor_mask.sum().item())} and {path_noisy_norm.shape[0]}."
            )

        ctx_hidden_pack = self._encode_context(
            agent_token_index=tokenized_agent["ctx_sampled_idx"],
            pos_a=tokenized_agent["ctx_sampled_pos"],
            head_a=tokenized_agent["ctx_sampled_heading"],
            mask=tokenized_agent["ctx_valid"],
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        if ctx_hidden_pack.shape[1] < 2:
            raise ValueError(
                "path_flow_velocity_for_anchor0 requires at least one leading context "
                "token and one anchor token."
            )
        anchor_hidden = ctx_hidden_pack[:, 1:2, :]
        single_anchor_mask = anchor_mask.bool().view(-1, 1)
        anchor_hidden_valid = self._pack_anchor_hidden(anchor_hidden, single_anchor_mask)
        velocity = self.flow_decoder(anchor_hidden_valid, path_noisy_norm, tau)
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
