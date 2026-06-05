from __future__ import annotations

from typing import Dict

import torch
from omegaconf import DictConfig
from torch_cluster import radius_graph

from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.flow_local_decoder import (
    ContinuousCommitBridge,
    HierarchicalFlowDecoder,
    LQRCommitBridgeConfig,
)
from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    MDG_STATE_DIM,
    MDG_STATE_SPEED_SCALE_MPS,
    POSE_FLOW_DIM,
    control_norm_to_mdg_state_norm,
    control_norm_to_pose_norm,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)
from src.smart.modules.dynamic_light_time import build_constant_light_time_delta_norm
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
        flow_solver_steps: int | None = None,
        flow_solver_method: str | None = None,
        flow_solver_eps: float | None = None,
        mdg_num_noise_levels: int = 5,
        mdg_state_speed_scale_mps: float = MDG_STATE_SPEED_SCALE_MPS,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        use_rolling_supervision: bool = True,
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
        detach_train_metric_clean: bool = False,
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
        if bool(use_holonomic_model_only):
            raise ValueError(
                "semi_mdg supports only all-agent non-holonomic control dynamics. "
                "Remove use_holonomic_model_only or set it to false."
            )
        self.use_holonomic_model_only = False
        self.use_rolling_supervision = bool(use_rolling_supervision)
        self.detach_train_metric_clean = bool(detach_train_metric_clean)
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
        if not self.use_kinematic_control_flow:
            raise ValueError("semi_mdg requires control-space targets: use_kinematic_control_flow=true.")
        self.mdg_num_noise_levels = int(mdg_num_noise_levels)
        if self.mdg_num_noise_levels <= 0:
            raise ValueError(f"mdg_num_noise_levels must be positive, got {self.mdg_num_noise_levels}.")
        self.mdg_state_speed_scale_mps = float(mdg_state_speed_scale_mps)
        if self.mdg_state_speed_scale_mps <= 0.0:
            raise ValueError(
                "mdg_state_speed_scale_mps must be positive, "
                f"got {self.mdg_state_speed_scale_mps}."
            )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=3,
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
            input_state_dim=MDG_STATE_DIM,
            output_dim=CONTROL_FLOW_DIM,
            num_noise_levels=self.mdg_num_noise_levels,
        )
        del flow_solver_steps, flow_solver_method, flow_solver_eps
        if closed_loop_rollout_mode != "raw_mdg":
            raise ValueError(
                "closed_loop_rollout_mode must be 'raw_mdg'. "
                f"got {closed_loop_rollout_mode!r}."
            )
        self.closed_loop_rollout_mode = "raw_mdg"
        self.use_lqr = bool(use_lqr)
        if self.use_lqr:
            raise ValueError("semi_mdg removes the LQR bridge; set decoder.use_lqr=false.")
        # Stop-motion gating is intentionally disabled for every experiment in
        # this branch. Keep the constructor argument for checkpoint/config
        # compatibility, but never let a Hydra override re-enable it.
        self.use_stop_motion = False
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
            use_stop_motion=False,
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

    def _run_attention_layer(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        r: torch.Tensor | None,
        edge_index: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del layer_idx
        return layer(x, r, edge_index)

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

    def _to_mdg_state_norm(
        self,
        control_norm: torch.Tensor,
        agent_type: torch.Tensor,
        agent_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert normalized 3D control to MDG's 5D state representation."""
        return control_norm_to_mdg_state_norm(
            control_norm=control_norm,
            agent_type=agent_type.to(device=control_norm.device),
            agent_length=(
                agent_length.to(device=control_norm.device, dtype=control_norm.dtype)
                if agent_length is not None
                else None
            ),
            pos_scale_m=self.control_pos_scale_m,
            vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            state_speed_scale_mps=getattr(self, "mdg_state_speed_scale_mps", MDG_STATE_SPEED_SCALE_MPS),
            use_holonomic_model_only=self.use_holonomic_model_only,
            vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
        )

    def _mdg_alpha_schedule(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        noisy = torch.linspace(
            0.99,
            0.01,
            int(getattr(self, "mdg_num_noise_levels", 5)),
            device=device,
            dtype=dtype,
        )
        return torch.cat([torch.ones(1, device=device, dtype=dtype), noisy], dim=0)

    def _mdg_alpha_from_mask_level(self, mask_level: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        schedule = self._mdg_alpha_schedule(mask_level.device, dtype)
        max_level = float(getattr(self, "mdg_num_noise_levels", 5))
        level = mask_level.to(dtype=dtype).clamp(0.0, max_level)
        lower = torch.floor(level).long()
        upper = torch.ceil(level).long()
        weight = level - lower.to(dtype=dtype)
        return torch.lerp(schedule[lower], schedule[upper], weight)

    def _mdg_mask_level_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        schedule = self._mdg_alpha_schedule(alpha.device, alpha.dtype)
        alpha = alpha.clamp(min=float(schedule[-1].item()), max=1.0)
        clean_span = (schedule[0] - schedule[1]).clamp_min(torch.finfo(alpha.dtype).eps)
        noisy_span = (schedule[1] - schedule[-1]).clamp_min(torch.finfo(alpha.dtype).eps)
        clean_level = (schedule[0] - alpha) / clean_span
        max_level = float(getattr(self, "mdg_num_noise_levels", 5))
        noisy_level = 1.0 + (schedule[1] - alpha) / noisy_span * (max_level - 1.0)
        return torch.where(alpha >= schedule[1], clean_level, noisy_level).clamp(
            0.0,
            max_level,
        )

    def _mdg_apply_noise(
        self,
        clean_control_norm: torch.Tensor,
        mask_level: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_control_norm)
        alpha = self._mdg_alpha_from_mask_level(mask_level, clean_control_norm.dtype).unsqueeze(-1)
        return torch.sqrt(alpha) * clean_control_norm + torch.sqrt(1.0 - alpha) * noise

    def _pack_anchor_values(
        self,
        values: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[:2] != anchor_mask.shape:
            raise ValueError(
                "values first two dimensions must match anchor_mask: "
                f"got {tuple(values.shape[:2])} and {tuple(anchor_mask.shape)}."
            )
        if not bool(anchor_mask.any()):
            return values.new_zeros((0,) + tuple(values.shape[2:]))
        permute_order = (1, 0) + tuple(range(2, values.ndim))
        return values.permute(permute_order)[anchor_mask.t()].contiguous()

    def _sample_mdg_train_mask_levels(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        anchor_mask: torch.Tensor,
        future_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample MDG time/agent noise levels and pack them in anchor-major order."""
        n_agent, n_anchor = anchor_mask.shape
        n_step = self.flow_window_steps
        device = anchor_mask.device
        dtype = (
            future_valid_mask.dtype
            if torch.is_floating_point(future_valid_mask)
            else torch.float32
        )
        full = torch.ones((n_agent, n_anchor, n_step), device=device, dtype=dtype)
        batch_value = tokenized_agent.get("batch")
        if batch_value is None:
            batch = torch.zeros(n_agent, device=device, dtype=torch.long)
        else:
            batch = batch_value.to(device=device)
        num_graphs = int(
            tokenized_agent.get(
                "num_graphs",
                int(batch.max().item()) + 1 if batch.numel() else 0,
            )
        )
        if num_graphs <= 0 or n_agent <= 0 or n_anchor <= 0:
            packed = self._pack_anchor_values(full, anchor_mask)
            return torch.where(
                future_valid_mask.to(device=device, dtype=torch.bool),
                packed,
                torch.zeros_like(packed),
            )
        max_level = float(getattr(self, "mdg_num_noise_levels", 5))

        safe_batch = batch.clamp(min=0, max=num_graphs - 1)
        anchor_idx = torch.arange(n_anchor, device=device)
        pair_id = safe_batch[:, None] * n_anchor + anchor_idx.view(1, n_anchor)
        n_pair = num_graphs * n_anchor
        valid_agent_anchor = anchor_mask.to(device=device, dtype=torch.bool)
        valid_pair_counts = torch.zeros(n_pair, device=device, dtype=torch.long)
        valid_pair_counts.scatter_add_(
            0,
            pair_id.reshape(-1),
            valid_agent_anchor.reshape(-1).to(dtype=torch.long),
        )
        active_pair = valid_pair_counts > 0

        delta = torch.rand(n_pair, device=device, dtype=dtype)
        temporal_pair = (torch.rand(n_pair, device=device) < 0.5) & active_pair
        agent_pair = (~temporal_pair) & active_pair
        rounded_step_count = torch.floor(delta * float(n_step) + 0.5).to(
            dtype=torch.long
        )

        time_idx = torch.arange(n_step, device=device)
        pair_remaining = (n_step - rounded_step_count).clamp(min=0, max=n_step)
        pair_remaining_2d = pair_remaining[pair_id]
        temporal_agent_anchor = temporal_pair[pair_id] & valid_agent_anchor
        temporal_prefix = time_idx.view(1, 1, n_step) < pair_remaining_2d.unsqueeze(-1)
        temporal_mask = temporal_agent_anchor.unsqueeze(-1)

        denom = (pair_remaining.to(dtype=dtype) - 1.0).clamp_min(1.0)
        progressive_max = (
            1.0
            + 3.0 * time_idx.to(dtype=dtype).view(1, n_step) / denom.view(-1, 1)
        )
        progressive_max = progressive_max.clamp(max=4.0)
        random_levels = 1.0 + torch.rand(
            (n_agent, n_anchor, n_step),
            device=device,
            dtype=dtype,
        ) * (progressive_max[pair_id] - 1.0)
        random_levels = torch.cummax(random_levels, dim=-1).values
        full = torch.where(
            temporal_mask & temporal_prefix,
            random_levels,
            full,
        )
        full = torch.where(
            temporal_mask & ~temporal_prefix,
            torch.full_like(full, max_level),
            full,
        )

        agent_agent_anchor = agent_pair[pair_id] & valid_agent_anchor
        if bool(agent_agent_anchor.any()):
            num_full_agents = torch.floor(
                delta * valid_pair_counts.to(dtype=dtype) + 0.5
            ).to(dtype=torch.long)
            random_score = torch.rand((n_agent, n_anchor), device=device, dtype=dtype)
            valid_flat = agent_agent_anchor.reshape(-1)
            flat_pair = pair_id.reshape(-1)[valid_flat]
            flat_score = random_score.reshape(-1)[valid_flat]
            if flat_score.numel() > 0:
                score_order = torch.argsort(flat_score, stable=True)
                pair_order = score_order[
                    torch.argsort(flat_pair[score_order], stable=True)
                ]
                sorted_pair = flat_pair[pair_order]
                new_group = torch.ones_like(sorted_pair, dtype=torch.bool)
                new_group[1:] = sorted_pair[1:] != sorted_pair[:-1]
                start_positions = torch.arange(sorted_pair.numel(), device=device)[
                    new_group
                ]
                group_lengths = torch.diff(
                    torch.cat(
                        [
                            start_positions,
                            sorted_pair.new_tensor([sorted_pair.numel()]),
                        ]
                    )
                )
                group_start = torch.repeat_interleave(start_positions, group_lengths)
                rank_sorted = (
                    torch.arange(sorted_pair.numel(), device=device) - group_start
                )
                selected_sorted = rank_sorted < num_full_agents[sorted_pair]
                selected_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
                selected_valid_order = torch.zeros_like(flat_score, dtype=torch.bool)
                selected_valid_order[pair_order] = selected_sorted
                selected_flat[valid_flat] = selected_valid_order
                selected = selected_flat.view(n_agent, n_anchor)
            else:
                selected = torch.zeros(
                    (n_agent, n_anchor),
                    device=device,
                    dtype=torch.bool,
                )

            low = (
                1.0
                + torch.rand((n_agent, n_anchor), device=device, dtype=dtype) * 3.0
            )
            agent_values = torch.where(
                selected,
                torch.full_like(low, max_level),
                low,
            ).unsqueeze(-1)
            full = torch.where(agent_agent_anchor.unsqueeze(-1), agent_values, full)

        packed = self._pack_anchor_values(full, anchor_mask)
        if tuple(packed.shape) != tuple(future_valid_mask.shape):
            raise ValueError(
                "Packed MDG mask shape must match future_valid_mask: "
                f"got {tuple(packed.shape)} and {tuple(future_valid_mask.shape)}."
            )
        return torch.where(
            future_valid_mask.to(device=device, dtype=torch.bool),
            packed,
            torch.zeros_like(packed),
        )

    def _mdg_closed_loop_mask_schedule(
        self,
        device: torch.device,
        dtype: torch.dtype,
        sample_steps: int,
    ) -> torch.Tensor:
        sample_steps = int(sample_steps)
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}.")
        if sample_steps == 1:
            max_level = float(getattr(self, "mdg_num_noise_levels", 5))
            return torch.full(
                (1, self.flow_window_steps),
                max_level,
                device=device,
                dtype=dtype,
            )
        schedule = []
        max_level = float(getattr(self, "mdg_num_noise_levels", 5))
        time_band = torch.div(
            torch.arange(self.flow_window_steps, device=device) * int(max_level - 1.0),
            self.flow_window_steps,
            rounding_mode="floor",
        ).to(dtype=dtype)
        base = torch.linspace(
            max_level,
            1.0,
            sample_steps,
            device=device,
            dtype=dtype,
        )
        base[0] = max_level
        base[-1] = 1.0
        for value in base:
            schedule.append(torch.clamp(value + time_band, max=max_level))
        return torch.stack(schedule, dim=0)

    def _mdg_shift_reuse_action(self, previous_action: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(previous_action)
        shift = int(self.shift)
        if previous_action.shape[1] > shift:
            shifted[:, : previous_action.shape[1] - shift] = previous_action[:, shift:]
        return shifted

    def _mdg_reuse_mask_template(
        self,
        device: torch.device,
        dtype: torch.dtype,
        sampling_scheme: DictConfig,
    ) -> torch.Tensor:
        alpha_values = getattr(sampling_scheme, "action_reuse_alpha", (0.70, 0.60, 0.50, 0.01))
        if len(alpha_values) != 4:
            raise ValueError("action_reuse_alpha must contain four values.")
        alpha = torch.tensor(alpha_values, device=device, dtype=dtype)
        mask_values = self._mdg_mask_level_from_alpha(alpha)
        template = torch.empty(self.flow_window_steps, device=device, dtype=dtype)
        chunk = self.shift
        template[:chunk] = mask_values[0]
        template[chunk : 2 * chunk] = mask_values[1]
        template[2 * chunk : 3 * chunk] = mask_values[2]
        template[3 * chunk :] = mask_values[3]
        return template

    def _mdg_denoise_control(
        self,
        anchor_hidden: torch.Tensor,
        initial_control_norm: torch.Tensor,
        mask_schedule: torch.Tensor,
        agent_type: torch.Tensor,
        agent_length: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current = initial_control_norm
        for step_idx, mask_level in enumerate(mask_schedule):
            expanded_mask = mask_level.view(1, -1).expand(current.shape[0], -1)
            noisy_state = self._to_mdg_state_norm(current, agent_type, agent_length)
            pred_control = self.flow_decoder(anchor_hidden, noisy_state, expanded_mask)
            if step_idx + 1 < mask_schedule.shape[0]:
                next_mask = mask_schedule[step_idx + 1].view(1, -1).expand_as(expanded_mask)
                noise = torch.randn(
                    pred_control.shape,
                    device=pred_control.device,
                    dtype=pred_control.dtype,
                    generator=generator,
                )
                current = self._mdg_apply_noise(pred_control, next_mask, noise=noise)
            else:
                current = pred_control
        return current

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
        agent_type: torch.Tensor | None = None,
        agent_length: torch.Tensor | None = None,
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
        if agent_type is None:
            raise ValueError("agent_type is required for MDG open-loop sampling.")

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
        mdg_sample_steps = int(getattr(sampling_scheme, "sample_steps", 1))
        mask_schedule = self._mdg_closed_loop_mask_schedule(
            device=x_init_norm.device,
            dtype=x_init_norm.dtype,
            sample_steps=mdg_sample_steps,
        )
        return self._mdg_denoise_control(
            anchor_hidden=anchor_hidden_valid,
            initial_control_norm=x_init_norm,
            mask_schedule=mask_schedule,
            agent_type=agent_type,
            agent_length=agent_length,
            generator=generator,
        )

    def sample_open_loop_future(
        self,
        anchor_hidden: torch.Tensor,
        anchor_mask: torch.Tensor,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        backprop_last_k: int | None = None,
        agent_type: torch.Tensor | None = None,
        agent_length: torch.Tensor | None = None,
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
            agent_type=agent_type,
            agent_length=agent_length,
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
        scenario_sampling_signs: torch.Tensor | None = None,
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
            scenario_sampling_signs: 시나리오별 noise 부호입니다.
                shape은 ``[n_scenario]`` 입니다. ``None`` 이면 모두 ``+1`` 입니다.
            agent_batch: 각 agent가 어느 시나리오에 속하는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            torch.Tensor:
                각 agent가 rollout 전체에서 공유할 긴 Gaussian 잡음입니다.
                shape은 ``[n_agent, tape_steps, flow_state_dim]`` 입니다.
        """
        noise_scale = float(getattr(sampling_scheme, "noise_scale", 1.0))
        if num_agent == 0:
            return torch.zeros((0, tape_steps, self.flow_state_dim), device=device, dtype=dtype)

        if scenario_sampling_seeds is not None:
            if agent_batch is None:
                raise ValueError("scenario별 잡음 테이프를 만들려면 agent_batch가 필요합니다.")
            if (
                scenario_sampling_signs is not None
                and scenario_sampling_signs.shape != scenario_sampling_seeds.shape
            ):
                raise ValueError(
                    "scenario_sampling_signs must match scenario_sampling_seeds shape, "
                    f"got {tuple(scenario_sampling_signs.shape)} and "
                    f"{tuple(scenario_sampling_seeds.shape)}."
                )
            noise_tape = torch.empty((num_agent, tape_steps, self.flow_state_dim), device=device, dtype=dtype)
            scenario_seed_list = scenario_sampling_seeds.detach().cpu().tolist()
            scenario_sign_list = (
                scenario_sampling_signs.detach().cpu().tolist()
                if scenario_sampling_signs is not None
                else None
            )
            for scenario_idx, scenario_seed in enumerate(scenario_seed_list):
                scenario_mask = agent_batch == scenario_idx
                if not bool(scenario_mask.any()):
                    continue
                scenario_sign = (
                    float(scenario_sign_list[scenario_idx])
                    if scenario_sign_list is not None
                    else 1.0
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(int(scenario_seed))
                noise_tape[scenario_mask] = torch.randn(
                    int(scenario_mask.sum().item()),
                    tape_steps,
                    self.flow_state_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ) * scenario_sign
            return self._apply_rollout_noise_scale(
                noise_tape=noise_tape,
                noise_scale=noise_scale,
            )

        generator = None
        if sampling_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(sampling_seed))
        noise_tape = torch.randn(
            num_agent,
            tape_steps,
            self.flow_state_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return self._apply_rollout_noise_scale(
            noise_tape=noise_tape,
            noise_scale=noise_scale,
        )

    def _apply_rollout_noise_scale(
        self,
        noise_tape: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        """Apply scalar closed-loop noise scale."""
        return noise_tape * float(noise_scale)

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

        feat_map = map_feature["pt_token"]
        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self._run_attention_layer(
                self.t_attn_layers[i],
                feat_a,
                r_t,
                edge_index_t,
                layer_idx=i,
            )
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self._run_attention_layer(
                self.pt2a_attn_layers[i],
                (feat_map, feat_a),
                r_pl2a,
                edge_index_pl2a,
                layer_idx=i,
            )
            feat_a = self._run_attention_layer(
                self.a2a_attn_layers[i],
                feat_a,
                r_a2a,
                edge_index_a2a,
                layer_idx=i,
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
            empty_state = flow_clean_norm.new_zeros((0, self.flow_window_steps, MDG_STATE_DIM))
            empty_mask = flow_clean_norm.new_zeros((0, self.flow_window_steps))
            output = {
                "flow_pred_norm": empty,
                "flow_target_norm": empty,
                "flow_pred_clean_norm": empty,
                "flow_clean_norm": empty,
                "mdg_pred_state_norm": empty_state,
                "mdg_clean_state_norm": empty_state,
                "mdg_mask_level": empty_mask,
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

        if flow_agent_type is None:
            raise ValueError("flow_agent_type is required for MDG control-state training.")
        mask_level = self._sample_mdg_train_mask_levels(
            tokenized_agent=tokenized_agent,
            anchor_mask=anchor_mask,
            future_valid_mask=(
                flow_loss_mask
                if flow_loss_mask is not None
                else torch.ones(
                    flow_clean_norm.shape[:2],
                    device=flow_clean_norm.device,
                    dtype=torch.bool,
                )
            ),
        )
        noisy_control_norm = self._mdg_apply_noise(flow_clean_norm, mask_level)
        noisy_state_norm = self._to_mdg_state_norm(
            noisy_control_norm,
            flow_agent_type,
            flow_agent_length,
        )
        flow_pred_clean_norm = self.flow_decoder(
            anchor_hidden_valid,
            noisy_state_norm,
            mask_level,
            future_valid_mask=flow_loss_mask,
        )
        if flow_pred_clean_norm.shape[-1] != CONTROL_FLOW_DIM:
            raise ValueError(
                "semi_mdg flow decoder must predict normalized 3D control, "
                f"got last dim {flow_pred_clean_norm.shape[-1]}."
            )
        mdg_pred_state_norm = self._to_mdg_state_norm(
            flow_pred_clean_norm,
            flow_agent_type,
            flow_agent_length,
        )
        mdg_clean_state_norm = self._to_mdg_state_norm(
            flow_clean_norm,
            flow_agent_type,
            flow_agent_length,
        )
        output = {
            "flow_pred_norm": mdg_pred_state_norm,
            "flow_target_norm": mdg_clean_state_norm,
            "flow_pred_clean_norm": flow_pred_clean_norm,
            "flow_clean_norm": flow_clean_norm,
            "mdg_pred_state_norm": mdg_pred_state_norm,
            "mdg_clean_state_norm": mdg_clean_state_norm,
            "mdg_mask_level": mask_level,
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

        min_executed_steps = int(getattr(random_cfg, "min_executed_steps", 16))
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
        scenario_sampling_signs: torch.Tensor | None = None,
        return_flow_2s_preview: bool = False,
        rollout_steps_2hz: int | None = None,
        self_forced_epoch: int | None = None,
        detach_block_transition: bool = False,
        use_stop_motion: bool | None = None,
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
            scenario_sampling_signs: 시나리오별 noise 부호입니다.
                shape은 ``[n_scenario]`` 입니다.
            self_forced_epoch: self-forced 학습 epoch입니다. ``None`` 이면 random terminal
                denoising step을 쓰지 않는 평가/추론 경로로 봅니다.

        Returns:
            Dict[str, torch.Tensor]:
                한 번의 rollout 결과입니다. 기존 inference 반환과 같은 키를 가집니다.
                ``return_flow_2s_preview=True`` 이면 step별 raw 2초 preview도
                함께 반환합니다.
        """
        state = self._clone_rollout_cache(rollout_cache)
        # Always keep stop-motion disabled, including self-forced rollout calls
        # that pass an explicit use_stop_motion argument.
        rollout_use_stop_motion = False

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
        previous_pred_action = None

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
            scenario_sampling_signs=scenario_sampling_signs,
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
                    light_time_delta_norm=build_constant_light_time_delta_norm(
                        num_agents=n_agent,
                        num_steps=1,
                        delta_seconds=float(t * self.shift) * 0.1,
                        device=pos_window.device,
                        dtype=pos_window.dtype,
                    ),
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
                mdg_sample_steps = int(getattr(
                    sampling_scheme,
                    "sample_steps",
                    1,
                ))
                if mdg_sample_steps <= 0:
                    raise ValueError(f"sample_steps must be positive, got {mdg_sample_steps}.")
                mask_schedule = self._mdg_closed_loop_mask_schedule(
                    device=x_init_norm.device,
                    dtype=x_init_norm.dtype,
                    sample_steps=mdg_sample_steps,
                )
                current_pos_act = pos_window[active_mask, -1]
                current_head_act = head_window[active_mask, -1]
                active_agent_type = tokenized_agent["type"][active_mask]
                active_agent_length = tokenized_agent["shape"][active_mask, 0]
                if bool(getattr(sampling_scheme, "action_reuse", False)) and previous_pred_action is not None:
                    shifted_reuse = self._mdg_shift_reuse_action(previous_pred_action)[active_mask]
                    reuse_template = self._mdg_reuse_mask_template(
                        device=x_init_norm.device,
                        dtype=x_init_norm.dtype,
                        sampling_scheme=sampling_scheme,
                    )
                    reuse_mask = reuse_template.view(1, -1).expand_as(mask_schedule)
                    mask_schedule = torch.minimum(mask_schedule, reuse_mask)
                    x_init_norm = self._mdg_apply_noise(
                        shifted_reuse,
                        reuse_template.view(1, -1).expand(shifted_reuse.shape[0], -1),
                        noise=x_init_norm,
                    )
                if terminal_step_by_agent is None:
                    y_hat_norm = self._mdg_denoise_control(
                        anchor_hidden=active_hidden,
                        initial_control_norm=x_init_norm,
                        mask_schedule=mask_schedule,
                        agent_type=active_agent_type,
                        agent_length=active_agent_length,
                    )
                else:
                    with torch.no_grad():
                        y_hat_norm = self._mdg_denoise_control(
                            anchor_hidden=active_hidden,
                            initial_control_norm=x_init_norm,
                            mask_schedule=mask_schedule,
                            agent_type=active_agent_type,
                            agent_length=active_agent_length,
                        )
                    y_hat_norm = y_hat_norm.detach()
                if previous_pred_action is None:
                    previous_pred_action = torch.zeros(
                        (n_agent, self.flow_window_steps, self.flow_state_dim),
                        device=y_hat_norm.device,
                        dtype=y_hat_norm.dtype,
                    )
                previous_pred_action[active_mask] = y_hat_norm.detach()
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
                        agent_length=active_agent_length[lqr_mask_act],
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
                commit_traj_step[active_mask] = commit_pos_act
                commit_head_step[active_mask] = commit_head_act
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
        scenario_sampling_signs: torch.Tensor | None = None,
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
            scenario_sampling_signs: scenario별 noise 부호입니다. shape은 ``[n_scenario]`` 입니다.
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
            scenario_sampling_signs=scenario_sampling_signs,
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
        scenario_sampling_signs: torch.Tensor | None = None,
        rollout_steps_2hz: int | None = None,
        self_forced_epoch: int | None = None,
        detach_block_transition: bool = False,
        use_stop_motion: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """self-forced 학습에서 gradient를 유지한 closed-loop rollout을 실행합니다.

        Args:
            rollout_cache: ``prepare_training_rollout_cache`` 가 만든 초기 상태입니다.
            tokenized_agent: 평가 모드 기준 토큰 사전입니다.
            map_feature: 현재 Generator의 지도 인코더 출력입니다.
            sampling_scheme: flow sampling 설정입니다.
            sampling_seed: batch 공통 seed입니다.
            scenario_sampling_seeds: scenario별 seed입니다. shape은 ``[n_scenario]`` 입니다.
            scenario_sampling_signs: scenario별 noise 부호입니다. shape은 ``[n_scenario]`` 입니다.
            rollout_steps_2hz: 실행할 0.5초 block 수입니다. 기본 self-forced 학습은
                ``flow_window_steps / 5`` 를 넘깁니다.
            self_forced_epoch: 현재 self-forced epoch입니다. ``None`` 이면 training
                random terminal denoising step을 끕니다.
            use_stop_motion: ``None``이면 decoder 기본 inference 설정을 사용합니다.
                self-forced 학습에서는 별도 config 값을 넘겨 inference 설정과 분리합니다.

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
            scenario_sampling_signs=scenario_sampling_signs,
            return_flow_2s_preview=False,
            rollout_steps_2hz=rollout_steps_2hz,
            self_forced_epoch=self_forced_epoch,
            detach_block_transition=detach_block_transition,
            use_stop_motion=use_stop_motion,
        )

    def path_flow_velocity_for_anchor0(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        path_noisy_norm: torch.Tensor,
        tau: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """semi_mdg에서 제거된 Flow Matching velocity API입니다.

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
        del tokenized_agent, map_feature, path_noisy_norm, tau, anchor_mask
        raise RuntimeError(
            "semi_mdg removes Flow Matching velocity prediction. "
            "Use MDG control-state denoising through sample_open_loop_future() "
            "or rollout_from_cache() instead."
        )

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
