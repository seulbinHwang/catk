from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    CYCLIST_TYPE_ID,
    DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    DEFAULT_CONTROL_POS_SCALE_M,
    DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    POSE_FLOW_DIM,
    VEHICLE_TYPE_ID,
    build_transition_aligned_control_trajectory,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)
from src.smart.tokens.control_alignment_cache import (
    CONTROL_ALIGNED_FUTURE_HEADING_KEY,
    CONTROL_ALIGNED_FUTURE_POS_KEY,
    CONTROL_ALIGNMENT_CACHE_CURRENT_STEP,
    CONTROL_TRANSITION_NORM_FUTURE_KEY,
    ControlAlignmentCacheConfig,
    validate_control_alignment_cache_fields,
)
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, validate_flow_window_steps


FLOW_CONTEXT_TOKEN_COUNT = 18
FLOW_TRAIN_ANCHOR_COUNT = 16
DEFAULT_CONTROL_ALIGNMENT_FILTER_CONFIG = {
    "enabled": True,
    "vehicle_max_error_m": 5.0,
    "cyclist_max_error_m": 2.0,
}


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습용 anchor 목표와 평가용 메타데이터를 만듭니다."""

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        flow_window_steps: int = 20,
        use_prefix_valid_future_loss_mask: bool = False,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        control_pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
        control_vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
        control_cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
        control_alignment_filter: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            map_token_file=map_token_file,
            agent_token_file=agent_token_file,
        )
        self.flow_window_steps = validate_flow_window_steps(
            flow_window_steps=flow_window_steps,
            commit_steps=self.shift,
        )
        self.use_prefix_valid_future_loss_mask = bool(use_prefix_valid_future_loss_mask)
        self.use_kinematic_control_flow = bool(use_kinematic_control_flow)
        self.use_holonomic_model_only = bool(use_holonomic_model_only)
        self.control_pos_scale_m = float(control_pos_scale_m)
        filter_config = dict(DEFAULT_CONTROL_ALIGNMENT_FILTER_CONFIG)
        if control_alignment_filter is not None:
            filter_config.update(dict(control_alignment_filter))
        self.control_alignment_filter_enabled = bool(filter_config["enabled"])
        self.control_alignment_filter_vehicle_max_error_m = float(filter_config["vehicle_max_error_m"])
        self.control_alignment_filter_cyclist_max_error_m = float(filter_config["cyclist_max_error_m"])
        if self.control_alignment_filter_vehicle_max_error_m <= 0.0:
            raise ValueError(
                "control_alignment_filter.vehicle_max_error_m must be positive, "
                f"got {self.control_alignment_filter_vehicle_max_error_m}."
            )
        if self.control_alignment_filter_cyclist_max_error_m <= 0.0:
            raise ValueError(
                "control_alignment_filter.cyclist_max_error_m must be positive, "
                f"got {self.control_alignment_filter_cyclist_max_error_m}."
            )
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
            self.control_alignment_cache_config = ControlAlignmentCacheConfig(
                current_step=CONTROL_ALIGNMENT_CACHE_CURRENT_STEP,
                pos_scale_m=self.control_pos_scale_m,
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=self.use_holonomic_model_only,
                vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
            )
        else:
            self.control_alignment_cache_config = None
        self.flow_target_dim = CONTROL_FLOW_DIM if self.use_kinematic_control_flow else POSE_FLOW_DIM

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """지도 토큰과 에이전트 토큰을 만들고 flow 목표를 붙입니다.

        Args:
            data: 원본 장면 배치입니다.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                지도 토큰 사전과 에이전트 토큰 사전입니다.
        """
        tokenized_map = self.tokenize_map(data)
        tokenized_agent, processed_agent = self.tokenize_agent(
            data,
            return_preprocessed=True,
            match_tokens=not self.use_kinematic_control_flow,
        )
        tokenized_agent = self._build_flow_targets(
            data=data,
            tokenized_agent=tokenized_agent,
            processed_agent=processed_agent,
        )
        return tokenized_map, tokenized_agent

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
        processed_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습/평가에 필요한 anchor별 미래와 메타데이터를 만듭니다.

        Args:
            data: 원본 장면 배치입니다.
            tokenized_agent: coarse token 기반 에이전트 토큰 사전입니다.
            processed_agent: 전처리된 실제 좌표와 방향 사전입니다.

        Returns:
            Dict[str, Tensor]:
                flow 관련 필드가 추가된 에이전트 토큰 사전입니다.
        """
        valid = processed_agent["valid"]
        pos = processed_agent["pos"]
        heading = processed_agent["heading"]

        target_pos = pos
        target_heading = heading
        transition_control_norm_by_step: Tensor | None = None
        if self.use_kinematic_control_flow:
            cached_transition = self._load_precomputed_transition_alignment(
                data=data,
                pos=pos,
                heading=heading,
            )
            if cached_transition is not None:
                (
                    target_pos,
                    target_heading,
                    transition_control_norm_by_step,
                ) = cached_transition
            else:
                (
                    target_pos,
                    target_heading,
                    transition_control_norm_by_step,
                ) = build_transition_aligned_control_trajectory(
                    pos=pos,
                    heading=heading,
                    agent_type=tokenized_agent["type"],
                    agent_length=tokenized_agent["shape"][:, 0],
                    current_step=self.shift * 2,
                    pos_scale_m=self.control_pos_scale_m,
                    vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                    pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                    cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                    use_holonomic_model_only=self.use_holonomic_model_only,
                    vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                    cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
                )
            tokenized_agent.update(
                self._match_agent_token(
                    valid=valid,
                    pos=target_pos,
                    heading=target_heading,
                    agent_type=tokenized_agent["type"],
                    agent_shape=tokenized_agent["token_agent_shape"],
                )
            )

        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        num_anchor = FLOW_TRAIN_ANCHOR_COUNT
        raw_current_steps = self.shift * (
            torch.arange(num_anchor, device=device, dtype=torch.long) + 2
        )

        if "train_mask" in data["agent"]:
            train_mask = data["agent"]["train_mask"].bool()
        else:
            train_mask = torch.ones(num_agent, device=device, dtype=torch.bool)

        tokenized_agent.update(
            {
                "ctx_sampled_idx": ctx_sampled_idx,
                "ctx_sampled_pos": ctx_sampled_pos,
                "ctx_sampled_heading": ctx_sampled_heading,
                "ctx_valid": ctx_valid,
            }
        )

        if self.training:
            future_loss_mask_all = self._build_all_anchor_future_loss_mask(
                valid=valid,
                raw_steps=raw_current_steps,
            )
            current_valid_all = valid[:, raw_current_steps]
            alignment_filter_mask_all = self._build_control_alignment_filter_mask_all(
                raw_pos=pos,
                aligned_pos=target_pos,
                agent_type=tokenized_agent["type"],
                raw_steps=raw_current_steps,
                future_loss_mask=future_loss_mask_all,
            )
            flow_train_mask = (
                current_valid_all
                & future_loss_mask_all.any(dim=-1)
                & alignment_filter_mask_all
                & train_mask.unsqueeze(1)
            )
            selected_anchor_idx, selected_agent_idx = flow_train_mask.t().nonzero(as_tuple=True)
            selected_future_loss_mask = future_loss_mask_all[selected_agent_idx, selected_anchor_idx]

            flow_train_clean_norm = self._build_selected_anchor_clean_norm(
                pos=target_pos,
                heading=target_heading,
                raw_steps=raw_current_steps,
                selected_agent_idx=selected_agent_idx,
                selected_anchor_idx=selected_anchor_idx,
                future_loss_mask=selected_future_loss_mask,
                transition_control_norm_by_step=transition_control_norm_by_step,
            )
            flow_train_metric_norm = (
                self._build_selected_anchor_clean_norm(
                    pos=target_pos,
                    heading=target_heading,
                    raw_steps=raw_current_steps,
                    selected_agent_idx=selected_agent_idx,
                    selected_anchor_idx=selected_anchor_idx,
                    future_loss_mask=selected_future_loss_mask,
                    force_pose_space=True,
                    transition_control_norm_by_step=transition_control_norm_by_step,
                )
                if self.use_kinematic_control_flow
                else flow_train_clean_norm
            )

            self._assert_flow_train_anchor_context_valid(
                flow_train_mask=flow_train_mask,
                ctx_valid=ctx_valid,
            )
            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": flow_train_clean_norm,
                    "flow_train_clean_metric_norm": flow_train_metric_norm,
                    "flow_train_loss_mask": selected_future_loss_mask,
                    "flow_train_agent_type": tokenized_agent["type"][selected_agent_idx],
                    "flow_train_agent_length": tokenized_agent["shape"][selected_agent_idx, 0],
                }
            )
            for key in [
                "valid_mask",
                "gt_idx",
                "gt_pos",
                "gt_heading",
                "sampled_idx",
                "sampled_pos",
                "sampled_heading",
            ]:
                tokenized_agent.pop(key, None)
            return tokenized_agent

        future_loss_mask_all = self._build_all_anchor_future_loss_mask(
            valid=valid,
            raw_steps=raw_current_steps,
        )
        current_valid_all = valid[:, raw_current_steps]
        flow_eval_mask = current_valid_all & future_loss_mask_all.all(dim=-1)
        selected_anchor_idx, selected_agent_idx = flow_eval_mask.t().nonzero(as_tuple=True)
        flow_eval_clean_norm = self._build_selected_anchor_clean_norm(
            pos=target_pos,
            heading=target_heading,
            raw_steps=raw_current_steps,
            selected_agent_idx=selected_agent_idx,
            selected_anchor_idx=selected_anchor_idx,
            future_loss_mask=None,
            transition_control_norm_by_step=transition_control_norm_by_step,
        )
        flow_eval_metric_norm = (
            self._build_selected_anchor_clean_norm(
                pos=target_pos,
                heading=target_heading,
                raw_steps=raw_current_steps,
                selected_agent_idx=selected_agent_idx,
                selected_anchor_idx=selected_anchor_idx,
                future_loss_mask=None,
                force_pose_space=True,
                transition_control_norm_by_step=transition_control_norm_by_step,
            )
            if self.use_kinematic_control_flow
            else flow_eval_clean_norm
        )

        tokenized_agent.update(
            {
                "flow_eval_mask": flow_eval_mask,
                "flow_eval_clean_norm": flow_eval_clean_norm,
                "flow_eval_clean_metric_norm": flow_eval_metric_norm,
                "flow_eval_agent_type": tokenized_agent["type"][selected_agent_idx],
                "flow_eval_agent_length": tokenized_agent["shape"][selected_agent_idx, 0],
            }
        )
        return tokenized_agent

    def _load_precomputed_transition_alignment(
        self,
        data: HeteroData,
        pos: Tensor,
        heading: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Use cache-time transition-aligned state/control when it is config-safe.

        The precomputed cache is only valid for unmodified raw ``valid_mask``.
        ``WaymoTargetBuilderTrain`` mutates validity when
        ``train_use_eval_agent_selection=false`` and adds ``train_mask``; in
        that case we deliberately fall back to online construction.
        """
        config = getattr(self, "control_alignment_cache_config", None)
        if config is None:
            return None
        if "train_mask" in data["agent"]:
            return None
        if int(config.current_step) != int(self.shift) * 2:
            return None

        agent_data = data["agent"]
        if not validate_control_alignment_cache_fields(
            agent_data,
            expected_config=config,
            expected_n_agent=int(pos.shape[0]),
            expected_n_step=int(pos.shape[1]),
            device=pos.device,
        ):
            return None

        future_start = int(config.current_step) + 1
        aligned_future_pos = agent_data[CONTROL_ALIGNED_FUTURE_POS_KEY].to(
            device=pos.device,
            dtype=pos.dtype,
        )
        aligned_future_heading = agent_data[CONTROL_ALIGNED_FUTURE_HEADING_KEY].to(
            device=heading.device,
            dtype=heading.dtype,
        )
        transition_control_future = agent_data[CONTROL_TRANSITION_NORM_FUTURE_KEY].to(
            device=pos.device,
            dtype=pos.dtype,
        )

        target_pos = pos.clone()
        target_heading = heading.clone()
        transition_control_norm_by_step = pos.new_zeros(
            (pos.shape[0], pos.shape[1], CONTROL_FLOW_DIM)
        )
        target_pos[:, future_start:] = aligned_future_pos
        target_heading[:, future_start:] = aligned_future_heading
        transition_control_norm_by_step[:, future_start:] = transition_control_future
        return target_pos, target_heading, transition_control_norm_by_step

    def _build_all_anchor_future_loss_mask(self, valid: Tensor, raw_steps: Tensor) -> Tensor:
        """모든 anchor의 future loss mask를 한 번에 만듭니다."""
        future_offsets = torch.arange(self.flow_window_steps, device=valid.device, dtype=torch.long)
        future_indices = raw_steps.unsqueeze(1) + 1 + future_offsets.unsqueeze(0)
        in_bounds = future_indices < valid.shape[1]
        safe_indices = future_indices.clamp(max=max(valid.shape[1] - 1, 0))
        future_valid = valid[:, safe_indices.reshape(-1)].view(
            valid.shape[0],
            raw_steps.shape[0],
            self.flow_window_steps,
        ).bool()
        future_valid = future_valid & in_bounds.unsqueeze(0)
        if self.use_prefix_valid_future_loss_mask:
            return future_valid.to(dtype=torch.long).cumprod(dim=-1).bool()
        full_future_valid = future_valid.all(dim=-1)
        return full_future_valid.unsqueeze(-1).expand(-1, -1, self.flow_window_steps)

    def _build_control_alignment_filter_mask_all(
        self,
        raw_pos: Tensor,
        aligned_pos: Tensor,
        agent_type: Tensor,
        raw_steps: Tensor,
        future_loss_mask: Tensor,
    ) -> Tensor:
        """모든 anchor의 raw-vs-aligned 위치 왜곡 filter를 한 번에 계산합니다."""
        if not self.use_kinematic_control_flow or not getattr(self, "control_alignment_filter_enabled", True):
            return torch.ones(
                (raw_pos.shape[0], raw_steps.shape[0]),
                device=raw_pos.device,
                dtype=torch.bool,
            )

        max_horizon = min(raw_pos.shape[1], aligned_pos.shape[1])
        if max_horizon <= 0:
            return torch.ones(
                (raw_pos.shape[0], raw_steps.shape[0]),
                device=raw_pos.device,
                dtype=torch.bool,
            )

        future_offsets = torch.arange(self.flow_window_steps, device=raw_pos.device, dtype=torch.long)
        future_indices = raw_steps.unsqueeze(1) + 1 + future_offsets.unsqueeze(0)
        in_bounds = future_indices < max_horizon
        safe_indices = future_indices.clamp(max=max_horizon - 1)
        raw_future = raw_pos[:, safe_indices.reshape(-1)].view(
            raw_pos.shape[0],
            raw_steps.shape[0],
            self.flow_window_steps,
            2,
        )
        aligned_future = aligned_pos[:, safe_indices.reshape(-1)].view(
            aligned_pos.shape[0],
            raw_steps.shape[0],
            self.flow_window_steps,
            2,
        )
        error = torch.linalg.vector_norm(aligned_future - raw_future, dim=-1)
        step_mask = future_loss_mask & in_bounds.unsqueeze(0)
        max_error = error.masked_fill(~step_mask, -torch.inf).amax(dim=-1)

        threshold = raw_pos.new_full((raw_pos.shape[0],), torch.inf)
        agent_type_device = agent_type.to(device=raw_pos.device)
        threshold[agent_type_device == VEHICLE_TYPE_ID] = float(
            getattr(self, "control_alignment_filter_vehicle_max_error_m", 5.0)
        )
        threshold[agent_type_device == CYCLIST_TYPE_ID] = float(
            getattr(self, "control_alignment_filter_cyclist_max_error_m", 2.0)
        )
        return max_error <= threshold.unsqueeze(1)

    def _build_selected_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        raw_steps: Tensor,
        selected_agent_idx: Tensor,
        selected_anchor_idx: Tensor,
        future_loss_mask: Tensor | None = None,
        force_pose_space: bool = False,
        transition_control_norm_by_step: Tensor | None = None,
    ) -> Tensor:
        """선택된 [agent, anchor] 쌍의 미래 target을 한 번에 만듭니다."""
        num_selected = int(selected_agent_idx.numel())
        if num_selected == 0:
            target_dim = POSE_FLOW_DIM if force_pose_space else self.flow_target_dim
            return pos.new_zeros((0, self.flow_window_steps, target_dim))

        current_steps = raw_steps[selected_anchor_idx]
        future_offsets = torch.arange(self.flow_window_steps, device=pos.device, dtype=torch.long)
        future_indices = current_steps.unsqueeze(1) + 1 + future_offsets.unsqueeze(0)

        if self.use_kinematic_control_flow and not force_pose_space:
            if transition_control_norm_by_step is None:
                raise ValueError(
                    "transition_control_norm_by_step is required for control-space flow targets."
                )
            if (
                transition_control_norm_by_step.ndim != 3
                or transition_control_norm_by_step.shape[-1] != CONTROL_FLOW_DIM
            ):
                raise ValueError(
                    "transition_control_norm_by_step must have shape [n_agent, n_step, 3], "
                    f"got {tuple(transition_control_norm_by_step.shape)}."
                )
            if transition_control_norm_by_step.shape[0] != pos.shape[0]:
                raise ValueError(
                    "transition_control_norm_by_step agent count must match pos: "
                    f"got {transition_control_norm_by_step.shape[0]} and {pos.shape[0]}."
                )

            in_bounds = future_indices < transition_control_norm_by_step.shape[1]
            if future_loss_mask is None and not bool(in_bounds.all().item()):
                raise ValueError(
                    "Requested control future window exceeds the available transition horizon."
                )
            safe_indices = future_indices.clamp(max=transition_control_norm_by_step.shape[1] - 1)
            control_target = transition_control_norm_by_step[
                selected_agent_idx.unsqueeze(1),
                safe_indices,
            ].clone()
            control_target = control_target.masked_fill(~in_bounds.unsqueeze(-1), 0.0)
            if future_loss_mask is not None:
                expected_shape = (num_selected, self.flow_window_steps)
                if tuple(future_loss_mask.shape) != expected_shape:
                    raise ValueError(
                        "future_loss_mask shape must match selected anchors and flow_window_steps: "
                        f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                    )
                future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
                if bool((future_loss_mask.long().sum(dim=1) <= 0).any().item()):
                    raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")
                control_target = control_target.masked_fill(~future_loss_mask.unsqueeze(-1), 0.0)
            return control_target

        current_pos = pos[selected_agent_idx, current_steps]
        current_head = heading[selected_agent_idx, current_steps]

        in_bounds = future_indices < pos.shape[1]
        if future_loss_mask is None:
            if not bool(in_bounds.all().item()):
                raise ValueError("Requested flow future window exceeds the available sequence length.")
            future_pos = pos[selected_agent_idx.unsqueeze(1), future_indices]
            future_head = heading[selected_agent_idx.unsqueeze(1), future_indices]
        else:
            expected_shape = (num_selected, self.flow_window_steps)
            if tuple(future_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "future_loss_mask shape must match selected anchors and flow_window_steps: "
                    f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                )
            future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
            valid_step_count = future_loss_mask.long().sum(dim=1)
            if bool((valid_step_count <= 0).any().item()):
                raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

            safe_indices = future_indices.clamp(max=pos.shape[1] - 1)
            future_pos = current_pos.unsqueeze(1).expand(-1, self.flow_window_steps, -1).clone()
            future_head = current_head.unsqueeze(1).expand(-1, self.flow_window_steps).clone()
            gathered_pos = pos[selected_agent_idx.unsqueeze(1), safe_indices]
            gathered_head = heading[selected_agent_idx.unsqueeze(1), safe_indices]
            future_pos = torch.where(in_bounds.unsqueeze(-1), gathered_pos, future_pos)
            future_head = torch.where(in_bounds, gathered_head, future_head)

            last_valid_index = valid_step_count - 1
            last_valid_pos = future_pos.gather(
                dim=1,
                index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
            ).squeeze(1)
            last_valid_head = future_head.gather(
                dim=1,
                index=last_valid_index.view(-1, 1),
            ).squeeze(1)
            invalid_future_mask = ~future_loss_mask
            future_pos = torch.where(
                invalid_future_mask.unsqueeze(-1),
                last_valid_pos.unsqueeze(1),
                future_pos,
            )
            future_head = torch.where(
                invalid_future_mask,
                last_valid_head.unsqueeze(1),
                future_head,
            )

        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=current_pos,
            head_now=current_head,
        )
        return torch.stack(
            [
                future_pos_local[..., 0] / 20.0,
                future_pos_local[..., 1] / 20.0,
                future_head_local.cos(),
                future_head_local.sin(),
            ],
            dim=-1,
        )

    def _build_control_alignment_filter_mask(
        self,
        raw_pos: Tensor,
        aligned_pos: Tensor,
        agent_type: Tensor,
        raw_step: int,
        future_loss_mask: Tensor,
    ) -> Tensor:
        """raw와 aligned 위치 차이가 너무 큰 control-space 학습 anchor를 제외합니다."""
        if not self.use_kinematic_control_flow or not getattr(self, "control_alignment_filter_enabled", True):
            return torch.ones(raw_pos.shape[0], device=raw_pos.device, dtype=torch.bool)

        future_start = int(raw_step) + 1
        available_len = min(
            int(self.flow_window_steps),
            max(0, raw_pos.shape[1] - future_start),
            max(0, aligned_pos.shape[1] - future_start),
            future_loss_mask.shape[1],
        )
        if available_len <= 0:
            return torch.ones(raw_pos.shape[0], device=raw_pos.device, dtype=torch.bool)

        step_mask = future_loss_mask[:, :available_len].to(device=raw_pos.device, dtype=torch.bool)
        error = torch.linalg.vector_norm(
            aligned_pos[:, future_start : future_start + available_len]
            - raw_pos[:, future_start : future_start + available_len],
            dim=-1,
        )
        max_error = error.masked_fill(~step_mask, -torch.inf).amax(dim=1)

        threshold = raw_pos.new_full((raw_pos.shape[0],), torch.inf)
        agent_type_device = agent_type.to(device=raw_pos.device)
        threshold[agent_type_device == VEHICLE_TYPE_ID] = float(
            getattr(self, "control_alignment_filter_vehicle_max_error_m", 5.0)
        )
        threshold[agent_type_device == CYCLIST_TYPE_ID] = float(
            getattr(self, "control_alignment_filter_cyclist_max_error_m", 2.0)
        )
        return max_error <= threshold

    def _assert_flow_train_anchor_context_valid(
        self,
        flow_train_mask: Tensor,
        ctx_valid: Tensor,
    ) -> None:
        """선택된 flow 학습 anchor의 현재 0.5초 context token 유효성을 확인합니다."""
        if flow_train_mask.numel() == 0:
            return

        required_ctx_steps = flow_train_mask.shape[1] + 1
        if ctx_valid.shape[1] < required_ctx_steps:
            raise ValueError(
                "Flow train context validity check requires one leading context token "
                f"plus all anchors: required={required_ctx_steps}, actual={ctx_valid.shape[1]}."
            )

        anchor_ctx_valid = ctx_valid[:, 1:required_ctx_steps]
        invalid_anchor_mask = flow_train_mask & ~anchor_ctx_valid
        if invalid_anchor_mask.any():
            invalid_count = int(invalid_anchor_mask.sum().item())
            selected_count = int(flow_train_mask.sum().item())
            raise ValueError(
                "Flow train invariant violated: selected training anchors include invalid "
                "current 0.5s context tokens. "
                f"invalid_count={invalid_count}, selected_count={selected_count}."
            )

    def _build_anchor_future_valid(self, valid: Tensor, raw_step: int) -> Tensor:
        future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
        return future_loss_mask.all(dim=1)

    def _build_anchor_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """현재 설정에 맞는 미래 loss mask를 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
        """
        if self.use_prefix_valid_future_loss_mask:
            return self._build_prefix_valid_future_loss_mask(valid=valid, raw_step=raw_step)
        return self._build_full_window_future_loss_mask(valid=valid, raw_step=raw_step)

    def _build_full_window_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """기존 방식처럼 전체 미래 window가 유효한 경우에만 loss mask를 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
                미래 전체가 유효한 agent만 모든 step이 ``True`` 입니다.
        """
        future_start = raw_step + 1
        # future_loss_mask: [n_agent, flow_window_steps]
        future_loss_mask = torch.zeros(
            (valid.shape[0], self.flow_window_steps),
            device=valid.device,
            dtype=torch.bool,
        )
        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))
        if available_len != self.flow_window_steps:
            return future_loss_mask

        # available_future_valid: [n_agent, flow_window_steps]
        available_future_valid = valid[:, future_start : future_start + available_len].bool()
        full_future_valid = available_future_valid.all(dim=1)
        future_loss_mask[full_future_valid] = True
        return future_loss_mask

    def _build_prefix_valid_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """가까운 미래부터 연속으로 유효한 구간만 loss mask로 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
                ``raw_step + 1``부터 처음 유효하지 않은 step 직전까지만
                ``True`` 입니다. 첫 미래 step이 유효하지 않으면 전부 ``False`` 입니다.
        """
        future_start = raw_step + 1
        # future_loss_mask: [n_agent, flow_window_steps]
        future_loss_mask = torch.zeros(
            (valid.shape[0], self.flow_window_steps),
            device=valid.device,
            dtype=torch.bool,
        )
        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))
        if available_len <= 0:
            return future_loss_mask

        # available_future_valid: [n_agent, available_len]
        available_future_valid = valid[:, future_start : future_start + available_len].bool()
        # prefix_valid: [n_agent, available_len]
        prefix_valid = available_future_valid.to(dtype=torch.long).cumprod(dim=1).bool()
        future_loss_mask[:, :available_len] = prefix_valid
        return future_loss_mask

    def _build_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
        future_loss_mask: Tensor | None = None,
        force_pose_space: bool = False,
        transition_control_norm_by_step: Tensor | None = None,
    ) -> Tensor:
        """한 anchor에서 실제로 쓰는 agent만 골라 미래 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.
            future_loss_mask: loss에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 window를 모두 사용합니다.
            force_pose_space: control-space 학습 중에도 transition-aligned pose-space
                target을 만들어 open-loop metric 정답으로 쓸 때 켭니다.
            transition_control_norm_by_step: ``use_kinematic_control_flow=True`` 일 때
                관측 현재 이후 전체 궤적을 한 번만 변환하며 만든 raw-step별 control입니다.
                shape은 ``[n_agent, n_step, 3]`` 입니다.

        Returns:
            Tensor:
                정규화된 미래 목표입니다.
                pose-space에서는 ``[n_valid_anchor, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_anchor, flow_window_steps, 3]`` 입니다.
        """
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            target_dim = POSE_FLOW_DIM if force_pose_space else self.flow_target_dim
            return pos.new_zeros((0, self.flow_window_steps, target_dim))

        future_start = raw_step + 1
        future_end = future_start + self.flow_window_steps

        if self.use_kinematic_control_flow and not force_pose_space:
            if future_loss_mask is None:
                if future_end > pos.shape[1]:
                    raise ValueError(
                        "Requested flow future window exceeds the available sequence length: "
                        f"raw_step={raw_step}, flow_window_steps={self.flow_window_steps}, "
                        f"n_step={pos.shape[1]}."
                    )
            else:
                expected_shape = (num_valid_anchor, self.flow_window_steps)
                if tuple(future_loss_mask.shape) != expected_shape:
                    raise ValueError(
                        "future_loss_mask shape must match selected anchors and flow_window_steps: "
                        f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                    )
                future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
                valid_step_count = future_loss_mask.long().sum(dim=1)
                if bool((valid_step_count <= 0).any().item()):
                    raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

            if transition_control_norm_by_step is None:
                raise ValueError(
                    "transition_control_norm_by_step is required for control-space flow targets."
                )
            if (
                transition_control_norm_by_step.ndim != 3
                or transition_control_norm_by_step.shape[-1] != CONTROL_FLOW_DIM
            ):
                raise ValueError(
                    "transition_control_norm_by_step must have shape [n_agent, n_step, 3], "
                    f"got {tuple(transition_control_norm_by_step.shape)}."
                )
            if transition_control_norm_by_step.shape[0] != pos.shape[0]:
                raise ValueError(
                    "transition_control_norm_by_step agent count must match pos: "
                    f"got {transition_control_norm_by_step.shape[0]} and {pos.shape[0]}."
                )
            control_start = raw_step + 1
            control_target = pos.new_zeros((num_valid_anchor, self.flow_window_steps, CONTROL_FLOW_DIM))
            available_len = min(
                self.flow_window_steps,
                max(0, transition_control_norm_by_step.shape[1] - control_start),
            )
            if available_len > 0:
                control_target[:, :available_len] = transition_control_norm_by_step[
                    anchor_mask,
                    control_start : control_start + available_len,
                ]
            if future_loss_mask is not None:
                control_target = control_target.masked_fill(
                    ~future_loss_mask.unsqueeze(-1),
                    0.0,
                )
            elif available_len != self.flow_window_steps:
                raise ValueError(
                    "Requested control future window exceeds the available transition horizon: "
                    f"raw_step={raw_step}, flow_window_steps={self.flow_window_steps}, "
                    f"n_step={transition_control_norm_by_step.shape[1]}."
                )
            return control_target

        selected_current_pos = current_pos[anchor_mask]
        selected_current_head = current_head[anchor_mask]

        if future_loss_mask is None:
            if future_end > pos.shape[1]:
                raise ValueError(
                    "Requested flow future window exceeds the available sequence length: "
                    f"raw_step={raw_step}, flow_window_steps={self.flow_window_steps}, "
                    f"n_step={pos.shape[1]}."
                )
            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = pos[anchor_mask, future_start:future_end]
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = heading[anchor_mask, future_start:future_end]
        else:
            expected_shape = (num_valid_anchor, self.flow_window_steps)
            if tuple(future_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "future_loss_mask shape must match selected anchors and flow_window_steps: "
                    f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                )
            future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
            valid_step_count = future_loss_mask.long().sum(dim=1)
            if bool((valid_step_count <= 0).any().item()):
                raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = selected_current_pos.unsqueeze(1).expand(-1, self.flow_window_steps, -1).clone()
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = selected_current_head.unsqueeze(1).expand(-1, self.flow_window_steps).clone()

            available_len = min(self.flow_window_steps, max(0, pos.shape[1] - future_start))
            if available_len > 0:
                future_pos[:, :available_len] = pos[anchor_mask, future_start : future_start + available_len]
                future_head[:, :available_len] = heading[anchor_mask, future_start : future_start + available_len]

            last_valid_index = valid_step_count - 1
            # last_valid_pos: [n_valid_anchor, 2]
            last_valid_pos = future_pos.gather(
                dim=1,
                index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
            ).squeeze(1)
            # last_valid_head: [n_valid_anchor]
            last_valid_head = future_head.gather(
                dim=1,
                index=last_valid_index.view(-1, 1),
            ).squeeze(1)
            invalid_future_mask = ~future_loss_mask
            future_pos = torch.where(
                invalid_future_mask.unsqueeze(-1),
                last_valid_pos.unsqueeze(1),
                future_pos,
            )
            future_head = torch.where(
                invalid_future_mask,
                last_valid_head.unsqueeze(1),
                future_head,
            )

        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=selected_current_pos,
            head_now=selected_current_head,
        )
        return torch.stack(
            [
                future_pos_local[..., 0] / 20.0,
                future_pos_local[..., 1] / 20.0,
                future_head_local.cos(),
                future_head_local.sin(),
            ],
            dim=-1,
        )

    def _concat_flow_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
        target_dim: int | None = None,
    ) -> Tensor:
        """빈 경우까지 포함해서 flow 목표 조각을 하나로 합칩니다.

        Args:
            chunks: 각 anchor에서 만든 목표 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 목표입니다. shape은 ``[n_total_valid_anchor, 20, 4]`` 입니다.
                유효한 anchor가 없으면 ``[0, 20, 4]`` 빈 텐서를 돌려줍니다.
        """
        if target_dim is None:
            target_dim = self.flow_target_dim
        if len(chunks) == 0:
            return torch.zeros((0, self.flow_window_steps, target_dim), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    def _concat_mask_chunks(
        self,
        chunks: List[Tensor],
        device: torch.device,
    ) -> Tensor:
        """미래 step별 loss mask 조각을 하나로 잇습니다.

        Args:
            chunks: 각 anchor에서 고른 mask 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 mask입니다.
                shape은 ``[n_total_valid_anchor, flow_window_steps]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, self.flow_window_steps), device=device, dtype=torch.bool)
        return torch.cat([chunk.to(device=device, dtype=torch.bool) for chunk in chunks], dim=0)

    def _concat_vector_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """1차원 조각 목록을 하나의 벡터로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 벡터입니다. shape은 ``[n_total_valid_anchor]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0,), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _concat_matrix_chunks(
        self,
        chunks: List[Tensor],
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """2차원 조각 목록을 하나의 행렬로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor, width]`` 입니다.
            width: 마지막 축 너비입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 행렬입니다. shape은 ``[n_total_valid_anchor, width]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, width), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
