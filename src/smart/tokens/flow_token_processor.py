from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    DEFAULT_CONTROL_POS_SCALE_M,
    DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M,
    DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    POSE_FLOW_DIM,
    build_rolling_control_target,
    build_rolling_control_target_with_round_trip_error,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, validate_flow_window_steps


FLOW_CONTEXT_TOKEN_COUNT = 18
FLOW_TRAIN_ANCHOR_COUNT = 16


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습용 anchor 목표와 평가용 메타데이터를 만듭니다."""

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling,
        agent_token_sampling,
        flow_window_steps: int = 20,
        use_prefix_valid_future_loss_mask: bool = False,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        use_rolling_supervision: bool = True,
        control_pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
        control_vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
        control_cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
        control_round_trip_max_position_error_m: float = DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M,
    ) -> None:
        super().__init__(
            map_token_file=map_token_file,
            agent_token_file=agent_token_file,
            map_token_sampling=map_token_sampling,
            agent_token_sampling=agent_token_sampling,
        )
        self.flow_window_steps = validate_flow_window_steps(
            flow_window_steps=flow_window_steps,
            commit_steps=self.shift,
        )
        self.use_prefix_valid_future_loss_mask = bool(use_prefix_valid_future_loss_mask)
        self.use_kinematic_control_flow = bool(use_kinematic_control_flow)
        self.use_holonomic_model_only = bool(use_holonomic_model_only)
        self.use_rolling_supervision = bool(use_rolling_supervision)
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
        self.control_round_trip_max_position_error_m = float(
            control_round_trip_max_position_error_m
        )
        if self.control_round_trip_max_position_error_m <= 0.0:
            raise ValueError(
                "control_round_trip_max_position_error_m must be positive, "
                f"got {self.control_round_trip_max_position_error_m}."
            )
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

        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        num_anchor = FLOW_TRAIN_ANCHOR_COUNT
        raw_current_steps = [
            self.shift * (anchor_idx + 2)
            for anchor_idx in range(num_anchor)
        ]

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
            flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
            flow_train_chunks: List[Tensor] = []
            flow_train_metric_chunks: List[Tensor] = []
            flow_train_loss_mask_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_agent_length_chunks: List[Tensor] = []

            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = valid[:, raw_step]
                future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
                anchor_mask = current_valid & future_loss_mask.any(dim=1)
                train_anchor_mask = anchor_mask & train_mask
                if not train_anchor_mask.any():
                    continue

                current_pos = pos[:, raw_step]
                current_head = heading[:, raw_step]
                selected_future_loss_mask = future_loss_mask[train_anchor_mask]
                flow_clean_result = self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=current_pos,
                    current_head=current_head,
                    agent_type=tokenized_agent["type"],
                    agent_length=tokenized_agent["shape"][:, 0],
                    anchor_mask=train_anchor_mask,
                    raw_step=raw_step,
                    future_loss_mask=selected_future_loss_mask,
                    return_round_trip_error=self.use_kinematic_control_flow,
                )
                if self.use_kinematic_control_flow:
                    flow_train_clean_norm, round_trip_error_m = flow_clean_result
                    keep_mask = self._build_control_round_trip_keep_mask(
                        round_trip_error_m=round_trip_error_m,
                        future_loss_mask=selected_future_loss_mask,
                    )
                    if not bool(keep_mask.all().item()):
                        selected_agent_index = train_anchor_mask.nonzero(as_tuple=False).flatten()
                        kept_agent_index = selected_agent_index[keep_mask]
                        filtered_train_anchor_mask = torch.zeros_like(train_anchor_mask)
                        filtered_train_anchor_mask[kept_agent_index] = True
                        train_anchor_mask = filtered_train_anchor_mask
                        flow_train_clean_norm = flow_train_clean_norm[keep_mask]
                        selected_future_loss_mask = selected_future_loss_mask[keep_mask]
                else:
                    flow_train_clean_norm = flow_clean_result

                flow_train_mask[:, anchor_offset] = train_anchor_mask
                if not train_anchor_mask.any():
                    continue

                flow_train_metric_norm = (
                    self._build_anchor_clean_norm(
                        pos=pos,
                        heading=heading,
                        current_pos=current_pos,
                        current_head=current_head,
                        agent_type=tokenized_agent["type"],
                        agent_length=tokenized_agent["shape"][:, 0],
                        anchor_mask=train_anchor_mask,
                        raw_step=raw_step,
                        future_loss_mask=selected_future_loss_mask,
                        force_pose_space=True,
                    )
                    if self.use_kinematic_control_flow
                    else flow_train_clean_norm
                )
                flow_train_chunks.append(flow_train_clean_norm)
                flow_train_metric_chunks.append(flow_train_metric_norm)
                flow_train_loss_mask_chunks.append(selected_future_loss_mask)
                flow_train_agent_type_chunks.append(tokenized_agent["type"][train_anchor_mask])
                flow_train_agent_length_chunks.append(tokenized_agent["shape"][train_anchor_mask, 0])

            self._assert_flow_train_anchor_context_valid(
                flow_train_mask=flow_train_mask,
                ctx_valid=ctx_valid,
            )
            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": self._concat_flow_chunks(
                        chunks=flow_train_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_clean_metric_norm": self._concat_flow_chunks(
                        chunks=flow_train_metric_chunks,
                        dtype=dtype,
                        device=device,
                        target_dim=POSE_FLOW_DIM,
                    ),
                    "flow_train_loss_mask": self._concat_mask_chunks(
                        chunks=flow_train_loss_mask_chunks,
                        device=device,
                    ),
                    "flow_train_agent_type": self._concat_vector_chunks(
                        chunks=flow_train_agent_type_chunks,
                        dtype=tokenized_agent["type"].dtype,
                        device=device,
                    ),
                    "flow_train_agent_length": self._concat_vector_chunks(
                        chunks=flow_train_agent_length_chunks,
                        dtype=dtype,
                        device=device,
                    ),
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

        flow_eval_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
        flow_eval_chunks: List[Tensor] = []
        flow_eval_metric_chunks: List[Tensor] = []
        flow_eval_agent_type_chunks: List[Tensor] = []
        flow_eval_agent_length_chunks: List[Tensor] = []
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = valid[:, raw_step]
            future_valid = self._build_anchor_future_valid(valid=valid, raw_step=raw_step)
            anchor_mask = current_valid & future_valid
            flow_eval_mask[:, anchor_offset] = anchor_mask
            if not anchor_mask.any():
                continue

            flow_eval_agent_type_chunks.append(tokenized_agent["type"][anchor_mask])
            flow_eval_agent_length_chunks.append(tokenized_agent["shape"][anchor_mask, 0])
            flow_eval_clean_norm = self._build_anchor_clean_norm(
                pos=pos,
                heading=heading,
                current_pos=pos[:, raw_step],
                current_head=heading[:, raw_step],
                agent_type=tokenized_agent["type"],
                agent_length=tokenized_agent["shape"][:, 0],
                anchor_mask=anchor_mask,
                raw_step=raw_step,
            )
            flow_eval_chunks.append(flow_eval_clean_norm)
            flow_eval_metric_chunks.append(
                self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=pos[:, raw_step],
                    current_head=heading[:, raw_step],
                    agent_type=tokenized_agent["type"],
                    agent_length=tokenized_agent["shape"][:, 0],
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                    force_pose_space=True,
                )
                if self.use_kinematic_control_flow
                else flow_eval_clean_norm
            )

        tokenized_agent.update(
            {
                "flow_eval_mask": flow_eval_mask,
                "flow_eval_clean_norm": self._concat_flow_chunks(
                    chunks=flow_eval_chunks,
                    dtype=dtype,
                    device=device,
                ),
                "flow_eval_clean_metric_norm": self._concat_flow_chunks(
                    chunks=flow_eval_metric_chunks,
                    dtype=dtype,
                    device=device,
                    target_dim=POSE_FLOW_DIM,
                ),
                "flow_eval_agent_type": self._concat_vector_chunks(
                    chunks=flow_eval_agent_type_chunks,
                    dtype=tokenized_agent["type"].dtype,
                    device=device,
                ),
                "flow_eval_agent_length": self._concat_vector_chunks(
                    chunks=flow_eval_agent_length_chunks,
                    dtype=dtype,
                    device=device,
                ),
            }
        )
        return tokenized_agent

    def _build_control_round_trip_keep_mask(
        self,
        round_trip_error_m: Tensor,
        future_loss_mask: Tensor,
    ) -> Tensor:
        """control 복원 위치 오차가 설정값 이하인 anchor만 남깁니다."""
        if round_trip_error_m.ndim != 2:
            raise ValueError(
                "round_trip_error_m must have shape [n_valid_anchor, flow_window_steps], "
                f"got {tuple(round_trip_error_m.shape)}."
            )
        if tuple(future_loss_mask.shape) != tuple(round_trip_error_m.shape):
            raise ValueError(
                "future_loss_mask shape must match round_trip_error_m: "
                f"expected={tuple(round_trip_error_m.shape)}, actual={tuple(future_loss_mask.shape)}."
            )
        if round_trip_error_m.shape[0] == 0:
            return torch.zeros((0,), device=round_trip_error_m.device, dtype=torch.bool)

        mask = future_loss_mask.to(device=round_trip_error_m.device, dtype=torch.bool)
        masked_error_m = torch.where(
            mask,
            round_trip_error_m,
            torch.zeros_like(round_trip_error_m),
        )
        max_position_error_m = masked_error_m.max(dim=1).values
        return max_position_error_m <= self.control_round_trip_max_position_error_m

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
        agent_type: Tensor,
        agent_length: Tensor | None,
        anchor_mask: Tensor,
        raw_step: int,
        future_loss_mask: Tensor | None = None,
        return_round_trip_error: bool = False,
        force_pose_space: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """한 anchor에서 실제로 쓰는 agent만 골라 미래 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_length: WOMD box length입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.
            future_loss_mask: loss에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 window를 모두 사용합니다.
            return_round_trip_error: control-space label의 복원 위치 오차도 함께 돌려줄지 정합니다.
            force_pose_space: control-space 학습 중에도 raw GT 기준 pose-space target을
                만들어 open-loop metric 정답으로 쓸 때 켭니다.

        Returns:
            Tensor | Tuple[Tensor, Tensor]:
                정규화된 미래 목표입니다.
                pose-space에서는 ``[n_valid_anchor, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_anchor, flow_window_steps, 3]`` 입니다.
                ``return_round_trip_error=True`` 이면 두 번째 값으로 meter 단위 복원 오차
                ``[n_valid_anchor, flow_window_steps]`` 를 함께 돌려줍니다.
        """
        if force_pose_space and return_round_trip_error:
            raise ValueError("force_pose_space cannot be combined with return_round_trip_error.")
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            target_dim = POSE_FLOW_DIM if force_pose_space else self.flow_target_dim
            empty_target = pos.new_zeros((0, self.flow_window_steps, target_dim))
            if return_round_trip_error:
                return empty_target, pos.new_zeros((0, self.flow_window_steps))
            return empty_target

        selected_current_pos = current_pos[anchor_mask]
        selected_current_head = current_head[anchor_mask]
        selected_agent_type = agent_type[anchor_mask]
        selected_agent_length = agent_length[anchor_mask] if agent_length is not None else None
        future_start = raw_step + 1
        future_end = future_start + self.flow_window_steps

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

        if self.use_kinematic_control_flow and not force_pose_space:
            if return_round_trip_error:
                return build_rolling_control_target_with_round_trip_error(
                    future_pos=future_pos,
                    future_head=future_head,
                    current_pos=selected_current_pos,
                    current_head=selected_current_head,
                    agent_type=selected_agent_type,
                    agent_length=selected_agent_length,
                    pos_scale_m=self.control_pos_scale_m,
                    vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                    pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                    cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                    use_holonomic_model_only=self.use_holonomic_model_only,
                    use_rolling_supervision=self.use_rolling_supervision,
                    vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                    cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
                )
            return build_rolling_control_target(
                future_pos=future_pos,
                future_head=future_head,
                current_pos=selected_current_pos,
                current_head=selected_current_head,
                agent_type=selected_agent_type,
                agent_length=selected_agent_length,
                pos_scale_m=self.control_pos_scale_m,
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=self.use_holonomic_model_only,
                use_rolling_supervision=self.use_rolling_supervision,
                vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
            )

        if return_round_trip_error:
            raise ValueError("return_round_trip_error is only supported for control-space flow targets.")

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
