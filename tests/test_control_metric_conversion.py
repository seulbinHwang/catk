from __future__ import annotations

import pytest
import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.modules.kinematic_control import (
    VEHICLE_TYPE_ID,
    control_norm_to_pose_norm,
)
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


CONTROL_YAW_SCALE_KWARGS = {
    "control_vehicle_yaw_scale_rad": 0.025,
    "control_pedestrian_yaw_scale_rad": 0.20,
    "control_cyclist_yaw_scale_rad": 0.06,
}


def test_control_metric_conversion_requires_agent_type() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.use_kinematic_control_flow = True

    control_norm = torch.zeros((2, 5, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="agent_type is required"):
        decoder.flow_norm_to_pose_metric_norm(value=control_norm, agent_type=None)


def test_pose_metric_conversion_allows_pose_space_without_agent_type() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.use_kinematic_control_flow = True

    pose_norm = torch.zeros((2, 5, 4), dtype=torch.float32)

    out = decoder.flow_norm_to_pose_metric_norm(value=pose_norm, agent_type=None)

    assert out is pose_norm


def _make_control_processor() -> FlowTokenProcessor:
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.flow_window_steps = 2
    processor.flow_target_dim = 3
    processor.use_kinematic_control_flow = True
    processor.use_holonomic_model_only = False
    processor.use_rolling_supervision = True
    processor.control_pos_scale_m = 1.0
    processor.control_vehicle_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_vehicle_yaw_scale_rad"]
    processor.control_pedestrian_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_pedestrian_yaw_scale_rad"]
    processor.control_cyclist_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_cyclist_yaw_scale_rad"]
    processor.control_vehicle_no_slip_point_ratio = 0.0
    processor.control_cyclist_no_slip_point_ratio = 0.0
    return processor


def test_control_metric_target_keeps_raw_gt_not_projection() -> None:
    processor = _make_control_processor()
    pos = torch.zeros((1, 3, 2), dtype=torch.float32)
    pos[0, 1] = torch.tensor([0.0, 1.0])
    pos[0, 2] = torch.tensor([0.0, 2.0])
    heading = torch.zeros((1, 3), dtype=torch.float32)
    current_pos = pos[:, 0]
    current_head = heading[:, 0]
    agent_type = torch.tensor([VEHICLE_TYPE_ID])
    agent_length = torch.tensor([4.0])
    anchor_mask = torch.tensor([True])

    control_target = processor._build_anchor_clean_norm(
        pos=pos,
        heading=heading,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        anchor_mask=anchor_mask,
        raw_step=0,
    )
    projection_target = control_norm_to_pose_norm(
        control_norm=control_target,
        agent_type=agent_type,
        pos_scale_m=processor.control_pos_scale_m,
        vehicle_yaw_scale_rad=processor.control_vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=processor.control_pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=processor.control_cyclist_yaw_scale_rad,
        vehicle_no_slip_point_ratio=processor.control_vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=processor.control_cyclist_no_slip_point_ratio,
    )
    raw_metric_target = processor._build_anchor_clean_norm(
        pos=pos,
        heading=heading,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        anchor_mask=anchor_mask,
        raw_step=0,
        force_pose_space=True,
    )

    assert tuple(control_target.shape) == (1, 2, 3)
    assert tuple(raw_metric_target.shape) == (1, 2, 4)
    torch.testing.assert_close(raw_metric_target[0, :, 1], torch.tensor([1.0 / 20.0, 2.0 / 20.0]))
    torch.testing.assert_close(projection_target[0, :, 1], torch.zeros(2))


class _DummyFlowSample:
    def __init__(self, clean: torch.Tensor) -> None:
        self.x_t = torch.zeros_like(clean)
        self.target = torch.zeros_like(clean)
        self.tau = torch.zeros(clean.shape[0], device=clean.device, dtype=clean.dtype)


class _DummyFlowODE:
    def sample(self, clean: torch.Tensor, target_type: str):
        return _DummyFlowSample(clean)

    def predict_clean_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return x_t + velocity


def test_decoder_uses_raw_metric_target_when_provided() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.use_kinematic_control_flow = True
    decoder.use_holonomic_model_only = False
    decoder.use_rolling_supervision = True
    decoder.flow_window_steps = 2
    decoder.flow_state_dim = 3
    decoder.control_pos_scale_m = 1.0
    decoder.control_vehicle_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_vehicle_yaw_scale_rad"]
    decoder.control_pedestrian_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_pedestrian_yaw_scale_rad"]
    decoder.control_cyclist_yaw_scale_rad = CONTROL_YAW_SCALE_KWARGS["control_cyclist_yaw_scale_rad"]
    decoder.control_vehicle_no_slip_point_ratio = 0.0
    decoder.control_cyclist_no_slip_point_ratio = 0.0
    decoder.flow_ode = _DummyFlowODE()
    decoder.flow_decoder = lambda hidden, x_t, tau, future_valid_mask=None: x_t
    decoder.build_anchor_context = lambda **kwargs: {
        "ctx_hidden_pack": torch.zeros((1, 2, 1)),
        "anchor_hidden": torch.zeros((1, 1, 1)),
    }
    decoder._pack_anchor_hidden = lambda anchor_hidden, anchor_mask: torch.zeros((1, 1))

    flow_clean_norm = torch.zeros((1, 2, 3), dtype=torch.float32)
    agent_type = torch.tensor([VEHICLE_TYPE_ID])
    raw_metric_target = torch.zeros((1, 2, 4), dtype=torch.float32)
    raw_metric_target[0, :, 1] = torch.tensor([1.0 / 20.0, 2.0 / 20.0])

    out = decoder.forward(
        tokenized_agent={},
        map_feature={},
        anchor_mask=torch.tensor([[True]]),
        flow_clean_norm=flow_clean_norm,
        flow_agent_type=agent_type,
        flow_clean_metric_norm=raw_metric_target,
    )

    torch.testing.assert_close(out["flow_clean_metric_norm"], raw_metric_target)


def test_agent_anchor_context_keeps_raw_metric_target() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder._encode_context = lambda **kwargs: torch.zeros((1, 2, 1))

    flow_clean_norm = torch.zeros((1, 2, 3), dtype=torch.float32)
    raw_metric_target = torch.zeros((1, 2, 4), dtype=torch.float32)
    raw_metric_target[0, :, 1] = torch.tensor([1.0 / 20.0, 2.0 / 20.0])
    tokenized_agent = {
        "ctx_sampled_idx": torch.zeros((1, 2), dtype=torch.long),
        "ctx_sampled_pos": torch.zeros((1, 2, 2), dtype=torch.float32),
        "ctx_sampled_heading": torch.zeros((1, 2), dtype=torch.float32),
        "ctx_valid": torch.ones((1, 2), dtype=torch.bool),
    }

    out = decoder.build_anchor_context(
        tokenized_agent=tokenized_agent,
        map_feature={},
        anchor_mask=torch.tensor([[True]]),
        flow_clean_norm=flow_clean_norm,
        flow_agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        flow_clean_metric_norm=raw_metric_target,
    )

    torch.testing.assert_close(out["flow_clean_metric_norm"], raw_metric_target)


def test_smart_flow_anchor_context_passes_raw_metric_target() -> None:
    class _DummyAgentEncoder:
        def build_anchor_context(self, **kwargs):
            return kwargs

    decoder = SMARTFlowDecoder.__new__(SMARTFlowDecoder)
    decoder.agent_encoder = _DummyAgentEncoder()

    flow_clean_norm = torch.zeros((1, 2, 3), dtype=torch.float32)
    raw_metric_target = torch.zeros((1, 2, 4), dtype=torch.float32)
    raw_metric_target[0, :, 1] = torch.tensor([1.0 / 20.0, 2.0 / 20.0])
    tokenized_agent = {
        "flow_eval_mask": torch.tensor([[True]]),
        "flow_eval_clean_norm": flow_clean_norm,
        "flow_eval_clean_metric_norm": raw_metric_target,
        "flow_eval_agent_type": torch.tensor([VEHICLE_TYPE_ID]),
        "flow_eval_agent_length": torch.tensor([4.0]),
    }

    out = decoder.build_anchor_context_from_map_feature(
        map_feature={},
        tokenized_agent=tokenized_agent,
        anchor_mask_key="flow_eval_mask",
    )

    torch.testing.assert_close(out["flow_clean_metric_norm"], raw_metric_target)
    torch.testing.assert_close(out["flow_agent_length"], tokenized_agent["flow_eval_agent_length"])
