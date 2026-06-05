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
    decoder.flow_decoder = lambda hidden, x_t, tau, future_valid_mask=None: x_t.new_zeros(
        (x_t.shape[0], x_t.shape[1], decoder.flow_state_dim)
    )
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


def test_mdg_mask_sampler_marks_invalid_future_steps_clean_zero() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.flow_window_steps = 4
    decoder.mdg_num_noise_levels = 5

    torch.manual_seed(7)
    tokenized_agent = {
        "batch": torch.zeros(2, dtype=torch.long),
        "num_graphs": 1,
    }
    anchor_mask = torch.tensor([[True], [True]])
    future_valid_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, False],
        ]
    )

    mask_level = decoder._sample_mdg_train_mask_levels(
        tokenized_agent=tokenized_agent,
        anchor_mask=anchor_mask,
        future_valid_mask=future_valid_mask,
    )

    assert tuple(mask_level.shape) == tuple(future_valid_mask.shape)
    torch.testing.assert_close(
        mask_level[~future_valid_mask],
        torch.zeros_like(mask_level[~future_valid_mask]),
    )
    assert bool((mask_level[future_valid_mask] >= 1.0).all().item())
    assert bool((mask_level[future_valid_mask] <= decoder.mdg_num_noise_levels).all().item())


def test_mdg_mask_plan_stratifies_rates_and_balances_axes() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    active_pair = torch.ones(20, dtype=torch.bool)

    torch.manual_seed(17)
    delta, temporal_pair = decoder._sample_mdg_training_mask_plan(
        active_pair=active_pair,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        torch.sort(delta).values,
        torch.linspace(0.0, 1.0, 20),
    )
    assert abs(int(temporal_pair.sum().item()) - 10) <= 1


def test_mdg_mask_plan_ignores_inactive_pairs() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    active_pair = torch.tensor([True, False, True, False, True])

    torch.manual_seed(23)
    delta, temporal_pair = decoder._sample_mdg_training_mask_plan(
        active_pair=active_pair,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        torch.sort(delta[active_pair]).values,
        torch.linspace(0.0, 1.0, 3),
    )
    torch.testing.assert_close(
        delta[~active_pair],
        torch.zeros_like(delta[~active_pair]),
    )
    assert not bool(temporal_pair[~active_pair].any().item())
    assert abs(int(temporal_pair[active_pair].sum().item()) - 2) <= 1


def test_mdg_multistep_final_clean_transition_uses_last_clean_estimate() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.flow_window_steps = 3
    decoder.flow_state_dim = 3
    decoder.mdg_num_noise_levels = 5
    decoder._to_mdg_state_norm = lambda current, agent_type, agent_length: current.new_zeros(
        current.shape[:-1] + (5,)
    )

    captured_masks = []

    def _fake_denoiser(anchor_hidden, noisy_state, mask_level, future_valid_mask=None):
        del future_valid_mask
        captured_masks.append(mask_level.detach().clone())
        return noisy_state.new_full(
            (anchor_hidden.shape[0], noisy_state.shape[1], decoder.flow_state_dim),
            float(len(captured_masks)),
        )

    decoder.flow_decoder = _fake_denoiser
    hidden = torch.zeros((2, 4), dtype=torch.float32)
    initial = torch.zeros((2, 3, 3), dtype=torch.float32)
    schedule = torch.tensor(
        [
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )

    out = decoder._mdg_denoise_control(
        anchor_hidden=hidden,
        initial_control_norm=initial,
        mask_schedule=schedule,
        agent_type=torch.zeros(2, dtype=torch.long),
        agent_length=None,
    )

    assert len(captured_masks) == schedule.shape[0]
    for actual, expected in zip(captured_masks, schedule, strict=True):
        torch.testing.assert_close(actual, expected.view(1, -1).expand(2, -1))
    torch.testing.assert_close(out, torch.full_like(out, float(schedule.shape[0])))


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
