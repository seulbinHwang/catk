from __future__ import annotations

import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def _build_processor(use_prefix_valid_future_loss_mask: bool) -> FlowTokenProcessor:
    """토큰 파일을 읽지 않고 loss mask helper만 테스트할 processor를 만듭니다.

    Args:
        use_prefix_valid_future_loss_mask: prefix-valid 방식을 사용할지 여부입니다.

    Returns:
        FlowTokenProcessor: ``flow_window_steps``와 옵션만 채운 테스트용 객체입니다.
    """
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.flow_window_steps = 5
    processor.use_prefix_valid_future_loss_mask = use_prefix_valid_future_loss_mask
    processor.control_round_trip_max_position_error_m = 5.0
    return processor


def _build_target_processor(use_prefix_valid_future_loss_mask: bool) -> FlowTokenProcessor:
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.training = True
    processor.shift = 5
    processor.flow_window_steps = 20
    processor.flow_target_dim = 4
    processor.use_prefix_valid_future_loss_mask = use_prefix_valid_future_loss_mask
    processor.use_kinematic_control_flow = False
    processor.control_round_trip_max_position_error_m = 5.0
    return processor


def _build_kinematic_target_processor(use_prefix_valid_future_loss_mask: bool) -> FlowTokenProcessor:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask)
    processor.flow_target_dim = 2
    processor.use_kinematic_control_flow = True
    processor.use_holonomic_model_only = False
    processor.use_rolling_supervision = True
    processor.control_pos_scale_m = 1.0
    processor.control_vehicle_yaw_scale_rad = 0.5
    processor.control_pedestrian_yaw_scale_rad = 0.5
    processor.control_cyclist_yaw_scale_rad = 0.5
    processor.control_vehicle_no_slip_point_ratio = 0.0
    processor.control_cyclist_no_slip_point_ratio = 0.0
    processor.control_round_trip_max_position_error_m = 0.5
    return processor


def _build_tokenized_agent_for_18_context() -> dict[str, torch.Tensor]:
    return {
        "sampled_idx": torch.zeros((1, 18), dtype=torch.long),
        "sampled_pos": torch.zeros((1, 18, 2), dtype=torch.float32),
        "sampled_heading": torch.zeros((1, 18), dtype=torch.float32),
        "valid_mask": torch.ones((1, 18), dtype=torch.bool),
        "type": torch.zeros((1,), dtype=torch.long),
        "shape": torch.tensor([[2.0, 4.8, 1.5]], dtype=torch.float32),
    }


def _clone_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in values.items()}


def _build_multi_agent_tokenized_agent() -> dict[str, torch.Tensor]:
    num_agent = 4
    return {
        "sampled_idx": torch.zeros((num_agent, 18), dtype=torch.long),
        "sampled_pos": torch.zeros((num_agent, 18, 2), dtype=torch.float32),
        "sampled_heading": torch.zeros((num_agent, 18), dtype=torch.float32),
        "valid_mask": torch.ones((num_agent, 18), dtype=torch.bool),
        "type": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "shape": torch.tensor(
            [
                [4.8, 2.0, 1.5],
                [0.8, 0.8, 1.7],
                [1.8, 0.6, 1.6],
                [4.4, 2.1, 1.5],
            ],
            dtype=torch.float32,
        ),
    }


def _build_multi_agent_processed_agent() -> dict[str, torch.Tensor]:
    num_agent = 4
    raw_step = torch.arange(91, dtype=torch.float32)
    pos = torch.zeros((num_agent, 91, 2), dtype=torch.float32)
    heading = torch.zeros((num_agent, 91), dtype=torch.float32)
    valid = torch.ones((num_agent, 91), dtype=torch.bool)

    pos[0, :, 0] = raw_step * 0.35
    pos[0, :, 1] = torch.sin(raw_step * 0.08) * 0.5
    heading[0] = raw_step * 0.01

    pos[1, :, 0] = raw_step * 0.05
    pos[1, :, 1] = raw_step * 0.08
    heading[1] = raw_step * 0.02

    pos[2, :, 0] = raw_step * 0.18
    pos[2, :, 1] = torch.cos(raw_step * 0.05) * 0.3
    heading[2] = raw_step * 0.015

    pos[3, :, 0] = raw_step * 0.2
    pos[3, :, 1] = torch.where(raw_step >= 71, torch.tensor(4.0), torch.zeros_like(raw_step))
    heading[3] = 0.0
    velocity = torch.zeros_like(pos)
    velocity[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / 0.1

    valid[1, 82:] = False
    valid[2, 76:] = False
    valid[3, 72:] = False
    return {"valid": valid, "pos": pos, "heading": heading, "velocity": velocity}


def _build_flow_targets_anchor_loop_reference(
    processor: FlowTokenProcessor,
    data: dict,
    tokenized_agent: dict[str, torch.Tensor],
    processed_agent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    valid = processed_agent["valid"]
    pos = processed_agent["pos"]
    heading = processed_agent["heading"]
    velocity = processed_agent["velocity"]
    ctx_valid = tokenized_agent["valid_mask"][:, :18].contiguous()
    num_agent = pos.shape[0]
    num_anchor = 16
    raw_current_steps = [processor.shift * (anchor_idx + 2) for anchor_idx in range(num_anchor)]
    train_mask = data["agent"].get(
        "train_mask",
        torch.ones(num_agent, device=pos.device, dtype=torch.bool),
    ).bool()

    flow_train_mask = torch.zeros(num_agent, num_anchor, device=pos.device, dtype=torch.bool)
    flow_train_chunks = []
    flow_train_metric_chunks = []
    flow_train_loss_mask_chunks = []
    flow_train_agent_type_chunks = []
    flow_train_agent_length_chunks = []
    flow_train_current_speed_chunks = []

    for anchor_offset, raw_step in enumerate(raw_current_steps):
        future_loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
        train_anchor_mask = valid[:, raw_step] & future_loss_mask.any(dim=1) & train_mask
        if not bool(train_anchor_mask.any().item()):
            continue

        selected_future_loss_mask = future_loss_mask[train_anchor_mask]
        flow_train_clean_norm = processor._build_anchor_clean_norm(
            pos=pos,
            heading=heading,
            velocity=velocity,
            current_pos=pos[:, raw_step],
            current_head=heading[:, raw_step],
            agent_type=tokenized_agent["type"],
            agent_length=tokenized_agent["shape"][:, 0],
            anchor_mask=train_anchor_mask,
            raw_step=raw_step,
            future_loss_mask=selected_future_loss_mask,
        )

        flow_train_mask[:, anchor_offset] = train_anchor_mask
        if not bool(train_anchor_mask.any().item()):
            continue

        flow_train_metric_norm = processor._build_anchor_clean_norm(
            pos=pos,
            heading=heading,
            velocity=velocity,
            current_pos=pos[:, raw_step],
            current_head=heading[:, raw_step],
            agent_type=tokenized_agent["type"],
            agent_length=tokenized_agent["shape"][:, 0],
            anchor_mask=train_anchor_mask,
            raw_step=raw_step,
            future_loss_mask=selected_future_loss_mask,
            force_pose_space=True,
        )
        flow_train_chunks.append(flow_train_clean_norm)
        flow_train_metric_chunks.append(flow_train_metric_norm)
        flow_train_loss_mask_chunks.append(selected_future_loss_mask)
        flow_train_agent_type_chunks.append(tokenized_agent["type"][train_anchor_mask])
        flow_train_agent_length_chunks.append(tokenized_agent["shape"][train_anchor_mask, 0])
        flow_train_current_speed_chunks.append(
            torch.linalg.vector_norm(velocity[train_anchor_mask, raw_step, :2], dim=-1)
        )

    processor._assert_flow_train_anchor_context_valid(
        flow_train_mask=flow_train_mask,
        ctx_valid=ctx_valid,
    )
    return {
        "flow_train_mask": flow_train_mask,
        "flow_train_clean_norm": processor._concat_flow_chunks(
            chunks=flow_train_chunks,
            dtype=pos.dtype,
            device=pos.device,
        ),
        "flow_train_clean_metric_norm": processor._concat_flow_chunks(
            chunks=flow_train_metric_chunks,
            dtype=pos.dtype,
            device=pos.device,
            target_dim=4,
        ),
        "flow_train_loss_mask": processor._concat_mask_chunks(
            chunks=flow_train_loss_mask_chunks,
            device=pos.device,
        ),
        "flow_train_agent_type": processor._concat_vector_chunks(
            chunks=flow_train_agent_type_chunks,
            dtype=tokenized_agent["type"].dtype,
            device=pos.device,
        ),
        "flow_train_agent_length": processor._concat_vector_chunks(
            chunks=flow_train_agent_length_chunks,
            dtype=pos.dtype,
            device=pos.device,
        ),
        "flow_train_current_speed": processor._concat_vector_chunks(
            chunks=flow_train_current_speed_chunks,
            dtype=pos.dtype,
            device=pos.device,
        ),
    }


def _build_processed_agent_for_full_womd_horizon() -> dict[str, torch.Tensor]:
    raw_step = torch.arange(91, dtype=torch.float32)
    pos = torch.zeros((1, 91, 2), dtype=torch.float32)
    pos[0, :, 0] = raw_step
    velocity = torch.zeros_like(pos)
    velocity[:, 1:, 0] = 10.0
    return {
        "valid": torch.ones((1, 91), dtype=torch.bool),
        "pos": pos,
        "heading": torch.zeros((1, 91), dtype=torch.float32),
        "velocity": velocity,
    }


def test_prefix_valid_future_loss_mask_keeps_only_continuous_prefix() -> None:
    """가까운 미래부터 처음 끊기기 전까지만 True로 남는지 확인합니다."""
    processor = _build_processor(use_prefix_valid_future_loss_mask=True)
    # valid: [n_agent, n_step]
    valid = torch.tensor(
        [
            [True, True, True, True, False, True, True],
            [True, True, False, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    # raw_step=1이면 future는 step 2부터 최대 5개입니다.
    loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=1)

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(loss_mask, expected)


def test_full_window_future_loss_mask_keeps_original_behavior() -> None:
    """옵션이 꺼져 있으면 전체 미래가 유효한 경우에만 모두 True인지 확인합니다."""
    processor = _build_processor(use_prefix_valid_future_loss_mask=False)
    # valid: [n_agent, n_step]
    valid = torch.tensor(
        [
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=1)

    expected = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(loss_mask, expected)


def test_control_round_trip_keep_mask_filters_only_large_valid_step_error() -> None:
    """유효한 미래 step에서만 5m 초과 복원 오차 anchor를 제거하는지 확인합니다."""
    processor = _build_processor(use_prefix_valid_future_loss_mask=True)

    round_trip_error_m = torch.tensor(
        [
            [0.0, 4.9, 9.0, 0.0, 0.0],
            [0.0, 5.1, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0],
        ]
    )
    future_loss_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    keep_mask = processor._build_control_round_trip_keep_mask(
        round_trip_error_m=round_trip_error_m,
        future_loss_mask=future_loss_mask,
    )

    expected = torch.tensor([True, False, True], dtype=torch.bool)
    assert torch.equal(keep_mask, expected)


def test_flow_targets_use_18_context_and_16_prefix_valid_anchors() -> None:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask=True)

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=_build_tokenized_agent_for_18_context(),
        processed_agent=_build_processed_agent_for_full_womd_horizon(),
    )

    assert tuple(out["ctx_sampled_idx"].shape) == (1, 18)
    assert tuple(out["flow_train_mask"].shape) == (1, 16)
    assert int(out["flow_train_mask"].sum().item()) == 16
    assert tuple(out["flow_train_clean_norm"].shape) == (16, 20, 4)
    torch.testing.assert_close(
        out["flow_train_loss_mask"].sum(dim=1).cpu(),
        torch.tensor([20] * 13 + [15, 10, 5]),
    )


def test_flow_targets_keep_16_anchor_slots_for_full_window_mode() -> None:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask=False)

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=_build_tokenized_agent_for_18_context(),
        processed_agent=_build_processed_agent_for_full_womd_horizon(),
    )

    assert tuple(out["ctx_sampled_idx"].shape) == (1, 18)
    assert tuple(out["flow_train_mask"].shape) == (1, 16)
    assert out["flow_train_mask"].tolist() == [[True] * 13 + [False] * 3]
    torch.testing.assert_close(
        out["flow_train_loss_mask"].sum(dim=1).cpu(),
        torch.tensor([20] * 13),
    )


def test_batched_kinematic_flow_targets_match_anchor_loop_reference() -> None:
    processor = _build_kinematic_target_processor(use_prefix_valid_future_loss_mask=True)
    data = {"agent": {"train_mask": torch.tensor([True, True, True, True])}}
    tokenized_agent = _build_multi_agent_tokenized_agent()
    processed_agent = _build_multi_agent_processed_agent()

    expected = _build_flow_targets_anchor_loop_reference(
        processor=processor,
        data=data,
        tokenized_agent=_clone_tensor_dict(tokenized_agent),
        processed_agent=_clone_tensor_dict(processed_agent),
    )
    actual = processor._build_flow_targets(
        data=data,
        tokenized_agent=_clone_tensor_dict(tokenized_agent),
        processed_agent=_clone_tensor_dict(processed_agent),
    )

    assert torch.equal(actual["flow_train_mask"], expected["flow_train_mask"])
    assert torch.equal(actual["flow_train_loss_mask"], expected["flow_train_loss_mask"])
    assert torch.equal(actual["flow_train_agent_type"], expected["flow_train_agent_type"])
    torch.testing.assert_close(actual["flow_train_agent_length"], expected["flow_train_agent_length"])
    torch.testing.assert_close(actual["flow_train_current_speed"], expected["flow_train_current_speed"])
    torch.testing.assert_close(actual["flow_train_clean_norm"], expected["flow_train_clean_norm"])
    torch.testing.assert_close(
        actual["flow_train_clean_metric_norm"],
        expected["flow_train_clean_metric_norm"],
    )


def test_anchor_context_uses_mask_width_and_ignores_extra_tail_context() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    encoded = torch.arange(2 * 18 * 3, dtype=torch.float32).view(2, 18, 3)
    decoder._encode_context = lambda **kwargs: encoded
    tokenized_agent = {
        "ctx_sampled_idx": torch.zeros((2, 18), dtype=torch.long),
        "ctx_sampled_pos": torch.zeros((2, 18, 2), dtype=torch.float32),
        "ctx_sampled_heading": torch.zeros((2, 18), dtype=torch.float32),
        "ctx_valid": torch.ones((2, 18), dtype=torch.bool),
    }

    out = decoder.build_anchor_context(
        tokenized_agent=tokenized_agent,
        map_feature={},
        anchor_mask=torch.ones((2, 16), dtype=torch.bool),
        flow_clean_norm=torch.zeros((32, 20, 4), dtype=torch.float32),
    )

    assert tuple(out["ctx_hidden_pack"].shape) == (2, 18, 3)
    assert tuple(out["anchor_hidden"].shape) == (2, 16, 3)
    torch.testing.assert_close(out["anchor_hidden"], encoded[:, 1:17])
