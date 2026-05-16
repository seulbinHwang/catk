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


def _build_tokenized_agent_for_18_context() -> dict[str, torch.Tensor]:
    return {
        "sampled_idx": torch.zeros((1, 18), dtype=torch.long),
        "sampled_pos": torch.zeros((1, 18, 2), dtype=torch.float32),
        "sampled_heading": torch.zeros((1, 18), dtype=torch.float32),
        "valid_mask": torch.ones((1, 18), dtype=torch.bool),
        "type": torch.zeros((1,), dtype=torch.long),
        "shape": torch.tensor([[2.0, 4.8, 1.5]], dtype=torch.float32),
    }


def _build_processed_agent_for_full_womd_horizon() -> dict[str, torch.Tensor]:
    raw_step = torch.arange(91, dtype=torch.float32)
    pos = torch.zeros((1, 91, 2), dtype=torch.float32)
    pos[0, :, 0] = raw_step
    return {
        "valid": torch.ones((1, 91), dtype=torch.bool),
        "pos": pos,
        "heading": torch.zeros((1, 91), dtype=torch.float32),
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
