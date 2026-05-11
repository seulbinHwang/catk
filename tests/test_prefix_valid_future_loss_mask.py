from __future__ import annotations

import torch

from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def _build_processor(
    use_prefix_valid_future_loss_mask: bool,
    flow_window_steps: int = 5,
) -> FlowTokenProcessor:
    """토큰 파일을 읽지 않고 loss mask helper만 테스트할 processor를 만듭니다.

    Args:
        use_prefix_valid_future_loss_mask: prefix-valid 방식을 사용할지 여부입니다.

    Returns:
        FlowTokenProcessor: ``flow_window_steps``와 옵션만 채운 테스트용 객체입니다.
    """
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.flow_window_steps = flow_window_steps
    processor.shift = 5
    processor.use_prefix_valid_future_loss_mask = use_prefix_valid_future_loss_mask
    processor.control_round_trip_max_position_error_m = 5.0
    return processor


def _valid_row_with_future_prefix(
    prefix_len: int,
    flow_window_steps: int,
    raw_step: int = 1,
) -> torch.Tensor:
    """raw_step 뒤 연속 future valid 길이가 prefix_len인 테스트 row를 만듭니다."""
    total_steps = raw_step + 1 + flow_window_steps
    row = torch.ones(total_steps, dtype=torch.bool)
    if prefix_len < flow_window_steps:
        row[raw_step + 1 + prefix_len] = False
    return row


def _loss_mask_row(usable_len: int, flow_window_steps: int) -> torch.Tensor:
    """앞 usable_len step만 True인 loss mask row를 만듭니다."""
    return torch.arange(flow_window_steps) < int(usable_len)


def test_prefix_valid_future_loss_mask_uses_only_complete_chunks() -> None:
    """연속 valid prefix를 5-step chunk 단위로 내림하는지 확인합니다."""
    flow_window_steps = 20
    processor = _build_processor(
        use_prefix_valid_future_loss_mask=True,
        flow_window_steps=flow_window_steps,
    )
    valid = torch.stack(
        [
            _valid_row_with_future_prefix(prefix_len, flow_window_steps=flow_window_steps)
            for prefix_len in [3, 6, 10, 15, 20]
        ],
        dim=0,
    )

    # raw_step=1이면 future는 step 2부터 최대 20개입니다.
    loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=1)

    expected = torch.stack(
        [
            _loss_mask_row(usable_len, flow_window_steps=flow_window_steps)
            for usable_len in [0, 5, 10, 15, 20]
        ],
        dim=0,
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
