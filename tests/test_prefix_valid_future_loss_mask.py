from __future__ import annotations

import torch

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
    return processor


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
