from __future__ import annotations

import torch

from src.smart.tokens.flow_token_processor import (
    FLOW_TRAIN_ANCHOR_COUNT,
    FlowTokenProcessor,
)


def _build_processor() -> FlowTokenProcessor:
    """토큰 파일을 읽지 않고 anchor rollout init helper만 테스트할 processor를 만듭니다."""
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.training = False
    processor.shift = 5
    processor.flow_window_steps = 20
    return processor


def _build_raw_agent_series(
    num_agent: int = 3,
    num_step: int = 91,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    torch.manual_seed(0)
    pos = torch.randn(num_agent, num_step, 2)
    heading = torch.randn(num_agent, num_step)
    valid = torch.rand(num_agent, num_step) > 0.2
    position_3d = torch.cat([pos, torch.randn(num_agent, num_step, 1)], dim=-1)
    data = {"agent": {"position": position_3d}}
    return pos, heading, valid, data


def _anchor_raw_steps(shift: int = 5) -> list[int]:
    return [shift * (anchor_idx + 2) for anchor_idx in range(FLOW_TRAIN_ANCHOR_COUNT)]


def test_anchor_rollout_init_states_shapes() -> None:
    processor = _build_processor()
    pos, heading, valid, data = _build_raw_agent_series()
    states = processor._build_anchor_rollout_init_states(
        data=data,
        valid=valid,
        pos=pos,
        heading=heading,
        raw_current_steps=_anchor_raw_steps(),
    )
    num_agent = pos.shape[0]
    n_anchor = FLOW_TRAIN_ANCHOR_COUNT
    history_len = processor.shift + 1
    assert states["sf_anchor_fine_pos_history"].shape == (num_agent, n_anchor, history_len, 2)
    assert states["sf_anchor_fine_head_history"].shape == (num_agent, n_anchor, history_len)
    assert states["sf_anchor_fine_valid_history"].shape == (num_agent, n_anchor, history_len)
    assert states["sf_anchor_z"].shape == (num_agent, n_anchor)


def test_anchor0_matches_existing_rollout_init_fine_history() -> None:
    """anchor 0 슬라이스는 기존 anchor0 전용 helper와 정확히 같아야 합니다."""
    processor = _build_processor()
    pos, heading, valid, data = _build_raw_agent_series()
    states = processor._build_anchor_rollout_init_states(
        data=data,
        valid=valid,
        pos=pos,
        heading=heading,
        raw_current_steps=_anchor_raw_steps(),
    )
    (
        expected_pos_history,
        expected_head_history,
        expected_valid_history,
    ) = processor._build_rollout_init_fine_history(valid=valid, pos=pos, heading=heading)
    (
        expected_pos_pair,
        expected_head_pair,
        expected_valid_pair,
    ) = processor._build_rollout_init_fine_pair(valid=valid, pos=pos, heading=heading)

    torch.testing.assert_close(states["sf_anchor_fine_pos_history"][:, 0], expected_pos_history)
    torch.testing.assert_close(states["sf_anchor_fine_head_history"][:, 0], expected_head_history)
    assert torch.equal(states["sf_anchor_fine_valid_history"][:, 0], expected_valid_history)
    torch.testing.assert_close(
        states["sf_anchor_fine_pos_history"][:, 0, -2:], expected_pos_pair
    )
    torch.testing.assert_close(
        states["sf_anchor_fine_head_history"][:, 0, -2:], expected_head_pair
    )
    assert torch.equal(states["sf_anchor_fine_valid_history"][:, 0, -2:], expected_valid_pair)


def test_anchor_k_matches_manual_raw_slice() -> None:
    """anchor k의 fine history는 raw step ``5k+5..5k+10`` 슬라이스와 같아야 합니다."""
    processor = _build_processor()
    pos, heading, valid, data = _build_raw_agent_series()
    states = processor._build_anchor_rollout_init_states(
        data=data,
        valid=valid,
        pos=pos,
        heading=heading,
        raw_current_steps=_anchor_raw_steps(),
    )
    for anchor_idx in [1, 4, 12, 15]:
        raw_step = 5 * (anchor_idx + 2)
        torch.testing.assert_close(
            states["sf_anchor_fine_pos_history"][:, anchor_idx],
            pos[:, raw_step - 5 : raw_step + 1],
        )
        torch.testing.assert_close(
            states["sf_anchor_fine_head_history"][:, anchor_idx],
            heading[:, raw_step - 5 : raw_step + 1],
        )
        assert torch.equal(
            states["sf_anchor_fine_valid_history"][:, anchor_idx],
            valid[:, raw_step - 5 : raw_step + 1],
        )
        torch.testing.assert_close(
            states["sf_anchor_z"][:, anchor_idx],
            data["agent"]["position"][:, raw_step, 2],
        )


def test_anchor_z_anchor0_matches_gt_z_raw_rule() -> None:
    """anchor 0의 z는 기존 ``gt_z_raw`` 규칙(raw step 10의 z)과 같아야 합니다."""
    processor = _build_processor()
    pos, heading, valid, data = _build_raw_agent_series()
    states = processor._build_anchor_rollout_init_states(
        data=data,
        valid=valid,
        pos=pos,
        heading=heading,
        raw_current_steps=_anchor_raw_steps(),
    )
    torch.testing.assert_close(
        states["sf_anchor_z"][:, 0],
        data["agent"]["position"][:, 10, 2],
    )
