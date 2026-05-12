from __future__ import annotations

import pytest
import torch

from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder


def test_motion_feature_marks_missing_motion_separately_from_stationary_motion() -> None:
    pos = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [2.0, 0.0]],
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, True],
            [True, True, True, True],
        ]
    )
    head_vector = torch.zeros_like(pos)
    head_vector[..., 0] = 1.0

    motion_vector = SMARTAgentEncoder._build_motion_vector(pos, valid)
    motion_valid = SMARTAgentEncoder._build_motion_valid_mask(pos, valid)
    motion_feature = SMARTAgentEncoder._build_motion_feature(pos, head_vector, valid)

    assert motion_feature.shape == (2, 4, 3)
    assert motion_valid.tolist() == [
        [False, True, False, False],
        [False, True, True, True],
    ]
    assert torch.allclose(motion_vector[0, 2], torch.zeros(2))
    assert torch.allclose(motion_vector[0, 3], torch.zeros(2))

    # Both have zero-valued motion, but the validity bit keeps them separable.
    assert motion_feature[0, 3, 2].item() == 0.0
    assert motion_feature[1, 3, 2].item() == 1.0


def test_recent_coarse_motion_returns_value_and_validity() -> None:
    pos_window = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 4.0]],
            [[0.0, 0.0], [10.0, 0.0]],
        ]
    )
    valid_window = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, True],
        ]
    )

    recent_motion, recent_motion_valid = SMARTFlowAgentDecoder._build_recent_coarse_motion(
        pos_window=pos_window,
        valid_window=valid_window,
    )

    assert recent_motion_valid.tolist() == [True, True, False]
    assert torch.allclose(recent_motion[0], torch.zeros(2))
    assert torch.allclose(recent_motion[1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(recent_motion[2], torch.zeros(2))


def test_external_motion_requires_validity_mask() -> None:
    pos = torch.zeros(2, 1, 2)
    head = torch.zeros(2, 1)
    head_vector = torch.zeros(2, 1, 2)
    head_vector[..., 0] = 1.0
    mask = torch.ones(2, 1, dtype=torch.bool)
    batch_s = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="motion_valid_a is required"):
        SMARTFlowAgentDecoder.build_interaction_edge(
            None,
            pos_a=pos,
            head_a=head,
            head_vector_a=head_vector,
            batch_s=batch_s,
            mask=mask,
            motion_a=pos,
        )
