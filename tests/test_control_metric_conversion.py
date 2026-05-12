from __future__ import annotations

import pytest
import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder


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
