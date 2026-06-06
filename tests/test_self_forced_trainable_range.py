from __future__ import annotations

import torch.nn as nn

from src.smart.modules.self_forced_trainable_range import (
    apply_self_forced_unfrozen_range,
    collect_trainable_parameter_names,
    resolve_self_forced_unfrozen_range,
)


class DummyAgentEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent_token_embedding = nn.Linear(4, 4)
        self.t_attn_layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.pt2a_attn_layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.a2a_attn_layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.flow_decoder = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )


class DummySMARTFlowDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = nn.Linear(4, 4)
        self.agent_encoder = DummyAgentEncoder()


def _trainable_count(model: nn.Module) -> int:
    """학습 가능한 파라미터 원소 개수를 셉니다.

    Args:
        model: 확인할 PyTorch 모듈입니다.

    Returns:
        int: ``requires_grad=True`` 인 파라미터 원소 개수입니다.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_resolve_self_forced_unfrozen_range_uses_default() -> None:
    assert resolve_self_forced_unfrozen_range(None) == "middle"


def test_except_map_encoder_freezes_only_map_encoder() -> None:
    model = DummySMARTFlowDecoder()

    apply_self_forced_unfrozen_range(model, "except_map_encoder")
    trainable_names = collect_trainable_parameter_names(model)

    assert all(not name.startswith("map_encoder") for name in trainable_names)
    assert any(name.startswith("agent_encoder.agent_token_embedding") for name in trainable_names)
    assert any(name.startswith("agent_encoder.flow_decoder") for name in trainable_names)


def test_full_flow_decoder_only_unfreezes_flow_decoder() -> None:
    model = DummySMARTFlowDecoder()

    apply_self_forced_unfrozen_range(model, "full_flow_decoder")
    trainable_names = collect_trainable_parameter_names(model)

    assert len(trainable_names) > 0
    assert all(name.startswith("agent_encoder.flow_decoder") for name in trainable_names)


def test_middle_is_between_except_map_encoder_and_full_flow_decoder() -> None:
    except_model = DummySMARTFlowDecoder()
    middle_model = DummySMARTFlowDecoder()
    full_flow_model = DummySMARTFlowDecoder()

    apply_self_forced_unfrozen_range(except_model, "except_map_encoder")
    apply_self_forced_unfrozen_range(middle_model, "middle")
    apply_self_forced_unfrozen_range(full_flow_model, "full_flow_decoder")

    except_count = _trainable_count(except_model)
    middle_count = _trainable_count(middle_model)
    full_flow_count = _trainable_count(full_flow_model)

    assert except_count > middle_count > full_flow_count

    trainable_names = collect_trainable_parameter_names(middle_model)
    assert any(name.startswith("agent_encoder.flow_decoder") for name in trainable_names)
    assert any(name.startswith("agent_encoder.t_attn_layers.1") for name in trainable_names)
    assert any(name.startswith("agent_encoder.pt2a_attn_layers.1") for name in trainable_names)
    assert any(name.startswith("agent_encoder.a2a_attn_layers.1") for name in trainable_names)
    assert not any(name.startswith("agent_encoder.t_attn_layers.0") for name in trainable_names)
    assert not any(name.startswith("agent_encoder.agent_token_embedding") for name in trainable_names)
