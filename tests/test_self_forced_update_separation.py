from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.smart.modules.self_forced_update_separation import (
    assert_no_module_gradients,
    clear_module_gradients,
    detach_tensor_tree,
    module_gradients_disabled,
)


def test_module_gradients_disabled_restores_trainable_mask() -> None:
    module = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    module[0].weight.requires_grad_(False)
    expected_requires_grad = [parameter.requires_grad for parameter in module.parameters()]

    with module_gradients_disabled(module):
        assert not any(parameter.requires_grad for parameter in module.parameters())

    assert [parameter.requires_grad for parameter in module.parameters()] == expected_requires_grad


def test_module_gradients_disabled_blocks_parameter_gradients() -> None:
    module = nn.Linear(2, 1)
    x = torch.ones(1, 2, requires_grad=True)

    with module_gradients_disabled(module):
        module(x).sum().backward()

    assert x.grad is not None
    assert all(parameter.grad is None for parameter in module.parameters())


def test_detach_tensor_tree_breaks_nested_autograd_history() -> None:
    source = torch.ones(2, requires_grad=True)
    nested = {
        "path": source * 2.0,
        "context": [source + 3.0, (source - 1.0, "metadata")],
    }

    detached = detach_tensor_tree(nested)
    estimator_weight = torch.ones(2, requires_grad=True)
    loss = (detached["path"] * estimator_weight).sum()
    loss.backward()

    assert source.grad is None
    assert estimator_weight.grad is not None
    assert detached["path"].grad_fn is None
    assert detached["context"][0].grad_fn is None
    assert detached["context"][1][0].grad_fn is None
    assert detached["context"][1][1] == "metadata"


def test_assert_no_module_gradients_reports_and_clear_removes_gradients() -> None:
    module = nn.Linear(2, 1)
    module(torch.ones(1, 2)).sum().backward()

    with pytest.raises(RuntimeError, match="Unexpected gradient on online Generator"):
        assert_no_module_gradients(module, "online Generator", "generated-estimator update")

    clear_module_gradients(module)
    assert_no_module_gradients(module, "online Generator", "generated-estimator update")
