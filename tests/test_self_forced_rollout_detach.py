from __future__ import annotations

import torch

from src.smart.modules.self_forced_rollout_detach import (
    detach_tensor_tree,
    detach_training_rollout_state,
)


def test_detach_tensor_tree_keeps_tensor_shape_and_value() -> None:
    source = torch.randn(3, 7, 4, requires_grad=True)

    detached = detach_tensor_tree(source)

    assert tuple(detached.shape) == tuple(source.shape)
    assert detached.data_ptr() == source.data_ptr()
    assert not detached.requires_grad
    assert torch.allclose(detached, source)


def test_detach_tensor_tree_handles_nested_containers() -> None:
    source = torch.ones(2, 5, 2, requires_grad=True)
    nested = {
        "pos_window": source * 2.0,
        "feat_a_t_dict": {
            "layer0": source[..., 0],
            "layer1": [source[..., 1]],
        },
        "plain_value": "keep",
    }

    detached = detach_tensor_tree(nested)

    assert not detached["pos_window"].requires_grad
    assert not detached["feat_a_t_dict"]["layer0"].requires_grad
    assert not detached["feat_a_t_dict"]["layer1"][0].requires_grad
    assert detached["plain_value"] == "keep"


def test_detach_training_rollout_state_detaches_all_explicit_state_values() -> None:
    source = torch.ones(2, 5, 2, requires_grad=True)
    rollout_state = {
        "pos_window": source * 2.0,
        "head_window": source[..., 0],
        "exec_pos_history_10hz": source * 3.0,
    }

    detached_state = detach_training_rollout_state(rollout_state)

    assert set(detached_state.keys()) == set(rollout_state.keys())
    assert not detached_state["pos_window"].requires_grad
    assert not detached_state["head_window"].requires_grad
    assert not detached_state["exec_pos_history_10hz"].requires_grad
