from __future__ import annotations

import torch

from src.smart.model.smart_flow import SMARTFlow


def test_parallel_rollout_map_feature_repeats_light_type_with_map_tokens() -> None:
    model = SMARTFlow.__new__(SMARTFlow)
    map_feature = {
        "pt_token": torch.tensor([[1.0], [2.0]]),
        "position": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        "orientation": torch.tensor([0.0, 1.0]),
        "light_type": torch.tensor([2, 0]),
        "batch": torch.tensor([0, 1]),
    }

    expanded = model._build_parallel_rollout_map_feature(
        map_feature=map_feature,
        repeat_count=3,
        num_graphs=2,
    )

    assert expanded["pt_token"].flatten().tolist() == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert expanded["light_type"].tolist() == [2, 0, 2, 0, 2, 0]
    assert expanded["batch"].tolist() == [0, 1, 2, 3, 4, 5]
