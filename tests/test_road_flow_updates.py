from types import SimpleNamespace

import torch

from src.smart.road.cache import build_selected_epoch_cache
from src.smart.road.generator import (
    RoadGenerationConfig,
    extract_rollout_prediction,
    select_epoch_source_paths,
)
from src.smart.utils.finetune import set_model_for_finetuning


class _DummyFlowEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = torch.nn.Linear(2, 2)
        self.agent_encoder = torch.nn.Module()
        self.agent_encoder.flow_decoder = torch.nn.Linear(2, 2)
        self.other = torch.nn.Linear(2, 2)


def test_road_finetune_freezes_only_smart_map_decoder() -> None:
    model = _DummyFlowEncoder()

    set_model_for_finetuning(
        model,
        SimpleNamespace(enabled=True, freeze_smart_map_decoder_only=True),
    )

    assert all(not p.requires_grad for p in model.map_encoder.parameters())
    trainable_non_map = [
        p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith("map_encoder.")
    ]
    assert trainable_non_map
    assert all(trainable_non_map)


def test_road_data_use_ratio_selects_epoch_subset(tmp_path) -> None:
    source_paths = []
    for idx in range(10):
        path = tmp_path / f"scenario_{idx:02d}.pkl"
        path.write_bytes(b"sample")
        source_paths.append(path)

    config = RoadGenerationConfig(road_data_use_ratio=0.25, seed=123)
    selected = select_epoch_source_paths(source_paths, config=config, epoch_idx=0)

    assert len(selected) == 3
    assert all(path in source_paths for path in selected)


def test_selected_epoch_cache_links_all_rollout_variants(tmp_path) -> None:
    variant_dirs = [tmp_path / f"variant_{idx:02d}" for idx in range(3)]
    for variant_idx, variant_dir in enumerate(variant_dirs):
        variant_dir.mkdir()
        for scenario_id in ("scenario_a", "scenario_b"):
            (variant_dir / f"{scenario_id}.pkl").write_text(
                f"{scenario_id}:{variant_idx}",
                encoding="utf-8",
            )

    selected_dir = tmp_path / "selected"
    made = build_selected_epoch_cache(
        variant_dirs=variant_dirs,
        selected_dir=selected_dir,
        epoch_idx=0,
        seed=817,
        rank=0,
        world_size=1,
    )

    assert made == 6
    assert sorted(path.name for path in selected_dir.glob("*.pkl")) == [
        "scenario_a__road_r00.pkl",
        "scenario_a__road_r01.pkl",
        "scenario_a__road_r02.pkl",
        "scenario_b__road_r00.pkl",
        "scenario_b__road_r01.pkl",
        "scenario_b__road_r02.pkl",
    ]


def test_rollout_prediction_expands_coarse_valid_to_10hz() -> None:
    prediction = {
        "pred_traj_10hz": torch.zeros(2, 20, 2),
        "pred_head_10hz": torch.zeros(2, 20),
        "pred_valid": torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
            ]
        ),
    }

    _, _, valid = extract_rollout_prediction(prediction)

    assert valid.shape == (2, 20)
    assert valid[0].tolist() == [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
    assert valid[1].tolist() == [False] * 5 + [True] * 5 + [False] * 5 + [True] * 5


def test_rollout_prediction_uses_future_coarse_valid_when_context_is_present() -> None:
    prediction = {
        "pred_traj_10hz": torch.zeros(1, 20, 2),
        "pred_head_10hz": torch.zeros(1, 20),
        "pred_valid": torch.tensor([[False, False, True, False, True, False]]),
    }

    _, _, valid = extract_rollout_prediction(prediction)

    assert valid.shape == (1, 20)
    assert valid[0].tolist() == [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
