from __future__ import annotations

from types import SimpleNamespace

import torch

from src.data_preprocess import decode_map_features_from_proto, get_map_features
from src.smart.modules.map_decoder import SMARTMapDecoder, _fold_legacy_surface_categories


class _MapFeature:
    def __init__(self, feature_id: int, feature_name: str) -> None:
        self.id = feature_id
        self._feature_name = feature_name
        square = [
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
            SimpleNamespace(x=4.0, y=0.0, z=0.0),
            SimpleNamespace(x=4.0, y=4.0, z=0.0),
            SimpleNamespace(x=0.0, y=4.0, z=0.0),
        ]
        setattr(self, feature_name, SimpleNamespace(polygon=square))

    def WhichOneof(self, _: str) -> str:
        return self._feature_name


def test_surface_map_categories_are_merged_into_crosswalk_cache() -> None:
    map_infos = decode_map_features_from_proto(
        [
            _MapFeature(1, "crosswalk"),
            _MapFeature(2, "speed_bump"),
            _MapFeature(3, "driveway"),
        ]
    )

    assert list(map_infos.keys()) == [
        "lane",
        "road_edge",
        "road_line",
        "crosswalk",
        "all_polylines",
    ]
    assert [entry["id"] for entry in map_infos["crosswalk"]] == [1, 2, 3]
    assert {entry["type"] for entry in map_infos["crosswalk"]} == {9}

    empty_lights = {"lane_id": torch.empty(0, dtype=torch.long), "state": torch.empty(0)}
    map_data = get_map_features(map_infos, empty_lights)
    assert int(map_data["map_point"]["type"].max()) == 9
    assert set(map_data["map_polygon"]["type"].tolist()) == {3}


def test_map_decoder_uses_pre_category_vocab_sizes() -> None:
    decoder = SMARTMapDecoder(
        hidden_dim=16,
        pl2pl_radius=50.0,
        num_freq_bands=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
    )

    assert decoder.type_pt_emb.num_embeddings == 10
    assert decoder.polygon_type_emb.num_embeddings == 4


def test_latest_cache_surface_categories_are_folded_before_embedding() -> None:
    point_type = torch.tensor([0, 8, 9, 10, 11], dtype=torch.long)
    polygon_type = torch.tensor([0, 2, 3, 4, 5], dtype=torch.long)

    folded_point_type, folded_polygon_type = _fold_legacy_surface_categories(
        point_type=point_type,
        polygon_type=polygon_type,
    )

    assert folded_point_type.tolist() == [0, 8, 9, 9, 9]
    assert folded_polygon_type.tolist() == [0, 2, 3, 3, 3]

    decoder = SMARTMapDecoder(
        hidden_dim=16,
        pl2pl_radius=50.0,
        num_freq_bands=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
    )
    decoder.type_pt_emb(folded_point_type)
    decoder.polygon_type_emb(folded_polygon_type)


def test_map_decoder_accepts_latest_cache_surface_category_ids() -> None:
    decoder = SMARTMapDecoder(
        hidden_dim=16,
        pl2pl_radius=50.0,
        num_freq_bands=4,
        num_layers=0,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
    )

    output = decoder(
        {
            "position": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "orientation": torch.zeros(2, dtype=torch.float32),
            "token_traj_src": torch.zeros(1, 22, dtype=torch.float32),
            "token_idx": torch.zeros(2, dtype=torch.long),
            "type": torch.tensor([10, 11], dtype=torch.long),
            "pl_type": torch.tensor([4, 5], dtype=torch.long),
            "light_type": torch.zeros(2, dtype=torch.long),
            "batch": torch.zeros(2, dtype=torch.long),
        }
    )

    assert output["pt_token"].shape == (2, 16)
