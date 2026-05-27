from __future__ import annotations

from types import SimpleNamespace

import torch

from src.data_preprocess import decode_map_features_from_proto, get_map_features
from src.smart.modules.map_decoder import SMARTMapDecoder


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
