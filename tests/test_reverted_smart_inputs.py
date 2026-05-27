import torch

from src.data_preprocess import decode_map_features_from_proto
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.map_decoder import SMARTMapDecoder, _fold_legacy_surface_categories


class _Point:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _Surface:
    def __init__(self):
        self.polygon = [
            _Point(0.0, 0.0),
            _Point(1.0, 0.0),
            _Point(1.0, 1.0),
            _Point(0.0, 1.0),
        ]


class _MapFeature:
    def __init__(self, feature_id: int, feature_type: str):
        self.id = feature_id
        self._feature_type = feature_type
        setattr(self, feature_type, _Surface())

    def WhichOneof(self, name: str) -> str:
        assert name == "feature_data"
        return self._feature_type


def test_surface_map_categories_are_folded_to_crosswalk_style_cache() -> None:
    map_infos = decode_map_features_from_proto(
        [
            _MapFeature(1, "crosswalk"),
            _MapFeature(2, "speed_bump"),
            _MapFeature(3, "driveway"),
        ]
    )

    assert sorted(map_infos.keys()) == [
        "all_polylines",
        "crosswalk",
        "lane",
        "road_edge",
        "road_line",
    ]
    assert len(map_infos["crosswalk"]) == 3
    assert {int(info["type"]) for info in map_infos["crosswalk"]} == {9}
    assert set(map_infos["all_polylines"][:, 3].astype(int).tolist()) == {9}


def test_smart_map_decoder_uses_coarse_surface_embedding_ranges() -> None:
    decoder = SMARTMapDecoder(
        hidden_dim=8,
        pl2pl_radius=10.0,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
    )

    assert decoder.type_pt_emb.num_embeddings == 10
    assert decoder.polygon_type_emb.num_embeddings == 4


def test_smart_map_decoder_folds_new_surface_cache_values_to_legacy_range() -> None:
    point_type, polygon_type = _fold_legacy_surface_categories(
        torch.tensor([0, 9, 10, 11]),
        torch.tensor([0, 3, 4, 5]),
    )

    assert point_type.tolist() == [0, 9, 9, 9]
    assert polygon_type.tolist() == [0, 3, 3, 3]


def test_smart_agent_decoder_uses_two_dimensional_motion_and_no_stale_light_relation() -> None:
    decoder = SMARTAgentDecoder(
        hidden_dim=8,
        num_historical_steps=11,
        num_future_steps=80,
        time_span=30,
        pl2a_radius=5.0,
        a2a_radius=60.0,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        hist_drop_prob=0.0,
        n_token_agent=5,
    )

    assert decoder.x_a_emb.input_dim == 2
    assert not hasattr(decoder, "light_pl2a_emb")
    assert not hasattr(decoder, "light_time_pl2a_emb")
