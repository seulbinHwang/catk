import pickle
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from src.smart.road.cache import (
    ROAD_UNUSED_AGENT_FIELDS,
    build_road_cache_sample,
    build_selected_epoch_cache,
)
from src.smart.road.generator import (
    RoadGenerationConfig,
    _split_repeated_rollout_by_sample,
    _to_repeated_batch_for_samples,
    extract_rollout_prediction,
    generate_road_epoch_cache,
    select_epoch_source_paths,
)
from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.utils.finetune import set_model_for_finetuning


class _DummyFlowEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = torch.nn.Linear(2, 2)
        self.agent_encoder = torch.nn.Module()
        self.agent_encoder.flow_decoder = torch.nn.Linear(2, 2)
        self.other = torch.nn.Linear(2, 2)


class _DummyGenerationModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


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


def test_road_flow_decoder_schema_matches_x5f9g0ce_pretrain_relations() -> None:
    decoder = SMARTFlowAgentDecoder(
        hidden_dim=128,
        num_historical_steps=11,
        num_future_steps=80,
        flow_window_steps=20,
        time_span=30,
        pl2a_radius=30.0,
        a2a_radius=60.0,
        num_freq_bands=64,
        num_layers=1,
        num_heads=8,
        head_dim=15,
        dropout=0.1,
        hist_drop_prob=0.1,
        n_token_agent=4,
        flow_dim=96,
        flow_num_chunk_heads=4,
        flow_num_chunk_layers=1,
        flow_solver_steps=16,
        flow_solver_method="euler",
        flow_solver_eps=1.0e-3,
        use_kinematic_control_flow=True,
        use_holonomic_model_only=False,
        use_rolling_supervision=True,
        control_pos_scale_m=1.0,
        control_vehicle_no_slip_point_ratio=0.2289518863,
        control_cyclist_no_slip_point_ratio=0.0495847873,
        control_vehicle_yaw_scale_rad=0.025,
        control_pedestrian_yaw_scale_rad=0.20,
        control_cyclist_yaw_scale_rad=0.06,
    )

    assert decoder.flow_state_dim == 3
    assert decoder.x_a_emb.freqs.weight.shape == (3, 64)
    assert decoder.r_pt2a_emb.freqs.weight.shape == (3, 64)
    assert decoder.r_a2a_emb.freqs.weight.shape == (3, 64)
    assert decoder.token_emb_veh.mlp[0].weight.shape == (128, 48)
    assert decoder.flow_decoder.noisy_future_encoder.step_proj.weight.shape == (96, 3)
    assert decoder.flow_decoder.noisy_future_encoder.step_embed.weight.shape == (20, 96)
    assert decoder.flow_decoder.velocity_head.net[2].weight.shape == (3, 96)
    assert hasattr(decoder, "light_pl2a_emb")
    assert decoder.light_time_pl2a_emb.freqs.weight.shape == (1, 64)


def test_split_repeated_rollout_by_sample_handles_variable_agent_counts() -> None:
    repeat_count = 2
    agent_counts = [2, 3]
    total_agents = repeat_count * sum(agent_counts)
    xy = torch.arange(total_agents * 4 * 2, dtype=torch.float32).reshape(total_agents, 4, 2)
    heading = torch.arange(total_agents * 4, dtype=torch.float32).reshape(total_agents, 4)
    valid = torch.ones(total_agents, 4, dtype=torch.bool)

    outputs = _split_repeated_rollout_by_sample(
        xy=xy,
        heading=heading,
        valid=valid,
        agent_counts=agent_counts,
        repeat_count=repeat_count,
    )

    assert len(outputs) == 2
    assert outputs[0][0].shape == (2, 2, 4, 2)
    assert outputs[1][0].shape == (2, 3, 4, 2)
    assert torch.equal(outputs[0][0].reshape(4, 4, 2), xy[:4])
    assert torch.equal(outputs[1][0].reshape(6, 4, 2), xy[4:])


def _make_source_sample(num_agents: int = 2) -> dict:
    position = torch.zeros(num_agents, 91, 3)
    heading = torch.zeros(num_agents, 91)
    velocity = torch.zeros(num_agents, 91, 2)
    valid_mask = torch.ones(num_agents, 91, dtype=torch.bool)
    shape = torch.ones(num_agents, 3)
    return {
        "agent": {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "valid_mask": valid_mask,
            "shape": shape,
        },
    }


def test_road_cache_sample_drops_stale_control_side_fields(tmp_path) -> None:
    sample = _make_source_sample(num_agents=2)
    sample["scenario_id"] = "scenario_with_control_sidecars"
    sample["agent"]["control_aligned_future_pos"] = torch.ones(2, 16, 20, 2)
    sample["agent"]["control_aligned_future_heading"] = torch.ones(2, 16, 20)
    sample["agent"]["control_transition_norm_future"] = torch.ones(2, 16, 3)
    sample["agent"]["control_alignment_cache_key"] = "stale-original-future"

    road_sample = build_road_cache_sample(
        source_sample=sample,
        rollout_xy=torch.full((2, 80, 2), 3.0),
        rollout_heading=torch.full((2, 80), 0.25),
        rollout_valid=torch.ones(2, 80, dtype=torch.bool),
        rollout_index=1,
        source_path=tmp_path / "scenario_with_control_sidecars.pkl",
    )

    for key in ROAD_UNUSED_AGENT_FIELDS:
        assert key not in road_sample["agent"]
    assert road_sample["scenario_id"] == "scenario_with_control_sidecars__road_r01"
    assert torch.equal(
        road_sample["agent"]["position"][:, 11:91, :2],
        torch.full((2, 80, 2), 3.0),
    )
    assert torch.equal(
        road_sample["agent"]["heading"][:, 11:91],
        torch.full((2, 80), 0.25),
    )


def test_generate_road_epoch_cache_uses_generation_batch_size(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for idx in range(5):
        sample = _make_source_sample(num_agents=1 + (idx % 2))
        sample["scenario_id"] = f"scenario_{idx:02d}"
        with (source_dir / f"scenario_{idx:02d}.pkl").open("wb") as handle:
            pickle.dump(sample, handle)

    seen_batch_sizes = []

    def fake_generate_road_rollout_batch(
        model,
        source_samples,
        transform,
        config,
        epoch_idx,
        rollout_idx,
        device,
    ):
        seen_batch_sizes.append(len(source_samples))
        outputs = []
        for sample in source_samples:
            num_agents = sample["agent"]["position"].shape[0]
            outputs.append(
                (
                    torch.zeros(num_agents, 80, 2),
                    torch.zeros(num_agents, 80),
                    torch.ones(num_agents, 80, dtype=torch.bool),
                )
            )
        return outputs

    monkeypatch.setattr(
        "src.smart.road.generator.generate_road_rollout_batch",
        fake_generate_road_rollout_batch,
    )

    generated = generate_road_epoch_cache(
        model=_DummyGenerationModel(),
        source_train_raw_dir=source_dir,
        epoch_dir=tmp_path / "road_epoch",
        transform=lambda sample: sample,
        config=RoadGenerationConfig(
            rollouts_per_scenario=1,
            generation_batch_size=2,
            road_data_use_ratio=1.0,
        ),
        epoch_idx=0,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )

    assert generated == 5
    assert seen_batch_sizes == [2, 2, 1]
    assert len(list((tmp_path / "road_epoch" / "all" / "variant_00").glob("*.pkl"))) == 5


def test_repeated_batch_transforms_each_source_sample_once() -> None:
    source_samples = [_make_source_sample(num_agents=1), _make_source_sample(num_agents=2)]
    transform_calls = 0

    def counted_transform(sample):
        nonlocal transform_calls
        transform_calls += 1
        return HeteroData(sample)

    batch = _to_repeated_batch_for_samples(
        samples=source_samples,
        repeat_count=4,
        transform=counted_transform,
        device=torch.device("cpu"),
    )

    assert transform_calls == 2
    assert batch.num_graphs == 8
    assert batch["agent"]["position"].shape[0] == 12
