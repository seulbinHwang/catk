import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, HeteroData

from src.smart.tokens.token_cache import empty_token_cache, load_token_cache
from src.smart.tokens.token_processor import TokenProcessor


def _make_scene(raw_path, scenario_idx: int) -> HeteroData:
    num_agents = 3 + scenario_idx
    num_map = 4 + scenario_idx
    num_steps = 91
    data = HeteroData()
    time = torch.arange(num_steps, dtype=torch.float32)

    positions = []
    headings = []
    velocities = []
    valid_masks = []
    for agent_idx in range(num_agents):
        x = scenario_idx * 100.0 + agent_idx + time * 0.2
        y = agent_idx * 0.5 + time * 0.01
        positions.append(torch.stack([x, y, torch.zeros_like(x)], dim=-1))
        headings.append(torch.full((num_steps,), 0.05 * agent_idx))
        velocities.append(torch.stack([torch.full_like(time, 2.0), torch.zeros_like(time)], dim=-1))
        valid_masks.append(torch.ones(num_steps, dtype=torch.bool))

    data["agent"]["position"] = torch.stack(positions)
    data["agent"]["heading"] = torch.stack(headings)
    data["agent"]["velocity"] = torch.stack(velocities)
    data["agent"]["valid_mask"] = torch.stack(valid_masks)
    data["agent"]["role"] = torch.zeros(num_agents, 3, dtype=torch.bool)
    data["agent"]["role"][0, 0] = True
    data["agent"]["type"] = torch.tensor([0, 1, 2, 0, 1][:num_agents], dtype=torch.long)
    data["agent"]["shape"] = torch.tensor([[4.8, 2.0, 1.6]] * num_agents)
    data["agent"]["id"] = torch.arange(num_agents) + scenario_idx * 1000
    data["agent"].num_nodes = num_agents

    map_base = torch.arange(num_map, dtype=torch.float32)
    map_pos = torch.stack(
        [
            torch.stack([scenario_idx * 100.0 + map_base, map_base * 0.2], dim=-1),
            torch.stack([scenario_idx * 100.0 + map_base + 1.0, map_base * 0.2], dim=-1),
            torch.stack([scenario_idx * 100.0 + map_base + 2.0, map_base * 0.2], dim=-1),
        ],
        dim=1,
    )
    data["map_save"]["traj_pos"] = map_pos
    data["map_save"]["traj_theta"] = torch.zeros(num_map)
    data["pt_token"]["type"] = torch.zeros(num_map, dtype=torch.long)
    data["pt_token"]["pl_type"] = torch.zeros(num_map, dtype=torch.long)
    data["pt_token"]["light_type"] = torch.zeros(num_map, dtype=torch.long)
    data["pt_token"].num_nodes = num_map
    data["map_save"].num_nodes = num_map
    data.scenario_id = f"scenario_{scenario_idx}"
    data.raw_path = str(raw_path)
    return data


def _make_processor() -> TokenProcessor:
    sampling = OmegaConf.create({"num_k": 1, "temp": 1.0})
    processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        map_token_sampling=sampling,
        agent_token_sampling=sampling,
    )
    processor.train()
    return processor


def test_smart_ntp_token_cache_replays_deterministic_tokenization(monkeypatch, tmp_path):
    monkeypatch.setenv("SMART_NTP_TOKEN_CACHE", "1")
    raw_paths = [tmp_path / "scene0.pkl", tmp_path / "scene1.pkl"]
    for raw_path in raw_paths:
        raw_path.write_bytes(b"placeholder")

    first_batch = Batch.from_data_list(
        [_make_scene(raw_paths[0], 0), _make_scene(raw_paths[1], 1)]
    )
    processor = _make_processor()
    first_map, first_agent = processor(first_batch)

    cached_scenes = []
    for scene_idx, raw_path in enumerate(raw_paths):
        scene = _make_scene(raw_path, scene_idx)
        scene.smart_ntp_token_cache = load_token_cache(str(raw_path))
        cached_scenes.append(scene)
    cached_batch = Batch.from_data_list(cached_scenes)
    second_map, second_agent = processor(cached_batch)

    for key in ["token_idx", "position", "orientation", "type", "pl_type", "light_type", "batch"]:
        assert torch.equal(first_map[key], second_map[key])
    for key in [
        "gt_pos_raw",
        "gt_head_raw",
        "gt_valid_raw",
        "valid_mask",
        "gt_idx",
        "gt_pos",
        "gt_heading",
        "sampled_idx",
        "sampled_pos",
        "sampled_heading",
        "token_agent_shape",
        "batch",
    ]:
        assert torch.equal(first_agent[key], second_agent[key])


def test_empty_token_cache_placeholder_can_collate_with_cached_scene(monkeypatch, tmp_path):
    monkeypatch.setenv("SMART_NTP_TOKEN_CACHE", "1")
    raw_path = tmp_path / "scene0.pkl"
    raw_path.write_bytes(b"placeholder")
    processor = _make_processor()
    first_scene = _make_scene(raw_path, 0)
    first_batch = Batch.from_data_list([first_scene])
    processor(first_batch)

    cached_scene = _make_scene(raw_path, 0)
    cached_scene.smart_ntp_token_cache = load_token_cache(str(raw_path))
    empty_scene = _make_scene(tmp_path / "scene1.pkl", 1)
    empty_scene.smart_ntp_token_cache = empty_token_cache(18)

    batch = Batch.from_data_list([cached_scene, empty_scene])

    assert "smart_ntp_token_cache" in batch
    assert batch.smart_ntp_token_cache["agent"]["gt_idx"].shape[0] == 3


def test_token_cache_is_ignored_when_raw_scene_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("SMART_NTP_TOKEN_CACHE", "1")
    raw_path = tmp_path / "scene0.pkl"
    raw_path.write_bytes(b"placeholder")

    processor = _make_processor()
    processor(Batch.from_data_list([_make_scene(raw_path, 0)]))
    assert load_token_cache(str(raw_path)) is not None

    raw_path.write_bytes(b"changed raw scene contents")

    assert load_token_cache(str(raw_path)) is None
