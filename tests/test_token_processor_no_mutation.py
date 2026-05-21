from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local


def _make_processor() -> TokenProcessor:
    processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
    )
    processor.train()
    return processor


def _make_agent_data() -> HeteroData:
    n_agent = 2
    n_step = 91
    data = HeteroData()
    data.num_graphs = 1

    valid_mask = torch.ones(n_agent, n_step, dtype=torch.bool)
    valid_mask[1, :10] = False

    heading = torch.zeros(n_agent, n_step)
    heading[0, 15] = 3.2

    position = torch.zeros(n_agent, n_step, 3)
    position[:, :, 0] = torch.arange(n_step, dtype=torch.float32)

    velocity = torch.ones(n_agent, n_step, 2) * 0.1
    role = torch.zeros(n_agent, 3, dtype=torch.bool)
    role[0, 0] = True

    data["agent"]["num_nodes"] = n_agent
    data["agent"]["valid_mask"] = valid_mask
    data["agent"]["role"] = role
    data["agent"]["id"] = torch.arange(n_agent, dtype=torch.long)
    data["agent"]["type"] = torch.zeros(n_agent, dtype=torch.uint8)
    data["agent"]["position"] = position
    data["agent"]["heading"] = heading
    data["agent"]["velocity"] = velocity
    data["agent"]["shape"] = torch.tensor(
        [[4.8, 2.0, 1.5], [4.8, 2.0, 1.5]],
        dtype=torch.float32,
    )
    data["agent"]["batch"] = torch.zeros(n_agent, dtype=torch.long)
    return data


def _make_map_data() -> HeteroData:
    n_pl = 4
    data = HeteroData()
    data["map_save"]["traj_pos"] = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            [[3.0, 0.0], [3.0, 1.0], [3.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    data["map_save"]["traj_theta"] = torch.zeros(n_pl, dtype=torch.float32)
    data["pt_token"]["type"] = torch.zeros(n_pl, dtype=torch.long)
    data["pt_token"]["pl_type"] = torch.zeros(n_pl, dtype=torch.long)
    data["pt_token"]["light_type"] = torch.zeros(n_pl, dtype=torch.long)
    data["pt_token"]["batch"] = torch.zeros(n_pl, dtype=torch.long)
    return data


def test_tokenize_agent_does_not_mutate_raw_agent_cache_fields() -> None:
    data = _make_agent_data()
    originals = {
        key: data["agent"][key].clone()
        for key in ("valid_mask", "heading", "position", "velocity")
    }

    tokenized_agent = _make_processor().tokenize_agent(data)

    assert bool(tokenized_agent["gt_valid_raw"][1, 0])
    assert float(tokenized_agent["gt_head_raw"][0, 2]) == 0.0
    for key, original in originals.items():
        torch.testing.assert_close(data["agent"][key], original)


def test_tokenize_map_always_uses_nearest_token() -> None:
    processor = _make_processor()
    data = _make_map_data()

    tokenized_map = processor.tokenize_map(data)

    traj_pos = data["map_save"]["traj_pos"]
    traj_theta = data["map_save"]["traj_theta"]
    traj_pos_local, _ = transform_to_local(
        pos_global=traj_pos,
        head_global=None,
        pos_now=traj_pos[:, 0],
        head_now=traj_theta,
    )
    dist = torch.sum(
        (processor.map_token_sample_pt - traj_pos_local.unsqueeze(1)) ** 2,
        dim=(-2, -1),
    )
    torch.testing.assert_close(tokenized_map["token_idx"], torch.argmin(dist, dim=-1))


def test_tokenize_agent_sampled_state_matches_default_nearest_gt_state() -> None:
    tokenized_agent = _make_processor().tokenize_agent(_make_agent_data())

    torch.testing.assert_close(tokenized_agent["sampled_idx"], tokenized_agent["gt_idx"])
    torch.testing.assert_close(tokenized_agent["sampled_pos"], tokenized_agent["gt_pos"])
    torch.testing.assert_close(
        tokenized_agent["sampled_heading"], tokenized_agent["gt_heading"]
    )
