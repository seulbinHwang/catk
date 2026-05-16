from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor


def _make_processor() -> TokenProcessor:
    processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        map_token_sampling=OmegaConf.create({"num_k": 1, "temp": 1.0}),
        agent_token_sampling=OmegaConf.create({"num_k": 1, "temp": 1.0}),
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
