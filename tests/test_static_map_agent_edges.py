import torch

from src.smart.modules.agent_decoder import SMARTAgentDecoder


def _make_decoder(pl2a_radius: float = 5.0) -> SMARTAgentDecoder:
    return SMARTAgentDecoder(
        hidden_dim=8,
        num_historical_steps=11,
        num_future_steps=80,
        time_span=30,
        pl2a_radius=pl2a_radius,
        a2a_radius=60.0,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        hist_drop_prob=0.0,
        n_token_agent=5,
    )


def test_map_to_agent_edges_use_static_map_tokens_across_time() -> None:
    decoder = _make_decoder()

    pos_pl = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    orient_pl = torch.zeros(2)
    pos_a = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[10.0, 0.0], [10.5, 0.0], [11.0, 0.0]],
        ]
    )
    head_a = torch.zeros(2, 3)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.ones(2, 3, dtype=torch.bool)
    agent_batch = torch.tensor([0, 1])
    batch_s = agent_batch.repeat(pos_a.shape[1])
    batch_pl = torch.tensor([0, 1])
    edge_index, edge_attr = decoder.build_map2agent_edge(
        pos_pl=pos_pl,
        orient_pl=orient_pl,
        pos_a=pos_a,
        head_a=head_a,
        head_vector_a=head_vector_a,
        mask=mask,
        batch_s=batch_s,
        batch_pl=batch_pl,
    )

    assert edge_index.shape[0] == 2
    assert edge_index[0].max().item() < pos_pl.shape[0]
    assert edge_index[1].max().item() < pos_a.shape[0] * pos_a.shape[1]
    assert edge_attr.shape == (edge_index.shape[1], decoder.hidden_dim)


def test_map_to_agent_edges_keep_all_same_scene_edges_with_repeated_agent_batches() -> None:
    decoder = _make_decoder(pl2a_radius=100.0)
    agents_per_scene = [6, 4, 8, 3]
    maps_per_scene = [3, 2, 5, 2]
    num_steps = 4

    agent_positions = []
    map_positions = []
    agent_batch = []
    map_batch = []
    for scene_idx, (num_agents, num_maps) in enumerate(
        zip(agents_per_scene, maps_per_scene)
    ):
        center = float(scene_idx * 1000)
        for agent_idx in range(num_agents):
            step_positions = []
            for step_idx in range(num_steps):
                step_positions.append(
                    [center + float(agent_idx), float(step_idx) * 0.25]
                )
            agent_positions.append(step_positions)
            agent_batch.append(scene_idx)
        for map_idx in range(num_maps):
            map_positions.append([center + float(map_idx) * 0.5, -1.0])
            map_batch.append(scene_idx)

    pos_a = torch.tensor(agent_positions, dtype=torch.float32)
    pos_pl = torch.tensor(map_positions, dtype=torch.float32)
    orient_pl = torch.zeros(pos_pl.shape[0])
    head_a = torch.zeros(pos_a.shape[0], num_steps)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.ones(pos_a.shape[0], num_steps, dtype=torch.bool)
    batch_s = torch.tensor(agent_batch).repeat(num_steps)
    batch_pl = torch.tensor(map_batch)

    edge_index, _ = decoder.build_map2agent_edge(
        pos_pl=pos_pl,
        orient_pl=orient_pl,
        pos_a=pos_a,
        head_a=head_a,
        head_vector_a=head_vector_a,
        mask=mask,
        batch_s=batch_s,
        batch_pl=batch_pl,
    )

    pos_s = pos_a.transpose(0, 1).flatten(0, 1)
    expected_edges = (
        (torch.cdist(pos_pl, pos_s) <= decoder.pl2a_radius)
        & (batch_pl[:, None] == batch_s[None, :])
    ).sum()

    assert not bool(torch.all(batch_s[:-1] <= batch_s[1:]))
    assert edge_index.shape[1] == int(expected_edges.item())
    assert bool(torch.all(batch_pl[edge_index[0]] == batch_s[edge_index[1]]))


def test_dynamic_light_bias_is_sparse_relation_feature_not_map_token_feature() -> None:
    decoder = _make_decoder()
    edge_index = torch.tensor(
        [
            [0, 1, 1],
            [0, 1, 5],
        ]
    )

    light_bias = decoder._build_light_relation_bias(
        edge_index_pl2a=edge_index,
        light_type=torch.tensor([0, 2]),
        light_time_delta_norm=None,
        num_map=2,
        num_agents=4,
        num_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert decoder.r_pt2a_emb.input_dim == 3
    torch.testing.assert_close(light_bias[0], torch.zeros_like(light_bias[0]))
    assert not bool(torch.allclose(light_bias[1], torch.zeros_like(light_bias[1])))
    assert not bool(torch.allclose(light_bias[1], light_bias[2]))
