import torch

from src.smart.modules.agent_decoder import SMARTAgentDecoder


def test_map_to_agent_edges_use_static_map_tokens_across_time() -> None:
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
    light_type = torch.tensor([1, 2])

    edge_index, edge_attr = decoder.build_map2agent_edge(
        pos_pl=pos_pl,
        orient_pl=orient_pl,
        pos_a=pos_a,
        head_a=head_a,
        head_vector_a=head_vector_a,
        mask=mask,
        batch_s=batch_s,
        batch_pl=batch_pl,
        light_type=light_type,
    )

    assert edge_index.shape[0] == 2
    assert edge_index[0].max().item() < pos_pl.shape[0]
    assert edge_index[1].max().item() < pos_a.shape[0] * pos_a.shape[1]
    assert edge_attr.shape == (edge_index.shape[1], decoder.hidden_dim)
