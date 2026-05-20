import torch

from src.smart.modules.agent_decoder import SMARTAgentDecoder


def _make_decoder(a2a_radius: float = 10.0) -> SMARTAgentDecoder:
    return SMARTAgentDecoder(
        hidden_dim=8,
        num_historical_steps=11,
        num_future_steps=80,
        time_span=30,
        pl2a_radius=5.0,
        a2a_radius=a2a_radius,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        hist_drop_prob=0.0,
        n_token_agent=5,
    )


def test_motion_feature_keeps_missingness_on_agent_node() -> None:
    pos_a = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [5.0, 0.0], [7.0, 0.0]],
        ]
    )
    head_vector_a = torch.zeros(2, 3, 2)
    head_vector_a[..., 0] = 1.0
    valid_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, True],
        ]
    )

    feature = SMARTAgentDecoder._build_motion_feature(
        pos_a=pos_a,
        head_vector_a=head_vector_a,
        valid_mask=valid_mask,
    )

    torch.testing.assert_close(feature[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(feature[0, :, 2], torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(feature[1, :, 0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(feature[1, :, 2], torch.tensor([0.0, 0.0, 0.0]))


def test_interaction_edge_uses_geometry_only_and_skips_invalid_nodes_before_radius() -> None:
    decoder = _make_decoder(a2a_radius=100.0)
    pos_a = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
        ]
    )
    head_a = torch.zeros(3, 1)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.tensor([[True], [False], [True]])
    batch_s = torch.zeros(3, dtype=torch.long)

    edge_index, edge_attr = decoder.build_interaction_edge(
        pos_a=pos_a,
        head_a=head_a,
        head_vector_a=head_vector_a,
        batch_s=batch_s,
        mask=mask,
    )

    assert decoder.r_a2a_emb.input_dim == 3
    assert edge_attr.shape == (edge_index.shape[1], decoder.hidden_dim)
    assert 1 not in edge_index.reshape(-1).tolist()
    assert edge_index.shape[1] == 2
