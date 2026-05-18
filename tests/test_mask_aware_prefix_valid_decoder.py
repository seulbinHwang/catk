from __future__ import annotations

import torch

from src.smart.metrics.flow_metrics import flow_matching_loss
from src.smart.modules.flow_local_decoder import HierarchicalFlowDecoder


def _build_decoder(flow_state_dim: int = 4) -> HierarchicalFlowDecoder:
    torch.manual_seed(1234)
    decoder = HierarchicalFlowDecoder(
        context_dim=8,
        flow_dim=16,
        num_future_steps=10,
        num_chunk_heads=4,
        num_chunk_layers=1,
        chunk_size=5,
        flow_state_dim=flow_state_dim,
    )
    decoder.eval()
    return decoder


def test_full_valid_mask_preserves_original_output() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(3, 8)
    x_t_norm = torch.randn(3, 10, 4)
    tau = torch.full((3,), 0.37)
    full_mask = torch.ones(3, 10, dtype=torch.bool)

    without_mask = decoder(anchor_hidden, x_t_norm, tau)
    with_full_mask = decoder(anchor_hidden, x_t_norm, tau, future_valid_mask=full_mask)

    assert torch.equal(without_mask, with_full_mask)


def test_invalid_tail_does_not_change_valid_prefix_output() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 4)
    tau = torch.full((2,), 0.61)
    prefix_mask = torch.zeros(2, 10, dtype=torch.bool)
    prefix_mask[:, :3] = True

    changed_tail = x_t_norm.clone()
    changed_tail[:, 3:] = torch.randn_like(changed_tail[:, 3:]) * 50.0

    base_output = decoder(anchor_hidden, x_t_norm, tau, future_valid_mask=prefix_mask)
    changed_output = decoder(anchor_hidden, changed_tail, tau, future_valid_mask=prefix_mask)

    assert torch.allclose(base_output[:, :3], changed_output[:, :3], atol=1.0e-6, rtol=1.0e-6)


def test_control_state_dim_prefix_mask() -> None:
    decoder = _build_decoder(flow_state_dim=3)
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 3)
    tau = torch.full((2,), 0.42)
    prefix_mask = torch.zeros(2, 10, dtype=torch.bool)
    prefix_mask[:, :6] = True

    output = decoder(anchor_hidden, x_t_norm, tau, future_valid_mask=prefix_mask)

    assert output.shape == (2, 10, 3)


def test_same_chunk_agent_attention_is_disabled_without_interaction_state() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 4)
    tau = torch.full((2,), 0.31)
    interaction_group = torch.tensor([0, 1], dtype=torch.long)

    base_output = decoder(
        anchor_hidden,
        x_t_norm,
        tau,
        interaction_group=interaction_group,
    )
    changed_anchor_hidden = anchor_hidden.clone()
    changed_x_t_norm = x_t_norm.clone()
    changed_anchor_hidden[1] = torch.randn_like(changed_anchor_hidden[1]) * 20.0
    changed_x_t_norm[1] = torch.randn_like(changed_x_t_norm[1]) * 20.0
    changed_output = decoder(
        changed_anchor_hidden,
        changed_x_t_norm,
        tau,
        interaction_group=interaction_group,
    )

    assert torch.allclose(base_output[0], changed_output[0], atol=1.0e-6, rtol=1.0e-6)


def test_same_chunk_agent_attention_uses_distance_graph() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 4)
    tau = torch.full((2,), 0.31)
    interaction_group = torch.tensor([0, 0], dtype=torch.long)
    far_pos = torch.tensor([[0.0, 0.0], [1000.0, 0.0]])
    head = torch.zeros(2)

    far_base = decoder(
        anchor_hidden,
        x_t_norm,
        tau,
        interaction_group=interaction_group,
        interaction_pos=far_pos,
        interaction_head=head,
    )
    changed_anchor_hidden = anchor_hidden.clone()
    changed_x_t_norm = x_t_norm.clone()
    changed_anchor_hidden[1] = torch.randn_like(changed_anchor_hidden[1]) * 20.0
    changed_x_t_norm[1] = torch.randn_like(changed_x_t_norm[1]) * 20.0
    far_changed = decoder(
        changed_anchor_hidden,
        changed_x_t_norm,
        tau,
        interaction_group=interaction_group,
        interaction_pos=far_pos,
        interaction_head=head,
    )

    assert torch.allclose(far_base[0], far_changed[0], atol=1.0e-6, rtol=1.0e-6)


def test_same_chunk_agent_attention_ignores_invalid_future_chunks() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 4)
    tau = torch.full((2,), 0.44)
    interaction_group = torch.tensor([0, 0], dtype=torch.long)
    close_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    head = torch.zeros(2)
    future_valid_mask = torch.ones(2, 10, dtype=torch.bool)
    future_valid_mask[1, 5:] = False

    base_output = decoder(
        anchor_hidden,
        x_t_norm,
        tau,
        future_valid_mask=future_valid_mask,
        interaction_group=interaction_group,
        interaction_pos=close_pos,
        interaction_head=head,
    )
    changed_x_t_norm = x_t_norm.clone()
    changed_x_t_norm[1, 5:] = torch.randn_like(changed_x_t_norm[1, 5:]) * 50.0
    changed_output = decoder(
        anchor_hidden,
        changed_x_t_norm,
        tau,
        future_valid_mask=future_valid_mask,
        interaction_group=interaction_group,
        interaction_pos=close_pos,
        interaction_head=head,
    )

    assert torch.allclose(base_output[0], changed_output[0], atol=1.0e-6, rtol=1.0e-6)


def test_same_chunk_agent_attention_ignores_invalid_steps_in_partial_chunk() -> None:
    decoder = _build_decoder()
    anchor_hidden = torch.randn(2, 8)
    x_t_norm = torch.randn(2, 10, 4)
    tau = torch.full((2,), 0.52)
    interaction_group = torch.tensor([0, 0], dtype=torch.long)
    close_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    head = torch.zeros(2)
    future_valid_mask = torch.ones(2, 10, dtype=torch.bool)
    future_valid_mask[1, 3:] = False

    base_output = decoder(
        anchor_hidden,
        x_t_norm,
        tau,
        future_valid_mask=future_valid_mask,
        interaction_group=interaction_group,
        interaction_pos=close_pos,
        interaction_head=head,
    )
    changed_x_t_norm = x_t_norm.clone()
    changed_x_t_norm[1, 3:] = torch.randn_like(changed_x_t_norm[1, 3:]) * 50.0
    changed_output = decoder(
        anchor_hidden,
        changed_x_t_norm,
        tau,
        future_valid_mask=future_valid_mask,
        interaction_group=interaction_group,
        interaction_pos=close_pos,
        interaction_head=head,
    )

    assert torch.allclose(
        base_output[future_valid_mask],
        changed_output[future_valid_mask],
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    target = torch.randn_like(base_output)
    changed_target = target.clone()
    changed_target[1, 3:] = torch.randn_like(changed_target[1, 3:]) * 50.0
    base_loss = flow_matching_loss(base_output, target, valid_mask=future_valid_mask)
    changed_loss = flow_matching_loss(changed_output, changed_target, valid_mask=future_valid_mask)

    assert torch.allclose(base_loss, changed_loss, atol=1.0e-6, rtol=1.0e-6)


def test_default_control_decoder_parameter_count_includes_agent_chunk_attention() -> None:
    decoder = HierarchicalFlowDecoder(
        context_dim=128,
        flow_dim=96,
        num_future_steps=20,
        num_chunk_heads=4,
        head_dim=15,
        num_freq_bands=64,
        num_chunk_layers=2,
        chunk_size=5,
        flow_state_dim=3,
        a2a_radius=60.0,
        dropout=0.1,
    )

    assert sum(param.numel() for param in decoder.parameters()) == 691_071
    assert sum(param.numel() for param in decoder.same_chunk_agent_mixer.parameters()) == 200_892
