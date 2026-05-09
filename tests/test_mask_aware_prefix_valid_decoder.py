from __future__ import annotations

import torch

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
