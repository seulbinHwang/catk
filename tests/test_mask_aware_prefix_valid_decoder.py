from __future__ import annotations

import torch

from src.smart.modules.flow_local_decoder import HierarchicalFlowDecoder


def _build_decoder() -> HierarchicalFlowDecoder:
    """작은 테스트용 flow decoder를 만듭니다.

    Returns:
        HierarchicalFlowDecoder: mask 동작만 확인하기 위한 작은 decoder입니다.
    """
    torch.manual_seed(1234)
    decoder = HierarchicalFlowDecoder(
        context_dim=8,
        flow_dim=16,
        num_future_steps=10,
        num_chunk_heads=4,
        num_chunk_layers=1,
        chunk_size=5,
    )
    decoder.eval()
    return decoder


def test_full_valid_mask_preserves_original_output() -> None:
    """모든 미래 step이 유효하면 mask를 주지 않은 출력과 완전히 같은지 확인합니다."""
    decoder = _build_decoder()
    anchor_hidden = torch.randn(3, 8)
    x_t_norm = torch.randn(3, 10, 4)
    tau = torch.full((3,), 0.37)
    full_mask = torch.ones(3, 10, dtype=torch.bool)

    without_mask = decoder(anchor_hidden, x_t_norm, tau)
    with_full_mask = decoder(anchor_hidden, x_t_norm, tau, future_valid_mask=full_mask)

    assert torch.equal(without_mask, with_full_mask)


def test_invalid_tail_does_not_change_valid_prefix_output() -> None:
    """invalid tail 값을 바꿔도 valid prefix 출력이 변하지 않는지 확인합니다."""
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
