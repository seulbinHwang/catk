from __future__ import annotations

import torch

from src.smart.metrics.flow_metrics import flow_matching_loss
from src.smart.modules.flow_local_decoder import (
    HierarchicalFlowDecoder,
    SameChunkAgentAttentionBlock,
)


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


def _build_same_chunk_packing(
    *,
    agents_per_scene: list[int],
    num_anchor: int,
    scene_spacing_m: float,
    within_scene_std_m: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """학습 패킹 그대로의 그룹/위치/방향 텐서를 만듭니다.

    실제 학습 입력처럼 장면별 차량 개수를 다르게 두고 위치는 장면 중심에서
    가우시안 잡음으로 흩뜨립니다. ``anchor_mask`` 도 약 70% 만 유효하게
    무작위로 만들어 anchor 단위 패킹에서 그룹 ID 가 단조 비감소가 아니게
    되는 production 패턴을 재현합니다.

    Args:
        agents_per_scene: 장면별 차량 개수 목록입니다.
        num_anchor: anchor 시점 개수입니다.
        scene_spacing_m: 장면 중심 사이 간격(m)입니다.
        within_scene_std_m: 장면 중심 주변 위치 분산(m)입니다.
        device: 텐서를 올릴 장치입니다.
        seed: 결정적 재현을 위한 시드입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            anchor 단위 이어붙인 그룹 ID ``[N]``, 위치 ``[N, 2]``,
            heading ``[N]`` 입니다.
    """
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    agent_batch = torch.cat(
        [
            torch.full((count,), scene_idx, dtype=torch.long)
            for scene_idx, count in enumerate(agents_per_scene)
        ]
    )
    num_agent = agent_batch.numel()
    anchor_mask = torch.rand(num_agent, num_anchor, generator=rng) > 0.3

    packed_group = torch.cat(
        [
            agent_batch[anchor_mask[:, anchor_idx]] * num_anchor + anchor_idx
            for anchor_idx in range(num_anchor)
            if anchor_mask[:, anchor_idx].any()
        ],
        dim=0,
    ).to(device=device)

    scene_centers = torch.stack(
        [
            torch.arange(len(agents_per_scene), dtype=torch.float32) * scene_spacing_m,
            torch.zeros(len(agents_per_scene), dtype=torch.float32),
        ],
        dim=-1,
    )
    agent_pos = scene_centers[agent_batch] + torch.randn(
        num_agent, 2, generator=rng
    ) * within_scene_std_m

    packed_pos = torch.cat(
        [
            agent_pos[anchor_mask[:, anchor_idx]]
            for anchor_idx in range(num_anchor)
            if anchor_mask[:, anchor_idx].any()
        ],
        dim=0,
    ).to(device=device)
    packed_head = torch.zeros(packed_pos.shape[0], device=device)
    return packed_group, packed_pos, packed_head


def _count_cross_group_edges(
    edge_index: torch.Tensor, packed_group: torch.Tensor, num_chunks: int
) -> int:
    """수정된 코드와 동일한 batch ID 인코딩으로 cross-group 엣지를 셉니다."""
    _, inverse_group = torch.unique(
        packed_group, sorted=True, return_inverse=True
    )
    num_groups = int(inverse_group.max().item()) + 1
    batch_s = (
        inverse_group.unsqueeze(0).expand(num_chunks, packed_group.numel())
        + torch.arange(num_chunks, device=packed_group.device).view(num_chunks, 1)
        * num_groups
    ).reshape(-1)
    src_batch = batch_s[edge_index[0]]
    dst_batch = batch_s[edge_index[1]]
    return int((src_batch != dst_batch).sum().item())


def test_same_chunk_agent_attention_blocks_cross_anchor_leakage_cpu() -> None:
    """CPU 에서 학습 패킹과 동일한 그룹 ID 입력에서 cross-group 엣지가 없어야 합니다."""
    block = SameChunkAgentAttentionBlock(
        flow_dim=16,
        num_heads=4,
        head_dim=4,
        num_freq_bands=64,
        radius=60.0,
    )

    packed_group, packed_pos, packed_head = _build_same_chunk_packing(
        agents_per_scene=[50, 25, 70, 12],
        num_anchor=16,
        scene_spacing_m=200.0,
        within_scene_std_m=30.0,
        device=torch.device("cpu"),
        seed=42,
    )
    num_chunks = 4
    edge_index, _ = block._build_same_chunk_edges(
        interaction_pos=packed_pos,
        interaction_head=packed_head,
        interaction_group=packed_group,
        chunk_valid_mask=None,
        num_chunks=num_chunks,
    )

    cross_group_edges = _count_cross_group_edges(
        edge_index=edge_index, packed_group=packed_group, num_chunks=num_chunks
    )
    assert cross_group_edges == 0, (
        f"CPU: cross-group edges leaked despite batch grouping: {cross_group_edges}"
    )


def test_same_chunk_agent_attention_blocks_cross_anchor_leakage_gpu() -> None:
    """GPU 에서도 cross-group 엣지가 없어야 합니다.

    GPU ``radius_graph`` 는 sorted batch 가정을 더 엄격하게 따르기 때문에
    수정 전 코드에서는 cross-group 엣지가 대량으로 발생합니다. 실제 학습
    배치 규모(장면별 차량 수가 다르고 anchor 마스크도 70% 정도 유효)를
    그대로 재현해 회귀 신호가 확실히 나오도록 합니다. CUDA 가 없는
    환경에서는 테스트를 건너뜁니다.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA unavailable; GPU regression cannot run.")

    device = torch.device("cuda")
    block = SameChunkAgentAttentionBlock(
        flow_dim=16,
        num_heads=4,
        head_dim=4,
        num_freq_bands=64,
        radius=60.0,
    ).to(device=device)

    packed_group, packed_pos, packed_head = _build_same_chunk_packing(
        agents_per_scene=[50, 25, 70, 12],
        num_anchor=16,
        scene_spacing_m=200.0,
        within_scene_std_m=30.0,
        device=device,
        seed=42,
    )
    num_chunks = 4
    edge_index, _ = block._build_same_chunk_edges(
        interaction_pos=packed_pos,
        interaction_head=packed_head,
        interaction_group=packed_group,
        chunk_valid_mask=None,
        num_chunks=num_chunks,
    )

    cross_group_edges = _count_cross_group_edges(
        edge_index=edge_index, packed_group=packed_group, num_chunks=num_chunks
    )
    assert cross_group_edges == 0, (
        f"GPU: cross-group edges leaked despite batch grouping: {cross_group_edges}"
    )


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
