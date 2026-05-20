from __future__ import annotations

import pytest
import torch

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.segmented_graph_attention import (
    build_graph_attention_metadata,
    segmented_graph_attention,
)


def _manual_graph_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    edge_index: torch.Tensor,
    r: torch.Tensor | None,
    relation_key_weight: torch.Tensor | None,
    relation_value_weight: torch.Tensor | None,
    relation_value_bias: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """작은 입력에서 attention 수식을 직접 계산합니다.

    Args:
        q: target query입니다. shape은 ``[N_dst, H, D]`` 입니다.
        k: source key입니다. shape은 ``[N_src, H, D]`` 입니다.
        v: source value입니다. shape은 ``[N_src, H, D]`` 입니다.
        edge_index: source에서 target으로 가는 edge 목록입니다. shape은 ``[2, E]`` 입니다.
        r: edge relation feature입니다. 있을 때 shape은 ``[E, R]`` 입니다.
        relation_key_weight: relation key projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_weight: relation value projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_bias: relation value projection bias입니다. shape은 ``[H * D]`` 입니다.
        scale: score scale입니다.

    Returns:
        torch.Tensor: attention 출력입니다. shape은 ``[N_dst, H, D]`` 입니다.
    """
    num_dst, num_heads, head_dim = q.shape
    out = q.new_zeros(num_dst, num_heads, head_dim)
    src_all = edge_index[0]
    dst_all = edge_index[1]
    for dst_idx in range(num_dst):
        edge_mask = dst_all == dst_idx
        if not bool(edge_mask.any()):
            continue
        src = src_all[edge_mask]
        k_edge = k[src]
        v_edge = v[src]
        if r is not None:
            r_edge = r[edge_mask]
            key_rel = torch.nn.functional.linear(r_edge, relation_key_weight).view(-1, num_heads, head_dim)
            value_rel = torch.nn.functional.linear(
                r_edge,
                relation_value_weight,
                relation_value_bias,
            ).view(-1, num_heads, head_dim)
            k_edge = k_edge + key_rel
            v_edge = v_edge + value_rel
        score = (q[dst_idx].unsqueeze(0) * k_edge).sum(dim=-1) * scale
        attn = torch.softmax(score, dim=0)
        out[dst_idx] = (attn.unsqueeze(-1) * v_edge).sum(dim=0)
    return out


def _make_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, ...]:
    """테스트용 작은 graph attention 입력을 만듭니다.

    Args:
        dtype: 입력 tensor dtype입니다.

    Returns:
        tuple[torch.Tensor, ...]: q, k, v, edge_index, r, relation weight/bias입니다.
    """
    generator = torch.Generator().manual_seed(7)
    num_src = 6
    num_dst = 6
    num_heads = 3
    head_dim = 16
    rel_dim = 16
    q = torch.randn(num_dst, num_heads, head_dim, dtype=dtype, generator=generator, requires_grad=True)
    k = torch.randn(num_src, num_heads, head_dim, dtype=dtype, generator=generator, requires_grad=True)
    v = torch.randn(num_src, num_heads, head_dim, dtype=dtype, generator=generator, requires_grad=True)
    edge_index = torch.tensor(
        [
            [0, 2, 3, 1, 4, 5, 2, 0, 3, 1, 5],
            [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    r = torch.randn(edge_index.size(1), rel_dim, dtype=dtype, generator=generator, requires_grad=True)
    relation_key_weight = torch.randn(num_heads * head_dim, rel_dim, dtype=dtype, generator=generator, requires_grad=True)
    relation_value_weight = torch.randn(num_heads * head_dim, rel_dim, dtype=dtype, generator=generator, requires_grad=True)
    relation_value_bias = torch.randn(num_heads * head_dim, dtype=dtype, generator=generator, requires_grad=True)
    return q, k, v, edge_index, r, relation_key_weight, relation_value_weight, relation_value_bias


def test_segmented_graph_attention_matches_manual_forward() -> None:
    q, k, v, edge_index, r, relation_key_weight, relation_value_weight, relation_value_bias = _make_inputs()
    scale = q.size(-1) ** -0.5
    actual = segmented_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=edge_index,
        r=r,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
        dropout_p=0.0,
        training=False,
    )
    expected = _manual_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=edge_index,
        r=r,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_segmented_graph_attention_matches_manual_backward() -> None:
    inputs_actual = _make_inputs()
    inputs_expected = tuple(
        item.detach().clone().requires_grad_(item.requires_grad) if torch.is_floating_point(item) else item.clone()
        for item in inputs_actual
    )
    q, k, v, edge_index, r, relation_key_weight, relation_value_weight, relation_value_bias = inputs_actual
    q_ref, k_ref, v_ref, edge_index_ref, r_ref, relation_key_weight_ref, relation_value_weight_ref, relation_value_bias_ref = inputs_expected
    scale = q.size(-1) ** -0.5
    probe = torch.randn_like(q)
    actual = segmented_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=edge_index,
        r=r,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
        dropout_p=0.0,
        training=False,
    )
    expected = _manual_graph_attention(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        edge_index=edge_index_ref,
        r=r_ref,
        relation_key_weight=relation_key_weight_ref,
        relation_value_weight=relation_value_weight_ref,
        relation_value_bias=relation_value_bias_ref,
        scale=scale,
    )
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual_tensor, expected_tensor in zip(
        [q, k, v, r, relation_key_weight, relation_value_weight, relation_value_bias],
        [q_ref, k_ref, v_ref, r_ref, relation_key_weight_ref, relation_value_weight_ref, relation_value_bias_ref],
    ):
        torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, atol=1e-10, rtol=1e-10)


def test_metadata_reuse_keeps_same_result() -> None:
    q, k, v, edge_index, r, relation_key_weight, relation_value_weight, relation_value_bias = _make_inputs()
    metadata = build_graph_attention_metadata(edge_index=edge_index, num_dst_nodes=q.size(0))
    r_sorted = metadata.reorder_edge_features(r)
    scale = q.size(-1) ** -0.5
    direct = segmented_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=edge_index,
        r=r,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
        dropout_p=0.0,
        training=False,
    )
    reused = segmented_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=metadata.sorted_edge_index,
        r=r_sorted,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
        dropout_p=0.0,
        training=False,
        metadata=metadata,
        r_is_sorted=True,
    )
    torch.testing.assert_close(direct, reused, atol=1e-10, rtol=1e-10)


def test_attention_layer_segmented_backend_matches_pyg(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(11)
    layer = AttentionLayer(
        hidden_dim=16,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
        bipartite=True,
        has_pos_emb=True,
    ).eval()
    x_src = torch.randn(7, 16)
    x_dst = torch.randn(5, 16)
    edge_index = torch.tensor(
        [
            [0, 3, 6, 1, 4, 5, 2, 0, 6],
            [0, 0, 1, 2, 2, 3, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    r = torch.randn(edge_index.size(1), 16)
    metadata = build_graph_attention_metadata(
        edge_index=edge_index,
        num_dst_nodes=x_dst.size(0),
    )
    r_sorted = metadata.reorder_edge_features(r)

    monkeypatch.setenv("CATK_ATTENTION_LAYER_BACKEND", "pyg")
    pyg_out = layer((x_src, x_dst), r_sorted, metadata.sorted_edge_index)

    monkeypatch.setenv("CATK_ATTENTION_LAYER_BACKEND", "segmented")
    segmented_out = layer(
        (x_src, x_dst),
        r_sorted,
        metadata.sorted_edge_index,
        attention_metadata=metadata,
        r_is_sorted=True,
    )
    torch.testing.assert_close(segmented_out, pyg_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the segmented graph attention smoke test.",
)
def test_cuda_segmented_kernel_matches_cpu_reference() -> None:
    q, k, v, edge_index, r, relation_key_weight, relation_value_weight, relation_value_bias = _make_inputs(dtype=torch.float32)
    scale = q.size(-1) ** -0.5
    expected = segmented_graph_attention(
        q=q,
        k=k,
        v=v,
        edge_index=edge_index,
        r=r,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        scale=scale,
        dropout_p=0.0,
        training=False,
    )
    cuda_inputs = (
        q.detach().cuda().to(torch.bfloat16).requires_grad_(True),
        k.detach().cuda().to(torch.bfloat16).requires_grad_(True),
        v.detach().cuda().to(torch.bfloat16).requires_grad_(True),
        r.detach().cuda().to(torch.bfloat16).requires_grad_(True),
        relation_key_weight.detach().cuda().requires_grad_(True),
        relation_value_weight.detach().cuda().requires_grad_(True),
        relation_value_bias.detach().cuda().requires_grad_(True),
    )
    q_cuda, k_cuda, v_cuda, r_cuda, relation_key_weight_cuda, relation_value_weight_cuda, relation_value_bias_cuda = cuda_inputs
    actual_cuda = segmented_graph_attention(
        q=q_cuda,
        k=k_cuda,
        v=v_cuda,
        edge_index=edge_index.cuda(),
        r=r_cuda,
        relation_key_weight=relation_key_weight_cuda,
        relation_value_weight=relation_value_weight_cuda,
        relation_value_bias=relation_value_bias_cuda,
        scale=scale,
        dropout_p=0.0,
        training=False,
    )
    torch.testing.assert_close(actual_cuda.float().detach().cpu(), expected.detach(), atol=3e-1, rtol=3e-1)

    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual_cuda * probe.cuda().to(actual_cuda.dtype)).sum().backward()
    for actual_tensor, expected_tensor in zip(
        cuda_inputs,
        [q, k, v, r, relation_key_weight, relation_value_weight, relation_value_bias],
    ):
        torch.testing.assert_close(actual_tensor.grad.float().cpu(), expected_tensor.grad, atol=2.0, rtol=2.0)
