from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraphAttentionMetadata:
    """edge attention에서 반복해서 쓰는 정렬 정보를 담습니다.

    Args:
        sorted_edge_index: target node 기준으로 정렬된 edge 목록입니다.
            shape은 ``[2, E]`` 입니다.
        sorted_src: 정렬된 source node index입니다. shape은 ``[E]`` 입니다.
        sorted_dst: 정렬된 target node index입니다. shape은 ``[E]`` 입니다.
        dst_ptr: 각 target node의 edge 시작 위치입니다. shape은 ``[N_dst + 1]`` 입니다.
        order: 원래 edge 순서에서 정렬된 edge 순서로 바꾸는 index입니다. shape은 ``[E]`` 입니다.
        num_dst_nodes: target node 개수입니다.
        max_degree: target node 하나가 갖는 최대 edge 수입니다.
    """

    sorted_edge_index: Tensor
    sorted_src: Tensor
    sorted_dst: Tensor
    dst_ptr: Tensor
    dst_ptr_cpu: tuple[int, ...]
    order: Tensor
    num_dst_nodes: int
    max_degree: int

    def reorder_edge_features(self, edge_features: Optional[Tensor]) -> Optional[Tensor]:
        """edge feature를 metadata의 정렬 순서와 같게 바꿉니다.

        Args:
            edge_features: edge별 feature입니다. 있을 때 첫 번째 차원 shape은 ``[E]`` 입니다.

        Returns:
            정렬된 edge feature입니다. 입력이 ``None`` 이면 ``None`` 을 반환합니다.
        """
        if edge_features is None:
            return None
        if edge_features.size(0) != self.order.numel():
            raise ValueError(
                "edge_features first dimension must match the number of edges, "
                f"got {edge_features.size(0)} and {self.order.numel()}."
            )
        return edge_features.index_select(0, self.order)

    def target_chunk_ranges(self, max_edges: int) -> tuple[tuple[int, int, int, int], ...]:
        """target node 연속 구간을 edge 수 상한에 맞춰 나눕니다."""
        num_edges = self.sorted_src.numel()
        if self.num_dst_nodes == 0:
            return ()
        if num_edges == 0 or num_edges <= max_edges:
            return ((0, self.num_dst_nodes, 0, num_edges),)

        ranges: list[tuple[int, int, int, int]] = []
        target_start = 0
        while target_start < self.num_dst_nodes:
            edge_start = self.dst_ptr_cpu[target_start]
            target_end = target_start + 1
            while target_end < self.num_dst_nodes:
                next_edge_end = self.dst_ptr_cpu[target_end + 1]
                if next_edge_end - edge_start > max_edges:
                    break
                target_end += 1
            edge_end = self.dst_ptr_cpu[target_end]
            ranges.append((target_start, target_end, edge_start, edge_end))
            target_start = target_end
        return tuple(ranges)


def build_graph_attention_metadata(
    edge_index: Tensor,
    num_dst_nodes: int,
) -> GraphAttentionMetadata:
    """target node 기준 edge 정렬 정보를 한 번만 만듭니다.

    같은 edge 구조를 여러 attention layer가 공유할 때 이 metadata를 재사용하면,
    layer마다 같은 정렬 작업을 반복하지 않습니다.

    Args:
        edge_index: source에서 target으로 가는 edge 목록입니다. shape은 ``[2, E]`` 입니다.
        num_dst_nodes: target node 개수입니다.

    Returns:
        GraphAttentionMetadata: 정렬된 edge와 target별 edge 위치 정보입니다.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.dtype != torch.long:
        raise ValueError(f"edge_index must be torch.long, got {edge_index.dtype}.")
    if num_dst_nodes < 0:
        raise ValueError(f"num_dst_nodes must be non-negative, got {num_dst_nodes}.")

    src = edge_index[0].contiguous()
    dst = edge_index[1].contiguous()
    if dst.numel() > 0:
        max_dst = int(dst.max().item())
        if max_dst >= num_dst_nodes:
            raise ValueError(
                f"edge_index target contains index {max_dst}, but num_dst_nodes={num_dst_nodes}."
            )
    order = torch.argsort(dst, stable=True)
    sorted_src = src.index_select(0, order).contiguous()
    sorted_dst = dst.index_select(0, order).contiguous()
    sorted_edge_index = torch.stack([sorted_src, sorted_dst], dim=0)

    counts = torch.bincount(sorted_dst, minlength=num_dst_nodes)
    dst_ptr = torch.zeros(num_dst_nodes + 1, device=edge_index.device, dtype=torch.long)
    dst_ptr[1:] = torch.cumsum(counts, dim=0)
    max_degree = int(counts.max().item()) if counts.numel() > 0 else 0
    return GraphAttentionMetadata(
        sorted_edge_index=sorted_edge_index.contiguous(),
        sorted_src=sorted_src,
        sorted_dst=sorted_dst,
        dst_ptr=dst_ptr.contiguous(),
        dst_ptr_cpu=tuple(int(value) for value in dst_ptr.detach().cpu().tolist()),
        order=order.contiguous(),
        num_dst_nodes=int(num_dst_nodes),
        max_degree=max_degree,
    )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    r: Optional[Tensor],
    relation_key_weight: Optional[Tensor],
    relation_value_weight: Optional[Tensor],
    relation_value_bias: Optional[Tensor],
    metadata: GraphAttentionMetadata,
    dropout_p: float,
) -> None:
    """attention 입력 shape과 설정을 검사합니다.

    Args:
        q: target node query입니다. shape은 ``[N_dst, H, D]`` 입니다.
        k: source node key입니다. shape은 ``[N_src, H, D]`` 입니다.
        v: source node value입니다. shape은 ``[N_src, H, D]`` 입니다.
        r: 정렬된 edge relation feature입니다. 있을 때 shape은 ``[E, R]`` 입니다.
        relation_key_weight: relation key projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_weight: relation value projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_bias: relation value projection bias입니다. shape은 ``[H * D]`` 입니다.
        metadata: target 기준으로 정렬된 edge metadata입니다.
        dropout_p: attention dropout 확률입니다.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must have shapes [N_dst, H, D], [N_src, H, D], [N_src, H, D].")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}.")
    if q.size(1) != k.size(1) or q.size(2) != k.size(2):
        raise ValueError(f"q, k, v must share H and D, got q={tuple(q.shape)}, k={tuple(k.shape)}.")
    if q.size(0) != metadata.num_dst_nodes:
        raise ValueError(f"q has {q.size(0)} target nodes, metadata has {metadata.num_dst_nodes}.")
    if metadata.sorted_src.numel() > 0 and int(metadata.sorted_src.max().item()) >= k.size(0):
        raise ValueError("metadata source index is out of range for k/v.")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")

    has_relation = r is not None
    if has_relation:
        num_edges = metadata.sorted_src.numel()
        if r.dim() != 2 or r.size(0) != num_edges:
            raise ValueError(f"r must have shape [E, R], got {tuple(r.shape)} with E={num_edges}.")
        if relation_key_weight is None or relation_value_weight is None:
            raise ValueError("relation weights are required when r is not None.")
        head_width = q.size(1) * q.size(2)
        if tuple(relation_key_weight.shape) != (head_width, r.size(1)):
            raise ValueError(
                "relation_key_weight must have shape [H * D, R], got "
                f"{tuple(relation_key_weight.shape)} and expected {(head_width, r.size(1))}."
            )
        if tuple(relation_value_weight.shape) != (head_width, r.size(1)):
            raise ValueError(
                "relation_value_weight must have shape [H * D, R], got "
                f"{tuple(relation_value_weight.shape)} and expected {(head_width, r.size(1))}."
            )
        if relation_value_bias is not None and tuple(relation_value_bias.shape) != (head_width,):
            raise ValueError(
                "relation_value_bias must have shape [H * D], got "
                f"{tuple(relation_value_bias.shape)} and expected {(head_width,)}."
            )


def _reference_graph_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    r: Optional[Tensor],
    relation_key_weight: Optional[Tensor],
    relation_value_weight: Optional[Tensor],
    relation_value_bias: Optional[Tensor],
    metadata: GraphAttentionMetadata,
    scale: float,
    dropout_p: float,
    training: bool,
) -> Tensor:
    """검증용 PyTorch graph attention입니다.

    CUDA 학습에서는 이 경로를 쓰지 않습니다. GPU에서 FlashAttention이 없으면 조용히 느린
    fallback을 타지 않고 에러를 냅니다. CPU unit test에서는 이 함수로 수식 일치성을
    확인합니다.

    Args:
        q: target node query입니다. shape은 ``[N_dst, H, D]`` 입니다.
        k: source node key입니다. shape은 ``[N_src, H, D]`` 입니다.
        v: source node value입니다. shape은 ``[N_src, H, D]`` 입니다.
        r: 정렬된 edge relation feature입니다. 있을 때 shape은 ``[E, R]`` 입니다.
        relation_key_weight: relation key projection weight입니다.
        relation_value_weight: relation value projection weight입니다.
        relation_value_bias: relation value projection bias입니다.
        metadata: target별 edge 위치 정보입니다.
        scale: score에 곱하는 값입니다.
        dropout_p: attention dropout 확률입니다.
        training: 학습 모드 여부입니다.

    Returns:
        torch.Tensor: target node별 attention 결과입니다. shape은 ``[N_dst, H, D]`` 입니다.
    """
    num_dst, num_heads, head_dim = q.shape
    src = metadata.sorted_src
    dst = metadata.sorted_dst
    num_edges = src.numel()
    output = q.new_zeros((num_dst, num_heads, head_dim))
    if num_edges == 0:
        return output

    k_edge = k.index_select(0, src)
    v_edge = v.index_select(0, src)
    if r is not None:
        relation_dtype = r.dtype
        key_rel = torch.nn.functional.linear(
            r,
            relation_key_weight.to(relation_dtype),
        ).view(num_edges, num_heads, head_dim)
        value_rel = torch.nn.functional.linear(
            r,
            relation_value_weight.to(relation_dtype),
            relation_value_bias.to(relation_dtype) if relation_value_bias is not None else None,
        ).view(num_edges, num_heads, head_dim)
        k_edge = k_edge + key_rel
        v_edge = v_edge + value_rel

    q_edge = q.index_select(0, dst)
    score = (q_edge * k_edge).sum(dim=-1) * float(scale)
    max_score = torch.full((num_dst, num_heads), -torch.inf, device=q.device, dtype=score.dtype)
    max_score.scatter_reduce_(0, dst[:, None].expand(-1, num_heads), score, reduce="amax", include_self=True)
    safe_max_score = torch.where(torch.isfinite(max_score), max_score, torch.zeros_like(max_score))
    exp_score = torch.exp(score - safe_max_score.index_select(0, dst))
    denom = torch.zeros((num_dst, num_heads), device=q.device, dtype=score.dtype)
    denom.scatter_add_(0, dst[:, None].expand(-1, num_heads), exp_score)
    attn = exp_score / denom.index_select(0, dst).clamp_min(torch.finfo(score.dtype).tiny)
    if training and dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p, training=True)
    output.scatter_add_(
        0,
        dst[:, None, None].expand(-1, num_heads, head_dim),
        v_edge * attn.to(dtype=v_edge.dtype).unsqueeze(-1),
    )
    return output


def _load_flash_varlen_impl():
    """FlashAttention varlen 구현을 지연 import합니다.

    Returns:
        module: FlashAttention varlen graph attention 함수입니다.

    Raises:
        RuntimeError: FlashAttention을 불러올 수 없을 때 발생합니다.
    """
    try:
        from src.smart.layers.graph_flash_attention_flash import graph_flash_attention_varlen
    except Exception as exc:  # pragma: no cover - CUDA 환경에서 확인됩니다.
        raise RuntimeError(
            "CUDA graph_flash_attention requires the FlashAttention varlen backend. "
            "Install the flash-attn wheel that matches the CUDA/PyTorch runtime."
        ) from exc
    return graph_flash_attention_varlen


def _load_cuda_segmented_impl():
    """C++/CUDA segmented graph attention 구현을 지연 import합니다."""
    try:
        from src.smart.layers.graph_flash_attention_cuda import graph_cuda_segmented_attention
    except Exception as exc:  # pragma: no cover - CUDA pod에서 확인됩니다.
        raise RuntimeError(
            "CUDA segmented graph attention backend failed to load. "
            "Install nvcc/ninja in the active PyTorch environment and ensure CUDA_HOME is valid."
        ) from exc
    return graph_cuda_segmented_attention


def graph_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edge_index: Optional[Tensor],
    r: Optional[Tensor],
    relation_key_weight: Optional[Tensor],
    relation_value_weight: Optional[Tensor],
    relation_value_bias: Optional[Tensor],
    scale: float,
    dropout_p: float,
    training: bool,
    metadata: Optional[GraphAttentionMetadata] = None,
    r_is_sorted: bool = False,
) -> Tensor:
    """CAT-K ``AttentionLayer``의 graph attention을 계산합니다.

    CUDA에서는 기본적으로 FlashAttention varlen backend를 사용합니다. 이 경로는
    target node별 neighbor edge를 variable-length sequence로 넘겨 edge별 score,
    attention weight, weighted value를 forward activation으로 저장하지 않습니다.
    CPU에서는 unit test를 위한 reference 경로만 제공합니다.

    Args:
        q: target node query입니다. shape은 ``[N_dst, H, D]`` 입니다.
        k: source node key입니다. shape은 ``[N_src, H, D]`` 입니다.
        v: source node value입니다. shape은 ``[N_src, H, D]`` 입니다.
        edge_index: source에서 target으로 가는 edge 목록입니다. shape은 ``[2, E]`` 입니다.
            ``metadata`` 가 없을 때만 필요합니다.
        r: edge relation feature입니다. 있을 때 shape은 ``[E, R]`` 입니다.
        relation_key_weight: relation key projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_weight: relation value projection weight입니다. shape은 ``[H * D, R]`` 입니다.
        relation_value_bias: relation value projection bias입니다. shape은 ``[H * D]`` 입니다.
        scale: score에 곱하는 값입니다.
        dropout_p: attention dropout 확률입니다.
        training: 학습 모드 여부입니다.
        metadata: target 기준 edge 정렬 정보입니다. 없으면 함수 안에서 한 번 만듭니다.
        r_is_sorted: ``r`` 이 이미 ``metadata`` 순서로 정렬됐는지 여부입니다.

    Returns:
        torch.Tensor: target node별 attention 결과입니다. shape은 ``[N_dst, H, D]`` 입니다.
    """
    if metadata is None:
        if edge_index is None:
            raise ValueError("edge_index is required when metadata is None.")
        metadata = build_graph_attention_metadata(edge_index=edge_index, num_dst_nodes=q.size(0))
        r_sorted = metadata.reorder_edge_features(r)
    else:
        r_sorted = r if r_is_sorted else metadata.reorder_edge_features(r)

    _validate_inputs(
        q=q,
        k=k,
        v=v,
        r=r_sorted,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        metadata=metadata,
        dropout_p=dropout_p,
    )

    if q.is_cuda:
        backend = os.environ.get("CATK_GRAPH_ATTENTION_BACKEND", "cuda_segmented").strip().lower()
        if backend in {"cuda", "cuda_segmented", "segmented_cuda", "segmented"}:
            cuda_graph_attention = _load_cuda_segmented_impl()
            return cuda_graph_attention(
                q=q,
                k=k,
                v=v,
                r=r_sorted,
                relation_key_weight=relation_key_weight,
                relation_value_weight=relation_value_weight,
                relation_value_bias=relation_value_bias,
                metadata=metadata,
                scale=float(scale),
                dropout_p=float(dropout_p),
                training=bool(training),
            )
        if backend in {"flash", "flash_varlen", "flash-attn", "flash_attention"}:
            flash_graph_attention = _load_flash_varlen_impl()
            return flash_graph_attention(
                q=q,
                k=k,
                v=v,
                r=r_sorted,
                relation_key_weight=relation_key_weight,
                relation_value_weight=relation_value_weight,
                relation_value_bias=relation_value_bias,
                metadata=metadata,
                scale=float(scale),
                dropout_p=float(dropout_p),
                training=bool(training),
            )
        if backend in {"reference", "torch", "torch_reference"}:
            return _reference_graph_attention(
                q=q,
                k=k,
                v=v,
                r=r_sorted,
                relation_key_weight=relation_key_weight,
                relation_value_weight=relation_value_weight,
                relation_value_bias=relation_value_bias,
                metadata=metadata,
                scale=float(scale),
                dropout_p=float(dropout_p),
                training=bool(training),
            )
        raise ValueError(
            "CATK_GRAPH_ATTENTION_BACKEND must be one of cuda_segmented, flash_varlen, or torch_reference, "
            f"got {backend!r}."
        )

    return _reference_graph_attention(
        q=q,
        k=k,
        v=v,
        r=r_sorted,
        relation_key_weight=relation_key_weight,
        relation_value_weight=relation_value_weight,
        relation_value_bias=relation_value_bias,
        metadata=metadata,
        scale=float(scale),
        dropout_p=float(dropout_p),
        training=bool(training),
    )
