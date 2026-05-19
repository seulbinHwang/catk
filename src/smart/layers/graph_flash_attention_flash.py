from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.smart.layers.graph_flash_attention import GraphAttentionMetadata


def _load_flash_attn_varlen_func():
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception as exc:  # pragma: no cover - exercised on GPU pods.
        raise RuntimeError(
            "FlashAttention graph attention requires flash-attn. Install the wheel "
            "matching the CUDA/PyTorch runtime before CUDA training."
        ) from exc
    return flash_attn_varlen_func


def _next_multiple(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _pad_head_dim(x: Tensor, padded_dim: int) -> Tensor:
    pad = int(padded_dim) - int(x.size(-1))
    if pad == 0:
        return x.contiguous()
    return F.pad(x, (0, pad)).contiguous()


def _project_relation(
    r: Tensor,
    relation_key_weight: Tensor,
    relation_value_weight: Tensor,
    relation_value_bias: Optional[Tensor],
    *,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    relation_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else r.dtype
    r_project = r.to(relation_dtype)
    key_weight = relation_key_weight.to(relation_dtype)
    value_weight = relation_value_weight.to(relation_dtype)
    value_bias = relation_value_bias.to(relation_dtype) if relation_value_bias is not None else None
    relation_key = F.linear(r_project, key_weight).view(-1, num_heads, head_dim)
    relation_value = F.linear(r_project, value_weight, value_bias).view(-1, num_heads, head_dim)
    return relation_key.contiguous(), relation_value.contiguous()


def _max_edges_per_flash_chunk() -> int:
    value = os.environ.get("CATK_FLASH_GRAPH_MAX_EDGES", "131072")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"CATK_FLASH_GRAPH_MAX_EDGES must be an integer, got {value!r}.") from exc
    return max(1, parsed)


def _flash_varlen_chunk_impl(
    q_chunk: Tensor,
    k: Tensor,
    v: Tensor,
    r: Tensor,
    relation_key_weight: Tensor,
    relation_value_weight: Tensor,
    relation_value_bias: Tensor,
    sorted_src: Tensor,
    dst_ptr: Tensor,
    target_start: int,
    target_end: int,
    edge_start: int,
    edge_end: int,
    max_degree: int,
    scale: float,
    dropout_p: float,
    training: bool,
    has_relation: bool,
    has_value_bias: bool,
) -> Tensor:
    flash_attn_varlen_func = _load_flash_attn_varlen_func()
    chunk_num_dst, num_heads, head_dim = q_chunk.shape
    output = q_chunk.new_zeros((chunk_num_dst, num_heads, head_dim))
    if edge_end <= edge_start:
        return output

    degrees = dst_ptr[target_start + 1 : target_end + 1] - dst_ptr[target_start:target_end]
    nonempty_dst = torch.nonzero(degrees > 0, as_tuple=False).flatten()
    if nonempty_dst.numel() == 0:
        return output

    sorted_src_chunk = sorted_src[edge_start:edge_end]
    k_edge = k.index_select(0, sorted_src_chunk)
    v_edge = v.index_select(0, sorted_src_chunk)
    if has_relation:
        relation_key, relation_value = _project_relation(
            r,
            relation_key_weight,
            relation_value_weight,
            relation_value_bias if has_value_bias else None,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=q_chunk.dtype,
        )
        k_edge = k_edge + relation_key
        v_edge = v_edge + relation_value

    q_pack = q_chunk.index_select(0, nonempty_dst)
    degrees_nonempty = degrees.index_select(0, nonempty_dst).to(torch.int32)
    cu_seqlens_q = torch.arange(
        nonempty_dst.numel() + 1,
        device=q_chunk.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.empty(nonempty_dst.numel() + 1, device=q_chunk.device, dtype=torch.int32)
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = torch.cumsum(degrees_nonempty, dim=0)

    padded_dim = max(8, _next_multiple(head_dim, 8))
    q_flash = _pad_head_dim(q_pack, padded_dim)
    k_flash = _pad_head_dim(k_edge, padded_dim)
    v_flash = _pad_head_dim(v_edge, padded_dim)

    out_pack = flash_attn_varlen_func(
        q_flash,
        k_flash,
        v_flash,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=int(max_degree),
        dropout_p=float(dropout_p) if training else 0.0,
        softmax_scale=float(scale),
        causal=False,
    )
    output.index_copy_(0, nonempty_dst, out_pack[..., :head_dim].to(output.dtype))
    return output


def _flash_varlen_forward_chunks(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    r: Tensor,
    relation_key_weight: Tensor,
    relation_value_weight: Tensor,
    relation_value_bias: Tensor,
    sorted_src: Tensor,
    dst_ptr: Tensor,
    chunk_ranges: tuple[tuple[int, int, int, int], ...],
    max_degree: int,
    scale: float,
    dropout_p: float,
    training: bool,
    has_relation: bool,
    has_value_bias: bool,
) -> Tensor:
    output = q.new_zeros(q.shape)
    for target_start, target_end, edge_start, edge_end in chunk_ranges:
        q_chunk = q[target_start:target_end]
        out_chunk = _flash_varlen_chunk_impl(
            q_chunk=q_chunk,
            k=k,
            v=v,
            r=r[edge_start:edge_end] if has_relation else r,
            relation_key_weight=relation_key_weight,
            relation_value_weight=relation_value_weight,
            relation_value_bias=relation_value_bias,
            sorted_src=sorted_src,
            dst_ptr=dst_ptr,
            target_start=int(target_start),
            target_end=int(target_end),
            edge_start=int(edge_start),
            edge_end=int(edge_end),
            max_degree=int(max_degree),
            scale=float(scale),
            dropout_p=float(dropout_p),
            training=bool(training),
            has_relation=bool(has_relation),
            has_value_bias=bool(has_value_bias),
        )
        output[target_start:target_end] = out_chunk
    return output


class _FlashVarlenGraphAttention(torch.autograd.Function):
    """Forward activation을 저장하지 않는 FlashAttention varlen wrapper입니다."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        r: Tensor,
        relation_key_weight: Tensor,
        relation_value_weight: Tensor,
        relation_value_bias: Tensor,
        sorted_src: Tensor,
        dst_ptr: Tensor,
        chunk_ranges: tuple[tuple[int, int, int, int], ...],
        max_degree: int,
        scale: float,
        dropout_p: float,
        training: bool,
        has_relation: bool,
        has_value_bias: bool,
    ) -> Tensor:
        use_dropout = bool(training) and float(dropout_p) > 0.0
        rng_state = torch.cuda.get_rng_state(q.device) if use_dropout else q.new_empty((0,), dtype=torch.uint8)
        output = _flash_varlen_forward_chunks(
            q=q,
            k=k,
            v=v,
            r=r,
            relation_key_weight=relation_key_weight,
            relation_value_weight=relation_value_weight,
            relation_value_bias=relation_value_bias,
            sorted_src=sorted_src,
            dst_ptr=dst_ptr,
            chunk_ranges=chunk_ranges,
            max_degree=int(max_degree),
            scale=float(scale),
            dropout_p=float(dropout_p),
            training=bool(training),
            has_relation=bool(has_relation),
            has_value_bias=bool(has_value_bias),
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            r,
            relation_key_weight,
            relation_value_weight,
            relation_value_bias,
            sorted_src,
            dst_ptr,
            rng_state,
        )
        ctx.chunk_ranges = chunk_ranges
        ctx.max_degree = int(max_degree)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.training = bool(training)
        ctx.has_relation = bool(has_relation)
        ctx.has_value_bias = bool(has_value_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (
            q,
            k,
            v,
            r,
            relation_key_weight,
            relation_value_weight,
            relation_value_bias,
            sorted_src,
            dst_ptr,
            rng_state,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        k_re = k.detach().requires_grad_(True)
        v_re = v.detach().requires_grad_(True)
        key_weight_re = relation_key_weight.detach().requires_grad_(ctx.has_relation)
        value_weight_re = relation_value_weight.detach().requires_grad_(ctx.has_relation)
        value_bias_re = relation_value_bias.detach().requires_grad_(ctx.has_relation and ctx.has_value_bias)

        grad_q = torch.zeros_like(q) if needs[0] else None
        grad_k = torch.zeros_like(k) if needs[1] else None
        grad_v = torch.zeros_like(v) if needs[2] else None
        grad_r = torch.zeros_like(r) if ctx.has_relation and needs[3] else None
        grad_key_weight = (
            torch.zeros_like(relation_key_weight) if ctx.has_relation and needs[4] else None
        )
        grad_value_weight = (
            torch.zeros_like(relation_value_weight) if ctx.has_relation and needs[5] else None
        )
        grad_value_bias = (
            torch.zeros_like(relation_value_bias)
            if ctx.has_relation and ctx.has_value_bias and needs[6]
            else None
        )

        use_dropout = bool(ctx.training) and float(ctx.dropout_p) > 0.0
        device_index = q.device.index
        fork_devices = [device_index] if device_index is not None else None
        with torch.enable_grad():
            with torch.random.fork_rng(devices=fork_devices, enabled=use_dropout):
                if use_dropout:
                    torch.cuda.set_rng_state(rng_state, q.device)
                for target_start, target_end, edge_start, edge_end in ctx.chunk_ranges:
                    q_chunk = q[target_start:target_end].detach().requires_grad_(True)
                    r_chunk = (
                        r[edge_start:edge_end].detach().requires_grad_(True)
                        if ctx.has_relation
                        else r.detach()
                    )
                    grad_targets = [q_chunk, k_re, v_re]
                    if ctx.has_relation:
                        grad_targets.extend([r_chunk, key_weight_re, value_weight_re])
                        if ctx.has_value_bias:
                            grad_targets.append(value_bias_re)
                    output = _flash_varlen_chunk_impl(
                        q_chunk=q_chunk,
                        k=k_re,
                        v=v_re,
                        r=r_chunk,
                        relation_key_weight=key_weight_re,
                        relation_value_weight=value_weight_re,
                        relation_value_bias=value_bias_re,
                        sorted_src=sorted_src,
                        dst_ptr=dst_ptr,
                        target_start=int(target_start),
                        target_end=int(target_end),
                        edge_start=int(edge_start),
                        edge_end=int(edge_end),
                        max_degree=int(ctx.max_degree),
                        scale=float(ctx.scale),
                        dropout_p=float(ctx.dropout_p),
                        training=bool(ctx.training),
                        has_relation=bool(ctx.has_relation),
                        has_value_bias=bool(ctx.has_value_bias),
                    )
                    grads = torch.autograd.grad(
                        outputs=output,
                        inputs=grad_targets,
                        grad_outputs=grad_output[target_start:target_end],
                        allow_unused=True,
                    )
                    if needs[0] and grads[0] is not None:
                        grad_q[target_start:target_end].add_(grads[0])
                    if needs[1] and grads[1] is not None:
                        grad_k.add_(grads[1])
                    if needs[2] and grads[2] is not None:
                        grad_v.add_(grads[2])
                    if ctx.has_relation:
                        if needs[3] and grads[3] is not None:
                            grad_r[edge_start:edge_end].add_(grads[3])
                        if needs[4] and grads[4] is not None:
                            grad_key_weight.add_(grads[4])
                        if needs[5] and grads[5] is not None:
                            grad_value_weight.add_(grads[5])
                        if ctx.has_value_bias and needs[6] and grads[6] is not None:
                            grad_value_bias.add_(grads[6])

        return (
            grad_q,
            grad_k,
            grad_v,
            grad_r,
            grad_key_weight,
            grad_value_weight,
            grad_value_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def graph_flash_attention_varlen(
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
    """FlashAttention varlen cross-attention으로 graph attention을 계산합니다.

    각 target node를 query 길이 1인 sequence로 보고, target에 연결된 edge들을
    variable-length key/value sequence로 넘깁니다. Custom autograd wrapper가
    forward의 edge별 K/V activation 저장을 막고 backward에서 현재 layer에 필요한
    edge tensor만 다시 만듭니다.
    """
    if not q.is_cuda:
        raise RuntimeError("graph_flash_attention_varlen requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "FlashAttention graph attention requires fp16 or bf16 tensors. "
            f"Got q.dtype={q.dtype}."
        )
    _load_flash_attn_varlen_func()

    has_relation = r is not None
    if has_relation:
        if relation_key_weight is None or relation_value_weight is None:
            raise ValueError("relation projection weights are required when r is not None.")
        r_tensor = r.contiguous()
        key_weight = relation_key_weight.contiguous()
        value_weight = relation_value_weight.contiguous()
        has_value_bias = relation_value_bias is not None
        value_bias = relation_value_bias.contiguous() if has_value_bias else q.new_empty((0,))
    else:
        r_tensor = q.new_empty((metadata.sorted_src.numel(), 0))
        key_weight = q.new_empty((0, 0))
        value_weight = q.new_empty((0, 0))
        value_bias = q.new_empty((0,))
        has_value_bias = False

    chunk_ranges = metadata.target_chunk_ranges(_max_edges_per_flash_chunk())
    return _FlashVarlenGraphAttention.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_tensor,
        key_weight,
        value_weight,
        value_bias,
        metadata.sorted_src.contiguous(),
        metadata.dst_ptr.contiguous(),
        chunk_ranges,
        int(metadata.max_degree),
        float(scale),
        float(dropout_p),
        bool(training),
        bool(has_relation),
        bool(has_value_bias),
    )
