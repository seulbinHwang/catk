from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.cpp_extension import load

from src.smart.layers.graph_flash_attention import GraphAttentionMetadata


@lru_cache(maxsize=1)
def _load_cuda_ext():
    """Build and load the CAT-K segmented attention CUDA extension."""
    source_dir = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    if "CUDA_HOME" not in os.environ and (Path(sys.prefix) / "bin" / "nvcc").exists():
        os.environ["CUDA_HOME"] = sys.prefix
    return load(
        name="catk_segmented_graph_attention_v2",
        sources=[
            str(source_dir / "cuda_graph_attention_ext.cpp"),
            str(source_dir / "cuda_graph_attention_ext_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("CATK_CUDA_EXT_VERBOSE", "0") == "1",
    )


def _relation_projection_dtype(q: Tensor, r: Tensor) -> torch.dtype:
    if q.dtype in {torch.float16, torch.bfloat16}:
        return q.dtype
    return r.dtype


def _relation_grad_chunk_edges() -> int:
    value = os.environ.get("CATK_CUDA_RELATION_GRAD_CHUNK_EDGES", "65536")
    try:
        return max(1, int(value))
    except ValueError as exc:
        raise ValueError(
            "CATK_CUDA_RELATION_GRAD_CHUNK_EDGES must be an integer, "
            f"got {value!r}."
        ) from exc


def _direct_relation_enabled() -> bool:
    value = os.environ.get("CATK_CUDA_SEGMENTED_DIRECT_RELATION", "0")
    return value.lower() in {"1", "true", "yes", "on"}


def _direct_forward_enabled() -> bool:
    value = os.environ.get("CATK_CUDA_SEGMENTED_DIRECT_FORWARD", "0")
    return value.lower() in {"1", "true", "yes", "on"} or _direct_relation_enabled()


def _edge_key_value(
    k: Tensor,
    v: Tensor,
    r: Tensor,
    relation_key_weight: Tensor,
    relation_value_weight: Tensor,
    relation_value_bias: Tensor,
    sorted_src: Tensor,
    *,
    num_heads: int,
    head_dim: int,
    has_relation: bool,
    has_value_bias: bool,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    k_edge = k.index_select(0, sorted_src)
    v_edge = v.index_select(0, sorted_src)
    if not has_relation:
        return k_edge.contiguous(), v_edge.contiguous()

    r_project = r.to(dtype)
    relation_key = F.linear(r_project, relation_key_weight.to(dtype)).view(-1, num_heads, head_dim)
    relation_value = F.linear(
        r_project,
        relation_value_weight.to(dtype),
        relation_value_bias.to(dtype) if has_value_bias else None,
    ).view(-1, num_heads, head_dim)
    return (k_edge + relation_key).contiguous(), (v_edge + relation_value).contiguous()


class _CudaSegmentedGraphAttention(torch.autograd.Function):
    """CUDA segmented graph attention with backward recompute.

    Forward stores only ``lse`` plus the compact original inputs. Edge K/V and
    relation projections are recomputed once in backward, avoiding the repeated
    chunk-level ``autograd.grad`` calls used by the FlashAttention varlen wrapper.
    """

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
        sorted_dst: Tensor,
        dst_ptr: Tensor,
        scale: float,
        dropout_p: float,
        training: bool,
        has_relation: bool,
        has_value_bias: bool,
    ) -> Tensor:
        ext = _load_cuda_ext()
        effective_dropout = float(dropout_p) if bool(training) else 0.0
        seed = random.getrandbits(63) if effective_dropout > 0.0 else 0
        num_heads = q.size(1)
        head_dim = q.size(2)
        projection_dtype = _relation_projection_dtype(q, r) if has_relation else q.dtype
        direct_forward = bool(has_relation) and _direct_forward_enabled()
        direct_backward = bool(has_relation) and _direct_relation_enabled()
        if direct_forward:
            output, lse = ext.forward_direct(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                r.to(projection_dtype).contiguous(),
                relation_key_weight.to(projection_dtype).contiguous(),
                relation_value_weight.to(projection_dtype).contiguous(),
                relation_value_bias.to(projection_dtype).contiguous()
                if bool(has_value_bias)
                else q.new_empty((0,)),
                sorted_src.contiguous(),
                sorted_dst.contiguous(),
                dst_ptr.contiguous(),
                float(scale),
                effective_dropout,
                int(seed),
                bool(has_value_bias),
            )
        else:
            k_edge, v_edge = _edge_key_value(
                k=k,
                v=v,
                r=r,
                relation_key_weight=relation_key_weight,
                relation_value_weight=relation_value_weight,
                relation_value_bias=relation_value_bias,
                sorted_src=sorted_src,
                num_heads=num_heads,
                head_dim=head_dim,
                has_relation=bool(has_relation),
                has_value_bias=bool(has_value_bias),
                dtype=projection_dtype,
            )
            output, lse = ext.forward(
                q.contiguous(),
                k_edge,
                v_edge,
                sorted_dst.contiguous(),
                dst_ptr.contiguous(),
                float(scale),
                effective_dropout,
                int(seed),
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
            sorted_dst,
            dst_ptr,
            lse,
        )
        ctx.scale = float(scale)
        ctx.dropout_p = effective_dropout
        ctx.seed = int(seed)
        ctx.has_relation = bool(has_relation)
        ctx.has_value_bias = bool(has_value_bias)
        ctx.direct_relation = bool(direct_backward)
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
            sorted_dst,
            dst_ptr,
            lse,
        ) = ctx.saved_tensors
        ext = _load_cuda_ext()
        needs = ctx.needs_input_grad
        num_heads = q.size(1)
        head_dim = q.size(2)
        projection_dtype = _relation_projection_dtype(q, r) if ctx.has_relation else q.dtype
        with torch.no_grad():
            if ctx.direct_relation:
                (
                    grad_q_float,
                    grad_k_float,
                    grad_v_float,
                    grad_r_float,
                    grad_key_weight_float,
                    grad_value_weight_float,
                    grad_value_bias_float,
                ) = ext.backward_direct(
                    grad_output.contiguous(),
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    r.to(projection_dtype).contiguous(),
                    relation_key_weight.to(projection_dtype).contiguous(),
                    relation_value_weight.to(projection_dtype).contiguous(),
                    relation_value_bias.to(projection_dtype).contiguous()
                    if ctx.has_value_bias
                    else q.new_empty((0,)),
                    sorted_src.contiguous(),
                    sorted_dst.contiguous(),
                    dst_ptr.contiguous(),
                    lse.contiguous(),
                    float(ctx.scale),
                    float(ctx.dropout_p),
                    int(ctx.seed),
                    bool(ctx.has_value_bias),
                )
                grad_q = grad_q_float.to(q.dtype) if needs[0] else None
                grad_k = grad_k_float.to(k.dtype) if needs[1] else None
                grad_v = grad_v_float.to(v.dtype) if needs[2] else None
                grad_r = grad_r_float.to(r.dtype) if needs[3] else None
                grad_key_weight = (
                    grad_key_weight_float.to(relation_key_weight.dtype) if needs[4] else None
                )
                grad_value_weight = (
                    grad_value_weight_float.to(relation_value_weight.dtype) if needs[5] else None
                )
                grad_value_bias = (
                    grad_value_bias_float.to(relation_value_bias.dtype)
                    if ctx.has_value_bias and needs[6]
                    else None
                )
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
                )

            k_edge, v_edge = _edge_key_value(
                k=k,
                v=v,
                r=r,
                relation_key_weight=relation_key_weight,
                relation_value_weight=relation_value_weight,
                relation_value_bias=relation_value_bias,
                sorted_src=sorted_src,
                num_heads=num_heads,
                head_dim=head_dim,
                has_relation=bool(ctx.has_relation),
                has_value_bias=bool(ctx.has_value_bias),
                dtype=projection_dtype,
            )
            grad_q_float, grad_k_edge, grad_v_edge = ext.backward(
                grad_output.contiguous(),
                q.contiguous(),
                k_edge,
                v_edge,
                sorted_dst.contiguous(),
                dst_ptr.contiguous(),
                lse.contiguous(),
                float(ctx.scale),
                float(ctx.dropout_p),
                int(ctx.seed),
            )

            grad_q = grad_q_float.to(q.dtype) if needs[0] else None
            grad_k = None
            grad_v = None
            if needs[1]:
                grad_k_float = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
                grad_k_float.scatter_add_(
                    0,
                    sorted_src[:, None, None].expand(-1, num_heads, head_dim),
                    grad_k_edge,
                )
                grad_k = grad_k_float.to(k.dtype)
            if needs[2]:
                grad_v_float = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
                grad_v_float.scatter_add_(
                    0,
                    sorted_src[:, None, None].expand(-1, num_heads, head_dim),
                    grad_v_edge,
                )
                grad_v = grad_v_float.to(v.dtype)

            grad_r = None
            grad_key_weight = None
            grad_value_weight = None
            grad_value_bias = None
            if ctx.has_relation:
                num_edges = sorted_src.numel()
                grad_key_rel = grad_k_edge.reshape(num_edges, num_heads * head_dim)
                grad_value_rel = grad_v_edge.reshape(num_edges, num_heads * head_dim)
                chunk_edges = _relation_grad_chunk_edges()
                key_weight_float = relation_key_weight.float()
                value_weight_float = relation_value_weight.float()
                if needs[3]:
                    grad_r_float = torch.empty(
                        r.shape,
                        device=r.device,
                        dtype=torch.float32,
                    )
                    for edge_start in range(0, num_edges, chunk_edges):
                        edge_end = min(num_edges, edge_start + chunk_edges)
                        grad_r_chunk = grad_key_rel[edge_start:edge_end].matmul(
                            key_weight_float
                        )
                        grad_r_chunk.add_(
                            grad_value_rel[edge_start:edge_end].matmul(value_weight_float)
                        )
                        grad_r_float[edge_start:edge_end] = grad_r_chunk
                    grad_r = grad_r_float.to(r.dtype)
                if needs[4]:
                    grad_key_weight_float = torch.zeros(
                        relation_key_weight.shape,
                        device=relation_key_weight.device,
                        dtype=torch.float32,
                    )
                    for edge_start in range(0, num_edges, chunk_edges):
                        edge_end = min(num_edges, edge_start + chunk_edges)
                        grad_key_weight_float.add_(
                            grad_key_rel[edge_start:edge_end].t().matmul(
                                r[edge_start:edge_end].float()
                            )
                        )
                    grad_key_weight = grad_key_weight_float.to(relation_key_weight.dtype)
                if needs[5]:
                    grad_value_weight_float = torch.zeros(
                        relation_value_weight.shape,
                        device=relation_value_weight.device,
                        dtype=torch.float32,
                    )
                    for edge_start in range(0, num_edges, chunk_edges):
                        edge_end = min(num_edges, edge_start + chunk_edges)
                        grad_value_weight_float.add_(
                            grad_value_rel[edge_start:edge_end].t().matmul(
                                r[edge_start:edge_end].float()
                            )
                        )
                    grad_value_weight = grad_value_weight_float.to(relation_value_weight.dtype)
                if ctx.has_value_bias and needs[6]:
                    grad_value_bias = grad_value_rel.sum(dim=0).to(relation_value_bias.dtype)

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
        )


def graph_cuda_segmented_attention(
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
    if not q.is_cuda:
        raise RuntimeError("graph_cuda_segmented_attention requires CUDA tensors.")
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

    return _CudaSegmentedGraphAttention.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_tensor,
        key_weight,
        value_weight,
        value_bias,
        metadata.sorted_src.contiguous(),
        metadata.sorted_dst.contiguous(),
        metadata.dst_ptr.contiguous(),
        float(scale),
        float(dropout_p),
        bool(training),
        bool(has_relation),
        bool(has_value_bias),
    )
