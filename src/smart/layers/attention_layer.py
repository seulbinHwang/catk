import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from src.smart.layers.segmented_graph_attention import GraphAttentionMetadata, segmented_graph_attention
from src.smart.utils import weight_init


class AttentionLayer(MessagePassing):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        **kwargs
    ) -> None:
        super(AttentionLayer, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    @staticmethod
    def _hybrid_edge_threshold() -> int:
        value = os.environ.get("CATK_HYBRID_SEGMENTED_EDGE_THRESHOLD", "100000")
        try:
            return max(0, int(value))
        except ValueError as exc:
            raise ValueError(
                "CATK_HYBRID_SEGMENTED_EDGE_THRESHOLD must be an integer, "
                f"got {value!r}."
            ) from exc

    @staticmethod
    def _attention_backend_policy() -> str:
        return os.environ.get("CATK_ATTENTION_LAYER_BACKEND", "hybrid").strip().lower()

    @staticmethod
    def _graph_attention_fp32_enabled() -> bool:
        value = os.environ.get("CATK_ATTENTION_GRAPH_FP32", "0")
        return value.lower() in {"1", "true", "yes", "on"}

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        self.attention_weight = attn.sum(-1).detach()
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        attention_metadata: Optional[GraphAttentionMetadata] = None,
        r_is_sorted: bool = False,
    ) -> torch.Tensor:
        """attention layer를 실행합니다.

        Args:
            x: node feature입니다. 일반 graph일 때 shape은 ``[N, C]`` 이고,
                bipartite graph일 때 ``([N_src, C], [N_dst, C])`` 입니다.
            r: edge relation feature입니다. 있을 때 shape은 ``[E, C]`` 입니다.
            edge_index: source에서 target으로 가는 edge 목록입니다. shape은 ``[2, E]`` 입니다.
            attention_metadata: target 기준 edge 정렬 정보입니다. 여러 layer에서 같은 edge를
                공유할 때 재사용합니다.
            r_is_sorted: ``r`` 이 이미 ``attention_metadata`` 순서로 정렬됐는지 여부입니다.

        Returns:
            torch.Tensor: target node feature입니다. shape은 ``[N_dst, C]`` 입니다.
        """
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(
            self._attn_block(
                x_src=x_src,
                x_dst=x_dst,
                r=r,
                edge_index=edge_index,
                attention_metadata=attention_metadata,
                r_is_sorted=r_is_sorted,
            )
        )
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        attention_metadata: Optional[GraphAttentionMetadata],
        r_is_sorted: bool,
    ) -> torch.Tensor:
        """큰 edge attention 중간 tensor 없이 attention block을 계산합니다.

        Args:
            x_src: source node feature입니다. shape은 ``[N_src, C]`` 입니다.
            x_dst: target node feature입니다. shape은 ``[N_dst, C]`` 입니다.
            r: edge relation feature입니다. 있을 때 shape은 ``[E, C]`` 입니다.
            edge_index: source에서 target으로 가는 edge 목록입니다. shape은 ``[2, E]`` 입니다.
            attention_metadata: target 기준 edge 정렬 정보입니다.
            r_is_sorted: ``r`` 이 이미 metadata 순서인지 여부입니다.

        Returns:
            torch.Tensor: attention block 출력입니다. shape은 ``[N_dst, C]`` 입니다.
        """
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)

        relation_key_weight = None
        relation_value_weight = None
        relation_value_bias = None
        if self.has_pos_emb and r is not None:
            relation_key_weight = self.to_k_r.weight
            relation_value_weight = self.to_v_r.weight
            relation_value_bias = self.to_v_r.bias

        self.attention_weight = None
        backend_policy = self._attention_backend_policy()
        if backend_policy not in {"hybrid", "pyg", "segmented", "cuda", "cuda_segmented"}:
            raise ValueError(
                "CATK_ATTENTION_LAYER_BACKEND must be one of hybrid, pyg, segmented, "
                f"cuda, or cuda_segmented, got {backend_policy!r}."
            )
        use_pyg = backend_policy == "pyg"
        if backend_policy == "hybrid":
            use_pyg = edge_index.size(1) < self._hybrid_edge_threshold()

        force_graph_fp32 = self._graph_attention_fp32_enabled()
        q_attn = q.float() if force_graph_fp32 else q
        k_attn = k.float() if force_graph_fp32 else k
        v_attn = v.float() if force_graph_fp32 else v
        r_attn = r.float() if force_graph_fp32 and r is not None else r
        relation_key_weight_attn = (
            relation_key_weight.float()
            if force_graph_fp32 and relation_key_weight is not None
            else relation_key_weight
        )
        relation_value_weight_attn = (
            relation_value_weight.float()
            if force_graph_fp32 and relation_value_weight is not None
            else relation_value_weight
        )
        relation_value_bias_attn = (
            relation_value_bias.float()
            if force_graph_fp32 and relation_value_bias is not None
            else relation_value_bias
        )

        if use_pyg:
            with torch.autocast(
                device_type=q_attn.device.type,
                enabled=not force_graph_fp32,
            ):
                agg = self.propagate(
                    edge_index=edge_index,
                    x_dst=x_dst,
                    q=q_attn,
                    k=k_attn,
                    v=v_attn,
                    r=r_attn,
                )
        else:
            with torch.autocast(
                device_type=q_attn.device.type,
                enabled=not force_graph_fp32,
            ):
                agg = segmented_graph_attention(
                    q=q_attn,
                    k=k_attn,
                    v=v_attn,
                    edge_index=edge_index,
                    r=r_attn if self.has_pos_emb else None,
                    relation_key_weight=relation_key_weight_attn,
                    relation_value_weight=relation_value_weight_attn,
                    relation_value_bias=relation_value_bias_attn,
                    scale=self.scale,
                    dropout_p=self.attn_drop.p,
                    training=self.training,
                    metadata=attention_metadata,
                    r_is_sorted=r_is_sorted,
                )
        agg = agg.to(dtype=x_dst.dtype)
        agg = self.update(agg, x_dst)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
