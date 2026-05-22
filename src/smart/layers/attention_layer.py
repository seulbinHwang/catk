import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

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
            self._compiled_relation_kv_project = None
            self._compile_relation_kv_project = os.environ.get(
                "CATK_COMPILE_ATTENTION_RELATION_KV", "1"
            ).lower() not in {"0", "false", "off", "no"}
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def _relation_kv_project_eager(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = F.layer_norm(
            r,
            self.attn_prenorm_r.normalized_shape,
            self.attn_prenorm_r.weight,
            self.attn_prenorm_r.bias,
            self.attn_prenorm_r.eps,
        )
        k_r = F.linear(r, self.to_k_r.weight, self.to_k_r.bias).view(-1, self.num_heads, self.head_dim)
        v_r = F.linear(r, self.to_v_r.weight, self.to_v_r.bias).view(-1, self.num_heads, self.head_dim)
        return k_r, v_r

    def _relation_kv_project(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._compile_relation_kv_project or not torch.cuda.is_available():
            return self._relation_kv_project_eager(r)
        try:
            if self._compiled_relation_kv_project is None:
                self._compiled_relation_kv_project = torch.compile(
                    self._relation_kv_project_eager,
                    dynamic=True,
                    mode="reduce-overhead",
                )
            return self._compiled_relation_kv_project(r)
        except Exception:
            self._compile_relation_kv_project = False
            self._compiled_relation_kv_project = None
            return self._relation_kv_project_eager(r)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r_k: Optional[torch.Tensor],
        r_v: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if r_k is not None and r_v is not None:
            k_j = k_j + r_k
            v_j = v_j + r_v
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

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
    ) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        r_k: Optional[torch.Tensor] = None
        r_v: Optional[torch.Tensor] = None
        if self.has_pos_emb and r is not None:
            r_k, r_v = self._relation_kv_project(r)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r_k=r_k, r_v=r_v)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
