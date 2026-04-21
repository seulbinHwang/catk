from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
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
        activation_recompute: bool = True,
        store_attention_weights: bool = False,
        **kwargs
    ) -> None:
        super(AttentionLayer, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.activation_recompute = bool(activation_recompute)
        self.store_attention_weights = bool(store_attention_weights)
        self.attention_weight: Optional[torch.Tensor] = None
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

    def set_activation_recompute(self, enabled: bool) -> None:
        """학습 중 attention 중간값 재계산 사용 여부를 바꿉니다.

        Args:
            enabled: ``True``이면 학습 forward에서 edge attention 중간값을 오래
                저장하지 않고, backward 때 필요한 부분을 다시 계산합니다.
                ``False``이면 기존처럼 forward 중간값을 저장합니다.

        Notes:
            이 함수는 tensor를 새로 만들지 않습니다. 학습 목표, 입력 길이,
            attention 반경, loss 계산은 바뀌지 않습니다.
        """
        self.activation_recompute = bool(enabled)

    def set_store_attention_weights(self, enabled: bool) -> None:
        """학습 중 attention weight 저장 여부를 바꿉니다.

        Args:
            enabled: ``True``이면 학습 중에도 edge별 attention weight를
                ``self.attention_weight``에 저장합니다. 메모리 절약이 목적이면
                기본값인 ``False``를 유지하는 편이 안전합니다.

        Notes:
            attention weight는 시각화나 디버깅용 값입니다. loss 계산에는 쓰지
            않으므로 기본 학습에서는 저장하지 않습니다.
        """
        self.store_attention_weights = bool(enabled)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src_raw = x_dst_raw = x
        else:
            x_src_raw, x_dst_raw = x

        attn_out = self._run_attn_block(
            x_src_raw=x_src_raw,
            x_dst_raw=x_dst_raw,
            r=r,
            edge_index=edge_index,
        )
        x_out = x_dst_raw + self.attn_postnorm(attn_out)
        x_out = x_out + self.ff_postnorm(self._ff_block(self.ff_prenorm(x_out)))
        return x_out

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
        if self.store_attention_weights or not self.training:
            self.attention_weight = attn.sum(-1).detach()
        else:
            self.attention_weight = None
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _should_recompute_attention(self) -> bool:
        """현재 forward에서 attention 중간값 재계산을 쓸지 정합니다.

        Returns:
            bool: 학습 중이고 gradient 계산이 켜져 있으며, 이 레이어의
            재계산 옵션이 켜져 있으면 ``True``입니다.

        Notes:
            tensor shape을 바꾸지 않는 순수 실행 방식 선택 함수입니다.
            따라서 예측 길이와 학습 loss는 그대로 유지됩니다.
        """
        return (
            self.activation_recompute
            and self.training
            and torch.is_grad_enabled()
        )

    def _run_attn_block(
        self,
        x_src_raw: torch.Tensor,
        x_dst_raw: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """attention 블록을 실행하고, 학습 시에는 중간값 저장을 줄입니다.

        Args:
            x_src_raw: source node feature입니다. shape은
                ``[num_source_nodes, hidden_dim]`` 입니다.
            x_dst_raw: destination node feature입니다. shape은
                ``[num_destination_nodes, hidden_dim]`` 입니다.
            r: edge feature입니다. ``None``이 아니면 shape은
                ``[num_edges, hidden_dim]`` 입니다.
            edge_index: source와 destination 연결 정보입니다. shape은
                ``[2, num_edges]`` 입니다.

        Returns:
            torch.Tensor: destination node별 attention 결과입니다. shape은
            ``[num_destination_nodes, hidden_dim]`` 입니다.

        Notes:
            재계산을 켠 경우에도 입력 tensor와 출력 tensor의 의미는 같습니다.
            forward에서 q/k/v, edge score, edge message 같은 큰 중간값을 오래
            저장하지 않고 backward에서 필요한 순간 다시 만듭니다.
        """
        if not self._should_recompute_attention():
            return self._attn_block_from_raw(
                x_src_raw=x_src_raw,
                x_dst_raw=x_dst_raw,
                r=r,
                edge_index=edge_index,
            )

        return activation_checkpoint(
            self._attn_block_from_raw,
            x_src_raw,
            x_dst_raw,
            r,
            edge_index,
            use_reentrant=False,
            preserve_rng_state=True,
        )

    def _attn_block_from_raw(
        self,
        x_src_raw: torch.Tensor,
        x_dst_raw: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """정규화부터 message passing까지 attention 계산을 한 번 수행합니다.

        Args:
            x_src_raw: source node feature입니다. shape은
                ``[num_source_nodes, hidden_dim]`` 입니다.
            x_dst_raw: destination node feature입니다. shape은
                ``[num_destination_nodes, hidden_dim]`` 입니다.
            r: edge feature입니다. ``None``이 아니면 shape은
                ``[num_edges, hidden_dim]`` 입니다.
            edge_index: source와 destination 연결 정보입니다. shape은
                ``[2, num_edges]`` 입니다.

        Returns:
            torch.Tensor: destination node별 attention 결과입니다. shape은
            ``[num_destination_nodes, hidden_dim]`` 입니다.
        """
        if (
            x_src_raw is x_dst_raw
            and self.attn_prenorm_x_src is self.attn_prenorm_x_dst
        ):
            x_src = x_dst = self.attn_prenorm_x_src(x_src_raw)
        else:
            x_src = self.attn_prenorm_x_src(x_src_raw)
            x_dst = self.attn_prenorm_x_dst(x_dst_raw)

        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)

        return self._attn_block(x_src, x_dst, r, edge_index)

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
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
