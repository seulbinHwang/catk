from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.utils import weight_init


class TauTemporalBlock(nn.Module):
    """시간 값 `tau`로 조절되는 아주 작은 시간축 블록이다.

    Args:
        hidden_dim: 내부 채널 수이다.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.tau_to_scale_shift = nn.Linear(hidden_dim, hidden_dim * 2)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self, x: Tensor, tau_emb: Tensor) -> Tensor:
        """시간축 특징을 한 번 다듬는다.

        Args:
            x: [n_query, n_future_step, hidden_dim] 모양의 입력이다.
            tau_emb: [n_query, hidden_dim] 모양의 시간 임베딩이다.

        Returns:
            [n_query, n_future_step, hidden_dim] 모양의 출력이다.
        """
        residual = x
        x = self.norm(x)
        scale, shift = self.tau_to_scale_shift(tau_emb).chunk(2, dim=-1)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.gelu(x)
        x = x + self.ffn(x)
        return residual + x


class FutureConditioner(nn.Module):
    """`noised future + tau`를 작은 조건 벡터로 바꾼다.

    구조는 `4 -> 128` step projection 이후, `tau`로 조절되는 시간축 블록 2개,
    마지막 평균 풀링으로 구성된다.
    """

    def __init__(
        self,
        future_dim: int,
        hidden_dim: int,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.future_dim = future_dim
        self.hidden_dim = hidden_dim
        self.step_proj = nn.Linear(future_dim, hidden_dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [TauTemporalBlock(hidden_dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def _flatten_input(self, noised_future: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
        """앞쪽 차원을 잠시 하나로 모은다.

        Args:
            noised_future: [*, n_future_step, 4] 모양의 미래이다.
            tau: [*] 모양의 시간 값이다.

        Returns:
            평탄화된 미래, 평탄화된 시간 값, 복원용 앞쪽 모양을 돌려준다.
        """
        prefix_shape = tuple(noised_future.shape[:-2])
        noised_future = noised_future.reshape(-1, noised_future.shape[-2], noised_future.shape[-1])
        tau = tau.reshape(-1)
        return noised_future, tau, prefix_shape

    def forward(self, noised_future: Tensor, tau: Tensor) -> Tensor:
        """noised future를 조건 벡터로 바꾼다.

        Args:
            noised_future: [*, n_future_step, 4] 모양의 미래이다.
            tau: [*] 모양의 시간 값이다.

        Returns:
            [*, hidden_dim] 모양의 조건 벡터이다.
        """
        noised_future, tau, prefix_shape = self._flatten_input(noised_future, tau)
        x = self.step_proj(noised_future)
        tau_emb = self.tau_mlp(tau.unsqueeze(-1))
        for block in self.blocks:
            x = block(x, tau_emb)
        x = self.out_norm(x.mean(dim=1))
        return x.view(*prefix_shape, self.hidden_dim)


class StructuredFlowHead(nn.Module):
    """시간 구조를 보존하는 작은 per-step flow head이다."""

    def __init__(self, hidden_dim: int, num_future_steps: int, output_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.step_emb = nn.Embedding(num_future_steps, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, anchor_feature: Tensor) -> Tensor:
        """anchor 특징 하나를 20개 미래 step velocity로 바꾼다.

        Args:
            anchor_feature: [*, hidden_dim] 모양의 anchor 특징이다.

        Returns:
            [*, n_future_step, 4] 모양의 step별 velocity이다.
        """
        prefix_shape = tuple(anchor_feature.shape[:-1])
        anchor_feature = anchor_feature.reshape(-1, self.hidden_dim)
        step_feature = self.step_emb.weight.unsqueeze(0)
        x = anchor_feature.unsqueeze(1) + step_feature
        x = self.mlp(x)
        return x.view(*prefix_shape, self.num_future_steps, -1)
