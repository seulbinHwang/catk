import math
import os
from typing import List, Optional

import torch
import torch.nn as nn

from src.smart.utils import weight_init


class FourierEmbedding(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._compiled_embed_continuous = None
        self._compile_continuous = os.environ.get(
            "CATK_COMPILE_FOURIER_EMBEDDING", "1"
        ).lower() not in {"0", "false", "off", "no"}
        self.apply(weight_init)

    def _embed_continuous_loop(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # Warning: if your data are noisy, don't use learnable sinusoidal embedding
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            continuous_embs[i] = self.mlps[i](x[:, i])
        return torch.stack(continuous_embs).sum(dim=0)

    def _embed_continuous_accumulated(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # Warning: if your data are noisy, don't use learnable sinusoidal embedding
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_emb = self.mlps[0](x[:, 0])
        for i in range(1, self.input_dim):
            continuous_emb = continuous_emb + self.mlps[i](x[:, i])
        return continuous_emb

    def _embed_continuous(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        if not self._compile_continuous or not torch.cuda.is_available():
            return self._embed_continuous_accumulated(continuous_inputs)
        try:
            if self._compiled_embed_continuous is None:
                self._compiled_embed_continuous = torch.compile(
                    self._embed_continuous_accumulated,
                    dynamic=True,
                    # Edge counts vary by SMART batch. CUDA graph capture can
                    # keep stale graph-pool storage alive across backward
                    # passes, so keep Inductor compilation but disable graphs.
                    options={"triton.cudagraphs": False},
                )
            return self._compiled_embed_continuous(continuous_inputs)
        except Exception:
            self._compile_continuous = False
            self._compiled_embed_continuous = None
            return self._embed_continuous_accumulated(continuous_inputs)

    @staticmethod
    def _sum_embeddings(embs: List[torch.Tensor]) -> torch.Tensor:
        x = embs[0]
        for emb in embs[1:]:
            x = x + emb
        return x

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = self._sum_embeddings(categorical_embs)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = self._embed_continuous(continuous_inputs)
            if categorical_embs is not None:
                x = x + self._sum_embeddings(categorical_embs)
        return self.to_out(x)


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(MLPEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = self.mlp(continuous_inputs)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return x
