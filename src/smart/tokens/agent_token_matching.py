from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


DEFAULT_TOKEN_MATCH_QUERY_CHUNK_SIZE = 4096


def build_agent_type_masks(agent_type: Tensor) -> Dict[str, Tensor]:
    return {
        "veh": agent_type == 0,
        "ped": agent_type == 1,
        "cyc": agent_type == 2,
    }


def _align_token_bank_and_query(
    token_bank: Tensor,
    contour_local: Tensor,
) -> Tuple[Tensor, Tensor]:
    if token_bank.dim() not in {3, 4}:
        raise ValueError(f"Unsupported token bank shape: {tuple(token_bank.shape)}")
    if contour_local.dim() not in {3, 4}:
        raise ValueError(f"Unsupported contour_local shape: {tuple(contour_local.shape)}")

    if token_bank.dim() == contour_local.dim():
        return token_bank, contour_local
    if token_bank.dim() == 4 and contour_local.dim() == 3:
        return token_bank[:, -1], contour_local
    return token_bank, contour_local[:, -1]


def _reduce_match_distance(dist: Tensor, reduction: str) -> Tensor:
    reduce_dims = tuple(range(2, dist.dim()))
    if reduction == "sum":
        return dist.sum(dim=reduce_dims)
    if reduction == "mean":
        return dist.mean(dim=reduce_dims)
    raise ValueError(f"Unsupported reduction: {reduction}")


def match_token_idx_from_local_contour(
    agent_type: Tensor,
    contour_local: Tensor,
    token_bank_all_veh: Tensor,
    token_bank_all_ped: Tensor,
    token_bank_all_cyc: Tensor,
    reduction: str,
    query_chunk_size: int = DEFAULT_TOKEN_MATCH_QUERY_CHUNK_SIZE,
) -> Tensor:
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}.")

    token_idx = torch.zeros(
        agent_type.shape[0],
        device=agent_type.device,
        dtype=torch.long,
    )
    token_banks = {
        "veh": token_bank_all_veh,
        "ped": token_bank_all_ped,
        "cyc": token_bank_all_cyc,
    }

    for token_key, mask in build_agent_type_masks(agent_type).items():
        if not mask.any():
            continue

        query_indices = mask.nonzero(as_tuple=True)[0]
        token_bank, contour_local_masked = _align_token_bank_and_query(
            token_bank=token_banks[token_key],
            contour_local=contour_local[mask],
        )
        matched_chunks = []
        for start in range(0, contour_local_masked.shape[0], query_chunk_size):
            contour_chunk = contour_local_masked[start : start + query_chunk_size]
            dist = torch.norm(
                token_bank.unsqueeze(0) - contour_chunk.unsqueeze(1),
                dim=-1,
            )
            dist = _reduce_match_distance(dist=dist, reduction=reduction)
            matched_chunks.append(torch.argmin(dist, dim=-1))
        token_idx[query_indices] = torch.cat(matched_chunks, dim=0)

    return token_idx
