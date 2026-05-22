from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


DEFAULT_TOKEN_MATCH_QUERY_CHUNK_SIZE = 4096


def build_agent_type_masks(agent_type: Tensor) -> Dict[str, Tensor]:
    """차종별 마스크를 만듭니다.

    Args:
        agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.

    Returns:
        Dict[str, Tensor]:
            ``veh``, ``ped``, ``cyc`` 키를 가지는 bool 마스크 사전입니다.
            각 마스크 shape은 ``[n_agent]`` 입니다.
    """
    return {
        "veh": agent_type == 0,
        "ped": agent_type == 1,
        "cyc": agent_type == 2,
    }


def _align_token_bank_and_query(
    token_bank: Tensor,
    contour_local: Tensor,
) -> Tuple[Tensor, Tensor]:
    """토큰 은행과 비교 대상의 시간 축 모양을 맞춥니다.

    Args:
        token_bank: 토큰 은행입니다. shape은 ``[n_token, 6, 4, 2]`` 또는
            ``[n_token, 4, 2]`` 입니다.
        contour_local: 로컬 좌표의 비교 대상입니다. shape은
            ``[n_agent, 6, 4, 2]`` 또는 ``[n_agent, 4, 2]`` 입니다.

    Returns:
        Tuple[Tensor, Tensor]:
            같은 시간 축 모양으로 맞춘 토큰 은행과 비교 대상입니다.
            반환 shape은 둘 다 ``[..., 6, 4, 2]`` 또는 둘 다 ``[..., 4, 2]`` 입니다.

    Raises:
        ValueError: 예상하지 못한 모양의 입력이 들어오면 발생합니다.
    """
    if token_bank.dim() not in {3, 4}:
        raise ValueError(
            f"Unsupported token bank shape: {tuple(token_bank.shape)}"
        )
    if contour_local.dim() not in {3, 4}:
        raise ValueError(
            f"Unsupported contour_local shape: {tuple(contour_local.shape)}"
        )

    if token_bank.dim() == contour_local.dim():
        return token_bank, contour_local

    if token_bank.dim() == 4 and contour_local.dim() == 3:
        return token_bank[:, -1], contour_local

    return token_bank, contour_local[:, -1]


def _reduce_match_distance(dist: Tensor, reduction: str) -> Tensor:
    """토큰 매칭 거리를 시간축과 사각형 점 축까지 함께 줄입니다.

    Args:
        dist: 점별 거리입니다. shape은 ``[n_agent, n_token, 4]`` 또는
            ``[n_agent, n_token, 6, 4]`` 입니다.
        reduction: ``sum`` 또는 ``mean`` 입니다.

    Returns:
        Tensor:
            토큰별 최종 거리입니다. shape은 ``[n_agent, n_token]`` 입니다.

    Raises:
        ValueError: 지원하지 않는 reduction 이면 발생합니다.
    """
    reduce_dims = tuple(range(2, dist.dim()))
    if reduction == "sum":
        return dist.sum(dim=reduce_dims)
    if reduction == "mean":
        return dist.mean(dim=reduce_dims)
    raise ValueError(f"Unsupported reduction: {reduction}")


def _flatten_match_xy(points: Tensor) -> Tuple[Tensor, Tensor]:
    """매칭용 점 좌표를 x/y별 contiguous matrix로 펼칩니다."""
    return (
        points[..., 0].flatten(1).contiguous(),
        points[..., 1].flatten(1).contiguous(),
    )


def _match_token_idx_from_flat_xy(
    token_x: Tensor,
    token_y: Tensor,
    contour_x: Tensor,
    contour_y: Tensor,
    reduction: str,
) -> Tensor:
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    dx = contour_x.unsqueeze(1) - token_x.unsqueeze(0)
    dx.square_()
    dy = contour_y.unsqueeze(1) - token_y.unsqueeze(0)
    dx.addcmul_(dy, dy)
    dx.sqrt_()
    return torch.argmin(dx.sum(dim=-1), dim=-1)


def match_token_idx_from_local_contour(
    agent_type: Tensor,
    contour_local: Tensor,
    token_bank_all_veh: Tensor,
    token_bank_all_ped: Tensor,
    token_bank_all_cyc: Tensor,
    reduction: str,
    query_chunk_size: int = DEFAULT_TOKEN_MATCH_QUERY_CHUNK_SIZE,
) -> Tensor:
    """로컬 좌표의 coarse 경로 전체를 기준으로 토큰 번호를 고릅니다.

    Args:
        agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
        contour_local: 현재 기준 좌표로 옮긴 비교 대상입니다. 기본 shape은
            ``[n_agent, 6, 4, 2]`` 이고, 이전 방식과의 호환을 위해
            ``[n_agent, 4, 2]`` 도 받을 수 있습니다.
        token_bank_all_veh: 차량 토큰 은행입니다. shape은
            ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        token_bank_all_ped: 보행자 토큰 은행입니다. shape은
            ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        token_bank_all_cyc: 자전거 토큰 은행입니다. shape은
            ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        reduction: 점별 거리를 ``sum`` 또는 ``mean`` 으로 줄이는 방법입니다.
        query_chunk_size: 한 번에 매칭할 query 개수입니다. 큰 batch로 묶은
            tokenization에서 peak memory가 튀지 않도록 query 축만 나눕니다.

    Returns:
        Tensor:
            선택된 토큰 번호입니다. shape은 ``[n_agent]`` 입니다.

    Raises:
        ValueError: reduction 값이 잘못됐거나 chunk 크기가 0 이하일 때 발생합니다.
    """
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
        token_x, token_y = _flatten_match_xy(token_bank)
        matched_chunks = []
        for start in range(0, contour_local_masked.shape[0], query_chunk_size):
            contour_chunk = contour_local_masked[start : start + query_chunk_size]
            contour_x, contour_y = _flatten_match_xy(contour_chunk)
            matched_chunks.append(
                _match_token_idx_from_flat_xy(
                    token_x=token_x,
                    token_y=token_y,
                    contour_x=contour_x,
                    contour_y=contour_y,
                    reduction=reduction,
                )
            )
        token_idx[query_indices] = torch.cat(matched_chunks, dim=0)

    return token_idx
