from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.distributions import Categorical


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


def _get_last_step_token_bank(token_bank: Tensor) -> Tensor:
    """토큰 은행에서 마지막 coarse 시점 사각형만 꺼냅니다.

    Args:
        token_bank: 토큰 은행입니다. shape은 ``[n_token, 6, 4, 2]`` 또는
            이미 마지막 시점만 남긴 ``[n_token, 4, 2]`` 입니다.

    Returns:
        Tensor:
            마지막 coarse 시점 사각형입니다. shape은 ``[n_token, 4, 2]`` 입니다.

    Raises:
        ValueError: 예상하지 못한 모양의 토큰 은행이 들어오면 발생합니다.
    """
    if token_bank.dim() == 4:
        return token_bank[:, -1]
    if token_bank.dim() == 3:
        return token_bank
    raise ValueError(
        f"Unsupported token bank shape: {tuple(token_bank.shape)}"
    )


def match_token_idx_from_local_contour(
    agent_type: Tensor,
    contour_local: Tensor,
    token_bank_all_veh: Tensor,
    token_bank_all_ped: Tensor,
    token_bank_all_cyc: Tensor,
    reduction: str,
    num_k: int = 1,
    sample_topk: bool = False,
    sampling_temp: float | None = None,
) -> Tensor:
    """로컬 좌표의 마지막 시점 사각형으로 토큰 번호를 고릅니다.

    Args:
        agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
        contour_local: 현재 기준 좌표로 옮긴 마지막 시점 사각형입니다.
            shape은 ``[n_agent, 4, 2]`` 입니다.
        token_bank_all_veh: 차량 토큰 은행입니다.
            shape은 ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        token_bank_all_ped: 보행자 토큰 은행입니다.
            shape은 ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        token_bank_all_cyc: 자전거 토큰 은행입니다.
            shape은 ``[n_token, 6, 4, 2]`` 또는 ``[n_token, 4, 2]`` 입니다.
        reduction: 점별 거리를 ``sum`` 또는 ``mean`` 으로 줄이는 방법입니다.
        num_k: 샘플 후보 개수입니다.
        sample_topk: True면 top-k 안에서 하나를 뽑습니다.
        sampling_temp: top-k 샘플링에 쓰는 온도값입니다.

    Returns:
        Tensor:
            선택된 토큰 번호입니다. shape은 ``[n_agent]`` 입니다.

    Raises:
        ValueError: reduction 값이 잘못됐거나 샘플링 온도가 없을 때 발생합니다.
    """
    token_idx = torch.zeros(
        agent_type.shape[0],
        device=agent_type.device,
        dtype=torch.long,
    )
    token_banks = {
        "veh": _get_last_step_token_bank(token_bank_all_veh),
        "ped": _get_last_step_token_bank(token_bank_all_ped),
        "cyc": _get_last_step_token_bank(token_bank_all_cyc),
    }

    for token_key, mask in build_agent_type_masks(agent_type).items():
        if not mask.any():
            continue

        token_bank = token_banks[token_key]
        dist = torch.norm(
            token_bank.unsqueeze(0) - contour_local[mask].unsqueeze(1),
            dim=-1,
        )
        if reduction == "sum":
            dist = dist.sum(-1)
        elif reduction == "mean":
            dist = dist.mean(-1)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        if sample_topk and (num_k > 1):
            if sampling_temp is None:
                raise ValueError("sampling_temp is required when sample_topk is True")
            top_k = min(num_k, dist.shape[1])
            topk_dists, topk_indices = torch.topk(
                dist,
                top_k,
                dim=-1,
                largest=False,
                sorted=False,
            )
            topk_logits = (-1.0 * topk_dists) / sampling_temp
            samples = Categorical(logits=topk_logits).sample()
            token_idx[mask] = topk_indices[
                torch.arange(samples.shape[0], device=samples.device),
                samples,
            ]
        else:
            token_idx[mask] = torch.argmin(dist, dim=-1)

    return token_idx
