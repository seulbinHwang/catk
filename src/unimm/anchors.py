from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch import Tensor

from src.smart.utils import transform_to_global, transform_to_local, wrap_angle

AGENT_TYPE_NAMES = ("veh", "ped", "cyc")
NUM_AGENT_TYPES = 3


@dataclass(frozen=True)
class AnchorSpec:
    """Fixed UniMM Anchor-Based-4s timing constants."""

    num_anchors: int = 2048
    num_future_steps: int = 80
    num_prediction_steps: int = 40
    num_commit_steps: int = 5
    num_match_steps: int = 5


def _as_type_tensor(values: Mapping[str, Tensor], dtype: torch.dtype = torch.float32) -> Tensor:
    tensors = []
    for name in AGENT_TYPE_NAMES:
        if name not in values:
            raise KeyError(f"anchor file is missing '{name}' anchors")
        tensors.append(torch.as_tensor(values[name], dtype=dtype))
    return torch.stack(tensors, dim=0)


def load_anchor_file(path: str | Path) -> tuple[Tensor, Tensor | None, dict]:
    """Load a UniMM anchor bank.

    Expected format is a pickle dictionary with either:
    - ``{"anchors": {"veh": [K, 80, 3], "ped": ..., "cyc": ...}}``
    - or the same category keys at the top level.

    Optional ``posterior_error_threshold`` is loaded as ``[3]``.
    """

    anchor_path = Path(path)
    with anchor_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"anchor file must contain a dict, got {type(payload)!r}")

    anchor_dict = payload.get("anchors", payload)
    anchors = _as_type_tensor(anchor_dict)
    if anchors.ndim != 4 or anchors.shape[0] != NUM_AGENT_TYPES or anchors.shape[-1] != 3:
        raise ValueError(
            "anchors must have shape [3, K, H, 3], "
            f"got {tuple(anchors.shape)} from {anchor_path}"
        )

    threshold = None
    threshold_payload = payload.get("posterior_error_threshold")
    if threshold_payload is not None:
        if isinstance(threshold_payload, Mapping):
            threshold = _as_type_tensor(
                {name: torch.as_tensor(value).reshape(1) for name, value in threshold_payload.items()}
            ).reshape(NUM_AGENT_TYPES)
        else:
            threshold = torch.as_tensor(threshold_payload, dtype=torch.float32).reshape(-1)
            if threshold.numel() != NUM_AGENT_TYPES:
                raise ValueError(
                    "posterior_error_threshold must have one value per agent type, "
                    f"got shape {tuple(threshold.shape)}"
                )

    return anchors.contiguous(), threshold, payload


def make_local_future(
    pos: Tensor,
    head: Tensor,
    ref_pos: Tensor,
    ref_head: Tensor,
) -> Tensor:
    """Convert global future states to the reference agent frame."""

    local_pos, local_head = transform_to_local(
        pos_global=pos,
        head_global=head,
        pos_now=ref_pos,
        head_now=ref_head,
    )
    return torch.cat([local_pos, wrap_angle(local_head).unsqueeze(-1)], dim=-1)


def anchor_distance(
    anchors: Tensor,
    target_local: Tensor,
    valid: Tensor | None = None,
    heading_weight: float = 1.0,
) -> Tensor:
    """Distance from each local target trajectory to every local anchor.

    Args:
        anchors: ``[K, H, 3]`` local anchor trajectories.
        target_local: ``[N, H, 3]`` local target trajectories.
        valid: optional ``[N, H]`` validity mask.

    Returns:
        ``[N, K]`` average distance over valid horizon steps.
    """

    pos_diff = anchors.unsqueeze(0)[..., :2] - target_local.unsqueeze(1)[..., :2]
    pos_sq = pos_diff.square().sum(dim=-1)
    head_diff = wrap_angle(
        anchors.unsqueeze(0)[..., 2] - target_local.unsqueeze(1)[..., 2]
    )
    per_step = pos_sq + float(heading_weight) * head_diff.square()
    if valid is None:
        return per_step.mean(dim=-1)

    weights = valid.to(dtype=per_step.dtype).unsqueeze(1)
    denom = weights.sum(dim=-1).clamp_min(1.0)
    return (per_step * weights).sum(dim=-1) / denom


def match_anchors_by_type(
    anchors_by_type: Tensor,
    agent_type: Tensor,
    target_local: Tensor,
    valid: Tensor | None,
    horizon_steps: int,
    heading_weight: float = 1.0,
    row_chunk_size: int = 4096,
) -> tuple[Tensor, Tensor]:
    """Find the nearest anchor index for every row, separated by agent type.

    ``target_local`` is flattened row-wise by the caller, for example
    ``[N_agent * N_context, H, 3]``.
    """

    if target_local.ndim != 3 or target_local.shape[-1] != 3:
        raise ValueError(f"target_local must be [N, H, 3], got {tuple(target_local.shape)}")
    if agent_type.shape != (target_local.shape[0],):
        raise ValueError(
            "agent_type must have one value per target row, "
            f"got {tuple(agent_type.shape)} and {tuple(target_local.shape)}"
        )

    device = target_local.device
    z = torch.zeros(target_local.shape[0], dtype=torch.long, device=device)
    best = torch.full(
        (target_local.shape[0],),
        float("inf"),
        dtype=target_local.dtype,
        device=device,
    )
    valid_h = valid[:, :horizon_steps] if valid is not None else None
    target_h = target_local[:, :horizon_steps]

    for type_idx in range(NUM_AGENT_TYPES):
        rows = torch.nonzero(agent_type == type_idx, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        anchors = anchors_by_type[type_idx, :, :horizon_steps].to(device=device, dtype=target_local.dtype)
        for start in range(0, rows.numel(), row_chunk_size):
            chunk_rows = rows[start : start + row_chunk_size]
            dist = anchor_distance(
                anchors=anchors,
                target_local=target_h[chunk_rows],
                valid=valid_h[chunk_rows] if valid_h is not None else None,
                heading_weight=heading_weight,
            )
            chunk_best, chunk_z = dist.min(dim=-1)
            z[chunk_rows] = chunk_z
            best[chunk_rows] = chunk_best

    return z, best


def gather_anchors_by_type(anchors_by_type: Tensor, agent_type: Tensor, z: Tensor) -> Tensor:
    """Gather category-specific anchors for each agent row."""

    return anchors_by_type[
        agent_type.long().clamp(0, NUM_AGENT_TYPES - 1),
        z.long(),
    ]


def execute_local_anchor(
    anchor: Tensor,
    ref_pos: Tensor,
    ref_head: Tensor,
    commit_steps: int,
) -> tuple[Tensor, Tensor]:
    """Transform the committed part of a local anchor to global coordinates."""

    local = anchor[:, :commit_steps]
    pos_global, head_global = transform_to_global(
        pos_local=local[..., :2],
        head_local=local[..., 2],
        pos_now=ref_pos,
        head_now=ref_head,
    )
    return pos_global, wrap_angle(head_global)
