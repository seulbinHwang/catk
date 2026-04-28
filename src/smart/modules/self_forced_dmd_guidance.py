from __future__ import annotations

import torch
from torch import Tensor


def build_clean_dmd_direction(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    normalizer_eps: float = 1.0e-3,
) -> Tensor:
    """Build a normalized DMD direction from clean path estimates."""
    expected_shape = tuple(committed_path_norm.shape)
    if tuple(target_clean_norm.shape) != expected_shape:
        raise ValueError(
            "target_clean_norm shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={tuple(target_clean_norm.shape)}."
        )
    if tuple(generated_clean_norm.shape) != expected_shape:
        raise ValueError(
            "generated_clean_norm shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={tuple(generated_clean_norm.shape)}."
        )
    if committed_path_norm.dim() < 2:
        raise ValueError(
            "committed_path_norm must have at least agent and path dimensions, "
            f"got shape={expected_shape}."
        )

    committed = committed_path_norm.float()
    target_clean = target_clean_norm.float()
    generated_clean = generated_clean_norm.float()

    clean_dmd_direction = target_clean - generated_clean
    reduce_dims = tuple(range(1, committed.dim()))
    agent_distance = (committed - target_clean).abs().mean(
        dim=reduce_dims,
        keepdim=True,
    )
    normalizer = agent_distance.clamp_min(float(normalizer_eps))

    normalized_direction = clean_dmd_direction / normalizer
    normalized_direction = torch.nan_to_num(
        normalized_direction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return normalized_direction.to(dtype=committed_path_norm.dtype)
