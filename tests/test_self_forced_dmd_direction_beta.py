"""build_clean_dmd_direction의 dmd_beta knob 검증.

reference Self-Forcing의 entropy knob ``g = (1/β)·F − R`` 와 우리 path-direction MSE
형식의 등가성을 확인합니다.
"""
from __future__ import annotations

import math

import torch

from src.smart.modules.self_forced_dmd_guidance import build_clean_dmd_direction


def _build_inputs(seed: int = 0):
    torch.manual_seed(seed)
    committed = torch.randn(4, 6, 4, dtype=torch.float32)
    target = torch.randn(4, 6, 4, dtype=torch.float32)
    generated = torch.randn(4, 6, 4, dtype=torch.float32)
    return committed, target, generated


def test_dmd_beta_default_matches_original_form() -> None:
    """β=1.0이면 기존 ``(R - F)/normalizer`` 와 정확히 같아야 합니다."""
    committed, target, generated = _build_inputs(seed=0)
    default_call = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
    )
    explicit_one = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        dmd_beta=1.0,
    )
    assert torch.equal(default_call, explicit_one)

    reduce_dims = tuple(range(1, committed.dim()))
    normalizer = (committed - target).abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1.0e-3)
    expected = (target - generated) / normalizer
    assert torch.allclose(default_call, expected, atol=1.0e-6)


def test_dmd_beta_half_keeps_normalizer_and_scales_fake() -> None:
    """β=0.5이면 fake 항만 1/β 배가 되고 normalizer는 그대로여야 합니다."""
    committed, target, generated = _build_inputs(seed=1)
    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        dmd_beta=0.5,
    )
    reduce_dims = tuple(range(1, committed.dim()))
    normalizer = (committed - target).abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1.0e-3)
    expected = (target - 2.0 * generated) / normalizer  # inv_beta = 2
    assert torch.allclose(direction, expected, atol=1.0e-6)


def test_dmd_beta_two_sharpens_fake() -> None:
    """β=2.0이면 fake 항이 0.5배가 됩니다 (sharpening)."""
    committed, target, generated = _build_inputs(seed=2)
    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        dmd_beta=2.0,
    )
    reduce_dims = tuple(range(1, committed.dim()))
    normalizer = (committed - target).abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1.0e-3)
    expected = (target - 0.5 * generated) / normalizer
    assert torch.allclose(direction, expected, atol=1.0e-6)


def test_dmd_beta_zero_raises() -> None:
    committed, target, generated = _build_inputs(seed=3)
    raised = False
    try:
        build_clean_dmd_direction(
            committed_path_norm=committed,
            target_clean_norm=target,
            generated_clean_norm=generated,
            dmd_beta=0.0,
        )
    except ValueError:
        raised = True
    assert raised, "dmd_beta=0.0 must raise ValueError."


def test_dmd_beta_negative_raises() -> None:
    committed, target, generated = _build_inputs(seed=4)
    raised = False
    try:
        build_clean_dmd_direction(
            committed_path_norm=committed,
            target_clean_norm=target,
            generated_clean_norm=generated,
            dmd_beta=-1.0,
        )
    except ValueError:
        raised = True
    assert raised, "dmd_beta=-1.0 must raise ValueError."


def test_dmd_beta_nan_raises() -> None:
    committed, target, generated = _build_inputs(seed=5)
    raised = False
    try:
        build_clean_dmd_direction(
            committed_path_norm=committed,
            target_clean_norm=target,
            generated_clean_norm=generated,
            dmd_beta=float("nan"),
        )
    except ValueError:
        raised = True
    assert raised, "dmd_beta=nan must raise ValueError."
    # Sanity: NaN comparison semantics.
    assert not (math.nan > 0.0)
