from __future__ import annotations

from types import SimpleNamespace

import torch

from src.smart.model.smart_flow import _build_commit_aligned_flow_loss_step_weights


def _cfg(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        commit_aligned_flow_loss=SimpleNamespace(
            enabled=enabled,
            weights=[1.25, 1.01, 0.87],
            boundary_steps=[5, 10],
        ),
    )


def test_commit_aligned_config_builds_expected_20_step_weights() -> None:
    weights = _build_commit_aligned_flow_loss_step_weights(_cfg(), flow_window_steps=20)

    expected = torch.tensor([1.25] * 5 + [1.01] * 5 + [0.87] * 10, dtype=torch.float32)

    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_commit_aligned_config_does_not_rescale_short_prefix_window() -> None:
    weights = _build_commit_aligned_flow_loss_step_weights(_cfg(), flow_window_steps=5)

    torch.testing.assert_close(weights, torch.full((5,), 1.25))


def test_commit_aligned_config_can_disable_weighted_loss() -> None:
    weights = _build_commit_aligned_flow_loss_step_weights(_cfg(enabled=False), flow_window_steps=20)

    assert weights.numel() == 0
