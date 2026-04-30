from __future__ import annotations

import pytest

from src.smart.modules.self_forced_estimator_warmup import (
    is_self_forced_estimator_warmup_epoch,
    resolve_self_forced_estimator_warmup_epochs,
)


class DummyConfig:
    def __init__(self, estimator_warmup_epochs: int | None = None) -> None:
        self.estimator_warmup_epochs = estimator_warmup_epochs


def test_resolve_self_forced_estimator_warmup_epochs_uses_default() -> None:
    assert resolve_self_forced_estimator_warmup_epochs(None) == 1


def test_resolve_self_forced_estimator_warmup_epochs_reads_config() -> None:
    assert resolve_self_forced_estimator_warmup_epochs(
        {"estimator_warmup_epochs": 2}
    ) == 2
    assert resolve_self_forced_estimator_warmup_epochs(DummyConfig(0)) == 0


def test_resolve_self_forced_estimator_warmup_epochs_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        resolve_self_forced_estimator_warmup_epochs(
            {"estimator_warmup_epochs": -1}
        )


def test_is_self_forced_estimator_warmup_epoch_respects_start_epoch() -> None:
    assert not is_self_forced_estimator_warmup_epoch(
        current_epoch=1,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
    )
    assert is_self_forced_estimator_warmup_epoch(
        current_epoch=2,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
    )
    assert not is_self_forced_estimator_warmup_epoch(
        current_epoch=3,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
    )


def test_is_self_forced_estimator_warmup_epoch_can_be_disabled() -> None:
    assert not is_self_forced_estimator_warmup_epoch(
        current_epoch=0,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=0,
    )
