from __future__ import annotations

import pytest

from src.smart.modules.self_forced_estimator_warmup import (
    is_self_forced_estimator_warmup_epoch,
    resolve_self_forced_estimator_warmup_epochs,
    should_compute_anchor_flow_matching_loss,
    should_run_self_forced_validation_after_epoch,
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


def test_anchor_flow_matching_loss_is_skipped_during_estimator_warmup() -> None:
    assert not should_compute_anchor_flow_matching_loss(
        use_anchor_flow_matching_loss=True,
        is_estimator_warmup_active=True,
    )


def test_anchor_flow_matching_loss_runs_after_estimator_warmup() -> None:
    assert should_compute_anchor_flow_matching_loss(
        use_anchor_flow_matching_loss=True,
        is_estimator_warmup_active=False,
    )


def test_anchor_flow_matching_loss_respects_disabled_config() -> None:
    assert not should_compute_anchor_flow_matching_loss(
        use_anchor_flow_matching_loss=False,
        is_estimator_warmup_active=False,
    )


def test_self_forced_validation_skips_warmup_epoch() -> None:
    assert not should_run_self_forced_validation_after_epoch(
        current_epoch=0,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=1,
    )
    assert should_run_self_forced_validation_after_epoch(
        current_epoch=1,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=1,
    )


def test_self_forced_validation_restarts_cadence_after_warmup() -> None:
    assert not should_run_self_forced_validation_after_epoch(
        current_epoch=1,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )
    assert should_run_self_forced_validation_after_epoch(
        current_epoch=2,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )


def test_self_forced_validation_keeps_pre_start_epoch_schedule() -> None:
    assert should_run_self_forced_validation_after_epoch(
        current_epoch=1,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )
    assert not should_run_self_forced_validation_after_epoch(
        current_epoch=2,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )
    assert not should_run_self_forced_validation_after_epoch(
        current_epoch=3,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )
    assert should_run_self_forced_validation_after_epoch(
        current_epoch=4,
        self_forced_start_epoch=2,
        estimator_warmup_epochs=1,
        check_val_every_n_epoch=2,
    )


def test_self_forced_validation_uses_default_schedule_without_warmup() -> None:
    assert not should_run_self_forced_validation_after_epoch(
        current_epoch=0,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=0,
        check_val_every_n_epoch=2,
    )
    assert should_run_self_forced_validation_after_epoch(
        current_epoch=1,
        self_forced_start_epoch=0,
        estimator_warmup_epochs=0,
        check_val_every_n_epoch=2,
    )
