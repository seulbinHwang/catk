"""self_forced_estimator_warmup의 step 기반 warmup helper 검증.

epoch 기반 helper는 기존 test_self_forced_estimator_warmup.py가 다루고, 본 모듈은
step 기반 분기 (`estimator_warmup_steps`, `is_self_forced_estimator_warmup_step`) 만
다룹니다.
"""
from __future__ import annotations

from src.smart.modules.self_forced_estimator_warmup import (
    DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_STEPS,
    is_self_forced_estimator_warmup_step,
    resolve_self_forced_estimator_warmup_steps,
)


def test_resolve_returns_default_when_config_none() -> None:
    assert resolve_self_forced_estimator_warmup_steps(None) == DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_STEPS


def test_resolve_reads_config_dict() -> None:
    assert resolve_self_forced_estimator_warmup_steps({"estimator_warmup_steps": 200}) == 200


def test_resolve_reads_attribute_style() -> None:
    class _Cfg:
        estimator_warmup_steps = 350

    assert resolve_self_forced_estimator_warmup_steps(_Cfg()) == 350


def test_resolve_treats_none_value_as_default() -> None:
    assert (
        resolve_self_forced_estimator_warmup_steps({"estimator_warmup_steps": None})
        == DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_STEPS
    )


def test_resolve_rejects_negative() -> None:
    raised = False
    try:
        resolve_self_forced_estimator_warmup_steps({"estimator_warmup_steps": -1})
    except ValueError:
        raised = True
    assert raised, "negative estimator_warmup_steps must raise ValueError"


def test_step_active_default_zero_disables() -> None:
    assert not is_self_forced_estimator_warmup_step(global_step=0, estimator_warmup_steps=0)
    assert not is_self_forced_estimator_warmup_step(global_step=10_000, estimator_warmup_steps=0)


def test_step_active_within_warmup_range() -> None:
    assert is_self_forced_estimator_warmup_step(global_step=0, estimator_warmup_steps=200)
    assert is_self_forced_estimator_warmup_step(global_step=199, estimator_warmup_steps=200)


def test_step_active_excludes_boundary_step() -> None:
    # warmup_steps 가 200 이면 step 0..199 가 warmup, step 200 부터 generator update.
    assert not is_self_forced_estimator_warmup_step(global_step=200, estimator_warmup_steps=200)
    assert not is_self_forced_estimator_warmup_step(global_step=10_000, estimator_warmup_steps=200)
