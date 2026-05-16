"""RoaD cache의 거리 클립과 autocast 정합 헬퍼를 검증한다."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.smart.road.cache import (
    resolve_autocast_dtype_from_precision,
    update_raw_data_with_road_rollout,
)


def _make_raw_data(n_agent: int, n_step: int = 91) -> dict:
    """ego 1개 + 비 ego 여러 개의 모의 raw data를 만든다."""
    pos = torch.zeros(n_agent, n_step, 3)
    heading = torch.zeros(n_agent, n_step)
    velocity = torch.zeros(n_agent, n_step, 2)
    valid_mask = torch.ones(n_agent, n_step, dtype=torch.bool)
    role = torch.zeros(n_agent, 3, dtype=torch.bool)
    role[0, 0] = True  # agent 0이 ego
    return {
        "scenario_id": "fake_scenario",
        "agent": {
            "id": torch.arange(n_agent, dtype=torch.int64),
            "type": torch.zeros(n_agent, dtype=torch.uint8),
            "shape": torch.tensor([[2.0, 4.8, 1.5]] * n_agent),
            "position": pos,
            "heading": heading,
            "velocity": velocity,
            "valid_mask": valid_mask,
            "role": role,
        },
    }


def test_distance_filter_disabled_when_threshold_nonpositive():
    raw = _make_raw_data(n_agent=3)
    n_agent = 3
    pred_traj = torch.zeros(n_agent, 80, 2)
    pred_traj[2] = 999.0  # 비 ego agent 2가 멀리 폭주
    pred_head = torch.zeros(n_agent, 80)
    future_valid = torch.ones(n_agent, dtype=torch.bool)

    updated = update_raw_data_with_road_rollout(
        raw_data=raw,
        scenario_id="fake_scenario",
        rollout_index=0,
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        future_valid=future_valid,
        num_historical_steps=11,
        max_distance_from_ego=0.0,
    )
    # 거리 필터가 꺼졌으니 모든 future step이 valid 유지.
    assert bool(updated["agent"]["valid_mask"][2, 11:].all()), \
        "filter off: runaway agent's future should remain valid"
    print("distance filter OFF: runaway stays valid (legacy behavior): OK")


def test_distance_filter_marks_runaway_invalid():
    raw = _make_raw_data(n_agent=3)
    n_agent = 3
    # ego는 (0,0)에 머무름, agent 1은 0.5m 떨어진 곳에 머무름, agent 2는 (200,0)으로 폭주.
    pred_traj = torch.zeros(n_agent, 80, 2)
    pred_traj[1, :, 0] = 0.5
    pred_traj[2, :, 0] = 200.0  # 150m 초과
    pred_head = torch.zeros(n_agent, 80)
    future_valid = torch.ones(n_agent, dtype=torch.bool)

    updated = update_raw_data_with_road_rollout(
        raw_data=raw,
        scenario_id="fake_scenario",
        rollout_index=0,
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        future_valid=future_valid,
        num_historical_steps=11,
        max_distance_from_ego=150.0,
    )
    valid_future = updated["agent"]["valid_mask"][:, 11:]
    assert bool(valid_future[0].all()), "ego itself must stay valid"
    assert bool(valid_future[1].all()), "close agent must stay valid"
    assert not bool(valid_future[2].any()), "runaway agent must be marked invalid"
    print("distance filter ON: runaway zeroed, close kept: OK")


def test_distance_filter_partial_invalidation_per_step():
    """rollout 도중에 거리가 점점 벌어지는 경우, 임계 이전만 valid로 남아야 한다."""
    raw = _make_raw_data(n_agent=2)
    pred_traj = torch.zeros(2, 80, 2)
    # 비 ego agent: x = 2.0 * step → 75 step (=150m)에서 임계 초과.
    pred_traj[1, :, 0] = torch.arange(1, 81, dtype=torch.float32) * 2.0
    pred_head = torch.zeros(2, 80)
    future_valid = torch.ones(2, dtype=torch.bool)

    updated = update_raw_data_with_road_rollout(
        raw_data=raw,
        scenario_id="partial",
        rollout_index=0,
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        future_valid=future_valid,
        num_historical_steps=11,
        max_distance_from_ego=150.0,
    )
    valid_future = updated["agent"]["valid_mask"][1, 11:]
    # 거리 = (step+1)*2. step 0..73이면 거리 2..148 (<150), step 74이면 150 (not <150).
    # 따라서 valid_future[0..73] = True, valid_future[74..79] = False.
    assert bool(valid_future[:74].all())
    assert not bool(valid_future[74:].any())
    print(
        "partial: valid through step 73, invalid from step 74: OK "
        f"(crossover at step {int((~valid_future).nonzero()[0].item())})"
    )


def test_distance_filter_skipped_when_role_ambiguous():
    """ego가 정확히 1개가 아니면 안전하게 fallback (이전 동작 유지)."""
    raw = _make_raw_data(n_agent=3)
    raw["agent"]["role"].zero_()  # ego가 0개
    pred_traj = torch.full((3, 80, 2), 999.0)
    pred_head = torch.zeros(3, 80)
    future_valid = torch.ones(3, dtype=torch.bool)

    updated = update_raw_data_with_road_rollout(
        raw_data=raw,
        scenario_id="no_ego",
        rollout_index=0,
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        future_valid=future_valid,
        num_historical_steps=11,
        max_distance_from_ego=150.0,
    )
    # ego 미상이라 거리 필터를 끄고 broadcast만 적용 → 모든 future valid.
    assert bool(updated["agent"]["valid_mask"][:, 11:].all())
    print("ambiguous-ego skip: filter falls back safely: OK")


def test_resolve_autocast_dtype_mapping():
    cases = {
        "bf16-mixed": torch.bfloat16,
        "BF16-MIXED": torch.bfloat16,
        "16-mixed": torch.float16,
        "bf16": torch.bfloat16,
        "16": torch.float16,
        "32-true": None,
        "32": None,
        None: None,
        "": None,
    }
    for prec, expected in cases.items():
        got = resolve_autocast_dtype_from_precision(prec)
        assert got == expected, f"precision={prec!r}: expected {expected}, got {got}"
    print(f"resolve_autocast_dtype_from_precision: {len(cases)} cases OK")


def main():
    test_distance_filter_disabled_when_threshold_nonpositive()
    test_distance_filter_marks_runaway_invalid()
    test_distance_filter_partial_invalidation_per_step()
    test_distance_filter_skipped_when_role_ambiguous()
    test_resolve_autocast_dtype_mapping()
    print("\nAll RoaD cache filter tests PASSED.")


if __name__ == "__main__":
    main()
