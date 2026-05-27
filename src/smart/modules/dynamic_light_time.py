from __future__ import annotations

DEFAULT_WAYMO_CURRENT_RAW_STEP = 10


def validate_observed_current_raw_step(
    current_time_index: int,
    *,
    expected_raw_step: int = DEFAULT_WAYMO_CURRENT_RAW_STEP,
    scenario_id: str | None = None,
) -> int:
    """Check that cached traffic lights use the expected WOMD current step."""
    current = int(current_time_index)
    expected = int(expected_raw_step)
    if current != expected:
        scenario_suffix = f" for scenario {scenario_id}" if scenario_id else ""
        raise ValueError(
            "Traffic-light cache construction assumes the observed current raw step "
            f"is {expected}, but got current_time_index={current}{scenario_suffix}. "
            "Regenerate the cache with the standard WOMD current step or make the "
            "observed traffic-light raw step explicit in the cache."
        )
    return current
