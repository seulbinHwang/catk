from __future__ import annotations


def validate_flow_window_steps(
    flow_window_steps: int,
    commit_steps: int,
    num_future_steps: int | None = None,
) -> int:
    """Validate the flow preview horizon against rollout chunk settings."""

    flow_window_steps = int(flow_window_steps)
    commit_steps = int(commit_steps)

    if commit_steps <= 0:
        raise ValueError(f"commit_steps must be positive, got {commit_steps}.")
    if flow_window_steps <= 0:
        raise ValueError(
            f"flow_window_steps must be positive, got {flow_window_steps}."
        )
    if flow_window_steps % commit_steps != 0:
        raise ValueError(
            "flow_window_steps must be divisible by commit_steps, "
            f"got {flow_window_steps} and {commit_steps}."
        )
    if num_future_steps is not None and flow_window_steps > int(num_future_steps):
        raise ValueError(
            "flow_window_steps cannot exceed num_future_steps, "
            f"got {flow_window_steps} and {int(num_future_steps)}."
        )
    return flow_window_steps


def format_flow_horizon_tag(flow_window_steps: int, hz: int = 10) -> str:
    """Convert a future horizon in steps into a compact metric tag."""

    flow_window_steps = int(flow_window_steps)
    hz = int(hz)
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}.")
    if flow_window_steps <= 0:
        raise ValueError(
            f"flow_window_steps must be positive, got {flow_window_steps}."
        )

    whole_seconds, remainder = divmod(flow_window_steps, hz)
    if remainder == 0:
        return f"{whole_seconds}s"

    decimal = f"{remainder / hz:.6f}".split(".", 1)[1].rstrip("0")
    return f"{whole_seconds}p{decimal}s"
