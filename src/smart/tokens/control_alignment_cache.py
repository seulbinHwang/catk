from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch
from torch import Tensor

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    DEFAULT_CONTROL_POS_SCALE_M,
    build_transition_aligned_control_trajectory,
)
from src.smart.tokens.trajectory_preprocess import (
    clean_heading_dense,
    extrapolate_agent_to_prev_token_step,
)


CONTROL_ALIGNMENT_CACHE_VERSION = 1
CONTROL_ALIGNMENT_CACHE_CURRENT_STEP = 10
CONTROL_ALIGNMENT_CACHE_SHIFT = 5

CONTROL_ALIGNED_FUTURE_POS_KEY = "control_aligned_future_pos"
CONTROL_ALIGNED_FUTURE_HEADING_KEY = "control_aligned_future_heading"
CONTROL_TRANSITION_NORM_FUTURE_KEY = "control_transition_norm_future"
CONTROL_ALIGNMENT_CACHE_KEY = "control_alignment_cache_key"
CONTROL_ALIGNMENT_CACHE_FIELD_KEYS = (
    CONTROL_ALIGNED_FUTURE_POS_KEY,
    CONTROL_ALIGNED_FUTURE_HEADING_KEY,
    CONTROL_TRANSITION_NORM_FUTURE_KEY,
    CONTROL_ALIGNMENT_CACHE_KEY,
)

DEFAULT_CACHE_CONTROL_VEHICLE_YAW_SCALE_RAD = 0.025
DEFAULT_CACHE_CONTROL_PEDESTRIAN_YAW_SCALE_RAD = 0.20
DEFAULT_CACHE_CONTROL_CYCLIST_YAW_SCALE_RAD = 0.06
DEFAULT_CACHE_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO = 0.2289518863
DEFAULT_CACHE_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO = 0.0495847873


@dataclass(frozen=True)
class ControlAlignmentCacheConfig:
    current_step: int = CONTROL_ALIGNMENT_CACHE_CURRENT_STEP
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M
    vehicle_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_VEHICLE_YAW_SCALE_RAD
    pedestrian_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_PEDESTRIAN_YAW_SCALE_RAD
    cyclist_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_CYCLIST_YAW_SCALE_RAD
    use_holonomic_model_only: bool = False
    vehicle_no_slip_point_ratio: float = DEFAULT_CACHE_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO
    cyclist_no_slip_point_ratio: float = DEFAULT_CACHE_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO


def control_alignment_cache_key_tensor(
    config: ControlAlignmentCacheConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Numeric cache key stored per agent so PyG batch collation stays tensor-only."""
    return torch.tensor(
        [
            float(CONTROL_ALIGNMENT_CACHE_VERSION),
            float(config.current_step),
            float(config.pos_scale_m),
            float(config.vehicle_yaw_scale_rad),
            float(config.pedestrian_yaw_scale_rad),
            float(config.cyclist_yaw_scale_rad),
            1.0 if config.use_holonomic_model_only else 0.0,
            float(config.vehicle_no_slip_point_ratio),
            float(config.cyclist_no_slip_point_ratio),
        ],
        device=device,
        dtype=dtype,
    )


def _get_agent_tensor(agent_data: Mapping[str, Tensor], key: str) -> Tensor:
    value = agent_data[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"agent[{key!r}] must be a torch.Tensor, got {type(value)!r}.")
    return value


def build_control_alignment_cache_fields(
    agent_data: Mapping[str, Tensor],
    config: ControlAlignmentCacheConfig | None = None,
) -> dict[str, Tensor]:
    """Build deterministic transition-aligned control-cache fields for one sample.

    The output intentionally stops at per-step aligned state/control. Anchor
    selection, loss masks, distortion thresholds, and final target gather remain
    online in ``FlowTokenProcessor``.
    """
    if config is None:
        config = ControlAlignmentCacheConfig()

    valid = _get_agent_tensor(agent_data, "valid_mask").clone()
    position = _get_agent_tensor(agent_data, "position")
    pos = position[..., :2].clone().contiguous()
    heading = _get_agent_tensor(agent_data, "heading").clone()
    vel = _get_agent_tensor(agent_data, "velocity").clone()
    agent_type = _get_agent_tensor(agent_data, "type")
    agent_length = _get_agent_tensor(agent_data, "shape")[:, 0]

    heading = clean_heading_dense(valid=valid, heading=heading)
    valid, pos, heading, _ = extrapolate_agent_to_prev_token_step(
        valid=valid,
        pos=pos,
        heading=heading,
        vel=vel,
        shift=CONTROL_ALIGNMENT_CACHE_SHIFT,
        current_step=int(config.current_step),
    )
    aligned_pos, aligned_heading, transition_control_norm_by_step = (
        build_transition_aligned_control_trajectory(
            pos=pos,
            heading=heading,
            agent_type=agent_type,
            agent_length=agent_length,
            current_step=int(config.current_step),
            pos_scale_m=float(config.pos_scale_m),
            vehicle_yaw_scale_rad=float(config.vehicle_yaw_scale_rad),
            pedestrian_yaw_scale_rad=float(config.pedestrian_yaw_scale_rad),
            cyclist_yaw_scale_rad=float(config.cyclist_yaw_scale_rad),
            use_holonomic_model_only=bool(config.use_holonomic_model_only),
            vehicle_no_slip_point_ratio=float(config.vehicle_no_slip_point_ratio),
            cyclist_no_slip_point_ratio=float(config.cyclist_no_slip_point_ratio),
        )
    )

    future_start = int(config.current_step) + 1
    n_agent = int(pos.shape[0])
    key = control_alignment_cache_key_tensor(config, dtype=torch.float64).view(1, -1)
    return {
        CONTROL_ALIGNED_FUTURE_POS_KEY: aligned_pos[:, future_start:].contiguous(),
        CONTROL_ALIGNED_FUTURE_HEADING_KEY: aligned_heading[:, future_start:].contiguous(),
        CONTROL_TRANSITION_NORM_FUTURE_KEY: transition_control_norm_by_step[
            :, future_start:
        ].contiguous(),
        CONTROL_ALIGNMENT_CACHE_KEY: key.expand(n_agent, -1).clone(),
    }


def attach_control_alignment_cache_fields(
    sample: dict,
    config: ControlAlignmentCacheConfig | None = None,
) -> dict:
    """Attach control-cache fields to a SMART cache sample in-place."""
    agent_data = sample["agent"]
    agent_data.update(build_control_alignment_cache_fields(agent_data, config=config))
    return sample


def get_control_alignment_cache_config_from_values(
    *,
    current_step: int = CONTROL_ALIGNMENT_CACHE_CURRENT_STEP,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_VEHICLE_YAW_SCALE_RAD,
    pedestrian_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_PEDESTRIAN_YAW_SCALE_RAD,
    cyclist_yaw_scale_rad: float = DEFAULT_CACHE_CONTROL_CYCLIST_YAW_SCALE_RAD,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CACHE_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CACHE_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> ControlAlignmentCacheConfig:
    return ControlAlignmentCacheConfig(
        current_step=int(current_step),
        pos_scale_m=float(pos_scale_m),
        vehicle_yaw_scale_rad=float(vehicle_yaw_scale_rad),
        pedestrian_yaw_scale_rad=float(pedestrian_yaw_scale_rad),
        cyclist_yaw_scale_rad=float(cyclist_yaw_scale_rad),
        use_holonomic_model_only=bool(use_holonomic_model_only),
        vehicle_no_slip_point_ratio=float(vehicle_no_slip_point_ratio),
        cyclist_no_slip_point_ratio=float(cyclist_no_slip_point_ratio),
    )


def has_control_alignment_cache_fields(agent_data: Mapping[str, Tensor]) -> bool:
    return all(key in agent_data for key in CONTROL_ALIGNMENT_CACHE_FIELD_KEYS)


def strip_control_alignment_cache_fields(agent_data: MutableMapping[str, Tensor]) -> None:
    """Remove optional control-cache fields from a sample if present."""
    for key in CONTROL_ALIGNMENT_CACHE_FIELD_KEYS:
        if key in agent_data:
            del agent_data[key]


def validate_control_alignment_cache_fields(
    agent_data: Mapping[str, Tensor],
    expected_config: ControlAlignmentCacheConfig,
    *,
    expected_n_agent: int,
    expected_n_step: int,
    device: torch.device,
) -> bool:
    """Return True only when cached fields exactly match the active config shape."""
    if not has_control_alignment_cache_fields(agent_data):
        return False

    future_len = max(0, int(expected_n_step) - int(expected_config.current_step) - 1)
    try:
        aligned_pos = _get_agent_tensor(agent_data, CONTROL_ALIGNED_FUTURE_POS_KEY)
        aligned_heading = _get_agent_tensor(
            agent_data, CONTROL_ALIGNED_FUTURE_HEADING_KEY
        )
        control_norm = _get_agent_tensor(agent_data, CONTROL_TRANSITION_NORM_FUTURE_KEY)
        cache_key = _get_agent_tensor(agent_data, CONTROL_ALIGNMENT_CACHE_KEY)
    except (KeyError, TypeError):
        return False

    if tuple(aligned_pos.shape) != (expected_n_agent, future_len, 2):
        return False
    if tuple(aligned_heading.shape) != (expected_n_agent, future_len):
        return False
    if tuple(control_norm.shape) != (expected_n_agent, future_len, CONTROL_FLOW_DIM):
        return False
    if cache_key.ndim != 2 or cache_key.shape[0] != expected_n_agent:
        return False

    expected_key = control_alignment_cache_key_tensor(
        expected_config,
        device=device,
        dtype=cache_key.dtype,
    )
    cache_key = cache_key.to(device=device)
    return bool(
        torch.allclose(
            cache_key,
            expected_key.expand_as(cache_key),
            atol=1.0e-9,
            rtol=0.0,
        )
    )
