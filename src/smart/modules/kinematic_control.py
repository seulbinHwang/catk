from __future__ import annotations

import torch
from torch import Tensor

from src.smart.utils import transform_to_local, wrap_angle


POSE_FLOW_DIM = 4
CONTROL_FLOW_DIM = 2
MDG_STATE_DIM = 5

DEFAULT_CONTROL_POS_SCALE_M = 1.0
DEFAULT_CONTROL_YAW_RATE_SCALE_RADPS = 0.5
DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M = 0.5
DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO = 0.0
DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO = 0.0

POSE_NORM_POS_SCALE_M = 20.0
MDG_STATE_POS_SCALE_M = 1.0
MDG_STATE_SPEED_SCALE_MPS = 1.0
CONTROL_DT_S = 0.1
YAW_RATE_SPEED_THRESHOLD_MPS = 0.1

VEHICLE_TYPE_ID = 0
PEDESTRIAN_TYPE_ID = 1
CYCLIST_TYPE_ID = 2
_VALID_AGENT_TYPE_IDS = (VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID)


def _validate_agent_type(agent_type: Tensor) -> None:
    if agent_type.numel() == 0:
        return
    type_min = int(agent_type.min().item())
    type_max = int(agent_type.max().item())
    if type_min < 0 or type_max > CYCLIST_TYPE_ID:
        raise ValueError(
            "agent_type must follow the repo convention "
            f"{{VEHICLE={VEHICLE_TYPE_ID}, PEDESTRIAN={PEDESTRIAN_TYPE_ID}, "
            f"CYCLIST={CYCLIST_TYPE_ID}}}; got values in [{type_min}, {type_max}]."
        )


def _reject_holonomic_model_only(use_holonomic_model_only: bool) -> None:
    if bool(use_holonomic_model_only):
        raise ValueError(
            "semi_mdg supports only MDG-style acceleration/yaw-rate dynamics. "
            "Remove use_holonomic_model_only or set it to false."
        )


def _validate_control_agent_type(control: Tensor, agent_type: Tensor) -> None:
    if agent_type.ndim != 1 or agent_type.shape[0] != control.shape[0]:
        raise ValueError(
            "agent_type must have shape [N] and match control batch, "
            f"got {tuple(agent_type.shape)} and {tuple(control.shape)}."
        )
    _validate_agent_type(agent_type)


def validate_control_yaw_scale_config(
    *,
    vehicle_yaw_scale_rad: float | None,
    pedestrian_yaw_scale_rad: float | None,
    cyclist_yaw_scale_rad: float | None,
) -> tuple[float, float, float]:
    """Validate the per-type config fields used as yaw-rate std values.

    The public config key names still say "yaw_scale_rad" for checkpoint/config
    compatibility, but MDG-style semi_mdg interprets the values as yaw-rate
    normalization scales in rad/s. Set all three to the same value to keep one
    shared dynamics formula for vehicles, pedestrians, and cyclists.
    """
    scales = {
        "control_vehicle_yaw_scale_rad": vehicle_yaw_scale_rad,
        "control_pedestrian_yaw_scale_rad": pedestrian_yaw_scale_rad,
        "control_cyclist_yaw_scale_rad": cyclist_yaw_scale_rad,
    }
    validated: list[float] = []
    for name, value in scales.items():
        if value is None:
            raise ValueError(f"{name} must be configured for control-space flow.")
        scale = float(value)
        if scale <= 0.0:
            raise ValueError(f"{name} must be positive, got {scale}.")
        validated.append(scale)
    return validated[0], validated[1], validated[2]


def validate_control_no_slip_ratio_config(
    *,
    vehicle_no_slip_point_ratio: float | None,
    cyclist_no_slip_point_ratio: float | None,
) -> tuple[float, float]:
    """Keep old config fields valid, although MDG-style dynamics ignores them."""
    ratios = {
        "control_vehicle_no_slip_point_ratio": vehicle_no_slip_point_ratio,
        "control_cyclist_no_slip_point_ratio": cyclist_no_slip_point_ratio,
    }
    validated: list[float] = []
    for name, value in ratios.items():
        if value is None:
            raise ValueError(f"{name} must be configured for control-space flow.")
        ratio = float(value)
        if ratio < 0.0:
            raise ValueError(f"{name} must be non-negative, got {ratio}.")
        validated.append(ratio)
    return validated[0], validated[1]


def resolve_control_yaw_scale(
    agent_type: Tensor,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    if agent_type.ndim != 1:
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")
    if device is None:
        device = agent_type.device
    if dtype is None:
        dtype = torch.float32

    agent_type_device = agent_type.to(device=device)
    _validate_agent_type(agent_type_device)
    vehicle_scale, pedestrian_scale, cyclist_scale = validate_control_yaw_scale_config(
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )

    yaw_scale = torch.empty(agent_type_device.shape, device=device, dtype=dtype)
    yaw_scale[agent_type_device == VEHICLE_TYPE_ID] = vehicle_scale
    yaw_scale[agent_type_device == PEDESTRIAN_TYPE_ID] = pedestrian_scale
    yaw_scale[agent_type_device == CYCLIST_TYPE_ID] = cyclist_scale
    return yaw_scale


def safe_sinc(x: Tensor, eps: float = 1.0e-6) -> Tensor:
    near_zero = x.abs() < eps
    safe_x = torch.where(near_zero, torch.ones_like(x), x)
    x2 = x * x
    return torch.where(
        near_zero,
        1.0 - x2 / 6.0 + x2 * x2 / 120.0,
        x.sin() / safe_x,
    )


def _validate_dynamics_inputs(
    control: Tensor,
    agent_type: Tensor,
    current_pos: Tensor | None,
    current_head: Tensor | None,
    current_speed: Tensor | None,
) -> None:
    if control.ndim != 3 or control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control must have shape [N, T, 2], got {tuple(control.shape)}.")
    if agent_type.ndim != 1 or agent_type.shape[0] != control.shape[0]:
        raise ValueError(
            "agent_type must have shape [N] and match control batch, "
            f"got {tuple(agent_type.shape)} and {tuple(control.shape)}."
        )
    _validate_agent_type(agent_type)
    n_agent = control.shape[0]
    if current_pos is not None and tuple(current_pos.shape) != (n_agent, 2):
        raise ValueError(f"current_pos must have shape [N, 2], got {tuple(current_pos.shape)}.")
    if current_head is not None and tuple(current_head.shape) != (n_agent,):
        raise ValueError(f"current_head must have shape [N], got {tuple(current_head.shape)}.")
    if current_speed is not None and tuple(current_speed.shape) != (n_agent,):
        raise ValueError(f"current_speed must have shape [N], got {tuple(current_speed.shape)}.")


def _default_current_pos(control: Tensor, current_pos: Tensor | None) -> Tensor:
    if current_pos is not None:
        return current_pos.to(device=control.device, dtype=control.dtype)
    return control.new_zeros((control.shape[0], 2))


def _default_current_head(control: Tensor, current_head: Tensor | None) -> Tensor:
    if current_head is not None:
        return current_head.to(device=control.device, dtype=control.dtype)
    return control.new_zeros((control.shape[0],))


def _default_current_speed(control: Tensor, current_speed: Tensor | None) -> Tensor:
    if current_speed is not None:
        return current_speed.to(device=control.device, dtype=control.dtype).clamp_min(0.0)
    return control.new_zeros((control.shape[0],))


def _integrate_accel_yaw_rate(
    control: Tensor,
    agent_type: Tensor,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
    current_speed: Tensor | None = None,
    *,
    dt: float = CONTROL_DT_S,
    use_holonomic_model_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    del agent_type
    _reject_holonomic_model_only(use_holonomic_model_only)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    pos0 = _default_current_pos(control, current_pos)
    head0 = _default_current_head(control, current_head)
    speed0 = _default_current_speed(control, current_speed)

    acc = control[..., 0]
    yaw_rate = control[..., 1]
    speed = torch.clamp(speed0.unsqueeze(1) + torch.cumsum(acc * float(dt), dim=1), min=0.0)
    yaw_rate = torch.where(
        speed > YAW_RATE_SPEED_THRESHOLD_MPS,
        yaw_rate,
        torch.zeros_like(yaw_rate),
    )
    head = wrap_angle(head0.unsqueeze(1) + torch.cumsum(yaw_rate * float(dt), dim=1))
    velocity = torch.stack([head.cos(), head.sin()], dim=-1) * speed.unsqueeze(-1)
    pos = pos0.unsqueeze(1) + torch.cumsum(velocity * float(dt), dim=1)
    return pos, head, speed


def normalize_control(
    control: Tensor,
    agent_type: Tensor,
    *,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
) -> Tensor:
    """Normalize MDG-style control [acceleration, yaw_rate]."""
    if control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control last dim must be 2, got {control.shape[-1]}.")
    if pos_scale_m <= 0.0:
        raise ValueError(f"pos_scale_m must be positive, got {pos_scale_m}.")
    _validate_control_agent_type(control=control, agent_type=agent_type)
    yaw_scale = resolve_control_yaw_scale(
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
        dtype=control.dtype,
        device=control.device,
    )
    view_shape = (yaw_scale.shape[0],) + (1,) * (control.ndim - 2)
    control_norm = torch.empty_like(control)
    control_norm[..., 0] = control[..., 0] / float(pos_scale_m)
    control_norm[..., 1] = control[..., 1] / yaw_scale.view(view_shape)
    return control_norm


def denormalize_control(
    control_norm: Tensor,
    agent_type: Tensor,
    *,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
) -> Tensor:
    """Denormalize MDG-style control [acceleration, yaw_rate]."""
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 2, got {control_norm.shape[-1]}.")
    if pos_scale_m <= 0.0:
        raise ValueError(f"pos_scale_m must be positive, got {pos_scale_m}.")
    _validate_control_agent_type(control=control_norm, agent_type=agent_type)
    yaw_scale = resolve_control_yaw_scale(
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
        dtype=control_norm.dtype,
        device=control_norm.device,
    )
    view_shape = (yaw_scale.shape[0],) + (1,) * (control_norm.ndim - 2)
    control = torch.empty_like(control_norm)
    control[..., 0] = control_norm[..., 0] * float(pos_scale_m)
    control[..., 1] = control_norm[..., 1] * yaw_scale.view(view_shape)
    return control


def decode_control_sequence(
    control: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
    *,
    current_speed: Tensor | None = None,
    dt: float = CONTROL_DT_S,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> tuple[Tensor, Tensor]:
    """Integrate [acceleration, yaw_rate] controls into pose sequence.

    agent_length and no-slip ratio arguments are accepted for compatibility but
    are intentionally unused; all agent types share the same MDG dynamics.
    """
    del agent_length, vehicle_no_slip_point_ratio, cyclist_no_slip_point_ratio
    _validate_dynamics_inputs(control, agent_type, current_pos, current_head, current_speed)
    return _integrate_accel_yaw_rate(
        control=control,
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
        current_speed=current_speed,
        dt=dt,
        use_holonomic_model_only=use_holonomic_model_only,
    )[:2]


def _denormalized_control_to_pose_state(
    control: Tensor,
    agent_type: Tensor,
    current_speed: Tensor | None = None,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
    *,
    dt: float = CONTROL_DT_S,
    use_holonomic_model_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    pos0 = _default_current_pos(control, current_pos)
    head0 = _default_current_head(control, current_head)
    pos, head, speed = _integrate_accel_yaw_rate(
        control=control,
        agent_type=agent_type,
        current_pos=pos0,
        current_head=head0,
        current_speed=current_speed,
        dt=dt,
        use_holonomic_model_only=use_holonomic_model_only,
    )
    local_pos, local_head = transform_to_local(
        pos_global=pos,
        head_global=head,
        pos_now=pos0,
        head_now=head0,
    )
    if local_head is None:
        raise RuntimeError("transform_to_local returned no heading despite heading input.")
    local_head = wrap_angle(local_head)
    return pos, head, speed, local_pos, local_head


def control_norm_to_pose_norm(
    control_norm: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    current_speed: Tensor | None = None,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    pose_pos_scale_m: float = POSE_NORM_POS_SCALE_M,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> Tensor:
    """Convert normalized control to the existing pose-space metric view."""
    del agent_length, vehicle_no_slip_point_ratio, cyclist_no_slip_point_ratio
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 2, got {control_norm.shape[-1]}.")
    _validate_control_agent_type(control=control_norm, agent_type=agent_type)
    control = denormalize_control(
        control_norm=control_norm,
        agent_type=agent_type,
        pos_scale_m=pos_scale_m,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )
    _, _, _, local_pos, local_head = _denormalized_control_to_pose_state(
        control=control,
        agent_type=agent_type,
        current_speed=current_speed,
        use_holonomic_model_only=use_holonomic_model_only,
    )
    return torch.stack(
        [
            local_pos[..., 0] / float(pose_pos_scale_m),
            local_pos[..., 1] / float(pose_pos_scale_m),
            local_head.cos(),
            local_head.sin(),
        ],
        dim=-1,
    )


def control_norm_to_mdg_state_norm(
    control_norm: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    current_speed: Tensor | None = None,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    state_pos_scale_m: float = MDG_STATE_POS_SCALE_M,
    state_speed_scale_mps: float = MDG_STATE_SPEED_SCALE_MPS,
    dt: float = CONTROL_DT_S,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> Tensor:
    """Convert normalized control to MDG's 5D state.

    The state follows the MDG branch: [local_x, local_y, cos(dyaw), sin(dyaw),
    speed]. The state_pos_scale_m/state_speed_scale_mps arguments are kept for
    old config compatibility and must remain 1.0 for MDG-style behavior.
    """
    del agent_length, vehicle_no_slip_point_ratio, cyclist_no_slip_point_ratio
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 2, got {control_norm.shape[-1]}.")
    _validate_control_agent_type(control=control_norm, agent_type=agent_type)
    if state_pos_scale_m <= 0.0:
        raise ValueError(f"state_pos_scale_m must be positive, got {state_pos_scale_m}.")
    if state_speed_scale_mps <= 0.0:
        raise ValueError(
            "state_speed_scale_mps must be positive, "
            f"got {state_speed_scale_mps}."
        )
    control = denormalize_control(
        control_norm=control_norm,
        agent_type=agent_type,
        pos_scale_m=pos_scale_m,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )
    _, _, speed, local_pos, local_head = _denormalized_control_to_pose_state(
        control=control,
        agent_type=agent_type,
        current_speed=current_speed,
        current_pos=current_pos,
        current_head=current_head,
        dt=dt,
        use_holonomic_model_only=use_holonomic_model_only,
    )
    return torch.stack(
        [
            local_pos[..., 0] / float(state_pos_scale_m),
            local_pos[..., 1] / float(state_pos_scale_m),
            local_head.cos(),
            local_head.sin(),
            speed / float(state_speed_scale_mps),
        ],
        dim=-1,
    )


def _future_velocity_or_displacement(
    future_pos: Tensor,
    current_pos: Tensor,
    future_velocity: Tensor | None,
    *,
    dt: float,
) -> Tensor:
    if future_velocity is not None:
        if tuple(future_velocity.shape) != tuple(future_pos.shape):
            raise ValueError(
                "future_velocity must have shape [N, T, 2] and match future_pos, "
                f"got {tuple(future_velocity.shape)} for {tuple(future_pos.shape)}."
            )
        return future_velocity.to(device=future_pos.device, dtype=future_pos.dtype)
    prev_pos = torch.cat([current_pos.unsqueeze(1), future_pos[:, :-1]], dim=1)
    return (future_pos - prev_pos) / float(dt)


def build_rolling_control_target(
    future_pos: Tensor,
    future_head: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    current_speed: Tensor | None = None,
    future_velocity: Tensor | None = None,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    use_holonomic_model_only: bool = False,
    use_rolling_supervision: bool = True,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    dt: float = CONTROL_DT_S,
) -> Tensor:
    """Build per-0.1s MDG-style [acceleration, yaw_rate] targets.

    There is no chunk averaging in semi_mdg. The target has one control vector
    per future 10Hz step, so a 2-second window produces shape [N, 20, 2].
    """
    del agent_length, use_rolling_supervision, vehicle_no_slip_point_ratio, cyclist_no_slip_point_ratio
    _reject_holonomic_model_only(use_holonomic_model_only)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if future_pos.ndim != 3 or future_pos.shape[-1] != 2:
        raise ValueError(f"future_pos must have shape [N, T, 2], got {tuple(future_pos.shape)}.")
    if tuple(future_head.shape) != tuple(future_pos.shape[:2]):
        raise ValueError(
            "future_head must have shape [N, T], "
            f"got {tuple(future_head.shape)} for future_pos {tuple(future_pos.shape)}."
        )
    n_agent = future_pos.shape[0]
    if tuple(current_pos.shape) != (n_agent, 2):
        raise ValueError(f"current_pos must have shape [N, 2], got {tuple(current_pos.shape)}.")
    if tuple(current_head.shape) != (n_agent,):
        raise ValueError(f"current_head must have shape [N], got {tuple(current_head.shape)}.")
    if tuple(agent_type.shape) != (n_agent,):
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")
    if current_speed is not None and tuple(current_speed.shape) != (n_agent,):
        raise ValueError(f"current_speed must have shape [N], got {tuple(current_speed.shape)}.")
    _validate_agent_type(agent_type)

    if future_pos.shape[1] == 0:
        return future_pos.new_zeros((n_agent, 0, CONTROL_FLOW_DIM))

    velocity = _future_velocity_or_displacement(
        future_pos=future_pos,
        current_pos=current_pos,
        future_velocity=future_velocity,
        dt=dt,
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    speed0 = (
        current_speed.to(device=future_pos.device, dtype=future_pos.dtype).clamp_min(0.0)
        if current_speed is not None
        else future_pos.new_zeros((n_agent,))
    )
    prev_speed = torch.cat([speed0.unsqueeze(1), speed[:, :-1]], dim=1)
    prev_head = torch.cat([current_head.unsqueeze(1), future_head[:, :-1]], dim=1)

    acc = (speed - prev_speed) / float(dt)
    yaw_rate = wrap_angle(future_head - prev_head) / float(dt)
    control = torch.stack([acc, yaw_rate], dim=-1)
    control = torch.where(torch.isfinite(control), control, torch.zeros_like(control))
    return normalize_control(
        control=control,
        pos_scale_m=pos_scale_m,
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )


def build_rolling_control_target_with_round_trip_error(
    future_pos: Tensor,
    future_head: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    current_speed: Tensor | None = None,
    future_velocity: Tensor | None = None,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    use_holonomic_model_only: bool = False,
    use_rolling_supervision: bool = True,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    dt: float = CONTROL_DT_S,
) -> tuple[Tensor, Tensor]:
    """Build MDG-style control target and its position reconstruction error."""
    del use_rolling_supervision
    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        current_speed=current_speed,
        future_velocity=future_velocity,
        pos_scale_m=pos_scale_m,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
        use_holonomic_model_only=use_holonomic_model_only,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
        dt=dt,
    )
    control = denormalize_control(
        control_norm=control_norm,
        pos_scale_m=pos_scale_m,
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )
    decoded_pos, _ = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        current_speed=current_speed,
        dt=dt,
        use_holonomic_model_only=use_holonomic_model_only,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
    )
    round_trip_error_m = torch.linalg.vector_norm(decoded_pos - future_pos, dim=-1)
    return control_norm, round_trip_error_m
