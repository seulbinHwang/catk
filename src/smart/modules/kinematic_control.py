from __future__ import annotations

import torch
from torch import Tensor


POSE_FLOW_DIM = 4
CONTROL_FLOW_DIM = 3
DEFAULT_CONTROL_POS_SCALE_M = 1.0
DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M = 0.5
DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO = 0.0
DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO = 0.0
POSE_NORM_POS_SCALE_M = 20.0

# repo의 agent encoder와 dataset 전처리가 공유하는 정수 매핑입니다.
# 기본 control-space는 "pedestrian만 holonomic, 나머지는 non-holonomic" 분기를 이 약속 위에서 직접 코딩하므로,
# 호출자가 다른 인덱싱을 넘기면 잘못된 디코더가 적용되어도 학습이 silent하게 진행됩니다.
# 매핑이 흔들리면 이 상수와 _validate_agent_type() 한 곳을 같이 고치도록 의도적으로 노출합니다.
VEHICLE_TYPE_ID = 0
PEDESTRIAN_TYPE_ID = 1
CYCLIST_TYPE_ID = 2
_VALID_AGENT_TYPE_IDS = (VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID)


def _validate_agent_type(agent_type: Tensor) -> None:
    """agent_type 값이 이 모듈이 가정한 정수 매핑 안에 있는지 확인합니다.

    Args:
        agent_type: 검사할 agent 종류 텐서입니다. shape은 임의입니다.
    """
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


def _validate_control_agent_type(control: Tensor, agent_type: Tensor) -> None:
    """control batch와 agent type batch가 서로 맞는지 확인합니다.

    Args:
        control: 정규화하거나 역정규화할 control입니다. shape은 ``[N, ..., 3]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.

    Raises:
        ValueError: batch 크기 또는 agent type 값이 올바르지 않은 경우 발생합니다.
    """
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
    """config에서 받은 agent별 yaw 정규화 scale을 검증합니다."""
    scales = {
        "control_vehicle_yaw_scale_rad": vehicle_yaw_scale_rad,
        "control_pedestrian_yaw_scale_rad": pedestrian_yaw_scale_rad,
        "control_cyclist_yaw_scale_rad": cyclist_yaw_scale_rad,
    }
    validated = []
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
    """vehicle/cyclist별 no-slip point offset ratio를 검증합니다."""
    ratios = {
        "control_vehicle_no_slip_point_ratio": vehicle_no_slip_point_ratio,
        "control_cyclist_no_slip_point_ratio": cyclist_no_slip_point_ratio,
    }
    validated = []
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
    """agent 종류별 yaw 정규화 scale을 고릅니다.

    Args:
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
            vehicle은 ``0``, pedestrian은 ``1``, cyclist는 ``2`` 입니다.
        vehicle_yaw_scale_rad: vehicle yaw 정규화 scale입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw 정규화 scale입니다.
        cyclist_yaw_scale_rad: cyclist yaw 정규화 scale입니다.
        dtype: 반환 tensor 자료형입니다. 값이 없으면 ``torch.float32`` 를 씁니다.
        device: 반환 tensor 장치입니다. 값이 없으면 ``agent_type`` 장치를 씁니다.

    Returns:
        Tensor: agent별 yaw scale입니다. shape은 ``[N]`` 입니다.
    """
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


def wrap_angle(angle: Tensor) -> Tensor:
    """각도를 안정적인 범위로 접습니다.

    Args:
        angle: 접을 각도입니다. shape은 임의입니다.

    Returns:
        Tensor: 입력과 같은 shape의 각도입니다. 값은 ``[-pi, pi]`` 범위에 있습니다.
    """
    return torch.atan2(angle.sin(), angle.cos())


def safe_sinc(x: Tensor, eps: float = 1.0e-6) -> Tensor:
    """작은 각도에서도 안전하게 ``sin(x) / x`` 를 계산합니다.

    Args:
        x: sinc 값을 계산할 입력입니다. shape은 임의입니다.
        eps: Taylor 분기로 바꿀 0 근처 판단 기준값입니다.

    Returns:
        Tensor: 입력과 같은 shape의 sinc 값입니다.
    """
    near_zero = x.abs() < eps
    safe_x = torch.where(near_zero, torch.ones_like(x), x)
    x2 = x * x
    return torch.where(
        near_zero,
        1.0 - x2 / 6.0 + x2 * x2 / 120.0,
        x.sin() / safe_x,
    )


def _resolve_no_slip_point_offset(
    agent_type: Tensor,
    agent_length: Tensor | None,
    *,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    use_holonomic_model_only: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """vehicle/cyclist box center 뒤쪽의 effective no-slip point offset을 고릅니다."""
    vehicle_ratio, cyclist_ratio = validate_control_no_slip_ratio_config(
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
    )
    if agent_type.ndim != 1:
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")
    if device is None:
        device = agent_type.device
    if dtype is None:
        dtype = torch.float32

    agent_type_device = agent_type.to(device=device)
    _validate_agent_type(agent_type_device)
    offset = torch.zeros(agent_type_device.shape, device=device, dtype=dtype)
    if use_holonomic_model_only:
        return offset

    ratio_by_agent = torch.zeros(agent_type_device.shape, device=device, dtype=dtype)
    ratio_by_agent[agent_type_device == VEHICLE_TYPE_ID] = vehicle_ratio
    ratio_by_agent[agent_type_device == CYCLIST_TYPE_ID] = cyclist_ratio
    ratio_mask = ratio_by_agent > 0.0
    if not bool(ratio_mask.any().item()):
        return offset
    if agent_length is None:
        raise ValueError(
            "agent_length is required when vehicle/cyclist no-slip point ratio > 0 for "
            "vehicle/cyclist control-space decoding."
        )
    if tuple(agent_length.shape) != tuple(agent_type.shape):
        raise ValueError(
            "agent_length must have shape [N] and match agent_type, "
            f"got {tuple(agent_length.shape)} and {tuple(agent_type.shape)}."
        )
    length = agent_length.to(device=device, dtype=dtype)
    if bool((length[ratio_mask] < 0.0).any().item()):
        raise ValueError("agent_length must be non-negative for vehicle/cyclist agents.")
    offset[ratio_mask] = ratio_by_agent[ratio_mask] * length[ratio_mask]
    return offset


def normalize_control(
    control: Tensor,
    agent_type: Tensor,
    *,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
) -> Tensor:
    """제어값을 Flow Matching 학습 스케일로 바꿉니다.

    Args:
        control: 실제 단위 제어값입니다. shape은 ``[N, ..., 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
        pos_scale_m: 이동량을 나눌 meter 단위 값입니다. 모든 agent에 공통 적용합니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        vehicle_yaw_scale_rad: vehicle yaw를 나눌 radian 단위 값입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw를 나눌 radian 단위 값입니다.
        cyclist_yaw_scale_rad: cyclist yaw를 나눌 radian 단위 값입니다.

    Returns:
        Tensor: 정규화된 제어값입니다. shape은 ``[N, ..., 3]`` 입니다.
    """
    if control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control last dim must be 3, got {control.shape[-1]}.")

    control_norm = control.clone()
    control_norm[..., :2] = control[..., :2] / float(pos_scale_m)
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
    control_norm[..., 2] = control[..., 2] / yaw_scale.view(view_shape)
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
    """정규화된 제어값을 실제 단위로 되돌립니다.

    Args:
        control_norm: 정규화된 제어값입니다. shape은 ``[N, ..., 3]`` 입니다.
        pos_scale_m: 이동량 정규화에 쓴 meter 단위 값입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        vehicle_yaw_scale_rad: vehicle yaw를 복원할 radian 단위 값입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw를 복원할 radian 단위 값입니다.
        cyclist_yaw_scale_rad: cyclist yaw를 복원할 radian 단위 값입니다.

    Returns:
        Tensor: 실제 단위 제어값입니다. shape은 ``[N, ..., 3]`` 입니다.
    """
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 3, got {control_norm.shape[-1]}.")

    control = control_norm.clone()
    control[..., :2] = control_norm[..., :2] * float(pos_scale_m)
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
    control[..., 2] = control_norm[..., 2] * yaw_scale.view(view_shape)
    return control


def _decode_control_components(
    delta_s: Tensor,
    delta_n: Tensor,
    delta_head: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None,
    current_pos: Tensor | None,
    current_head: Tensor | None,
    *,
    use_holonomic_model_only: bool,
    vehicle_no_slip_point_ratio: float,
    cyclist_no_slip_point_ratio: float,
) -> tuple[Tensor, Tensor]:
    num_agent = delta_s.shape[0]
    device = delta_s.device
    dtype = delta_s.dtype
    if current_pos is None:
        roll_pos = torch.zeros((num_agent, 2), device=device, dtype=dtype)
    else:
        roll_pos = current_pos.to(device=device, dtype=dtype)
    if current_head is None:
        roll_head = torch.zeros((num_agent,), device=device, dtype=dtype)
    else:
        roll_head = current_head.to(device=device, dtype=dtype)

    if delta_s.shape[1] == 0:
        return (
            delta_s.new_zeros((num_agent, 0, 2)),
            delta_s.new_zeros((num_agent, 0)),
        )

    holonomic_mask = agent_type.to(device=device) == PEDESTRIAN_TYPE_ID
    if use_holonomic_model_only:
        holonomic_mask = torch.ones_like(holonomic_mask)
    no_slip_offset = _resolve_no_slip_point_offset(
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
        use_holonomic_model_only=use_holonomic_model_only,
        dtype=dtype,
        device=device,
    )

    head = wrap_angle(roll_head.unsqueeze(1) + delta_head.cumsum(dim=1))
    head_prev = torch.cat([roll_head.unsqueeze(1), head[:, :-1]], dim=1)
    cos_head = head_prev.cos()
    sin_head = head_prev.sin()
    delta_pos_ped = torch.stack(
        [
            delta_s * cos_head - delta_n * sin_head,
            delta_s * sin_head + delta_n * cos_head,
        ],
        dim=-1,
    )

    mid_head = head_prev + 0.5 * delta_head
    arc_scale = delta_s * safe_sinc(0.5 * delta_head)
    delta_pos_nonhol = torch.stack(
        [arc_scale * mid_head.cos(), arc_scale * mid_head.sin()],
        dim=-1,
    )
    current_heading_vec = torch.stack([cos_head, sin_head], dim=-1)
    next_heading_vec = torch.stack([head.cos(), head.sin()], dim=-1)
    delta_pos_nonhol = delta_pos_nonhol + no_slip_offset[:, None, None] * (
        next_heading_vec - current_heading_vec
    )

    delta_pos = torch.where(holonomic_mask[:, None, None], delta_pos_ped, delta_pos_nonhol)
    pos = roll_pos.unsqueeze(1) + delta_pos.cumsum(dim=1)
    return pos, head


def decode_control_sequence(
    control: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
    *,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> tuple[Tensor, Tensor]:
    """제어 시퀀스를 pose 시퀀스로 바꿉니다.

    Args:
        control: 실제 단위 제어값입니다. shape은 ``[N, T, 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
            ``VEHICLE_TYPE_ID``, ``PEDESTRIAN_TYPE_ID``, ``CYCLIST_TYPE_ID`` 안에 있어야 합니다.
        agent_length: WOMD box length입니다. shape은 ``[N]`` 입니다.
            vehicle/cyclist no-slip point offset ratio가 0보다 클 때 씁니다.
        current_pos: 시작 위치입니다. shape은 ``[N, 2]`` 입니다.
            값이 없으면 원점에서 시작합니다.
        current_head: 시작 방향입니다. shape은 ``[N]`` 입니다.
            값이 없으면 0 rad에서 시작합니다.
        use_holonomic_model_only: ``True`` 이면 vehicle/cyclist도 pedestrian과 같은
            holonomic decoder를 사용합니다. ``False`` 이면 기존처럼 vehicle/cyclist는
            non-holonomic decoder를 사용합니다.
        vehicle_no_slip_point_ratio: vehicle box length에 곱해 no-slip point가 box center
            뒤쪽으로 얼마나 떨어져 있는지 정합니다.
        cyclist_no_slip_point_ratio: cyclist box length에 곱해 no-slip point가 box center
            뒤쪽으로 얼마나 떨어져 있는지 정합니다.

    Returns:
        tuple[Tensor, Tensor]:
            복원된 위치와 방향입니다. 위치 shape은 ``[N, T, 2]`` 이고,
            방향 shape은 ``[N, T]`` 입니다.
    """
    if control.ndim != 3 or control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control must have shape [N, T, 3], got {tuple(control.shape)}.")
    if agent_type.ndim != 1 or agent_type.shape[0] != control.shape[0]:
        raise ValueError(
            "agent_type must have shape [N] and match control batch, "
            f"got {tuple(agent_type.shape)} and {tuple(control.shape)}."
        )
    _validate_agent_type(agent_type)

    return _decode_control_components(
        delta_s=control[..., 0],
        delta_n=control[..., 1],
        delta_head=control[..., 2],
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        use_holonomic_model_only=use_holonomic_model_only,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
    )


def control_norm_to_pose_norm(
    control_norm: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    pose_pos_scale_m: float = POSE_NORM_POS_SCALE_M,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> Tensor:
    """정규화된 제어 시퀀스를 기존 pose-space 표현으로 바꿉니다.

    Args:
        control_norm: 정규화된 제어 시퀀스입니다. shape은 ``[N, T, 3]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
            yaw 역정규화는 agent type별 scale을 사용합니다.
        agent_length: WOMD box length입니다. shape은 ``[N]`` 입니다.
            vehicle/cyclist no-slip point offset ratio가 0보다 클 때 씁니다.
        pos_scale_m: 이동량 정규화에 쓴 meter 단위 값입니다.
        vehicle_yaw_scale_rad: vehicle yaw를 복원할 radian 단위 값입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw를 복원할 radian 단위 값입니다.
        cyclist_yaw_scale_rad: cyclist yaw를 복원할 radian 단위 값입니다.
        pose_pos_scale_m: 기존 pose-space Flow 표현의 위치 정규화 meter 값입니다.
        use_holonomic_model_only: ``True`` 이면 모든 agent type에 holonomic decoder를 씁니다.
        vehicle_no_slip_point_ratio: vehicle box length에 곱하는 no-slip point offset 비율입니다.
        cyclist_no_slip_point_ratio: cyclist box length에 곱하는 no-slip point offset 비율입니다.

    Returns:
        Tensor: 기존 Flow Matching 평가/추론 경로가 쓰는 pose 표현입니다.
            shape은 ``[N, T, 4]`` 이고, 마지막 차원은
            ``[x / pose_pos_scale_m, y / pose_pos_scale_m, cos(yaw), sin(yaw)]`` 입니다.
    """
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 3, got {control_norm.shape[-1]}.")
    _validate_control_agent_type(control=control_norm, agent_type=agent_type)
    yaw_scale = resolve_control_yaw_scale(
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
        dtype=control_norm.dtype,
        device=control_norm.device,
    )
    pos, head = _decode_control_components(
        delta_s=control_norm[..., 0] * float(pos_scale_m),
        delta_n=control_norm[..., 1] * float(pos_scale_m),
        delta_head=control_norm[..., 2] * yaw_scale[:, None],
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=None,
        current_head=None,
        use_holonomic_model_only=use_holonomic_model_only,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
    )
    return torch.stack(
        [
            pos[..., 0] / float(pose_pos_scale_m),
            pos[..., 1] / float(pose_pos_scale_m),
            head.cos(),
            head.sin(),
        ],
        dim=-1,
    )


def build_rolling_control_target(
    future_pos: Tensor,
    future_head: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
    agent_type: Tensor,
    agent_length: Tensor | None = None,
    *,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    use_holonomic_model_only: bool = False,
    use_rolling_supervision: bool = True,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> Tensor:
    """GT pose를 control label로 바꿉니다.

    Args:
        future_pos: GT 미래 위치입니다. shape은 ``[N, T, 2]`` 입니다.
        future_head: GT 미래 방향입니다. shape은 ``[N, T]`` 입니다.
        current_pos: anchor 현재 위치입니다. shape은 ``[N, 2]`` 입니다.
        current_head: anchor 현재 방향입니다. shape은 ``[N]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        agent_length: WOMD box length입니다. shape은 ``[N]`` 입니다.
            vehicle/cyclist no-slip point offset ratio가 0보다 클 때 씁니다.
        pos_scale_m: 이동량 정규화에 쓸 meter 단위 값입니다.
        vehicle_yaw_scale_rad: vehicle yaw 정규화 scale입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw 정규화 scale입니다.
        cyclist_yaw_scale_rad: cyclist yaw 정규화 scale입니다.
        use_holonomic_model_only: ``True`` 이면 vehicle/cyclist도 pedestrian과 같은
            holonomic inverse/decoder projection을 사용합니다.
        use_rolling_supervision: ``True`` 이면 decoder-consistent rolling supervision을
            사용합니다. ``False`` 이면 각 step의 raw GT pose pair만으로 inverse control을
            만듭니다. ``use_holonomic_model_only=True`` 에서는 두 방식이 같은 target을
            만듭니다.
        vehicle_no_slip_point_ratio: vehicle box length에 곱해 no-slip point가 box center
            뒤쪽으로 얼마나 떨어져 있는지 정합니다.
        cyclist_no_slip_point_ratio: cyclist box length에 곱해 no-slip point가 box center
            뒤쪽으로 얼마나 떨어져 있는지 정합니다.

    Returns:
        Tensor: 정규화된 rolling control label입니다. shape은 ``[N, T, 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
    """
    if future_pos.ndim != 3 or future_pos.shape[-1] != 2:
        raise ValueError(f"future_pos must have shape [N, T, 2], got {tuple(future_pos.shape)}.")
    if tuple(future_head.shape) != tuple(future_pos.shape[:2]):
        raise ValueError(
            "future_head must have shape [N, T], "
            f"got {tuple(future_head.shape)} for future_pos {tuple(future_pos.shape)}."
        )
    if tuple(current_pos.shape) != (future_pos.shape[0], 2):
        raise ValueError(f"current_pos must have shape [N, 2], got {tuple(current_pos.shape)}.")
    if tuple(current_head.shape) != (future_pos.shape[0],):
        raise ValueError(f"current_head must have shape [N], got {tuple(current_head.shape)}.")
    if tuple(agent_type.shape) != (future_pos.shape[0],):
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")
    _validate_agent_type(agent_type)

    roll_pos = current_pos.clone()
    roll_head = current_head.clone()
    holonomic_mask = agent_type.to(device=future_pos.device) == PEDESTRIAN_TYPE_ID
    if use_holonomic_model_only:
        holonomic_mask = torch.ones_like(holonomic_mask)
    no_slip_offset = _resolve_no_slip_point_offset(
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
        use_holonomic_model_only=use_holonomic_model_only,
        dtype=future_pos.dtype,
        device=future_pos.device,
    )
    control_steps: list[Tensor] = []

    for step_idx in range(future_pos.shape[1]):
        target_pos = future_pos[:, step_idx]
        target_head = future_head[:, step_idx]
        if use_rolling_supervision:
            source_pos = roll_pos
            source_head = roll_head
        elif step_idx == 0:
            source_pos = current_pos
            source_head = current_head
        else:
            source_pos = future_pos[:, step_idx - 1]
            source_head = future_head[:, step_idx - 1]
        delta_head = wrap_angle(target_head - source_head)
        delta_vec = target_pos - source_pos

        cos_head = source_head.cos()
        sin_head = source_head.sin()
        source_heading_vec = torch.stack([cos_head, sin_head], dim=-1)
        target_heading_vec = torch.stack([target_head.cos(), target_head.sin()], dim=-1)

        # pedestrian: holonomic — control은 현재 heading body frame의 GT 변위를 그대로 담는다.
        ped_delta_s = delta_vec[:, 0] * cos_head + delta_vec[:, 1] * sin_head
        ped_delta_n = -delta_vec[:, 0] * sin_head + delta_vec[:, 1] * cos_head

        # vehicle/cyclist: non-holonomic — no-slip point의 h_mid 방향 투영분만 살린다.
        # 이 inverse 결정이 곧 다음 가상 pose를 정의하므로(decoder를 따로 호출하지 않는다),
        # nonhol_proj 는 같은 한 번의 계산이 control과 다음 roll_pos 양쪽에 쓰인다.
        mid_head = source_head + 0.5 * delta_head
        h_mid = torch.stack([mid_head.cos(), mid_head.sin()], dim=-1)
        source_no_slip_pos = source_pos - no_slip_offset.unsqueeze(-1) * source_heading_vec
        target_no_slip_pos = target_pos - no_slip_offset.unsqueeze(-1) * target_heading_vec
        nonhol_delta_vec = target_no_slip_pos - source_no_slip_pos
        nonhol_proj = (nonhol_delta_vec * h_mid).sum(dim=-1)
        nonhol_delta_s = nonhol_proj / safe_sinc(0.5 * delta_head)

        delta_s = torch.where(holonomic_mask, ped_delta_s, nonhol_delta_s)
        delta_n = torch.where(holonomic_mask, ped_delta_n, torch.zeros_like(ped_delta_n))
        step_control = torch.stack([delta_s, delta_n, delta_head], dim=-1)
        control_steps.append(step_control)

        if use_rolling_supervision:
            nonhol_next_pos = (
                roll_pos
                + nonhol_proj.unsqueeze(-1) * h_mid
                + no_slip_offset.unsqueeze(-1) * (target_heading_vec - source_heading_vec)
            )
            roll_pos = torch.where(holonomic_mask.unsqueeze(-1), target_pos, nonhol_next_pos)
            roll_head = wrap_angle(roll_head + delta_head)

    if len(control_steps) == 0:
        return future_pos.new_zeros((future_pos.shape[0], 0, CONTROL_FLOW_DIM))
    control = torch.stack(control_steps, dim=1)
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
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    use_holonomic_model_only: bool = False,
    use_rolling_supervision: bool = True,
    vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
) -> tuple[Tensor, Tensor]:
    """GT pose를 control label로 바꾸고 복원 위치 오차를 함께 계산합니다.

    Args:
        future_pos: GT 미래 위치입니다. shape은 ``[N, T, 2]`` 입니다.
        future_head: GT 미래 방향입니다. shape은 ``[N, T]`` 입니다.
        current_pos: anchor 현재 위치입니다. shape은 ``[N, 2]`` 입니다.
        current_head: anchor 현재 방향입니다. shape은 ``[N]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        agent_length: WOMD box length입니다. shape은 ``[N]`` 입니다.
            vehicle/cyclist no-slip point offset ratio가 0보다 클 때 씁니다.
        pos_scale_m: 이동량 정규화에 쓸 meter 단위 값입니다.
        vehicle_yaw_scale_rad: vehicle yaw 정규화 scale입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw 정규화 scale입니다.
        cyclist_yaw_scale_rad: cyclist yaw 정규화 scale입니다.
        use_holonomic_model_only: ``True`` 이면 모든 agent type에 holonomic inverse/decoder를 씁니다.
        use_rolling_supervision: ``True`` 이면 decoder-consistent rolling supervision을
            사용하고, ``False`` 이면 raw GT pose pair inverse를 사용합니다.
        vehicle_no_slip_point_ratio: vehicle box length에 곱하는 no-slip point offset 비율입니다.
        cyclist_no_slip_point_ratio: cyclist box length에 곱하는 no-slip point offset 비율입니다.

    Returns:
        tuple[Tensor, Tensor]:
            정규화된 control label과 step별 위치 복원 오차입니다.
            shape은 각각 ``[N, T, 3]`` 과 ``[N, T]`` 입니다.
    """
    if not use_rolling_supervision:
        control_norm = build_rolling_control_target(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=current_pos,
            current_head=current_head,
            agent_type=agent_type,
            agent_length=agent_length,
            pos_scale_m=pos_scale_m,
            vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
            use_holonomic_model_only=use_holonomic_model_only,
            use_rolling_supervision=False,
            vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
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
            use_holonomic_model_only=use_holonomic_model_only,
            vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
        )
        round_trip_error_m = torch.linalg.vector_norm(decoded_pos - future_pos, dim=-1)
        return control_norm, round_trip_error_m

    if future_pos.ndim != 3 or future_pos.shape[-1] != 2:
        raise ValueError(f"future_pos must have shape [N, T, 2], got {tuple(future_pos.shape)}.")
    if tuple(future_head.shape) != tuple(future_pos.shape[:2]):
        raise ValueError(
            "future_head must have shape [N, T], "
            f"got {tuple(future_head.shape)} for future_pos {tuple(future_pos.shape)}."
        )
    if tuple(current_pos.shape) != (future_pos.shape[0], 2):
        raise ValueError(f"current_pos must have shape [N, 2], got {tuple(current_pos.shape)}.")
    if tuple(current_head.shape) != (future_pos.shape[0],):
        raise ValueError(f"current_head must have shape [N], got {tuple(current_head.shape)}.")
    if tuple(agent_type.shape) != (future_pos.shape[0],):
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")
    _validate_agent_type(agent_type)

    if future_pos.shape[1] == 0:
        return (
            future_pos.new_zeros((future_pos.shape[0], 0, CONTROL_FLOW_DIM)),
            future_pos.new_zeros((future_pos.shape[0], 0)),
        )

    roll_pos = current_pos.clone()
    roll_head = current_head.clone()
    holonomic_mask = agent_type.to(device=future_pos.device) == PEDESTRIAN_TYPE_ID
    if use_holonomic_model_only:
        holonomic_mask = torch.ones_like(holonomic_mask)
    no_slip_offset = _resolve_no_slip_point_offset(
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
        use_holonomic_model_only=use_holonomic_model_only,
        dtype=future_pos.dtype,
        device=future_pos.device,
    )
    control_steps: list[Tensor] = []
    round_trip_error_steps: list[Tensor] = []

    for step_idx in range(future_pos.shape[1]):
        target_pos = future_pos[:, step_idx]
        target_head = future_head[:, step_idx]
        source_pos = roll_pos
        source_head = roll_head
        delta_head = wrap_angle(target_head - source_head)
        delta_vec = target_pos - source_pos

        cos_head = source_head.cos()
        sin_head = source_head.sin()
        source_heading_vec = torch.stack([cos_head, sin_head], dim=-1)
        target_heading_vec = torch.stack([target_head.cos(), target_head.sin()], dim=-1)

        ped_delta_s = delta_vec[:, 0] * cos_head + delta_vec[:, 1] * sin_head
        ped_delta_n = -delta_vec[:, 0] * sin_head + delta_vec[:, 1] * cos_head

        mid_head = source_head + 0.5 * delta_head
        h_mid = torch.stack([mid_head.cos(), mid_head.sin()], dim=-1)
        source_no_slip_pos = source_pos - no_slip_offset.unsqueeze(-1) * source_heading_vec
        target_no_slip_pos = target_pos - no_slip_offset.unsqueeze(-1) * target_heading_vec
        nonhol_delta_vec = target_no_slip_pos - source_no_slip_pos
        nonhol_proj = (nonhol_delta_vec * h_mid).sum(dim=-1)
        nonhol_delta_s = nonhol_proj / safe_sinc(0.5 * delta_head)

        delta_s = torch.where(holonomic_mask, ped_delta_s, nonhol_delta_s)
        delta_n = torch.where(holonomic_mask, ped_delta_n, torch.zeros_like(ped_delta_n))
        control_steps.append(torch.stack([delta_s, delta_n, delta_head], dim=-1))

        nonhol_next_pos = (
            roll_pos
            + nonhol_proj.unsqueeze(-1) * h_mid
            + no_slip_offset.unsqueeze(-1) * (target_heading_vec - source_heading_vec)
        )
        roll_pos = torch.where(holonomic_mask.unsqueeze(-1), target_pos, nonhol_next_pos)
        roll_head = wrap_angle(roll_head + delta_head)
        round_trip_error_steps.append(torch.linalg.vector_norm(roll_pos - target_pos, dim=-1))

    control = torch.stack(control_steps, dim=1)
    control_norm = normalize_control(
        control=control,
        pos_scale_m=pos_scale_m,
        agent_type=agent_type,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
    )
    return control_norm, torch.stack(round_trip_error_steps, dim=1)
