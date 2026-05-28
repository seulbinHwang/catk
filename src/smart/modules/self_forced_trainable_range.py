from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch.nn as nn


SELF_FORCED_UNFROZEN_RANGES: Tuple[str, ...] = (
    "except_map_encoder",
    "middle",
    "full_flow_decoder",
    "velocity_head_only",
)
DEFAULT_SELF_FORCED_UNFROZEN_RANGE = "except_map_encoder"


def _get_config_value(config: object | None, key: str, default: object) -> object:
    """설정 객체에서 값을 안전하게 꺼냅니다.

    Args:
        config: OmegaConf DictConfig, dict, 일반 객체 또는 ``None`` 입니다.
        key: 읽을 설정 이름입니다.
        default: 설정이 없을 때 사용할 값입니다.

    Returns:
        object: 설정에서 읽은 값입니다. 값이 없으면 ``default`` 를 돌려줍니다.
    """
    if config is None:
        return default

    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter(key, default)
    elif isinstance(config, dict):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


def resolve_self_forced_unfrozen_range(config: object | None) -> str:
    """self-forcing에서 학습할 파라미터 범위를 확정합니다.

    Args:
        config: ``model.model_config.self_forced`` 설정입니다.

    Returns:
        str: ``except_map_encoder``, ``middle``, ``full_flow_decoder`` 중 하나입니다.

    설명:
        기본값은 ``except_map_encoder`` 입니다. 이 값은 기존 self-forcing의
        ``freeze_map_encoder=true`` 와 같은 의도입니다. 즉 지도 처리부는 고정하고,
        나머지는 학습할 수 있게 둡니다.
    """
    raw_range = _get_config_value(
        config=config,
        key="unfrozen_range",
        default=DEFAULT_SELF_FORCED_UNFROZEN_RANGE,
    )
    resolved_range = str(raw_range).strip().lower()
    if resolved_range not in SELF_FORCED_UNFROZEN_RANGES:
        valid_values = ", ".join(SELF_FORCED_UNFROZEN_RANGES)
        raise ValueError(
            "self_forced.unfrozen_range must be one of "
            f"{{{valid_values}}}, got {resolved_range!r}."
        )
    return resolved_range


def _set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    """모듈 안의 모든 파라미터 학습 여부를 바꿉니다.

    Args:
        module: 학습 여부를 바꿀 PyTorch 모듈입니다. ``None`` 이면 아무 일도 하지 않습니다.
        requires_grad: ``True`` 면 optimizer가 업데이트할 수 있고, ``False`` 면 고정됩니다.

    Returns:
        None
    """
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _get_agent_encoder(model: nn.Module) -> nn.Module | None:
    """SMARTFlowDecoder 안의 agent encoder를 가져옵니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        nn.Module | None: ``agent_encoder`` 가 있으면 돌려주고, 없으면 ``None`` 입니다.
    """
    return getattr(model, "agent_encoder", None)


def _get_flow_decoder(model: nn.Module) -> nn.Module | None:
    """agent encoder 안의 flow decoder를 가져옵니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        nn.Module | None: ``agent_encoder.flow_decoder`` 가 있으면 돌려주고, 없으면 ``None`` 입니다.
    """
    agent_encoder = _get_agent_encoder(model)
    if agent_encoder is None:
        return None
    return getattr(agent_encoder, "flow_decoder", None)


def _iter_last_context_blocks(agent_encoder: nn.Module | None) -> Iterable[nn.Module]:
    """생성부 바로 앞의 마지막 agent 문맥 블록들을 순서대로 찾습니다.

    Args:
        agent_encoder: SMART agent encoder입니다. ``None`` 이면 빈 목록처럼 동작합니다.

    Yields:
        nn.Module: 마지막 temporal / map-to-agent / agent-to-agent 블록입니다.

    설명:
        ``middle`` 범위는 전체 문맥부를 전부 열지 않습니다. 대신 마지막 궤적 생성부와
        바로 맞닿은 얕은 연결부만 엽니다. CATK 구조에서는 마지막 temporal attention,
        마지막 map-to-agent attention, 마지막 agent-to-agent attention이 여기에 해당합니다.
    """
    if agent_encoder is None:
        return

    for layer_group_name in ("t_attn_layers", "pt2a_attn_layers", "a2a_attn_layers"):
        layer_group = getattr(agent_encoder, layer_group_name, None)
        if layer_group is None:
            continue
        if len(layer_group) == 0:
            continue
        yield layer_group[-1]


def _apply_except_map_encoder_range(model: nn.Module) -> None:
    """지도 처리부만 고정하고 나머지 self-forcing 파라미터는 학습 가능하게 둡니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        None
    """
    _set_requires_grad(model, True)
    _set_requires_grad(getattr(model, "map_encoder", None), False)


def _apply_middle_range(model: nn.Module) -> None:
    """flow decoder와 마지막 agent 문맥 블록만 학습 가능하게 둡니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        None

    설명:
        이 범위는 ``except_map_encoder`` 보다 더 보수적입니다. 지도 처리부와 대부분의
        agent 문맥부는 고정하고, flow decoder와 생성부 바로 앞의 마지막 문맥 블록만 엽니다.
        따라서 pretrained scene understanding을 크게 흔들지 않으면서, 자기 rollout 상태에
        맞는 얕은 연결부와 최종 궤적 생성부를 함께 조정할 수 있습니다.
    """
    _set_requires_grad(model, False)
    _set_requires_grad(_get_flow_decoder(model), True)

    agent_encoder = _get_agent_encoder(model)
    for context_block in _iter_last_context_blocks(agent_encoder):
        _set_requires_grad(context_block, True)


def _apply_full_flow_decoder_range(model: nn.Module) -> None:
    """마지막 궤적 생성부만 학습 가능하게 둡니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        None

    설명:
        지도 처리부와 agent 문맥부는 그대로 보존하고, ``agent_encoder.flow_decoder`` 만
        자기 rollout 분포 차이를 흡수하게 합니다.
    """
    _set_requires_grad(model, False)
    _set_requires_grad(_get_flow_decoder(model), True)


def _apply_velocity_head_only_range(model: nn.Module) -> None:
    """flow_decoder.velocity_head (마지막 flow MLP) 만 학습 가능하게 둡니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.

    Returns:
        None

    설명:
        가장 보수적인 범위.  encoder/agent context/flow chunk 까지 전부 freeze 하고
        flow velocity 를 만드는 최종 MLP 만 self-forcing 학습 신호로 흡수.
        debug 또는 layer-wise 학습 영향 분석 용도.
    """
    _set_requires_grad(model, False)
    flow_decoder = _get_flow_decoder(model)
    if flow_decoder is None:
        return
    velocity_head = getattr(flow_decoder, "velocity_head", None)
    if velocity_head is None:
        raise AttributeError(
            "velocity_head_only range expects model.agent_encoder.flow_decoder.velocity_head; "
            "got flow_decoder without velocity_head attribute."
        )
    _set_requires_grad(velocity_head, True)


def count_trainable_parameters(model: nn.Module) -> int:
    """현재 학습 가능한 파라미터 개수를 셉니다.

    Args:
        model: 파라미터를 셀 PyTorch 모듈입니다.

    Returns:
        int: ``requires_grad=True`` 인 파라미터 원소 개수입니다.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def collect_trainable_parameter_names(model: nn.Module) -> Tuple[str, ...]:
    """현재 학습 가능한 파라미터 이름을 모읍니다.

    Args:
        model: 파라미터 이름을 읽을 PyTorch 모듈입니다.

    Returns:
        Tuple[str, ...]: ``requires_grad=True`` 인 파라미터 이름 목록입니다.
    """
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def apply_self_forced_unfrozen_range(
    model: nn.Module,
    unfrozen_range: str,
) -> Dict[str, int]:
    """self-forcing에서 학습할 파라미터 범위를 모델에 적용합니다.

    Args:
        model: ``SMARTFlowDecoder`` 또는 같은 속성 구조를 가진 모듈입니다.
        unfrozen_range: ``except_map_encoder``, ``middle``, ``full_flow_decoder`` 중 하나입니다.

    Returns:
        Dict[str, int]: 적용 결과 요약입니다.
            ``trainable_parameters`` 는 학습 가능한 파라미터 원소 개수입니다.
            ``frozen_parameters`` 는 고정된 파라미터 원소 개수입니다.

    설명:
        세 범위의 의도는 아래와 같습니다.
        ``except_map_encoder`` 는 기존 self-forcing의 ``freeze_map_encoder=true`` 와 같습니다.
        ``middle`` 은 flow decoder와 마지막 agent 문맥 블록만 엽니다.
        ``full_flow_decoder`` 는 flow decoder만 엽니다.
    """
    resolved_range = str(unfrozen_range).strip().lower()
    if resolved_range == "except_map_encoder":
        _apply_except_map_encoder_range(model)
    elif resolved_range == "middle":
        _apply_middle_range(model)
    elif resolved_range == "full_flow_decoder":
        _apply_full_flow_decoder_range(model)
    elif resolved_range == "velocity_head_only":
        _apply_velocity_head_only_range(model)
    else:
        valid_values = ", ".join(SELF_FORCED_UNFROZEN_RANGES)
        raise ValueError(
            "self_forced.unfrozen_range must be one of "
            f"{{{valid_values}}}, got {resolved_range!r}."
        )

    trainable_parameters = count_trainable_parameters(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
    }
