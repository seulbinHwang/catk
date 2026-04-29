from __future__ import annotations

import torch.nn as nn


def clear_module_gradients(module: nn.Module | None) -> None:
    """모듈 안에 남아 있는 gradient를 모두 비웁니다.

    Args:
        module: gradient를 비울 PyTorch 모듈입니다. 값이 ``None``이면 아무 일도 하지 않습니다.

    Returns:
        None.

    설명:
        optimizer가 step을 끝내도 PyTorch는 각 파라미터의 ``grad`` 값을 자동으로
        지우지 않습니다. self-forcing DMD에서는 Generator update와 generated estimator
        update가 서로 섞이지 않아야 하므로, 두 update 경계에서 이전 update의 gradient를
        명확히 비워 줍니다.
    """
    if module is None:
        return
    for parameter in module.parameters():
        parameter.grad = None


def find_first_gradient_name(module: nn.Module | None) -> str | None:
    """모듈 안에서 gradient가 남아 있는 첫 파라미터 이름을 찾습니다.

    Args:
        module: 확인할 PyTorch 모듈입니다. 값이 ``None``이면 gradient가 없다고 봅니다.

    Returns:
        str | None: gradient가 남아 있는 첫 파라미터 이름입니다.
        없으면 ``None``을 반환합니다.

    설명:
        여기서는 gradient 값이 0인지가 아니라 ``grad`` 객체가 생겼는지를 봅니다.
        DMD의 평가자 역할을 하는 모델은 Generator loss backward에 아예 참여하지
        않아야 하므로, 0 gradient tensor가 생겨도 분리 실패로 판단합니다.
    """
    if module is None:
        return None
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            return name
    return None


def assert_no_module_gradients(
    module: nn.Module | None,
    module_name: str,
    stage_name: str,
) -> None:
    """특정 update 단계에서 모듈에 gradient가 없는지 검사합니다.

    Args:
        module: 검사할 PyTorch 모듈입니다. 값이 ``None``이면 통과시킵니다.
        module_name: 오류 메시지에 표시할 모듈 이름입니다.
        stage_name: 오류 메시지에 표시할 현재 update 단계 이름입니다.

    Returns:
        None.

    Raises:
        RuntimeError: 해당 모듈에 gradient가 남아 있으면 발생합니다.

    설명:
        self-forcing DMD에서는 두 가지가 반드시 지켜져야 합니다.
        Generator update 중에는 target teacher와 generated estimator에 gradient가
        생기면 안 됩니다. 반대로 generated estimator update 중에는 Generator에
        gradient가 생기면 안 됩니다. 이 함수는 그 원칙을 코드 실행 중 바로 확인합니다.
    """
    gradient_name = find_first_gradient_name(module)
    if gradient_name is None:
        return
    raise RuntimeError(
        f"Unexpected gradient on {module_name} during {stage_name}: {gradient_name}. "
        "Self-forcing DMD requires generator and generated-estimator updates "
        "to be fully separated."
    )
