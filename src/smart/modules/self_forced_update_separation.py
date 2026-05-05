from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


def detach_tensor_tree(value: Any) -> Any:
    """nested container 안의 tensor들을 autograd graph에서 분리합니다.

    Args:
        value: tensor, dict, list, tuple 또는 그 밖의 값을 담은 객체입니다.

    Returns:
        Any: 원래 container 구조는 유지하되 tensor leaf만 ``detach()`` 한 값입니다.

    설명:
        generated estimator update는 현재 Generator가 만든 값들을 학습 target/context로만
        봐야 합니다. tensor container 전체를 boundary에서 detach해 두면 estimator backward가
        online Generator rollout graph로 되돌아가는 것을 구조적으로 막을 수 있습니다.
    """
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):
            return type(value)(*(detach_tensor_tree(item) for item in value))
        return tuple(detach_tensor_tree(item) for item in value)
    return value


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


@contextmanager
def module_gradients_disabled(*modules: nn.Module | None) -> Iterator[None]:
    """주어진 모듈들의 parameter gradient 누적을 잠시 비활성화합니다.

    Args:
        *modules: gradient 누적을 막을 PyTorch 모듈들입니다. 값이 ``None``이면 건너뜁니다.

    Returns:
        Iterator[None]: ``with`` 문에서 쓰는 context manager입니다.

    설명:
        ``torch.no_grad`` 는 block 안의 모든 autograd를 꺼 버리므로 generated estimator
        자체도 학습할 수 없습니다. 이 helper는 지정한 모듈의 parameter ``requires_grad`` 만
        잠시 꺼서, estimator update 중 online Generator / frozen teacher에 gradient가
        누적되는 것을 구조적으로 막고 block이 끝나면 원래 trainable mask를 복원합니다.
    """
    previous_states: list[tuple[Parameter, bool]] = []
    try:
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                previous_states.append((parameter, bool(parameter.requires_grad)))
                parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in reversed(previous_states):
            parameter.requires_grad_(requires_grad)


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
