from __future__ import annotations

import torch

DEFAULT_LENGTH_SCALE: float = 20.0


def normalize_length_values(
    values: torch.Tensor,
    length_scale: float = DEFAULT_LENGTH_SCALE,
) -> torch.Tensor:
    """meter 단위 길이값을 공통 기준 길이로 나눕니다.

    Args:
        values: 길이와 길이 차이처럼 meter 단위로 해석되는 입력입니다.
            shape은 ``[...]`` 또는 ``[..., d]`` 입니다.
        length_scale: 나눌 기준 길이입니다. 기본값은 ``20.0`` 입니다.
            0보다 커야 합니다.

    Returns:
        torch.Tensor:
            입력과 같은 shape의 정규화 결과입니다.
            shape은 ``values.shape`` 와 같습니다.
    """
    if length_scale <= 0:
        raise ValueError(f"length_scale must be positive, got {length_scale}")
    return values / length_scale
