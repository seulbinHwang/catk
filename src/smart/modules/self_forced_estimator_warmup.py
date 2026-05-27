from __future__ import annotations


DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_EPOCHS = 1
DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_STEPS = 0


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


def resolve_self_forced_estimator_warmup_epochs(config: object | None) -> int:
    """generated estimator만 먼저 학습할 epoch 수를 확정합니다.

    Args:
        config: ``model.model_config.self_forced`` 설정입니다.

    Returns:
        int: generator update를 건너뛰고 generated estimator만 학습할 epoch 수입니다.

    설명:
        기본값은 ``1`` 입니다. 첫 epoch 동안 현재 generator의 self-rollout 분포를
        generated estimator가 먼저 보게 한 뒤, 다음 epoch부터 기존 self-forcing
        generator update를 그대로 시작하기 위한 값입니다.
    """
    raw_epochs = _get_config_value(
        config=config,
        key="estimator_warmup_epochs",
        default=DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_EPOCHS,
    )
    warmup_epochs = int(raw_epochs)
    if warmup_epochs < 0:
        raise ValueError(
            "self_forced.estimator_warmup_epochs must be non-negative, "
            f"got {warmup_epochs}."
        )
    return warmup_epochs


def is_self_forced_estimator_warmup_epoch(
    *,
    current_epoch: int,
    self_forced_start_epoch: int,
    estimator_warmup_epochs: int,
) -> bool:
    """현재 epoch가 generated estimator 사전 적응 구간인지 판단합니다.

    Args:
        current_epoch: 현재 학습 epoch입니다.
        self_forced_start_epoch: self-forcing을 시작할 epoch입니다.
        estimator_warmup_epochs: generated estimator만 학습할 epoch 수입니다.

    Returns:
        bool: 현재 epoch가 사전 적응 구간이면 ``True`` 입니다.

    설명:
        warmup은 self-forcing 시작 epoch부터 계산합니다. 예를 들어
        ``self_forced_start_epoch=2`` 이고 ``estimator_warmup_epochs=1`` 이면,
        epoch 2에서만 generator update를 건너뛰고 epoch 3부터 기존 self-forcing을 실행합니다.
    """
    current_epoch = int(current_epoch)
    start_epoch = int(self_forced_start_epoch)
    warmup_epochs = int(estimator_warmup_epochs)
    if warmup_epochs <= 0:
        return False
    return start_epoch <= current_epoch < start_epoch + warmup_epochs


def resolve_self_forced_estimator_warmup_steps(config: object | None) -> int:
    """generated estimator만 먼저 학습할 step 수를 확정합니다.

    Args:
        config: ``model.model_config.self_forced`` 설정입니다.

    Returns:
        int: generator update를 건너뛰고 generated estimator만 학습할 global step 수입니다.
        ``0`` 이면 step 기반 warmup을 끕니다.

    설명:
        ``estimator_warmup_epochs`` 와 별개로 사용할 수 있는 step 기반 hook 입니다.
        잘 되는 세팅을 빠르게 탐색할 때 (epoch 1 이 수천 step 인 경우) 짧게 잡고 싶을 때
        쓰며, 두 값 모두 양수면 둘 중 하나라도 활성이면 warmup 으로 봅니다.
    """
    raw_steps = _get_config_value(
        config=config,
        key="estimator_warmup_steps",
        default=DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_STEPS,
    )
    warmup_steps = int(raw_steps)
    if warmup_steps < 0:
        raise ValueError(
            "self_forced.estimator_warmup_steps must be non-negative, "
            f"got {warmup_steps}."
        )
    return warmup_steps


def is_self_forced_estimator_warmup_step(
    *,
    global_step: int,
    estimator_warmup_steps: int,
) -> bool:
    """현재 global step 이 generated estimator 사전 적응 구간인지 판단합니다.

    Args:
        global_step: Lightning ``self.global_step`` 입니다 (학습 시작부터 0).
        estimator_warmup_steps: generated estimator만 학습할 step 수입니다.

    Returns:
        bool: 현재 step 이 사전 적응 구간이면 ``True`` 입니다.

    설명:
        epoch 단위 warmup 과 달리 항상 학습 시작 ``global_step=0`` 부터 셉니다.
        ``self_forced_start_epoch`` 이 0 이 아닌 경우에는 epoch warmup 과 OR 로 결합
        하므로, step warmup 만 단독 사용할 때는 ``self_forced_start_epoch=0`` 으로
        두는 게 자연스럽습니다.
    """
    warmup_steps = int(estimator_warmup_steps)
    if warmup_steps <= 0:
        return False
    return int(global_step) < warmup_steps
