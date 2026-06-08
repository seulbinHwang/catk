from __future__ import annotations


DEFAULT_SELF_FORCED_ESTIMATOR_WARMUP_EPOCHS = 1


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


def resolve_self_forced_zone_steps(config: object | None) -> tuple[int, int]:
    """반복 warmup/joint zone 스케줄의 step 길이를 확정합니다.

    Args:
        config: ``model.model_config.self_forced`` 설정입니다.

    Returns:
        tuple[int, int]: ``(warmup_zone_steps, joint_zone_steps)`` 입니다.
        둘 중 하나라도 0 이하이면 zone 스케줄은 비활성(기존 epoch 기반 warmup 사용)입니다.

    설명:
        warmup zone 에서는 generator update 를 건너뛰고 fake(critic) 만 학습하고,
        joint zone 에서는 기존 cadence 기반 self-forcing(fake + generator)을 실행합니다.
        ``step % (W + J) < W`` 면 warmup zone 입니다. 두 zone 을 step 기준으로 무한 반복합니다.
    """
    warmup_zone_steps = int(_get_config_value(config, "warmup_zone_steps", 0))
    joint_zone_steps = int(_get_config_value(config, "joint_zone_steps", 0))
    if warmup_zone_steps < 0 or joint_zone_steps < 0:
        raise ValueError(
            "self_forced.warmup_zone_steps / joint_zone_steps must be non-negative, "
            f"got warmup={warmup_zone_steps}, joint={joint_zone_steps}."
        )
    return warmup_zone_steps, joint_zone_steps


def is_self_forced_warmup_zone_step(
    *,
    step: int,
    warmup_zone_steps: int,
    joint_zone_steps: int,
) -> bool:
    """반복 zone 스케줄에서 현재 step 이 warmup zone 인지 판단합니다.

    Args:
        step: self-forced 학습 step(배치) 인덱스(0-based)입니다.
        warmup_zone_steps: 한 cycle 의 warmup zone step 수입니다.
        joint_zone_steps: 한 cycle 의 joint(동시 튜닝) zone step 수입니다.

    Returns:
        bool: 현재 step 이 warmup zone 이면 ``True`` 입니다.
        ``warmup_zone_steps`` 또는 ``joint_zone_steps`` 가 0 이하이면 항상 ``False`` 입니다.

    설명:
        cycle 길이는 ``warmup_zone_steps + joint_zone_steps`` 이고,
        ``step % cycle < warmup_zone_steps`` 면 warmup zone 입니다. 즉 각 cycle 의
        앞부분이 warmup, 뒷부분이 joint 입니다.
    """
    warmup = int(warmup_zone_steps)
    joint = int(joint_zone_steps)
    if warmup <= 0 or joint <= 0:
        return False
    cycle = warmup + joint
    return (int(step) % cycle) < warmup


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
