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


def should_compute_anchor_flow_matching_loss(
    *,
    use_anchor_flow_matching_loss: bool,
    is_estimator_warmup_active: bool,
) -> bool:
    """self-forced step에서 anchor flow-matching loss를 계산할지 판단합니다.

    Args:
        use_anchor_flow_matching_loss: anchor flow-matching 보조 loss 사용 여부입니다.
        is_estimator_warmup_active: 현재 epoch가 generated estimator warmup 구간인지입니다.

    Returns:
        bool: anchor flow-matching forward/loss를 계산해야 하면 ``True`` 입니다.

    설명:
        generated estimator warmup 구간에서는 Generator를 업데이트하지 않으므로,
        anchor flow-matching loss도 계산하지 않습니다.
    """
    return bool(use_anchor_flow_matching_loss) and not bool(is_estimator_warmup_active)


def should_run_self_forced_validation_after_epoch(
    *,
    current_epoch: int,
    self_forced_start_epoch: int,
    estimator_warmup_epochs: int,
    check_val_every_n_epoch: int,
) -> bool:
    """self-forced 학습 중 현재 epoch 끝 validation 실행 여부를 판단합니다.

    Args:
        current_epoch: 현재 학습 epoch입니다.
        self_forced_start_epoch: self-forcing을 시작할 epoch입니다.
        estimator_warmup_epochs: generated estimator만 학습할 epoch 수입니다.
        check_val_every_n_epoch: generator 학습 구간에서 적용할 validation 주기입니다.

    Returns:
        bool: 현재 epoch 끝에 validation을 실행해야 하면 ``True`` 입니다.

    설명:
        estimator warmup 구간에서는 epoch 끝 validation을 실행하지 않습니다.
        warmup이 끝나고 generator 업데이트가 다시 시작되는 epoch부터
        ``check_val_every_n_epoch`` 주기를 새로 셉니다. self-forcing 시작 전
        epoch에서는 Lightning의 기존 epoch 기준 주기를 그대로 따릅니다.
    """
    current_epoch = int(current_epoch)
    start_epoch = int(self_forced_start_epoch)
    warmup_epochs = int(estimator_warmup_epochs)
    check_interval = int(check_val_every_n_epoch)
    if warmup_epochs < 0:
        raise ValueError(
            "self_forced.estimator_warmup_epochs must be non-negative, "
            f"got {warmup_epochs}."
        )
    if check_interval <= 0:
        raise ValueError(
            "trainer.check_val_every_n_epoch must be positive, "
            f"got {check_interval}."
        )
    if warmup_epochs <= 0:
        return (current_epoch + 1) % check_interval == 0

    dmd_start_epoch = start_epoch + warmup_epochs
    if current_epoch < start_epoch:
        return (current_epoch + 1) % check_interval == 0
    if current_epoch < dmd_start_epoch:
        return False

    generator_epoch_count = current_epoch - dmd_start_epoch + 1
    return generator_epoch_count % check_interval == 0
