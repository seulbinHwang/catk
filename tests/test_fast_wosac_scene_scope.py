from types import SimpleNamespace

from src.smart.model.smart_flow import SMARTFlow


def _model_for_scope(*, scorer_scene_num: int, n_metric_batches: int, trainer):
    model = object.__new__(SMARTFlow)
    model.scorer_scene_num = scorer_scene_num
    model.n_batch_sim_agents_metric = n_metric_batches
    model._fabric = None
    model._jit_is_scripting = False
    model._trainer = trainer
    model._scorer_scene_num_last_key = None
    model._scorer_val_limit_last_key = None
    return model


def test_scorer_scene_num_expands_int_val_limit_to_requested_scene_scope() -> None:
    trainer = SimpleNamespace(
        world_size=1,
        global_rank=0,
        limit_val_batches=60,
        datamodule=SimpleNamespace(val_batch_size=4),
        is_global_zero=False,
    )
    model = _model_for_scope(scorer_scene_num=1680, n_metric_batches=100, trainer=trainer)

    model._configure_fast_wosac_validation_scope()

    assert model.n_batch_sim_agents_metric == 420
    assert trainer.limit_val_batches == 420


def test_scorer_scene_num_expands_fractional_val_limit_when_too_small() -> None:
    trainer = SimpleNamespace(
        world_size=6,
        global_rank=0,
        limit_val_batches=0.01,
        datamodule=SimpleNamespace(val_batch_size=12, val_dataset=range(44097)),
        is_global_zero=False,
    )
    model = _model_for_scope(scorer_scene_num=1680, n_metric_batches=10, trainer=trainer)

    model._configure_fast_wosac_validation_scope()

    assert model.n_batch_sim_agents_metric == 24
    assert trainer.limit_val_batches == 24


def test_scorer_scene_num_keeps_existing_val_limit_when_large_enough() -> None:
    trainer = SimpleNamespace(
        world_size=6,
        global_rank=0,
        limit_val_batches=0.1,
        datamodule=SimpleNamespace(val_batch_size=12, val_dataset=range(44097)),
        is_global_zero=False,
    )
    model = _model_for_scope(scorer_scene_num=1680, n_metric_batches=10, trainer=trainer)

    model._configure_fast_wosac_validation_scope()

    assert model.n_batch_sim_agents_metric == 24
    assert trainer.limit_val_batches == 0.1
