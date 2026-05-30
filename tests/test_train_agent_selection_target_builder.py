from __future__ import annotations

import torch
from torch.utils.data.distributed import DistributedSampler

from src.smart.datamodules.scalable_datamodule import (
    MultiDataModule,
    build_train_agent_target_builder,
)
from src.smart.datamodules.target_builder import (
    WaymoTargetBuilderTrain,
    WaymoTargetBuilderVal,
)
from src.smart.metrics.cross_entropy import CrossEntropy


def test_target_builders_implement_basetransform_forward() -> None:
    train_builder = WaymoTargetBuilderTrain(max_num=32)
    val_builder = WaymoTargetBuilderVal()

    assert callable(train_builder.forward)
    assert callable(val_builder.forward)


def test_train_agent_target_builder_factory_respects_eval_selection() -> None:
    legacy_builder = build_train_agent_target_builder(
        train_max_num=32,
        train_use_eval_agent_selection=False,
    )
    eval_builder = build_train_agent_target_builder(
        train_max_num=32,
        train_use_eval_agent_selection=True,
    )

    assert isinstance(legacy_builder, WaymoTargetBuilderTrain)
    assert isinstance(eval_builder, WaymoTargetBuilderVal)


def _make_datamodule(train_use_eval_agent_selection: bool = False) -> MultiDataModule:
    return MultiDataModule(
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        train_raw_dir="/tmp/catk_train",
        val_raw_dir="/tmp/catk_val",
        test_raw_dir="/tmp/catk_test",
        val_tfrecords_splitted="/tmp/catk_val_tfrecords",
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        train_max_num=32,
        train_use_eval_agent_selection=train_use_eval_agent_selection,
    )


def test_datamodule_constructs_both_train_selection_modes() -> None:
    legacy_dm = _make_datamodule(train_use_eval_agent_selection=False)
    eval_dm = _make_datamodule(train_use_eval_agent_selection=True)

    assert isinstance(legacy_dm.train_transform, WaymoTargetBuilderTrain)
    assert isinstance(eval_dm.train_transform, WaymoTargetBuilderVal)


def test_train_dataloader_shards_ddp_with_distributed_sampler(
    monkeypatch,
) -> None:
    dm = _make_datamodule(train_use_eval_agent_selection=True)
    dm.shuffle = True
    dm.train_dataset = list(range(32))
    dm._train_dataset_raw_dir = dm.train_raw_dir
    dm._train_dataset_road_group_size = dm.road_num_rollouts_per_scenario
    monkeypatch.setattr(dm, "_get_trainer_world_info", lambda: (8, 3))

    loader = dm.train_dataloader()

    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 8
    assert loader.sampler.rank == 3


def _make_agent_data() -> dict:
    n_agent = 4
    n_step = 91
    position = torch.zeros(n_agent, n_step, 2)
    position[1, :, 0] = 120.0
    position[2, :, 0] = 180.0
    position[3, :, 0] = 300.0

    role = torch.zeros(n_agent, 3, dtype=torch.bool)
    role[0, 0] = True

    return {
        "agent": {
            "position": position,
            "valid_mask": torch.ones(n_agent, n_step, dtype=torch.bool),
            "role": role,
        }
    }


def test_eval_selection_transform_is_no_op_for_agent_population() -> None:
    out = WaymoTargetBuilderVal()(_make_agent_data())

    assert "train_mask" not in out["agent"]
    assert bool(out["agent"].valid_mask[2].all())
    assert bool(out["agent"].valid_mask[3].all())


def test_legacy_train_transform_keeps_existing_clip_and_train_mask() -> None:
    out = WaymoTargetBuilderTrain(max_num=32)(_make_agent_data())

    assert "train_mask" in out["agent"]
    assert not bool(out["agent"].valid_mask[2].any())
    assert not bool(out["agent"].valid_mask[3].any())


def _make_cross_entropy_inputs(
    n_agent: int = 4,
    n_action: int = 16,
    n_token: int = 5,
) -> dict:
    torch.manual_seed(0)
    return {
        "next_token_logits": torch.randn(n_agent, n_action, n_token),
        "next_token_valid": torch.ones(n_agent, n_action, dtype=torch.bool),
        "pred_pos": torch.zeros(n_agent, 18, 2),
        "pred_head": torch.zeros(n_agent, 18),
        "pred_valid": torch.ones(n_agent, 18, dtype=torch.bool),
        "gt_pos_raw": torch.zeros(n_agent, 18, 2),
        "gt_head_raw": torch.zeros(n_agent, 18),
        "gt_valid_raw": torch.ones(n_agent, 18, dtype=torch.bool),
        "gt_pos": torch.zeros(n_agent, 18, 2),
        "gt_head": torch.zeros(n_agent, 18),
        "gt_valid": torch.ones(n_agent, 18, dtype=torch.bool),
        "token_agent_shape": torch.full((n_agent, 2), 4.0),
        "token_traj": torch.randn(n_agent, n_token, 4, 2) * 0.1,
    }


def _make_cross_entropy() -> CrossEntropy:
    metric = CrossEntropy(
        use_gt_raw=False,
        gt_thresh_scale_length=5.0,
        label_smoothing=0.0,
        rollout_as_gt=False,
    )
    metric.train()
    return metric


def test_cross_entropy_none_train_mask_matches_all_true_mask() -> None:
    inputs = _make_cross_entropy_inputs()
    none_mask_metric = _make_cross_entropy()
    all_true_metric = _make_cross_entropy()

    none_mask_metric.update(train_mask=None, **inputs)
    all_true_metric.update(
        train_mask=torch.ones(4, dtype=torch.bool),
        **inputs,
    )

    torch.testing.assert_close(none_mask_metric.loss_sum, all_true_metric.loss_sum)
    torch.testing.assert_close(none_mask_metric.count, all_true_metric.count)
