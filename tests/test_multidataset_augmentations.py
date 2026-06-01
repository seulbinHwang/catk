from __future__ import annotations

import pickle

import torch

import src.smart.datasets.scalable_dataset as scalable_dataset
from src.smart.datamodules.scalable_datamodule import MultiDataModule
from src.smart.datasets.scalable_dataset import MultiDataset


def _make_time_shift_data() -> dict:
    num_agent = 2
    num_step = 20
    position = torch.arange(num_agent * num_step * 3, dtype=torch.float32).reshape(
        num_agent, num_step, 3
    )
    velocity = position + 1000.0
    heading = torch.arange(num_agent * num_step, dtype=torch.float32).reshape(
        num_agent, num_step
    )
    valid_mask = torch.ones(num_agent, num_step, dtype=torch.bool)
    role = torch.zeros(num_agent, 3, dtype=torch.bool)
    role[0, 2] = True
    return {
        "agent": {
            "position": position.clone(),
            "velocity": velocity.clone(),
            "heading": heading.clone(),
            "valid_mask": valid_mask.clone(),
            "role": role,
        }
    }


def test_random_scene_scale_changes_only_xy_and_velocity_xy() -> None:
    data = {
        "map_save": {"traj_pos": torch.ones(3, 2)},
        "agent": {
            "position": torch.tensor([[[1.0, 2.0, 3.0]]]),
            "velocity": torch.tensor([[[4.0, 5.0, 6.0]]]),
            "heading": torch.tensor([[0.25]]),
            "shape": torch.tensor([[2.0, 4.0, 1.5]]),
        },
    }

    scaled = MultiDataset.random_scene_scale({"SCALE_RANGE": [1.5, 1.5]}, data)

    assert torch.allclose(scaled["map_save"]["traj_pos"], torch.full((3, 2), 1.5))
    assert torch.allclose(
        scaled["agent"]["position"], torch.tensor([[[1.5, 3.0, 3.0]]])
    )
    assert torch.allclose(
        scaled["agent"]["velocity"], torch.tensor([[[6.0, 7.5, 6.0]]])
    )
    assert torch.equal(scaled["agent"]["heading"], torch.tensor([[0.25]]))
    assert torch.equal(scaled["agent"]["shape"], torch.tensor([[2.0, 4.0, 1.5]]))


def test_random_time_shift_positive_moves_sequences_left(monkeypatch) -> None:
    data = _make_time_shift_data()
    original = {key: value.clone() for key, value in data["agent"].items()}
    monkeypatch.setattr(scalable_dataset.np.random, "choice", lambda values: 7)

    shifted = MultiDataset.random_time_shift({"MAX_TIME_SHIFT": 5}, data)

    assert torch.equal(shifted["agent"]["position"][:, :-2], original["position"][:, 2:])
    assert torch.equal(shifted["agent"]["velocity"][:, :-2], original["velocity"][:, 2:])
    assert torch.equal(shifted["agent"]["heading"][:, :-2], original["heading"][:, 2:])
    assert torch.equal(shifted["agent"]["position"][:, -2:], torch.zeros(2, 2, 3))
    assert torch.equal(shifted["agent"]["velocity"][:, -2:], torch.zeros(2, 2, 3))
    assert torch.equal(shifted["agent"]["heading"][:, -2:], torch.zeros(2, 2))
    assert not shifted["agent"]["valid_mask"][:, -2:].any()


def test_random_time_shift_negative_moves_sequences_right(monkeypatch) -> None:
    data = _make_time_shift_data()
    original = {key: value.clone() for key, value in data["agent"].items()}
    monkeypatch.setattr(scalable_dataset.np.random, "choice", lambda values: 3)

    shifted = MultiDataset.random_time_shift({"MAX_TIME_SHIFT": 5}, data)

    assert torch.equal(shifted["agent"]["position"][:, 2:], original["position"][:, :-2])
    assert torch.equal(shifted["agent"]["velocity"][:, 2:], original["velocity"][:, :-2])
    assert torch.equal(shifted["agent"]["heading"][:, 2:], original["heading"][:, :-2])
    assert torch.equal(shifted["agent"]["position"][:, :2], torch.zeros(2, 2, 3))
    assert torch.equal(shifted["agent"]["velocity"][:, :2], torch.zeros(2, 2, 3))
    assert torch.equal(shifted["agent"]["heading"][:, :2], torch.zeros(2, 2))
    assert not shifted["agent"]["valid_mask"][:, :2].any()


def test_datamodule_applies_random_augmentation_to_train_only(tmp_path) -> None:
    train_dir = tmp_path / "training"
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "testing"
    tfrecord_dir = tmp_path / "validation_tfrecords"
    for directory in [train_dir, val_dir, test_dir, tfrecord_dir]:
        directory.mkdir()
    for directory in [train_dir, val_dir, test_dir]:
        with open(directory / "sample.pkl", "wb") as handle:
            pickle.dump({"scenario_id": "sample"}, handle)

    datamodule = MultiDataModule(
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        train_raw_dir=train_dir.as_posix(),
        val_raw_dir=val_dir.as_posix(),
        test_raw_dir=test_dir.as_posix(),
        val_tfrecords_splitted=tfrecord_dir.as_posix(),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        train_max_num=32,
        random_scene_scale_config={"SCALE_RANGE": [0.8, 1.2]},
        random_time_shift_config={"MAX_TIME_SHIFT": 5},
    )

    datamodule.setup("fit")

    assert datamodule.train_dataset.random_scene_scale_config == {
        "SCALE_RANGE": [0.8, 1.2]
    }
    assert datamodule.train_dataset.random_time_shift_config == {"MAX_TIME_SHIFT": 5}
    assert datamodule.val_dataset.random_scene_scale_config is None
    assert datamodule.val_dataset.random_time_shift_config is None
