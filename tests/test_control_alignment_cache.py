from __future__ import annotations

import pickle

import pytest
import torch

from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.tokens.control_alignment_cache import (
    CONTROL_ALIGNED_FUTURE_POS_KEY,
    CONTROL_ALIGNMENT_CACHE_KEY,
    CONTROL_TRANSITION_NORM_FUTURE_KEY,
    ControlAlignmentCacheConfig,
    attach_control_alignment_cache_fields,
    has_control_alignment_cache_fields,
)
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def _make_processor() -> FlowTokenProcessor:
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.training = True
    processor.shift = 5
    processor.flow_window_steps = 20
    processor.flow_target_dim = 3
    processor.use_prefix_valid_future_loss_mask = True
    processor.use_kinematic_control_flow = True
    processor.use_holonomic_model_only = False
    processor.control_pos_scale_m = 1.0
    processor.control_vehicle_yaw_scale_rad = 0.025
    processor.control_pedestrian_yaw_scale_rad = 0.20
    processor.control_cyclist_yaw_scale_rad = 0.06
    processor.control_vehicle_no_slip_point_ratio = 0.0
    processor.control_cyclist_no_slip_point_ratio = 0.0
    processor.control_alignment_filter_enabled = True
    processor.control_alignment_filter_vehicle_max_error_m = 5.0
    processor.control_alignment_filter_cyclist_max_error_m = 2.0
    processor.control_alignment_cache_config = ControlAlignmentCacheConfig(
        current_step=10,
        pos_scale_m=1.0,
        vehicle_yaw_scale_rad=0.025,
        pedestrian_yaw_scale_rad=0.20,
        cyclist_yaw_scale_rad=0.06,
        use_holonomic_model_only=False,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
    )

    def match_from_passed_trajectory(valid, pos, heading, agent_type, agent_shape):
        coarse_steps = list(range(processor.shift, valid.shape[1], processor.shift))
        shape = (pos.shape[0], len(coarse_steps))
        token_idx = torch.zeros(shape, dtype=torch.long, device=pos.device)
        return {
            "valid_mask": valid[:, coarse_steps],
            "gt_idx": token_idx,
            "gt_pos": pos[:, coarse_steps],
            "gt_heading": heading[:, coarse_steps],
            "sampled_idx": token_idx,
            "sampled_pos": pos[:, coarse_steps],
            "sampled_heading": heading[:, coarse_steps],
        }

    processor._match_agent_token = match_from_passed_trajectory
    return processor


def _make_sample() -> dict:
    n_agent = 2
    n_step = 91
    step = torch.arange(n_step, dtype=torch.float32)
    position = torch.zeros((n_agent, n_step, 3), dtype=torch.float32)
    position[0, :, 0] = step * 0.7
    position[0, :, 1] = 0.2 * torch.sin(step / 8.0)
    position[1, :, 0] = step * 0.3
    position[1, :, 1] = step * 0.1
    heading = torch.zeros((n_agent, n_step), dtype=torch.float32)
    heading[0] = 0.01 * step
    heading[1] = 0.02 * step
    sample = {
        "agent": {
            "valid_mask": torch.ones((n_agent, n_step), dtype=torch.bool),
            "position": position,
            "heading": heading,
            "velocity": torch.zeros((n_agent, n_step, 2), dtype=torch.float32),
            "type": torch.tensor([0, 1], dtype=torch.uint8),
            "shape": torch.tensor(
                [[4.8, 2.0, 1.5], [1.0, 1.0, 1.8]],
                dtype=torch.float32,
            ),
        }
    }
    return sample


def _make_tokenized_agent() -> dict[str, torch.Tensor]:
    return {
        "type": torch.tensor([0, 1], dtype=torch.long),
        "shape": torch.tensor([[4.8, 2.0, 1.5], [1.0, 1.0, 1.8]], dtype=torch.float32),
        "token_agent_shape": torch.tensor([[2.0, 4.8], [1.0, 1.0]], dtype=torch.float32),
    }


def _make_processed_agent(sample: dict) -> dict[str, torch.Tensor]:
    return {
        "valid": sample["agent"]["valid_mask"].clone(),
        "pos": sample["agent"]["position"][..., :2].clone(),
        "heading": sample["agent"]["heading"].clone(),
    }


def test_cache_generation_stores_future_state_control_and_key() -> None:
    sample = _make_sample()
    attach_control_alignment_cache_fields(
        sample,
        config=ControlAlignmentCacheConfig(
            current_step=10,
            pos_scale_m=1.0,
            vehicle_yaw_scale_rad=0.025,
            pedestrian_yaw_scale_rad=0.20,
            cyclist_yaw_scale_rad=0.06,
            use_holonomic_model_only=False,
            vehicle_no_slip_point_ratio=0.0,
            cyclist_no_slip_point_ratio=0.0,
        ),
    )

    assert tuple(sample["agent"][CONTROL_ALIGNED_FUTURE_POS_KEY].shape) == (2, 80, 2)
    assert tuple(sample["agent"][CONTROL_TRANSITION_NORM_FUTURE_KEY].shape) == (2, 80, 3)
    assert tuple(sample["agent"][CONTROL_ALIGNMENT_CACHE_KEY].shape) == (2, 9)


def test_flow_targets_match_online_path_when_cache_is_valid() -> None:
    processor = _make_processor()
    raw_sample = _make_sample()
    cached_sample = _make_sample()
    attach_control_alignment_cache_fields(
        cached_sample,
        config=processor.control_alignment_cache_config,
    )

    online = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=_make_tokenized_agent(),
        processed_agent=_make_processed_agent(raw_sample),
    )
    cached = processor._build_flow_targets(
        data={"agent": cached_sample["agent"]},
        tokenized_agent=_make_tokenized_agent(),
        processed_agent=_make_processed_agent(raw_sample),
    )

    for key in (
        "ctx_sampled_pos",
        "ctx_sampled_heading",
        "flow_train_mask",
        "flow_train_clean_norm",
        "flow_train_clean_metric_norm",
        "flow_train_loss_mask",
    ):
        torch.testing.assert_close(cached[key], online[key])


def test_precomputed_cache_is_ignored_when_train_transform_mutates_valid_mask() -> None:
    processor = _make_processor()
    sample = _make_sample()
    attach_control_alignment_cache_fields(
        sample,
        config=processor.control_alignment_cache_config,
    )
    data = {"agent": dict(sample["agent"])}
    data["agent"]["train_mask"] = torch.ones(2, dtype=torch.bool)

    loaded = processor._load_precomputed_transition_alignment(
        data=data,
        pos=sample["agent"]["position"][..., :2],
        heading=sample["agent"]["heading"],
    )

    assert loaded is None


def test_dataset_strips_partial_new_cache_when_split_starts_with_old_cache(tmp_path) -> None:
    old_sample = _make_sample()
    new_sample = _make_sample()
    attach_control_alignment_cache_fields(
        new_sample,
        config=ControlAlignmentCacheConfig(
            current_step=10,
            pos_scale_m=1.0,
            vehicle_yaw_scale_rad=0.025,
            pedestrian_yaw_scale_rad=0.20,
            cyclist_yaw_scale_rad=0.06,
            use_holonomic_model_only=False,
            vehicle_no_slip_point_ratio=0.0,
            cyclist_no_slip_point_ratio=0.0,
        ),
    )
    with open(tmp_path / "000.pkl", "wb") as handle:
        pickle.dump(old_sample, handle)
    with open(tmp_path / "001.pkl", "wb") as handle:
        pickle.dump(new_sample, handle)

    dataset = MultiDataset(raw_dir=tmp_path.as_posix(), transform=None)
    loaded = dataset.get(1)

    assert not has_control_alignment_cache_fields(loaded["agent"])


def test_dataset_rejects_partial_old_cache_when_split_starts_with_new_cache(tmp_path) -> None:
    new_sample = _make_sample()
    attach_control_alignment_cache_fields(
        new_sample,
        config=ControlAlignmentCacheConfig(
            current_step=10,
            pos_scale_m=1.0,
            vehicle_yaw_scale_rad=0.025,
            pedestrian_yaw_scale_rad=0.20,
            cyclist_yaw_scale_rad=0.06,
            use_holonomic_model_only=False,
            vehicle_no_slip_point_ratio=0.0,
            cyclist_no_slip_point_ratio=0.0,
        ),
    )
    old_sample = _make_sample()
    with open(tmp_path / "000.pkl", "wb") as handle:
        pickle.dump(new_sample, handle)
    with open(tmp_path / "001.pkl", "wb") as handle:
        pickle.dump(old_sample, handle)

    dataset = MultiDataset(raw_dir=tmp_path.as_posix(), transform=None)

    with pytest.raises(ValueError, match="Mixed SMART cache formats"):
        dataset.get(1)
