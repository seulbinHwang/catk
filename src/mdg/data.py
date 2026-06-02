from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.mdg.memory_balanced_batch_sampler import (
    MemoryBalancedDistributedBatchSampler,
    load_or_build_memory_metadata,
    memory_balance_weights,
)
from src.smart.datamodules.exact_distributed_sampler import ExactDistributedSampler


_HISTORY_STEPS = 11
_TOTAL_STEPS = 91
_FUTURE_STEPS = 80


def _as_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


MDG_MAP_SAMPLING_VERSION = "arclength_v1"
MDG_TRAFFIC_SIGNAL_VERSION = "time_indexed_v1"
_DEFAULT_CURRENT_INDEX = _HISTORY_STEPS - 1
_DEFAULT_TRAIN_ANCHOR_STEPS = tuple(range(_DEFAULT_CURRENT_INDEX, 81, 10))


def _select_nearest(
    positions: Tensor,
    valid: Tensor,
    center: Tensor,
    max_count: int,
    force_index: Optional[int] = None,
) -> Tensor:
    if positions.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    distances = torch.linalg.norm(positions[:, :2] - center[:2], dim=-1)
    distances = torch.where(valid, distances, torch.full_like(distances, float("inf")))
    order = torch.argsort(distances, stable=True)
    if force_index is not None and 0 <= force_index < positions.shape[0]:
        force = torch.tensor([force_index], dtype=torch.long, device=positions.device)
        order = torch.cat((force, order[order != force_index]), dim=0)
    order = order[torch.isfinite(distances[order])]
    return order[:max_count].cpu()


def _select_current_valid(valid: Tensor, current_index: int) -> Tensor:
    if valid.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.where(valid[:, current_index])[0].cpu()


def _time_shift_view(value: Tensor, anchor_step: int, pad_value: float | int | bool = 0) -> Tensor:
    offset = int(anchor_step) - _DEFAULT_CURRENT_INDEX
    if offset == 0:
        return value
    out = torch.full_like(value, pad_value)
    src_start = max(0, offset)
    dst_start = max(0, -offset)
    length = min(int(value.shape[1]) - src_start, _TOTAL_STEPS - dst_start)
    if length > 0:
        out[:, dst_start : dst_start + length] = value[:, src_start : src_start + length]
    return out


def _pad_first_dim(value: Tensor, size: int, pad_value: float | int | bool = 0) -> tuple[Tensor, Tensor]:
    out_shape = (size,) + tuple(value.shape[1:])
    out = torch.full(out_shape, pad_value, dtype=value.dtype)
    valid = torch.zeros(size, dtype=torch.bool)
    n = min(size, int(value.shape[0]))
    if n > 0:
        out[:n] = value[:n].cpu()
        valid[:n] = True
    return out, valid


def _extract_map(data: Dict[str, Any]) -> Dict[str, Tensor]:
    if "mdg_map" not in data:
        raise ValueError("MDG cache is missing the required 'mdg_map' field. Regenerate the cache with the MDG branch.")
    mdg_map = data["mdg_map"]
    required = ("position", "heading", "id", "type", "light_type", "valid", "sampling")
    missing = [key for key in required if key not in mdg_map]
    if missing:
        raise ValueError(f"MDG cache mdg_map is missing required keys: {missing}")
    sampling = mdg_map["sampling"]
    if sampling != MDG_MAP_SAMPLING_VERSION:
        raise ValueError(
            "MDG cache must use arclength_v1 map sampling. "
            f"Found sampling={sampling!r}; regenerate MDG_cache from raw WOMD TFRecords."
        )
    light_type = _as_tensor(mdg_map["light_type"], torch.long)
    return {
        "position": _as_tensor(mdg_map["position"], torch.float32),
        "heading": _as_tensor(mdg_map["heading"], torch.float32),
        "id": _as_tensor(mdg_map["id"], torch.long),
        "type": _as_tensor(mdg_map["type"], torch.long),
        "light_type": light_type,
        "valid": _as_tensor(mdg_map["valid"], torch.bool),
    }


def _extract_traffic_signals(data: Dict[str, Any]) -> Dict[str, Tensor]:
    if "mdg_traffic_signal" not in data:
        raise ValueError(
            "MDG cache is missing the required 'mdg_traffic_signal' field. "
            "Regenerate the cache with the MDG branch."
        )
    signal = data["mdg_traffic_signal"]
    required = ("position", "heading", "state", "valid", "lane_id", "time_step", "version")
    missing = [key for key in required if key not in signal]
    if missing:
        raise ValueError(f"MDG cache mdg_traffic_signal is missing required keys: {missing}")
    version = signal["version"]
    if version != MDG_TRAFFIC_SIGNAL_VERSION:
        raise ValueError(
            "MDG cache mdg_traffic_signal must be time_indexed_v1. "
            f"Found version={version!r}; regenerate MDG_cache from raw WOMD TFRecords."
        )
    return {
        "position": _as_tensor(signal["position"], torch.float32),
        "heading": _as_tensor(signal["heading"], torch.float32),
        "state": _as_tensor(signal["state"], torch.long),
        "valid": _as_tensor(signal["valid"], torch.bool),
        "lane_id": _as_tensor(signal["lane_id"], torch.long),
        "time_step": _as_tensor(signal["time_step"], torch.long),
        "version": version,
    }


def _select_signal_at_anchor(signal: Dict[str, Any], anchor_step: int) -> Dict[str, Tensor]:
    valid = signal["valid"] & (signal["time_step"] == int(anchor_step))
    return {
        "position": signal["position"][valid],
        "heading": signal["heading"][valid],
        "state": signal["state"][valid],
        "valid": signal["valid"][valid],
        "lane_id": signal["lane_id"][valid],
        "time_indexed": True,
    }


def _anchor_map_light_type(mdg_map: Dict[str, Tensor], signal_at_anchor: Dict[str, Tensor]) -> Tensor:
    if mdg_map["id"].numel() == 0 or signal_at_anchor["lane_id"].numel() == 0:
        return torch.zeros_like(mdg_map["light_type"])
    by_lane = {
        int(lane_id.item()): int(state.item())
        for lane_id, state, valid in zip(
            signal_at_anchor["lane_id"],
            signal_at_anchor["state"],
            signal_at_anchor["valid"],
        )
        if bool(valid.item())
    }
    if not by_lane:
        return torch.zeros_like(mdg_map["light_type"])
    return torch.as_tensor(
        [by_lane.get(int(polyline_id.item()), 0) for polyline_id in mdg_map["id"]],
        dtype=torch.long,
    )


class MDGDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        max_agents: Optional[int],
        max_map_polylines: int,
        map_waypoints: int,
        max_traffic_lights: int,
        training: bool,
        tfrecord_dir: Optional[str] = None,
        train_anchor_steps: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.raw_paths = [
            path
            for path in sorted(self.raw_dir.glob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]
        self.max_agents = int(max_agents) if max_agents is not None else None
        self.max_map_polylines = int(max_map_polylines)
        self.map_waypoints = int(map_waypoints)
        self.max_traffic_lights = int(max_traffic_lights)
        self.training = bool(training)
        self.tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        if self.training:
            anchors = tuple(int(step) for step in (train_anchor_steps or _DEFAULT_TRAIN_ANCHOR_STEPS))
        else:
            anchors = (_DEFAULT_CURRENT_INDEX,)
        if not anchors:
            raise ValueError("MDGDataset requires at least one anchor step.")
        if min(anchors) < _DEFAULT_CURRENT_INDEX or max(anchors) >= _TOTAL_STEPS:
            raise ValueError(
                f"MDG anchor steps must be in [{_DEFAULT_CURRENT_INDEX}, {_TOTAL_STEPS - 1}], got {anchors}."
            )
        self.anchor_steps = anchors

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.raw_paths[index]
        with path.open("rb") as handle:
            data = pickle.load(handle)
        anchor_step = self._sample_anchor_step()
        return self._build_sample(data, anchor_step=anchor_step)

    def _sample_anchor_step(self) -> int:
        if len(self.anchor_steps) == 1:
            return self.anchor_steps[0]
        idx = int(torch.randint(len(self.anchor_steps), (1,)).item())
        return self.anchor_steps[idx]

    def _build_sample(self, data: Dict[str, Any], anchor_step: Optional[int] = None) -> Dict[str, Any]:
        if anchor_step is None:
            anchor_step = self.anchor_steps[0]
        agent = data["agent"]
        position = _as_tensor(agent["position"], torch.float32)
        heading = _as_tensor(agent["heading"], torch.float32)
        velocity = _as_tensor(agent["velocity"], torch.float32)
        valid = _as_tensor(agent["valid_mask"], torch.bool)
        shape = _as_tensor(agent["shape"], torch.float32)
        agent_type = _as_tensor(agent["type"], torch.long).clamp(min=0, max=2)
        role = _as_tensor(agent["role"], torch.bool)
        agent_id = _as_tensor(agent["id"], torch.long)

        position = _time_shift_view(position, anchor_step, pad_value=0.0)
        heading = _time_shift_view(heading, anchor_step, pad_value=0.0)
        velocity = _time_shift_view(velocity, anchor_step, pad_value=0.0)
        valid = _time_shift_view(valid, anchor_step, pad_value=False)

        current_index = _DEFAULT_CURRENT_INDEX
        sdc_candidates = torch.where(role[:, 0])[0]
        sdc_index = int(sdc_candidates[0].item()) if len(sdc_candidates) else 0
        center = position[sdc_index, current_index, :2]
        if self.max_agents is None:
            selected_agents = _select_current_valid(valid, current_index)
            agent_tensor_size = int(selected_agents.numel())
        else:
            selected_agents = _select_nearest(
                positions=position[:, current_index, :2],
                valid=valid[:, current_index],
                center=center,
                max_count=self.max_agents,
                force_index=sdc_index,
            )
            agent_tensor_size = self.max_agents

        position = position[selected_agents]
        heading = heading[selected_agents]
        velocity = velocity[selected_agents]
        valid = valid[selected_agents]
        shape = shape[selected_agents]
        agent_type = agent_type[selected_agents]
        agent_id = agent_id[selected_agents]

        position_pad, agent_present = _pad_first_dim(position, agent_tensor_size)
        heading_pad, _ = _pad_first_dim(heading, agent_tensor_size)
        velocity_pad, _ = _pad_first_dim(velocity, agent_tensor_size)
        valid_pad, _ = _pad_first_dim(valid, agent_tensor_size)
        shape_pad, _ = _pad_first_dim(shape, agent_tensor_size)
        type_pad, _ = _pad_first_dim(agent_type, agent_tensor_size)
        id_pad, _ = _pad_first_dim(agent_id, agent_tensor_size, pad_value=-1)
        agent_valid = agent_present & valid_pad[:, current_index]

        mdg_map = _extract_map(data)
        signal = _extract_traffic_signals(data)
        signal_at_anchor = _select_signal_at_anchor(signal, anchor_step=anchor_step)
        map_light_type_all = _anchor_map_light_type(mdg_map, signal_at_anchor)
        map_anchor = mdg_map["position"][:, 0, :2] if mdg_map["position"].numel() else torch.zeros(0, 2)
        selected_map = _select_nearest(
            positions=map_anchor,
            valid=mdg_map["valid"],
            center=center,
            max_count=self.max_map_polylines,
        )
        map_position, map_valid = _pad_first_dim(
            mdg_map["position"][selected_map],
            self.max_map_polylines,
        )
        map_heading, _ = _pad_first_dim(
            mdg_map["heading"][selected_map],
            self.max_map_polylines,
        )
        map_type, _ = _pad_first_dim(
            mdg_map["type"][selected_map].clamp(min=0, max=15),
            self.max_map_polylines,
        )
        map_light_type, _ = _pad_first_dim(
            map_light_type_all[selected_map].clamp(min=0, max=8),
            self.max_map_polylines,
        )

        selected_signal = _select_nearest(
            positions=signal_at_anchor["position"],
            valid=signal_at_anchor["valid"],
            center=center,
            max_count=self.max_traffic_lights,
        )
        signal_position, signal_valid = _pad_first_dim(
            signal_at_anchor["position"][selected_signal],
            self.max_traffic_lights,
        )
        signal_heading, _ = _pad_first_dim(
            signal_at_anchor["heading"][selected_signal],
            self.max_traffic_lights,
        )
        signal_state, _ = _pad_first_dim(
            signal_at_anchor["state"][selected_signal].clamp(min=0, max=8),
            self.max_traffic_lights,
        )

        scenario_id = str(data["scenario_id"])
        tfrecord_path = None
        if self.tfrecord_dir is not None:
            tfrecord_path = (self.tfrecord_dir / f"{scenario_id}.tfrecords").as_posix()
        elif "tfrecord_path" in data:
            tfrecord_path = data["tfrecord_path"]

        return {
            "scenario_id": scenario_id,
            "tfrecord_path": tfrecord_path,
            "anchor_step": torch.tensor(int(anchor_step), dtype=torch.long),
            "agent_id": id_pad,
            "agent_type": type_pad.long(),
            "agent_shape": shape_pad,
            "agent_valid": agent_valid,
            "agent_position": position_pad,
            "agent_heading": heading_pad,
            "agent_velocity": velocity_pad,
            "agent_valid_mask": valid_pad,
            "map_position": map_position,
            "map_heading": map_heading,
            "map_type": map_type.long(),
            "map_light_type": map_light_type.long(),
            "map_valid": map_valid,
            "signal_position": signal_position,
            "signal_heading": signal_heading,
            "signal_state": signal_state.long(),
            "signal_valid": signal_valid,
        }


_AGENT_PAD_VALUES: Dict[str, float | int | bool] = {
    "agent_id": -1,
    "agent_type": 0,
    "agent_shape": 0.0,
    "agent_valid": False,
    "agent_position": 0.0,
    "agent_heading": 0.0,
    "agent_velocity": 0.0,
    "agent_valid_mask": False,
}


def _pad_agent_tensor(value: Tensor, size: int, pad_value: float | int | bool) -> Tensor:
    if int(value.shape[0]) == size:
        return value
    out_shape = (size,) + tuple(value.shape[1:])
    out = torch.full(out_shape, pad_value, dtype=value.dtype)
    if value.shape[0] > 0:
        out[: value.shape[0]] = value
    return out


def _collate_values(key: str, values: Iterable[Any]) -> Any:
    values = list(values)
    first = values[0]
    if isinstance(first, Tensor):
        if key in _AGENT_PAD_VALUES:
            max_size = max(int(value.shape[0]) for value in values)
            values = [
                _pad_agent_tensor(value, max_size, _AGENT_PAD_VALUES[key])
                for value in values
            ]
        return torch.stack(values, dim=0)
    return values


def collate_mdg_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty MDG batch.")
    keys = samples[0].keys()
    return {key: _collate_values(key, (sample[key] for sample in samples)) for key in keys}


class MDGDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_tfrecords_splitted: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_agents: int = 64,
        max_map_polylines: int = 320,
        map_waypoints: int = 16,
        max_traffic_lights: int = 16,
        train_memory_balanced_batching: bool = False,
        train_memory_balanced_batches: Optional[bool] = None,
        train_memory_balance_metadata_cache: Optional[str] = None,
        train_memory_balance_metadata_num_workers: int = 8,
        train_memory_balance_build_on_missing: bool = True,
        train_memory_balance_agent_weight: float = 1.0,
        train_memory_balance_current_valid_agent_weight: float = 1.0,
        train_memory_balance_valid_agent_step_weight: float = 0.0,
        train_memory_balance_map_weight: float = 0.02,
        train_memory_balance_seed: int = 0,
        train_anchor_steps: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_max_agents = train_max_agents
        self.max_map_polylines = max_map_polylines
        self.map_waypoints = map_waypoints
        self.max_traffic_lights = max_traffic_lights
        if train_memory_balanced_batches is not None:
            train_memory_balanced_batching = bool(train_memory_balanced_batches)
        self.train_memory_balanced_batching = bool(train_memory_balanced_batching)
        self.train_memory_balance_metadata_cache = train_memory_balance_metadata_cache
        self.train_memory_balance_metadata_num_workers = int(train_memory_balance_metadata_num_workers)
        self.train_memory_balance_build_on_missing = bool(train_memory_balance_build_on_missing)
        self.train_memory_balance_agent_weight = float(train_memory_balance_agent_weight)
        self.train_memory_balance_current_valid_agent_weight = float(
            train_memory_balance_current_valid_agent_weight
        )
        self.train_memory_balance_valid_agent_step_weight = float(
            train_memory_balance_valid_agent_step_weight
        )
        self.train_memory_balance_map_weight = float(train_memory_balance_map_weight)
        self.train_memory_balance_seed = int(train_memory_balance_seed)
        self.train_anchor_steps = (
            tuple(int(step) for step in train_anchor_steps)
            if train_anchor_steps is not None
            else _DEFAULT_TRAIN_ANCHOR_STEPS
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in {"fit", None}:
            self.train_dataset = MDGDataset(
                raw_dir=self.train_raw_dir,
                max_agents=self.train_max_agents,
                max_map_polylines=self.max_map_polylines,
                map_waypoints=self.map_waypoints,
                max_traffic_lights=self.max_traffic_lights,
                training=True,
                train_anchor_steps=self.train_anchor_steps,
            )
            self.val_dataset = MDGDataset(
                raw_dir=self.val_raw_dir,
                max_agents=None,
                max_map_polylines=self.max_map_polylines,
                map_waypoints=self.map_waypoints,
                max_traffic_lights=self.max_traffic_lights,
                training=False,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "validate":
            self.val_dataset = MDGDataset(
                raw_dir=self.val_raw_dir,
                max_agents=None,
                max_map_polylines=self.max_map_polylines,
                map_waypoints=self.map_waypoints,
                max_traffic_lights=self.max_traffic_lights,
                training=False,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "test":
            self.test_dataset = MDGDataset(
                raw_dir=self.test_raw_dir,
                max_agents=None,
                max_map_polylines=self.max_map_polylines,
                map_waypoints=self.map_waypoints,
                max_traffic_lights=self.max_traffic_lights,
                training=False,
            )
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def _world_info(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        return int(getattr(trainer, "world_size", 1) or 1), int(getattr(trainer, "global_rank", 0) or 0)

    def _dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, exact_eval: bool = False) -> DataLoader:
        world_size, rank = self._world_info()
        sampler = None
        if world_size > 1:
            if exact_eval:
                sampler = ExactDistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=False,
                )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_mdg_samples,
        )

    def _build_memory_balanced_batch_sampler(self) -> MemoryBalancedDistributedBatchSampler:
        world_size, rank = self._world_info()
        metadata = load_or_build_memory_metadata(
            self.train_dataset.raw_paths,
            cache_path=self.train_memory_balance_metadata_cache,
            num_workers=self.train_memory_balance_metadata_num_workers,
            build_on_missing=self.train_memory_balance_build_on_missing,
        )
        sample_weight = memory_balance_weights(
            metadata,
            agent_weight=self.train_memory_balance_agent_weight,
            current_valid_agent_weight=self.train_memory_balance_current_valid_agent_weight,
            valid_agent_step_weight=self.train_memory_balance_valid_agent_step_weight,
            map_weight=self.train_memory_balance_map_weight,
        )
        return MemoryBalancedDistributedBatchSampler(
            sample_weight=sample_weight,
            batch_size=self.train_batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=self.shuffle,
            seed=self.train_memory_balance_seed,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_memory_balanced_batching:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._build_memory_balanced_batch_sampler(),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=collate_mdg_samples,
            )
        return self._dataloader(self.train_dataset, self.train_batch_size, self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, self.val_batch_size, False, exact_eval=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, self.test_batch_size, False, exact_eval=True)
