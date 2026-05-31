from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _interp_polyline(points: Tensor, headings: Tensor, num_waypoints: int) -> tuple[Tensor, Tensor]:
    """Resample a polyline to a fixed number of waypoints by index interpolation."""
    n = int(points.shape[0])
    if n <= 0:
        return (
            torch.zeros(num_waypoints, 2, dtype=torch.float32),
            torch.zeros(num_waypoints, dtype=torch.float32),
        )
    if n == 1:
        return (
            points[0:1].repeat(num_waypoints, 1).float(),
            headings[0:1].repeat(num_waypoints).float(),
        )

    src = torch.linspace(0, n - 1, steps=num_waypoints, dtype=torch.float32)
    low = torch.floor(src).long()
    high = torch.clamp(low + 1, max=n - 1)
    weight = (src - low.float()).unsqueeze(-1)
    out_points = points[low] * (1.0 - weight) + points[high] * weight

    heading_low = headings[low]
    heading_high = headings[high]
    delta = (heading_high - heading_low + math.pi) % (2 * math.pi) - math.pi
    out_heading = heading_low + delta * weight.squeeze(-1)
    return out_points.float(), out_heading.float()


def _sample_polyline_from_points(points: Tensor, num_waypoints: int) -> tuple[Tensor, Tensor]:
    if points.shape[0] <= 1:
        heading = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    else:
        delta = points[1:] - points[:-1]
        heading = torch.atan2(delta[:, 1], delta[:, 0])
        heading = torch.cat((heading, heading[-1:]), dim=0)
    return _interp_polyline(points[:, :2], heading, num_waypoints)


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


def _pad_first_dim(value: Tensor, size: int, pad_value: float | int | bool = 0) -> tuple[Tensor, Tensor]:
    out_shape = (size,) + tuple(value.shape[1:])
    out = torch.full(out_shape, pad_value, dtype=value.dtype)
    valid = torch.zeros(size, dtype=torch.bool)
    n = min(size, int(value.shape[0]))
    if n > 0:
        out[:n] = value[:n].cpu()
        valid[:n] = True
    return out, valid


def _fallback_map_from_smart_cache(data: Dict[str, Any], num_waypoints: int) -> Dict[str, Tensor]:
    traj_pos = _as_tensor(data["map_save"]["traj_pos"], torch.float32)
    traj_theta = _as_tensor(data["map_save"]["traj_theta"], torch.float32)
    poly_pos = []
    poly_head = []
    for idx in range(int(traj_pos.shape[0])):
        pos_i, head_i = _interp_polyline(
            traj_pos[idx, :, :2].float(),
            traj_theta[idx : idx + 1].repeat(traj_pos.shape[1]).float(),
            num_waypoints,
        )
        poly_pos.append(pos_i)
        poly_head.append(head_i)

    return {
        "position": torch.stack(poly_pos, dim=0) if poly_pos else torch.zeros(0, num_waypoints, 2),
        "heading": torch.stack(poly_head, dim=0) if poly_head else torch.zeros(0, num_waypoints),
        "type": _as_tensor(data["pt_token"]["pl_type"], torch.long),
        "light_type": _as_tensor(data["pt_token"]["light_type"], torch.long),
        "valid": torch.ones(int(traj_pos.shape[0]), dtype=torch.bool),
    }


def _extract_map(data: Dict[str, Any], num_waypoints: int) -> Dict[str, Tensor]:
    if "mdg_map" in data:
        mdg_map = data["mdg_map"]
        return {
            "position": _as_tensor(mdg_map["position"], torch.float32),
            "heading": _as_tensor(mdg_map["heading"], torch.float32),
            "type": _as_tensor(mdg_map["type"], torch.long),
            "light_type": _as_tensor(mdg_map.get("light_type", torch.zeros(0)), torch.long),
            "valid": _as_tensor(mdg_map.get("valid", torch.ones(len(mdg_map["position"]))), torch.bool),
        }
    return _fallback_map_from_smart_cache(data, num_waypoints)


def _extract_traffic_signals(
    data: Dict[str, Any],
    map_position: Tensor,
    map_heading: Tensor,
    map_light_type: Tensor,
) -> Dict[str, Tensor]:
    if "mdg_traffic_signal" in data:
        signal = data["mdg_traffic_signal"]
        return {
            "position": _as_tensor(signal["position"], torch.float32),
            "heading": _as_tensor(signal.get("heading", torch.zeros(len(signal["position"]))), torch.float32),
            "state": _as_tensor(signal["state"], torch.long),
            "valid": _as_tensor(signal.get("valid", torch.ones(len(signal["position"]))), torch.bool),
        }

    signal_mask = map_light_type > 0
    if not bool(signal_mask.any()):
        return {
            "position": torch.zeros(0, 2, dtype=torch.float32),
            "heading": torch.zeros(0, dtype=torch.float32),
            "state": torch.zeros(0, dtype=torch.long),
            "valid": torch.zeros(0, dtype=torch.bool),
        }
    return {
        "position": map_position[signal_mask, 0, :2].float(),
        "heading": map_heading[signal_mask, 0].float(),
        "state": map_light_type[signal_mask].long(),
        "valid": torch.ones(int(signal_mask.sum()), dtype=torch.bool),
    }


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

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.raw_paths[index]
        with path.open("rb") as handle:
            data = pickle.load(handle)
        return self._build_sample(data)

    def _build_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        agent = data["agent"]
        position = _as_tensor(agent["position"], torch.float32)
        heading = _as_tensor(agent["heading"], torch.float32)
        velocity = _as_tensor(agent["velocity"], torch.float32)
        valid = _as_tensor(agent["valid_mask"], torch.bool)
        shape = _as_tensor(agent["shape"], torch.float32)
        agent_type = _as_tensor(agent["type"], torch.long).clamp(min=0, max=2)
        role = _as_tensor(agent["role"], torch.bool)
        agent_id = _as_tensor(agent["id"], torch.long)

        current_index = _HISTORY_STEPS - 1
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

        mdg_map = _extract_map(data, self.map_waypoints)
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
            mdg_map["light_type"][selected_map].clamp(min=0, max=8),
            self.max_map_polylines,
        )

        signal = _extract_traffic_signals(
            data=data,
            map_position=mdg_map["position"],
            map_heading=mdg_map["heading"],
            map_light_type=mdg_map["light_type"],
        )
        selected_signal = _select_nearest(
            positions=signal["position"],
            valid=signal["valid"],
            center=center,
            max_count=self.max_traffic_lights,
        )
        signal_position, signal_valid = _pad_first_dim(
            signal["position"][selected_signal],
            self.max_traffic_lights,
        )
        signal_heading, _ = _pad_first_dim(
            signal["heading"][selected_signal],
            self.max_traffic_lights,
        )
        signal_state, _ = _pad_first_dim(
            signal["state"][selected_signal].clamp(min=0, max=8),
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
        eval_max_agents: Optional[int] = None,
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
        if eval_max_agents is not None:
            warnings.warn(
                "data.eval_max_agents is deprecated and ignored. "
                "MDG eval/test/submission now keep all current-valid cached agents "
                "so Fast WOSAC sim_agent_ids are not truncated.",
                stacklevel=2,
            )
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

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in {"fit", None}:
            self.train_dataset = MDGDataset(
                raw_dir=self.train_raw_dir,
                max_agents=self.train_max_agents,
                max_map_polylines=self.max_map_polylines,
                map_waypoints=self.map_waypoints,
                max_traffic_lights=self.max_traffic_lights,
                training=True,
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
