from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass(frozen=True)
class MetricFeaturesTorch:
    """Torch version of Waymo `MetricFeatures`.

    Shape contract (SIM_AGENTS):
    - `object_id`: (n_objects,) int64/32
    - `object_type`: (n_samples, n_objects) int
    - `valid`: (n_samples, n_objects, n_steps) bool
    - time-series floats: (n_samples, n_objects, n_steps) float
    - per-object floats: (n_samples, n_objects) float
    - per-step booleans: (n_samples, n_objects, n_steps) bool

    For log features, `n_samples=1`. For simulation rollouts, `n_samples=G`.
    """

    object_id: Tensor
    object_type: Tensor
    valid: Tensor
    average_displacement_error: Tensor
    linear_speed: Tensor
    linear_acceleration: Tensor
    angular_speed: Tensor
    angular_acceleration: Tensor
    distance_to_nearest_object: Tensor
    collision_per_step: Tensor
    time_to_collision: Tensor
    distance_to_road_edge: Tensor
    offroad_per_step: Tensor
    traffic_light_violation_per_step: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        # Keep key names identical to TF MetricFeatures.
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "valid": self.valid,
            "average_displacement_error": self.average_displacement_error,
            "linear_speed": self.linear_speed,
            "linear_acceleration": self.linear_acceleration,
            "angular_speed": self.angular_speed,
            "angular_acceleration": self.angular_acceleration,
            "distance_to_nearest_object": self.distance_to_nearest_object,
            "collision_per_step": self.collision_per_step,
            "time_to_collision": self.time_to_collision,
            "distance_to_road_edge": self.distance_to_road_edge,
            "offroad_per_step": self.offroad_per_step,
            "traffic_light_violation_per_step": self.traffic_light_violation_per_step,
        }


def ensure_same_device(*tensors: Tensor) -> torch.device:
    dev = tensors[0].device
    for t in tensors[1:]:
        if t.device != dev:
            raise ValueError(f"device mismatch: {dev} vs {t.device}")
    return dev

