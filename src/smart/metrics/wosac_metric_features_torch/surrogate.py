from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurrogateConfig:
    """Controls differentiable surrogates for discrete per-step events."""

    collision_temperature: float = 0.10
    offroad_temperature: float = 0.10
    red_light_crossing_temperature: float = 0.05

