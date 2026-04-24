from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GodFMConfig:
    enabled: bool = False
    pair_dir: str = ""
    p_aug: float = 0.5
    goal_weight: float = 5.0
    inpaint_steps: int = 10
    n_rollout_collect: int = 4


def _r(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_godfm_config(godfm: Any) -> GodFMConfig:
    if godfm is None or godfm is False:
        return GodFMConfig(enabled=False)
    if godfm is True:
        return GodFMConfig(enabled=True)
    return GodFMConfig(
        enabled=bool(_r(godfm, "enabled", False)),
        pair_dir=str(_r(godfm, "pair_dir", "")),
        p_aug=float(_r(godfm, "p_aug", 0.5)),
        goal_weight=float(_r(godfm, "goal_weight", 5.0)),
        inpaint_steps=int(_r(godfm, "inpaint_steps", 10)),
        n_rollout_collect=int(_r(godfm, "n_rollout_collect", 4)),
    )
