from __future__ import annotations

from typing import Dict

from torch import Tensor


def build_agent_type_masks(agent_type: Tensor) -> Dict[str, Tensor]:
    return {
        "veh": agent_type == 0,
        "ped": agent_type == 1,
        "cyc": agent_type == 2,
    }
