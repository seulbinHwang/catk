from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DynamicLimitTable:
    """Closed-loop commit bridge에서 쓰는 agent type별 동역학 제한값입니다."""

    v_max_mps: Tuple[float, float, float]
    a_max_mps2: Tuple[float, float, float]
    a_lat_max_mps2: Tuple[float, float, float]
    alpha_max_radps2: Tuple[float, float, float] = (1.75, 14.0, 6.0)
    r_min_m: Tuple[float, float, float] = (4.50, 1.0e-5, 0.5)
    omega_max_abs_radps: Tuple[float, float, float] = (0.9, 3.3, 2.0)
    beta_max_rad: Tuple[float, float, float] = (0.27, 0.0, 0.70)


DEFAULT_LIMITS = DynamicLimitTable(
    v_max_mps=(35.0, 5.0, 22.0),
    a_max_mps2=(8.0, 4.7, 5.5),
    a_lat_max_mps2=(4.2, 0.0, 4.4),
    alpha_max_radps2=(1.75, 14.0, 6.0),
    r_min_m=(4.50, 1.0e-5, 0.5),
    omega_max_abs_radps=(0.9, 3.3, 2.0),
    beta_max_rad=(0.27, 0.0, 0.70),
)
