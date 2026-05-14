from __future__ import annotations

import pickle

import numpy as np
import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    VEHICLE_TYPE_ID,
    decode_control_sequence,
)
from tools.estimate_control_no_slip_rho import (
    EstimatorConfig,
    _estimate_agent_rhos_for_type,
    estimate_file_agent_stats,
    weighted_median,
)


def _synthetic_agent_record(
    *,
    agent_type: int,
    rho: float,
    lengths: list[float],
    step_count: int = 40,
) -> dict[str, torch.Tensor]:
    num_agent = len(lengths)
    agent_type_tensor = torch.full((num_agent,), agent_type, dtype=torch.long)
    length = torch.tensor(lengths, dtype=torch.float32)
    current_pos = torch.zeros((num_agent, 2), dtype=torch.float32)
    current_head = torch.linspace(-0.6, 0.6, num_agent, dtype=torch.float32)

    control = torch.zeros((num_agent, step_count, 3), dtype=torch.float32)
    control[..., 0] = torch.linspace(0.55, 0.80, num_agent).unsqueeze(1)
    control[..., 2] = torch.linspace(0.04, 0.08, num_agent).unsqueeze(1)

    pos_future, head_future = decode_control_sequence(
        control=control,
        agent_type=agent_type_tensor,
        agent_length=length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=rho,
        cyclist_no_slip_point_ratio=rho,
    )

    position = torch.zeros((num_agent, step_count + 1, 3), dtype=torch.float32)
    position[:, 1:, :2] = pos_future
    heading = torch.zeros((num_agent, step_count + 1), dtype=torch.float32)
    heading[:, 0] = current_head
    heading[:, 1:] = head_future
    valid = torch.ones((num_agent, step_count + 1), dtype=torch.bool)
    shape = torch.zeros((num_agent, 3), dtype=torch.float32)
    shape[:, 0] = length
    shape[:, 1] = 2.0
    shape[:, 2] = 1.5

    return {
        "position": position,
        "heading": heading,
        "valid_mask": valid,
        "type": agent_type_tensor.to(torch.uint8),
        "shape": shape,
    }


def test_weighted_median_ignores_invalid_values() -> None:
    values = np.array([100.0, 0.2, np.nan, 0.3, 0.4])
    weights = np.array([0.0, 1.0, 1.0, 3.0, np.inf])

    assert weighted_median(values, weights) == 0.3


def test_agent_level_rho_recovers_known_no_slip_offset() -> None:
    rho = 0.27
    record = _synthetic_agent_record(agent_type=VEHICLE_TYPE_ID, rho=rho, lengths=[4.4, 4.7, 5.0])
    cfg = EstimatorConfig(min_agent_segments=5, min_agent_info_m=0.5)

    stats = _estimate_agent_rhos_for_type(
        pos=record["position"][..., :2].numpy(),
        heading=record["heading"].numpy(),
        valid=record["valid_mask"].numpy(),
        length=record["shape"][:, 0].numpy(),
        cfg=cfg,
    )

    assert stats["rho"].shape == (3,)
    assert np.allclose(stats["rho"], rho, atol=1.0e-4)


def test_file_level_estimator_reports_vehicle_and_cyclist_rhos(tmp_path) -> None:
    vehicle_rho = 0.18
    cyclist_rho = 0.34
    vehicle = _synthetic_agent_record(
        agent_type=VEHICLE_TYPE_ID,
        rho=vehicle_rho,
        lengths=[4.3, 4.9],
    )
    cyclist = _synthetic_agent_record(
        agent_type=CYCLIST_TYPE_ID,
        rho=cyclist_rho,
        lengths=[1.7, 2.0],
    )

    agent = {}
    for key in vehicle:
        agent[key] = torch.cat([vehicle[key], cyclist[key]], dim=0)
    path = tmp_path / "synthetic.pkl"
    with path.open("wb") as handle:
        pickle.dump({"agent": agent}, handle)

    stats = estimate_file_agent_stats(path, EstimatorConfig(min_agent_segments=5, min_agent_info_m=0.5))

    assert np.allclose(stats["vehicle"]["rho"], vehicle_rho, atol=1.0e-4)
    assert np.allclose(stats["cyclist"]["rho"], cyclist_rho, atol=1.0e-4)
    assert stats["vehicle"]["informative_segments"] > 0
    assert stats["cyclist"]["informative_segments"] > 0
