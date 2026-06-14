from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from src.smart.metrics.sim_agents_metrics import (
    _load_gt_scenario_for_device,
    _load_waymo_sim_agents_2025_config,
    _prediction_arrays_to_fast_bundle,
)
from src.smart.metrics.wosac_fast_eval_tool.fast_sim_agents_metrics import (
    metrics as fast_sim_agents_metrics,
)


@dataclass(frozen=True)
class RLFTSimRewardBatch:
    rewards: Tensor
    leave_one_out_rmm: Tensor
    full_rmm: Tensor


def compute_mloo_rewards_from_leave_one_out(leave_one_out_rmm: Tensor) -> Tensor:
    """Convert leave-one-out RMM values into per-rollout MLOO rewards."""
    if leave_one_out_rmm.ndim != 2:
        raise ValueError(
            "leave_one_out_rmm must have shape [n_scenario, n_rollout], "
            f"got {tuple(leave_one_out_rmm.shape)}."
        )
    if int(leave_one_out_rmm.shape[1]) < 2:
        raise ValueError("MLOO requires at least two rollouts.")
    return leave_one_out_rmm.mean(dim=1, keepdim=True) - leave_one_out_rmm


class RLFTSimMLOOReward:
    """Fast WOSAC/RMM based MLOO reward for goal-free RLFTSim fine-tuning."""

    def __init__(self, *, ego_only: bool = False, version: str = "2025") -> None:
        if version != "2025":
            raise ValueError(f"Only WOSAC 2025 fast RMM is supported, got {version!r}.")
        self.ego_only = bool(ego_only)
        self.version = version
        self.sim_agents_config = _load_waymo_sim_agents_2025_config()

    def _load_gt_scenario(self, *, scenario_file: str, device: torch.device) -> dict:
        if not Path(str(scenario_file)).is_file():
            raise FileNotFoundError(
                "RLFTSim reward requires split Waymo TFRecord files. "
                f"Missing scenario_file={scenario_file!r}."
            )
        return _load_gt_scenario_for_device(
            scenario_file=str(scenario_file),
            ego_only=self.ego_only,
            device=device,
        )

    def _compute_scenario_rmm(
        self,
        *,
        gt_scenario: dict,
        agent_id,
        pred_traj,
        pred_z,
        pred_head,
        device: torch.device,
    ) -> float:
        scenario_rollouts = _prediction_arrays_to_fast_bundle(
            agent_id=agent_id,
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            device=device,
        )
        result = fast_sim_agents_metrics.compute_scenario_metrics_for_bundle(
            config=self.sim_agents_config,
            gt_scenario=gt_scenario,
            scenario_rollouts=scenario_rollouts,
            version=self.version,
        )
        return float(result["metametric"])

    @torch.no_grad()
    def compute_from_prediction_tensors(
        self,
        *,
        scenario_files: Sequence[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> RLFTSimRewardBatch:
        if pred_traj.ndim != 4:
            raise ValueError(
                "pred_traj must have shape [n_agent, n_rollout, n_step, 2], "
                f"got {tuple(pred_traj.shape)}."
            )
        n_rollout = int(pred_traj.shape[1])
        if n_rollout < 2:
            raise ValueError(f"MLOO requires at least two rollouts, got {n_rollout}.")
        if len(scenario_files) == 0:
            raise ValueError("scenario_files must not be empty for RLFTSim reward.")

        device = pred_traj.device
        agent_batch_cpu = agent_batch.detach().to(device="cpu", dtype=torch.long)
        sizes = torch.bincount(
            agent_batch_cpu,
            minlength=len(scenario_files),
        ).tolist()

        leave_one_out_rows: list[list[float]] = []
        full_rmm_values: list[float] = []
        rollout_indices = torch.arange(n_rollout, device=device)
        start = 0
        for scenario_file, size in zip(scenario_files, sizes):
            end = start + int(size)
            scenario_agent_id = agent_id[start:end]
            scenario_pred_traj = pred_traj[start:end]
            scenario_pred_z = pred_z[start:end]
            scenario_pred_head = pred_head[start:end]
            gt_scenario = self._load_gt_scenario(
                scenario_file=str(scenario_file),
                device=device,
            )

            full_rmm_values.append(
                self._compute_scenario_rmm(
                    gt_scenario=gt_scenario,
                    agent_id=scenario_agent_id,
                    pred_traj=scenario_pred_traj,
                    pred_z=scenario_pred_z,
                    pred_head=scenario_pred_head,
                    device=device,
                )
            )

            scenario_leave_one_out: list[float] = []
            for held_out in range(n_rollout):
                keep = rollout_indices != held_out
                scenario_leave_one_out.append(
                    self._compute_scenario_rmm(
                        gt_scenario=gt_scenario,
                        agent_id=scenario_agent_id,
                        pred_traj=scenario_pred_traj[:, keep],
                        pred_z=scenario_pred_z[:, keep],
                        pred_head=scenario_pred_head[:, keep],
                        device=device,
                    )
                )
            leave_one_out_rows.append(scenario_leave_one_out)
            start = end

        leave_one_out_rmm = torch.tensor(
            leave_one_out_rows,
            dtype=torch.float32,
            device=device,
        )
        full_rmm = torch.tensor(full_rmm_values, dtype=torch.float32, device=device)
        rewards = compute_mloo_rewards_from_leave_one_out(leave_one_out_rmm)
        return RLFTSimRewardBatch(
            rewards=rewards,
            leave_one_out_rmm=leave_one_out_rmm,
            full_rmm=full_rmm,
        )
