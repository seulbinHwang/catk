# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric


_AGENT_TYPE_COUNT = 3
_DEFAULT_EPS = 1.0e-6


def _as_float_tensor(value: Tensor, *, device: torch.device) -> Tensor:
    """입력 텐서를 계산하기 좋은 실수 텐서로 바꿉니다.

    Args:
        value: 변환할 텐서입니다. shape은 호출 위치에서 유지됩니다.
        device: 결과 텐서를 올릴 장치입니다.

    Returns:
        Tensor: ``float32`` 타입으로 바뀐 텐서입니다. shape은 입력과 같습니다.
    """
    return value.detach().to(device=device, dtype=torch.float32)


def _as_bool_tensor(value: Tensor, *, device: torch.device) -> Tensor:
    """입력 텐서를 참/거짓 텐서로 바꿉니다.

    Args:
        value: 변환할 텐서입니다. shape은 호출 위치에서 유지됩니다.
        device: 결과 텐서를 올릴 장치입니다.

    Returns:
        Tensor: ``bool`` 타입으로 바뀐 텐서입니다. shape은 입력과 같습니다.
    """
    return value.detach().to(device=device, dtype=torch.bool)


def _as_long_tensor(value: Tensor, *, device: torch.device) -> Tensor:
    """입력 텐서를 정수 텐서로 바꿉니다.

    Args:
        value: 변환할 텐서입니다. shape은 호출 위치에서 유지됩니다.
        device: 결과 텐서를 올릴 장치입니다.

    Returns:
        Tensor: ``long`` 타입으로 바뀐 텐서입니다. shape은 입력과 같습니다.
    """
    return value.detach().to(device=device, dtype=torch.long)


def _safe_cat_state(
    value: Tensor | Iterable[Tensor],
    *,
    device: torch.device,
    empty_shape: tuple[int, ...],
) -> Tensor:
    """TorchMetrics list state를 하나의 텐서로 합칩니다.

    Args:
        value: 텐서이거나 텐서 목록입니다. 각 텐서는 첫 번째 차원으로 합쳐집니다.
        device: 결과 텐서를 둘 장치입니다.
        empty_shape: 저장된 값이 없을 때 만들 빈 텐서 shape입니다.

    Returns:
        Tensor: 첫 번째 차원으로 합쳐진 텐서입니다.
    """
    if isinstance(value, Tensor):
        return value.to(device=device)

    tensors = [tensor.to(device=device) for tensor in value if isinstance(tensor, Tensor)]
    if not tensors:
        return torch.zeros(empty_shape, dtype=torch.float64, device=device)
    return torch.cat(tensors, dim=0)


def _sum_square_by_type(
    square_distance: Tensor,
    valid_mask: Tensor,
    agent_type: Tensor,
    *,
    num_agent_types: int,
) -> Tensor:
    """agent 종류별 거리 제곱합을 계산합니다.

    Args:
        square_distance: 각 agent와 미래 시점별 거리 제곱입니다.
            shape은 ``[n_agent, n_step]`` 입니다.
        valid_mask: 계산에 넣을 위치인지 나타냅니다.
            shape은 ``[n_agent, n_step]`` 입니다.
        agent_type: agent 종류입니다. 0은 차량, 1은 보행자, 2는 자전거입니다.
            shape은 ``[n_agent]`` 입니다.
        num_agent_types: 지원하는 agent 종류 개수입니다.

    Returns:
        Tensor: 종류별 거리 제곱합입니다. shape은 ``[num_agent_types]`` 입니다.
    """
    weighted_square = torch.where(
        valid_mask,
        square_distance,
        torch.zeros_like(square_distance),
    )
    sums = []
    for type_index in range(num_agent_types):
        type_mask = agent_type == type_index
        if bool(type_mask.any()):
            sums.append(weighted_square[type_mask].sum(dtype=torch.float64))
        else:
            sums.append(torch.zeros((), dtype=torch.float64, device=square_distance.device))
    return torch.stack(sums, dim=0)


def _trim_prediction_and_gt(
    pred_traj: Tensor,
    gt_traj: Optional[Tensor],
    gt_valid_mask: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """예측과 정답의 미래 길이를 서로 맞춥니다.

    Args:
        pred_traj: 모델이 만든 미래 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step_pred, 2]`` 입니다.
        gt_traj: 정답 미래 위치입니다. 값이 있으면 shape은
            ``[n_agent, n_step_gt, 2]`` 입니다.
        gt_valid_mask: 정답 미래 valid mask입니다. 값이 있으면 shape은
            ``[n_agent, n_step_gt]`` 입니다.

    Returns:
        tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            미래 길이를 맞춘 예측, 정답, valid mask입니다.
    """
    if gt_traj is None or gt_valid_mask is None:
        return pred_traj, gt_traj, gt_valid_mask

    n_step = min(int(pred_traj.shape[2]), int(gt_traj.shape[1]), int(gt_valid_mask.shape[1]))
    return pred_traj[:, :, :n_step], gt_traj[:, :n_step], gt_valid_mask[:, :n_step]


def _compute_pair_square_summaries(
    pred_traj: Tensor,
    valid_mask: Tensor,
    agent_type: Tensor,
    *,
    num_agent_types: int,
) -> tuple[Tensor, Tensor]:
    """같은 scenario 안 rollout 쌍별 거리 제곱합을 저장합니다.

    Args:
        pred_traj: 한 scenario의 예측 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
        valid_mask: 계산에 넣을 agent와 시점입니다.
            shape은 ``[n_agent, n_step]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
        num_agent_types: 지원하는 agent 종류 개수입니다.

    Returns:
        tuple[Tensor, Tensor]:
            첫 번째 텐서는 rollout 쌍별, agent 종류별 거리 제곱합입니다.
            shape은 ``[n_pair, num_agent_types]`` 입니다.
            두 번째 텐서는 rollout 쌍별 valid 개수입니다. shape은 ``[n_pair]`` 입니다.
    """
    n_rollout = int(pred_traj.shape[1])
    n_pair = n_rollout * max(n_rollout - 1, 0) // 2
    pair_sq_by_type = torch.zeros(
        (n_pair, num_agent_types),
        dtype=torch.float64,
        device=pred_traj.device,
    )
    pair_count = torch.zeros((n_pair,), dtype=torch.float64, device=pred_traj.device)

    if n_pair == 0 or int(pred_traj.shape[0]) == 0 or not bool(valid_mask.any()):
        return pair_sq_by_type, pair_count

    valid_count = valid_mask.sum(dtype=torch.float64)
    pair_index = 0
    for first_rollout in range(n_rollout - 1):
        first_traj = pred_traj[:, first_rollout]
        for second_rollout in range(first_rollout + 1, n_rollout):
            diff = first_traj - pred_traj[:, second_rollout]
            square_distance = diff.square().sum(dim=-1)
            pair_sq_by_type[pair_index] = _sum_square_by_type(
                square_distance=square_distance,
                valid_mask=valid_mask,
                agent_type=agent_type,
                num_agent_types=num_agent_types,
            )
            pair_count[pair_index] = valid_count
            pair_index += 1

    return pair_sq_by_type, pair_count


def _compute_gt_square_summaries(
    pred_traj: Tensor,
    gt_traj: Tensor,
    gt_valid_mask: Tensor,
    agent_type: Tensor,
    *,
    num_agent_types: int,
) -> tuple[Tensor, Tensor]:
    """rollout과 정답 사이의 거리 제곱합을 저장합니다.

    Args:
        pred_traj: 한 scenario의 예측 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
        gt_traj: 한 scenario의 정답 미래 위치입니다.
            shape은 ``[n_agent, n_step, 2]`` 입니다.
        gt_valid_mask: 계산에 넣을 정답 위치입니다.
            shape은 ``[n_agent, n_step]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
        num_agent_types: 지원하는 agent 종류 개수입니다.

    Returns:
        tuple[Tensor, Tensor]:
            첫 번째 텐서는 rollout별, agent 종류별 거리 제곱합입니다.
            shape은 ``[n_rollout, num_agent_types]`` 입니다.
            두 번째 텐서는 rollout별 valid 개수입니다. shape은 ``[n_rollout]`` 입니다.
    """
    n_rollout = int(pred_traj.shape[1])
    gt_sq_by_type = torch.zeros(
        (n_rollout, num_agent_types),
        dtype=torch.float64,
        device=pred_traj.device,
    )
    gt_count = torch.zeros((n_rollout,), dtype=torch.float64, device=pred_traj.device)

    if int(pred_traj.shape[0]) == 0 or not bool(gt_valid_mask.any()):
        return gt_sq_by_type, gt_count

    valid_count = gt_valid_mask.sum(dtype=torch.float64)
    for rollout_index in range(n_rollout):
        diff = pred_traj[:, rollout_index] - gt_traj
        square_distance = diff.square().sum(dim=-1)
        gt_sq_by_type[rollout_index] = _sum_square_by_type(
            square_distance=square_distance,
            valid_mask=gt_valid_mask,
            agent_type=agent_type,
            num_agent_types=num_agent_types,
        )
        gt_count[rollout_index] = valid_count

    return gt_sq_by_type, gt_count


def _scenario_index_values(agent_batch: Tensor) -> Tensor:
    """batch 안에 들어 있는 scenario 번호를 정렬해서 돌려줍니다.

    Args:
        agent_batch: 각 agent가 속한 scenario 번호입니다. shape은 ``[n_agent]`` 입니다.

    Returns:
        Tensor: 중복을 제거한 scenario 번호입니다. shape은 ``[n_scenario]`` 입니다.
    """
    if agent_batch.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=agent_batch.device)
    return torch.unique(agent_batch, sorted=True)


class WOSACDistributionMetrics(Metric):
    """WOSAC closed-loop rollout의 CPD와 CES를 누적 계산합니다."""

    full_state_update = False

    def __init__(
        self,
        prefix: str,
        cpd_reference: Optional[float] = None,
        type_scale: Optional[Sequence[float]] = None,
        eps: float = _DEFAULT_EPS,
        num_agent_types: int = _AGENT_TYPE_COUNT,
    ) -> None:
        """metric 누적 상태를 만듭니다.

        Args:
            prefix: W&B와 Lightning log에서 사용할 앞부분 이름입니다.
            cpd_reference: CPD 보존율을 계산할 기준 CPD입니다. 값이 없으면
                CPD 보존율은 기록하지 않습니다.
            type_scale: agent 종류별 CPD/CES 정규화 scale입니다. 값이 있으면
                validation/test 모두 이 고정 scale을 우선 사용하고, 값이 없을 때만
                기존처럼 validation GT에서 scale을 계산합니다.
            eps: 0으로 나누는 일을 막기 위한 작은 값입니다.
            num_agent_types: agent 종류 개수입니다. WOMD 전처리 기준 기본값은 3입니다.
        """
        super().__init__(sync_on_compute=True)
        if num_agent_types <= 0:
            raise ValueError(f"num_agent_types must be positive, got {num_agent_types}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.prefix = str(prefix).rstrip("/")
        self.cpd_reference = None if cpd_reference is None else float(cpd_reference)
        self.eps = float(eps)
        self.num_agent_types = int(num_agent_types)
        self._fixed_type_scale = self._parse_fixed_type_scale(type_scale)

        self.add_state(
            "scale_sq_sum",
            default=torch.zeros(self.num_agent_types, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "scale_count",
            default=torch.zeros(self.num_agent_types, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("pair_sq_by_type", default=[], dist_reduce_fx="cat")
        self.add_state("pair_count", default=[], dist_reduce_fx="cat")
        self.add_state("gt_sq_by_type", default=[], dist_reduce_fx="cat")
        self.add_state("gt_count", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pred_traj: Tensor,
        agent_type: Tensor,
        agent_batch: Tensor,
        current_pos: Optional[Tensor] = None,
        gt_traj: Optional[Tensor] = None,
        gt_valid_mask: Optional[Tensor] = None,
        agent_valid_mask: Optional[Tensor] = None,
    ) -> None:
        """한 validation 또는 test batch의 rollout을 누적합니다.

        Args:
            pred_traj: 모델이 만든 closed-loop 미래 위치입니다.
                shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
            agent_type: agent 종류입니다. 0은 차량, 1은 보행자, 2는 자전거입니다.
                shape은 ``[n_agent]`` 입니다.
            agent_batch: 각 agent가 속한 scenario 번호입니다. shape은 ``[n_agent]`` 입니다.
            current_pos: simulation 시작 위치입니다. 값이 있으면 shape은 ``[n_agent, 2]`` 입니다.
            gt_traj: 정답 미래 위치입니다. validation에서만 의미가 있으며 shape은
                ``[n_agent, n_step, 2]`` 입니다.
            gt_valid_mask: 정답 미래 valid mask입니다. 값이 있으면 shape은
                ``[n_agent, n_step]`` 입니다.
            agent_valid_mask: simulation 시작 시점에 유효한 agent인지 나타냅니다.
                값이 있으면 shape은 ``[n_agent]`` 입니다.
        """
        if pred_traj.ndim != 4 or int(pred_traj.shape[-1]) != 2:
            raise ValueError(
                "pred_traj must have shape [n_agent, n_rollout, n_step, 2], "
                f"got {tuple(pred_traj.shape)}."
            )
        if pred_traj.shape[0] == 0:
            return

        device = self.scale_sq_sum.device
        pred_traj = _as_float_tensor(pred_traj, device=device)
        agent_type = _as_long_tensor(agent_type, device=device).clamp(0, self.num_agent_types - 1)
        agent_batch = _as_long_tensor(agent_batch, device=device)

        if agent_type.shape[0] != pred_traj.shape[0] or agent_batch.shape[0] != pred_traj.shape[0]:
            raise ValueError(
                "agent_type and agent_batch must have shape [n_agent] matching pred_traj. "
                f"got pred={tuple(pred_traj.shape)}, type={tuple(agent_type.shape)}, "
                f"batch={tuple(agent_batch.shape)}."
            )

        current_pos = (
            _as_float_tensor(current_pos, device=device) if current_pos is not None else None
        )
        gt_traj = _as_float_tensor(gt_traj, device=device) if gt_traj is not None else None
        gt_valid_mask = (
            _as_bool_tensor(gt_valid_mask, device=device)
            if gt_valid_mask is not None
            else None
        )
        agent_valid_mask = (
            _as_bool_tensor(agent_valid_mask, device=device)
            if agent_valid_mask is not None
            else torch.ones(pred_traj.shape[0], dtype=torch.bool, device=device)
        )

        pred_traj, gt_traj, gt_valid_mask = _trim_prediction_and_gt(
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            gt_valid_mask=gt_valid_mask,
        )
        if pred_traj.shape[2] == 0:
            return

        if (
            self._fixed_type_scale is None
            and gt_traj is not None
            and gt_valid_mask is not None
            and current_pos is not None
        ):
            self._update_type_scale(
                gt_traj=gt_traj,
                gt_valid_mask=gt_valid_mask,
                current_pos=current_pos,
                agent_type=agent_type,
                agent_valid_mask=agent_valid_mask,
            )

        pair_sq_chunks = []
        pair_count_chunks = []
        gt_sq_chunks = []
        gt_count_chunks = []

        for scenario_index in _scenario_index_values(agent_batch):
            scenario_mask = (agent_batch == scenario_index) & agent_valid_mask
            if not bool(scenario_mask.any()):
                continue

            scenario_pred = pred_traj[scenario_mask]
            scenario_type = agent_type[scenario_mask]

            if gt_valid_mask is not None and bool(gt_valid_mask[scenario_mask].any()):
                pair_valid_mask = gt_valid_mask[scenario_mask]
            else:
                pair_valid_mask = torch.ones(
                    scenario_pred.shape[0],
                    scenario_pred.shape[2],
                    dtype=torch.bool,
                    device=device,
                )

            pair_sq_by_type, pair_count = _compute_pair_square_summaries(
                pred_traj=scenario_pred,
                valid_mask=pair_valid_mask,
                agent_type=scenario_type,
                num_agent_types=self.num_agent_types,
            )
            if pair_sq_by_type.shape[0] > 0:
                pair_sq_chunks.append(pair_sq_by_type.unsqueeze(0))
                pair_count_chunks.append(pair_count.unsqueeze(0))

            if gt_traj is not None and gt_valid_mask is not None:
                scenario_gt_valid = gt_valid_mask[scenario_mask]
                if bool(scenario_gt_valid.any()):
                    gt_sq_by_type, gt_count = _compute_gt_square_summaries(
                        pred_traj=scenario_pred,
                        gt_traj=gt_traj[scenario_mask],
                        gt_valid_mask=scenario_gt_valid,
                        agent_type=scenario_type,
                        num_agent_types=self.num_agent_types,
                    )
                    gt_sq_chunks.append(gt_sq_by_type.unsqueeze(0))
                    gt_count_chunks.append(gt_count.unsqueeze(0))

        if pair_sq_chunks:
            self.pair_sq_by_type.append(torch.cat(pair_sq_chunks, dim=0))
            self.pair_count.append(torch.cat(pair_count_chunks, dim=0))
        if gt_sq_chunks:
            self.gt_sq_by_type.append(torch.cat(gt_sq_chunks, dim=0))
            self.gt_count.append(torch.cat(gt_count_chunks, dim=0))

    def _update_type_scale(
        self,
        gt_traj: Tensor,
        gt_valid_mask: Tensor,
        current_pos: Tensor,
        agent_type: Tensor,
        agent_valid_mask: Tensor,
    ) -> None:
        """validation GT로 agent 종류별 이동 scale을 누적합니다.

        Args:
            gt_traj: 정답 미래 위치입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            gt_valid_mask: 정답 미래 valid mask입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: simulation 시작 위치입니다. shape은 ``[n_agent, 2]`` 입니다.
            agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_valid_mask: 시작 시점 valid mask입니다. shape은 ``[n_agent]`` 입니다.
        """
        valid_mask = gt_valid_mask & agent_valid_mask[:, None]
        if not bool(valid_mask.any()):
            return

        displacement = gt_traj - current_pos[:, None, :]
        square_distance = displacement.square().sum(dim=-1)
        for type_index in range(self.num_agent_types):
            type_valid = valid_mask & (agent_type[:, None] == type_index)
            if bool(type_valid.any()):
                self.scale_sq_sum[type_index] += square_distance[type_valid].sum(
                    dtype=torch.float64
                )
                self.scale_count[type_index] += type_valid.sum(dtype=torch.float64)

    def _compute_type_scale(self) -> Tensor:
        """누적된 GT 이동량으로 agent 종류별 scale을 계산합니다.

        Returns:
            Tensor: agent 종류별 scale입니다. shape은 ``[num_agent_types]`` 입니다.
        """
        if self._fixed_type_scale is not None:
            return self._fixed_type_scale.to(
                device=self.scale_sq_sum.device,
                dtype=torch.float64,
            ).clamp_min(self.eps)

        scale = torch.ones_like(self.scale_sq_sum, dtype=torch.float64)
        valid_type = self.scale_count > 0
        if bool(valid_type.any()):
            scale[valid_type] = torch.sqrt(
                self.scale_sq_sum[valid_type] / self.scale_count[valid_type].clamp_min(self.eps)
            )
        return scale.clamp_min(self.eps)

    def _parse_fixed_type_scale(
        self,
        type_scale: Optional[Sequence[float]],
    ) -> Optional[Tensor]:
        """config에 지정된 agent type scale을 검증하고 텐서로 보관합니다."""
        if type_scale is None:
            return None
        values = [float(value) for value in type_scale]
        if len(values) != self.num_agent_types:
            raise ValueError(
                "type_scale length must match num_agent_types: "
                f"got {len(values)} values for {self.num_agent_types} types."
            )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(f"type_scale values must be finite positive numbers, got {values}.")
        return torch.tensor(values, dtype=torch.float64)

    def compute(self) -> Dict[str, Tensor]:
        """누적된 WOSAC-CPD와 WOSAC-CES를 계산합니다.

        Returns:
            Dict[str, Tensor]: Lightning/W&B에 넘길 스칼라 metric 사전입니다.
        """
        device = self.scale_sq_sum.device
        pair_sq_by_type = _safe_cat_state(
            self.pair_sq_by_type,
            device=device,
            empty_shape=(0, 0, self.num_agent_types),
        )
        pair_count = _safe_cat_state(
            self.pair_count,
            device=device,
            empty_shape=(0, 0),
        )
        if pair_sq_by_type.numel() == 0 or pair_count.numel() == 0:
            return {}

        type_scale = self._compute_type_scale()
        inv_scale_square = type_scale.reciprocal().square()
        pair_distance = torch.sqrt(
            (pair_sq_by_type * inv_scale_square[None, None, :]).sum(dim=-1)
            / pair_count.clamp_min(self.eps)
        )
        pair_valid = pair_count > 0
        scenario_pair_valid = pair_valid.any(dim=1)
        if not bool(scenario_pair_valid.any()):
            return {}

        scenario_pair_sum = torch.where(
            pair_valid,
            pair_distance,
            torch.zeros_like(pair_distance),
        ).sum(dim=1)
        scenario_pair_count = pair_valid.sum(dim=1).to(dtype=torch.float64)
        scenario_cpd = scenario_pair_sum / scenario_pair_count.clamp_min(1.0)
        cpd = scenario_cpd[scenario_pair_valid].mean()

        metric_dict: Dict[str, Tensor] = {
            f"{self.prefix}/WOSAC-CPD/value": cpd.to(dtype=torch.float32),
        }
        if self.cpd_reference is not None and self.cpd_reference > 0.0:
            metric_dict[f"{self.prefix}/WOSAC-CPD/DPR"] = (
                cpd / float(self.cpd_reference)
            ).to(dtype=torch.float32)

        gt_sq_by_type = _safe_cat_state(
            self.gt_sq_by_type,
            device=device,
            empty_shape=(0, 0, self.num_agent_types),
        )
        gt_count = _safe_cat_state(
            self.gt_count,
            device=device,
            empty_shape=(0, 0),
        )
        if gt_sq_by_type.numel() > 0 and gt_count.numel() > 0:
            n_ces_scenario = min(int(gt_sq_by_type.shape[0]), int(pair_distance.shape[0]))
            if n_ces_scenario > 0:
                ces = self._compute_ces(
                    gt_sq_by_type=gt_sq_by_type[:n_ces_scenario],
                    gt_count=gt_count[:n_ces_scenario],
                    pair_distance=pair_distance[:n_ces_scenario],
                    pair_valid=pair_valid[:n_ces_scenario],
                    inv_scale_square=inv_scale_square,
                )
                if ces is not None:
                    metric_dict[f"{self.prefix}/WOSAC-CES/value"] = ces.to(dtype=torch.float32)

        return metric_dict

    def _compute_ces(
        self,
        gt_sq_by_type: Tensor,
        gt_count: Tensor,
        pair_distance: Tensor,
        pair_valid: Tensor,
        inv_scale_square: Tensor,
    ) -> Optional[Tensor]:
        """GT가 있는 scenario에 대해 WOSAC-CES를 계산합니다.

        Args:
            gt_sq_by_type: rollout과 GT 사이의 종류별 거리 제곱합입니다.
                shape은 ``[n_scenario, n_rollout, num_agent_types]`` 입니다.
            gt_count: rollout과 GT 사이의 valid 개수입니다.
                shape은 ``[n_scenario, n_rollout]`` 입니다.
            pair_distance: rollout 쌍 거리입니다. shape은 ``[n_scenario, n_pair]`` 입니다.
            pair_valid: rollout 쌍 거리 valid mask입니다. shape은 ``[n_scenario, n_pair]`` 입니다.
            inv_scale_square: agent 종류별 scale의 역제곱입니다.
                shape은 ``[num_agent_types]`` 입니다.

        Returns:
            Optional[Tensor]: CES 평균입니다. 계산할 수 없으면 ``None`` 입니다.
        """
        gt_distance = torch.sqrt(
            (gt_sq_by_type * inv_scale_square[None, None, :]).sum(dim=-1)
            / gt_count.clamp_min(self.eps)
        )
        gt_valid = gt_count > 0
        scenario_gt_valid = gt_valid.any(dim=1)
        scenario_pair_valid = pair_valid.any(dim=1)
        scenario_valid = scenario_gt_valid & scenario_pair_valid
        if not bool(scenario_valid.any()):
            return None

        gt_term = torch.where(gt_valid, gt_distance, torch.zeros_like(gt_distance)).sum(dim=1)
        gt_term = gt_term / gt_valid.sum(dim=1).to(dtype=torch.float64).clamp_min(1.0)

        n_rollout = int(gt_sq_by_type.shape[1])
        pair_sum = torch.where(
            pair_valid,
            pair_distance,
            torch.zeros_like(pair_distance),
        ).sum(dim=1)
        pair_term = pair_sum / float(max(n_rollout * n_rollout, 1))
        scenario_ces = gt_term - pair_term
        return scenario_ces[scenario_valid].mean()


def _get_agent_store(data: Any) -> Any:
    """HeteroData에서 agent 저장소를 가져옵니다.

    Args:
        data: PyG ``HeteroData`` batch입니다.

    Returns:
        Any: ``data["agent"]`` 저장소입니다.
    """
    return data["agent"]


def _get_agent_field(agent_store: Any, key: str) -> Tensor:
    """agent 저장소에서 필요한 텐서를 읽습니다.

    Args:
        agent_store: PyG agent 저장소입니다.
        key: 가져올 필드 이름입니다.

    Returns:
        Tensor: 요청한 필드 텐서입니다.
    """
    try:
        return agent_store[key]
    except (KeyError, TypeError):
        return getattr(agent_store, key)


def _get_optional_agent_batch(agent_store: Any, *, n_agent: int, device: torch.device) -> Tensor:
    """agent batch 텐서를 가져오고 없으면 단일 scenario로 처리합니다.

    Args:
        agent_store: PyG agent 저장소입니다.
        n_agent: agent 개수입니다.
        device: 결과 텐서를 둘 장치입니다.

    Returns:
        Tensor: 각 agent의 scenario 번호입니다. shape은 ``[n_agent]`` 입니다.
    """
    try:
        return _get_agent_field(agent_store, "batch")
    except AttributeError:
        return torch.zeros(n_agent, dtype=torch.long, device=device)


def update_wosac_distribution_metric_from_batch(
    metric: WOSACDistributionMetrics,
    data: Any,
    pred_traj: Tensor,
    num_historical_steps: int,
    include_gt: bool = True,
) -> None:
    """PyG batch와 closed-loop rollout으로 WOSAC 분포 metric을 갱신합니다.

    Args:
        metric: 갱신할 WOSAC 분포 metric입니다.
        data: validation 또는 test batch입니다.
        pred_traj: 모델이 만든 closed-loop 미래 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
        num_historical_steps: history step 개수입니다. WOMD 기본값은 11입니다.
        include_gt: ``True``이면 validation GT로 CES와 type scale을 계산합니다.
            ``False``이면 test처럼 CPD만 계산합니다.
    """
    agent_store = _get_agent_store(data)
    n_agent = int(pred_traj.shape[0])
    device = pred_traj.device

    position = _get_agent_field(agent_store, "position")
    valid_mask = _get_agent_field(agent_store, "valid_mask")
    agent_type = _get_agent_field(agent_store, "type")
    agent_batch = _get_optional_agent_batch(agent_store, n_agent=n_agent, device=device)

    current_index = max(0, int(num_historical_steps) - 1)
    future_start = int(num_historical_steps)
    n_step = int(pred_traj.shape[2])
    n_available_future = max(0, int(position.shape[1]) - future_start)
    n_gt_step = min(n_step, n_available_future)

    current_pos = position[:, current_index, :2]
    agent_valid_mask = valid_mask[:, current_index] if valid_mask.ndim == 2 else None
    if include_gt and n_gt_step > 0:
        gt_traj = position[:, future_start : future_start + n_gt_step, :2]
        gt_valid = valid_mask[:, future_start : future_start + n_gt_step]
    else:
        gt_traj = None
        gt_valid = None

    metric.update(
        pred_traj=pred_traj,
        agent_type=agent_type,
        agent_batch=agent_batch,
        current_pos=current_pos,
        gt_traj=gt_traj,
        gt_valid_mask=gt_valid,
        agent_valid_mask=agent_valid_mask,
    )


def _trainer_is_testing(model: Any) -> bool:
    """현재 Lightning loop가 test인지 확인합니다.

    Args:
        model: LightningModule 객체입니다.

    Returns:
        bool: test loop이면 ``True`` 입니다.
    """
    trainer = getattr(model, "trainer", None)
    state = getattr(trainer, "state", None)
    state_fn = str(getattr(state, "fn", "")).lower()
    if "test" in state_fn:
        return True
    return bool(getattr(trainer, "testing", False))


def _prediction_update_key(
    metric: WOSACDistributionMetrics,
    data: Any,
    pred_traj: Tensor,
) -> tuple[int, int, int]:
    """같은 rollout을 두 번 누적하지 않기 위한 키를 만듭니다.

    Args:
        metric: 갱신 대상 metric입니다.
        data: validation 또는 test batch 객체입니다.
        pred_traj: closed-loop rollout 위치 텐서입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.

    Returns:
        tuple[int, int, int]: metric, batch, 예측 텐서의 식별자입니다.
    """
    data_ptr = int(pred_traj.data_ptr()) if pred_traj.numel() > 0 else 0
    return (id(metric), id(data), data_ptr)


def update_wosac_distribution_metric_from_model(
    model: Any,
    data: Any,
    pred_traj: Tensor,
) -> None:
    """현재 Lightning loop에 맞는 metric을 골라 WOSAC 분포 metric을 갱신합니다.

    Args:
        model: ``SMARTFlow`` 같은 LightningModule 객체입니다.
        data: validation 또는 test batch입니다.
        pred_traj: 모델이 만든 closed-loop 미래 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
    """
    if pred_traj.ndim != 4 or int(pred_traj.shape[-1]) != 2:
        return

    is_testing = _trainer_is_testing(model)
    metric_name = (
        "test_wosac_distribution_metrics"
        if is_testing
        else "wosac_distribution_metrics"
    )
    metric = getattr(model, metric_name, None)
    if metric is None:
        return

    update_key = _prediction_update_key(metric, data, pred_traj)
    if getattr(model, "_last_wosac_distribution_update_key", None) == update_key:
        return
    setattr(model, "_last_wosac_distribution_update_key", update_key)

    update_wosac_distribution_metric_from_batch(
        metric=metric,
        data=data,
        pred_traj=pred_traj,
        num_historical_steps=int(getattr(model, "num_historical_steps")),
        include_gt=not is_testing,
    )


def log_and_reset_wosac_distribution_metric(
    model: Any,
    metric: WOSACDistributionMetrics,
) -> None:
    """누적된 WOSAC 분포 metric을 log에 남기고 상태를 초기화합니다.

    Args:
        model: LightningModule 객체입니다.
        metric: 계산과 초기화를 수행할 metric입니다.
    """
    metric_dict = metric.compute()
    if metric_dict:
        model.log_dict(
            metric_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
    metric.reset()
