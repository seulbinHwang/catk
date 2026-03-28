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

from src.smart.metrics.cross_entropy import CrossEntropy
from src.smart.metrics.ego_nll import EgoNLL
from src.smart.metrics.gmm_ade import GMMADE
from src.smart.metrics.min_ade import minADE
from src.smart.metrics.next_token_cls import TokenCls
from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_submission import WOSACSubmission

# NOTE:
# `SMARTFlow`는 SimAgents 기반 metric/submission 인터페이스를 기대하지만,
# 현재 코드베이스에는 해당 심볼들이 직접 정의되어 있지 않습니다.
# 학습/스모크 목적에서는 `is_active=false` 및 `n_vis_batch=0` 환경이 많으므로,
# 아래 placeholder는 필요한 메서드/속성을 최소 제공하기 위해 포함합니다.
import torch
from typing import Any, Dict, List


class SimAgentsMetrics:
    def __init__(self, prefix: str, max_workers: int = 0) -> None:
        self.prefix = prefix
        self._metric_key = f"{prefix}/sim_agents_2025/realism_meta_metric"

    def update_from_prediction_tensors(
        self,
        *,
        scenario_files: List[str],
        agent_id: Any,
        agent_batch: Any,
        pred_traj: torch.Tensor,
        pred_z: torch.Tensor,
        pred_head: torch.Tensor,
    ) -> None:
        # terminal_cost_final_step 스모크에서는 메트릭 계산이 필수가 아니므로 no-op 처리
        return None

    def _drain_completed_futures(self, wait: bool = True, drain_all: bool = True) -> None:
        return None

    def get_state_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=torch.float32)

    def compute_from_state_tensor(self, reduced_metric_state: torch.Tensor) -> Dict[str, Any]:
        return {self._metric_key: reduced_metric_state.detach().clone()}

    def compute(self) -> Dict[str, Any]:
        return {self._metric_key: torch.tensor(0.0)}

    def reset(self) -> None:
        return None


class SimAgentsSubmission:
    def __init__(
        self,
        is_active: bool,
        method_name: str,
        authors: Any,
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        self.is_active = bool(is_active)

    def update(self, **kwargs: Any) -> None:
        return None

    def aggregate_current_batch(self) -> List[Any]:
        return []

    def save_sub_file(self) -> None:
        return None
