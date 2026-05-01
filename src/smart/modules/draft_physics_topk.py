from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from src.smart.modules.draft_physics import (
    BICYCLE_TYPE,
    DRAFT_PHYSICS_ACTUAL_UNIT_KEYS,
    DraftPhysicsRegularizer,
    PEDESTRIAN_TYPE,
    VEHICLE_TYPE,
    _build_zero_output,
)


class TopKDraftPhysicsRegularizer(DraftPhysicsRegularizer):
    """상위 위반 중심 DRaFT 물리 손실을 계산합니다.

    각 agent의 미래 시점별 위반량을 한 번 계산해서 두 가지 집계를 동시에
    얻습니다. 시간 평균과 시간축 상위-K 평균이며 둘을 절반씩 섞어
    최종 손실로 씁니다. 기본값 ``topk_violation_k=4`` 에서 한두 프레임의
    급가속/급회전이 평균에 묻히지 않도록 곧바로 강조됩니다.

    부모 클래스의 시점별 헬퍼(``_compute_*_per_step_penalties``)를 클래스
    당 한 번만 호출하므로, 기존 구현이 가졌던 mean 경로 + topk 경로의
    이중 forward 비용이 없습니다.

    Args:
        *args: 기존 ``DraftPhysicsRegularizer``에 그대로 넘길 위치 인자입니다.
        topk_violation_k: 한 agent 안에서 가장 큰 위반을 몇 개 시점까지 볼지
            정합니다. ``T`` 이상이면 상위-K 집계가 시간 평균과 같아져 단일
            mean 경로로만 동작합니다. 기본값은 ``4`` 이며 README 권장값과
            일치합니다.
        **kwargs: 기존 ``DraftPhysicsRegularizer``에 그대로 넘길 이름 인자입니다.
    """

    def __init__(
        self,
        *args: object,
        topk_violation_k: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(topk_violation_k) < 1:
            raise ValueError("topk_violation_k must be >= 1.")
        self.topk_violation_k = int(topk_violation_k)

    def _topk_mean_over_time(self, value: Tensor) -> Tensor:
        """시간축에서 큰 값 K개만 골라 평균합니다.

        Args:
            value: 시점별 위반량입니다. shape은 ``[n_agent, T]`` 입니다.

        Returns:
            Tensor: agent별 상위 위반 평균입니다. shape은 ``[n_agent]`` 입니다.
        """
        if value.shape[-1] == 0:
            return value.new_zeros((value.shape[0],))
        topk = min(self.topk_violation_k, int(value.shape[-1]))
        if topk >= int(value.shape[-1]):
            return self._mean_over_time(value)
        return value.topk(topk, dim=-1, largest=True, sorted=False).values.mean(dim=-1)

    def _aggregate_class_per_step(
        self,
        per_step: Dict[str, Tensor],
        class_id: int,
        topk: bool,
    ) -> Dict[str, Tensor]:
        """클래스별 시점 위반을 mean 또는 topk 로 집계합니다.

        손실 항(hard/slip/soft/head)만 ``topk`` 인자에 따라 집계 방식이
        달라지며, 로깅용 excess 키들은 항상 시간 평균을 씁니다 (학습 신호와
        분리).

        Args:
            per_step: ``_compute_*_per_step_penalties`` 의 출력입니다.
            class_id: ``VEHICLE``/``BICYCLE``/``PEDESTRIAN`` 클래스입니다.
            topk: ``True`` 면 hard/slip/soft/head 에 상위-K 평균을 적용합니다.

        Returns:
            Dict[str, Tensor]: 집계된 스칼라 사전입니다 (각 값 shape은 ``[n_agent]``).
        """
        agg = self._topk_mean_over_time if topk else self._mean_over_time
        if class_id == PEDESTRIAN_TYPE:
            return {
                "hard": agg(per_step["hard"]),
                "soft": agg(per_step["soft"]),
                "head": agg(per_step["head"]),
                "speed_excess_mps": self._mean_over_time(per_step["speed_excess_mps"]),
                "accel_excess_mps2": self._mean_over_time(per_step["accel_excess_mps2"]),
                "heading_error_deg": self._masked_mean_over_time(
                    per_step["heading_error_deg_unmasked"], per_step["heading_mask"]
                ),
            }
        return {
            "hard": agg(per_step["hard"]),
            "slip": agg(per_step["slip"]),
            "soft": agg(per_step["soft"]),
            "speed_excess_mps": self._mean_over_time(per_step["speed_excess_mps"]),
            "slip_beta_excess_deg": self._mean_over_time(per_step["slip_beta_excess_deg"]),
            "accel_excess_mps2": self._mean_over_time(per_step["accel_excess_mps2"]),
            "steer_excess_deg": self._mean_over_time(per_step["steer_excess_deg"]),
            "steer_rate_excess_degps": self._mean_over_time(per_step["steer_rate_excess_degps"]),
            "lat_accel_excess_mps2": self._mean_over_time(per_step["lat_accel_excess_mps2"]),
        }

    def _compute_pedestrian_totals(
        self,
        pred_stats: Dict[str, Tensor],
        gt_stats: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """사람 클래스의 effective_total / raw_total / soft_effective 를 만듭니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (effective_total, raw_total, soft_effective).
        """
        if self.compare_softness_to_gt:
            soft_effective = torch.relu(pred_stats["soft"] - gt_stats["soft"])
        else:
            soft_effective = pred_stats["soft"]
        effective_total = (
            pred_stats["hard"]
            + self.soft_weight * soft_effective
            + self.pedestrian_heading_weight * pred_stats["head"]
        )
        raw_total = (
            pred_stats["hard"]
            + self.soft_weight * pred_stats["soft"]
            + self.pedestrian_heading_weight * pred_stats["head"]
        )
        return effective_total, raw_total, soft_effective

    def _compute_vehicle_like_totals(
        self,
        pred_stats: Dict[str, Tensor],
        gt_stats: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """차량/자전거 클래스의 effective_total / raw_total / soft_effective 를 만듭니다."""
        if self.compare_softness_to_gt:
            soft_effective = torch.relu(pred_stats["soft"] - gt_stats["soft"])
        else:
            soft_effective = pred_stats["soft"]
        effective_total = pred_stats["hard"] + pred_stats["slip"] + self.soft_weight * soft_effective
        raw_total = pred_stats["hard"] + pred_stats["slip"] + self.soft_weight * pred_stats["soft"]
        return effective_total, raw_total, soft_effective

    def forward(
        self,
        pred_future_norm: Tensor,
        target_future_norm: Tensor,
        packed_agent_type: Tensor,
        packed_agent_length: Tensor,
        packed_prev_control: Tensor,
        packed_prev_control_valid: Tensor,
    ) -> Dict[str, Tensor]:
        """시점 위반을 한 번만 계산해서 mean 과 상위-K 손실을 동시에 얻고 섞습니다.

        Args:
            pred_future_norm: 모델이 생성한 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            target_future_norm: 같은 anchor의 GT 정규화 미래입니다.
                shape은 ``[n_valid_anchor, T, 4]`` 입니다.
            packed_agent_type: anchor별 agent 종류입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_agent_length: anchor별 agent 길이입니다.
                shape은 ``[n_valid_anchor]`` 입니다.
            packed_prev_control: anchor 직전 제어값입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            packed_prev_control_valid: 직전 제어값 유효 여부입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Dict[str, Tensor]: 부모 클래스와 같은 키 구조의 사전입니다. ``loss``,
                ``raw_pred_loss``, class별 component (``hard``/``slip``/``soft``/
                ``head``/``total``) 는 ``K < T`` 일 때 ``0.5 * (mean + topk)`` 로
                블렌드되며, 로깅용 excess 키들은 항상 시간 평균입니다.
        """
        if pred_future_norm.numel() == 0:
            return _build_zero_output(pred_future_norm)

        agent_type = packed_agent_type.to(
            device=pred_future_norm.device, dtype=torch.long
        ).clamp(min=0, max=2)
        agent_length = packed_agent_length.to(
            device=pred_future_norm.device, dtype=pred_future_norm.dtype
        )
        prev_control = packed_prev_control.to(
            device=pred_future_norm.device, dtype=pred_future_norm.dtype
        )
        prev_control_valid = packed_prev_control_valid.to(
            device=pred_future_norm.device, dtype=torch.bool
        )

        T = int(pred_future_norm.shape[1])
        # K >= T 면 topk == mean 이므로 블렌드를 건너뜁니다 (단일 mean 경로).
        do_blend = self.topk_violation_k < T

        output = _build_zero_output(pred_future_norm)
        pred_class_losses: List[Tensor] = []
        raw_pred_class_losses: List[Tensor] = []
        pred_actual_buckets: Dict[str, List[Tensor]] = {
            key: [] for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS
        }
        gt_actual_buckets: Dict[str, List[Tensor]] = {
            key: [] for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS
        }

        class_specs = (
            (VEHICLE_TYPE, "vehicle"),
            (PEDESTRIAN_TYPE, "pedestrian"),
            (BICYCLE_TYPE, "bicycle"),
        )
        for class_id, class_name in class_specs:
            class_mask = agent_type == class_id
            if not class_mask.any():
                continue

            pred_class_future = pred_future_norm[class_mask]
            gt_class_future = target_future_norm[class_mask].detach()
            class_prev_control = prev_control[class_mask]
            class_prev_valid = prev_control_valid[class_mask]
            class_length = agent_length[class_mask]

            # 시점별 위반을 클래스 당 한 번씩만 계산합니다 (pred + gt).
            if class_id == PEDESTRIAN_TYPE:
                pred_per_step = self._compute_pedestrian_per_step_penalties(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                )
                gt_per_step = self._compute_pedestrian_per_step_penalties(
                    future_norm=gt_class_future,
                    prev_control=class_prev_control.detach(),
                    prev_control_valid=class_prev_valid,
                )
            else:
                pred_per_step = self._compute_vehicle_like_per_step_penalties(
                    future_norm=pred_class_future,
                    prev_control=class_prev_control,
                    prev_control_valid=class_prev_valid,
                    agent_length=class_length,
                    class_id=class_id,
                )
                gt_per_step = self._compute_vehicle_like_per_step_penalties(
                    future_norm=gt_class_future,
                    prev_control=class_prev_control.detach(),
                    prev_control_valid=class_prev_valid,
                    agent_length=class_length,
                    class_id=class_id,
                )

            pred_mean = self._aggregate_class_per_step(pred_per_step, class_id, topk=False)
            gt_mean = self._aggregate_class_per_step(gt_per_step, class_id, topk=False)
            if do_blend:
                pred_topk = self._aggregate_class_per_step(pred_per_step, class_id, topk=True)
                gt_topk = self._aggregate_class_per_step(gt_per_step, class_id, topk=True)

            if class_id == PEDESTRIAN_TYPE:
                eff_mean, raw_mean, soft_eff_mean = self._compute_pedestrian_totals(pred_mean, gt_mean)
                if do_blend:
                    eff_topk, raw_topk, soft_eff_topk = self._compute_pedestrian_totals(pred_topk, gt_topk)
                    effective_total = 0.5 * (eff_mean + eff_topk)
                    raw_total = 0.5 * (raw_mean + raw_topk)
                    output["pedestrian_hard"] = (0.5 * (pred_mean["hard"] + pred_topk["hard"])).mean()
                    output["pedestrian_soft"] = (0.5 * (soft_eff_mean + soft_eff_topk)).mean()
                    output["pedestrian_head"] = (0.5 * (pred_mean["head"] + pred_topk["head"])).mean()
                    output["pedestrian_total"] = effective_total.mean()
                else:
                    effective_total = eff_mean
                    raw_total = raw_mean
                    output["pedestrian_hard"] = pred_mean["hard"].mean()
                    output["pedestrian_soft"] = soft_eff_mean.mean()
                    output["pedestrian_head"] = pred_mean["head"].mean()
                    output["pedestrian_total"] = effective_total.mean()
                pred_actual_buckets["speed_excess_mps"].append(pred_mean["speed_excess_mps"].mean())
                gt_actual_buckets["speed_excess_mps"].append(gt_mean["speed_excess_mps"].mean())
                pred_actual_buckets["accel_excess_mps2"].append(pred_mean["accel_excess_mps2"].mean())
                gt_actual_buckets["accel_excess_mps2"].append(gt_mean["accel_excess_mps2"].mean())
                pred_actual_buckets["heading_error_deg"].append(pred_mean["heading_error_deg"].mean())
                gt_actual_buckets["heading_error_deg"].append(gt_mean["heading_error_deg"].mean())
            else:
                eff_mean, raw_mean, soft_eff_mean = self._compute_vehicle_like_totals(pred_mean, gt_mean)
                if do_blend:
                    eff_topk, raw_topk, soft_eff_topk = self._compute_vehicle_like_totals(pred_topk, gt_topk)
                    effective_total = 0.5 * (eff_mean + eff_topk)
                    raw_total = 0.5 * (raw_mean + raw_topk)
                    output[f"{class_name}_hard"] = (0.5 * (pred_mean["hard"] + pred_topk["hard"])).mean()
                    output[f"{class_name}_slip"] = (0.5 * (pred_mean["slip"] + pred_topk["slip"])).mean()
                    output[f"{class_name}_soft"] = (0.5 * (soft_eff_mean + soft_eff_topk)).mean()
                    output[f"{class_name}_total"] = effective_total.mean()
                else:
                    effective_total = eff_mean
                    raw_total = raw_mean
                    output[f"{class_name}_hard"] = pred_mean["hard"].mean()
                    output[f"{class_name}_slip"] = pred_mean["slip"].mean()
                    output[f"{class_name}_soft"] = soft_eff_mean.mean()
                    output[f"{class_name}_total"] = effective_total.mean()
                for key in (
                    "speed_excess_mps",
                    "slip_beta_excess_deg",
                    "accel_excess_mps2",
                    "steer_excess_deg",
                    "steer_rate_excess_degps",
                    "lat_accel_excess_mps2",
                ):
                    pred_actual_buckets[key].append(pred_mean[key].mean())
                    gt_actual_buckets[key].append(gt_mean[key].mean())

            pred_class_losses.append(effective_total.mean())
            raw_pred_class_losses.append(raw_total.mean())

        output["loss"] = self._mean_list_or_zero(pred_class_losses, pred_future_norm)
        output["raw_pred_loss"] = self._mean_list_or_zero(raw_pred_class_losses, pred_future_norm)
        for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            output[f"pred_{key}"] = self._mean_list_or_zero(pred_actual_buckets[key], pred_future_norm)
            output[f"gt_{key}"] = self._mean_list_or_zero(gt_actual_buckets[key], pred_future_norm)
            output[key] = output[f"pred_{key}"]
        return output
