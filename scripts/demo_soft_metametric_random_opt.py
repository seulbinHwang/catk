#!/usr/bin/env python3
"""
합성 ``log_features``(고정 GT 패턴) 대비, 무작위 초기화된 ``sim_features``를 Adam으로 올려
``compute_wosac_metametric_soft`` 의 metametric 을 최대화하는 toy 실험.

- 실제 Waymo 궤적이 아니라 **이미 metric 공간에 올라간 텐서**를 직접 학습한다.
- Flow/BPTT 대신 “미분 경로가 살아 있으면 최적화가 되는지”만 빠르게 보려는 용도.

실행 예::

    python scripts/demo_soft_metametric_random_opt.py --steps 300 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waymo_open_dataset.protos import scenario_pb2

from src.smart.metrics.wosac_metametric_pytorch_differentiable import compute_wosac_metametric_soft
from src.smart.metrics.wosac_metrics import WOSACMetrics


def _hist_bounds(fc) -> tuple[float, float]:
    w = fc.WhichOneof("estimator")
    if w == "histogram":
        h = fc.histogram
        return float(h.min_val), float(h.max_val)
    return -0.5, 1.5


def _build_fixed_log(
    config,
    n_objects: int,
    n_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """[1,N,T] 또는 [1,N] 형태의 log 측 특징 (MetricFeatures 와 동일한 squeeze 규약)."""
    valid = torch.ones(1, n_objects, n_steps, device=device, dtype=dtype)
    object_type = torch.full(
        (1, n_objects),
        int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE),
        device=device,
        dtype=torch.long,
    )
    out: dict[str, torch.Tensor] = {"valid": valid, "object_type": object_type}

    t = torch.linspace(0, 3.14159, n_steps, device=device, dtype=dtype).view(1, 1, n_steps)
    n_idx = torch.arange(n_objects, device=device, dtype=dtype).view(1, n_objects, 1)

    ts_configs = [
        ("linear_speed", config.linear_speed),
        ("angular_speed", config.angular_speed),
        ("linear_acceleration", config.linear_acceleration),
        ("angular_acceleration", config.angular_acceleration),
        ("distance_to_nearest_object", config.distance_to_nearest_object),
        ("distance_to_road_edge", config.distance_to_road_edge),
        ("time_to_collision", config.time_to_collision),
    ]
    for name, fc in ts_configs:
        lo, hi = _hist_bounds(fc)
        mid = (lo + hi) * 0.5
        span = (hi - lo) * 0.15
        v = (mid + span * torch.sin(t + 0.07 * n_idx)).clamp(lo, hi)
        out[name] = v.expand(1, n_objects, n_steps).clone()

    for name in ("collision_per_step", "offroad_per_step", "traffic_light_violation_per_step"):
        out[name] = torch.zeros(1, n_objects, n_steps, device=device, dtype=dtype)

    return out


class ToyLearnableSim(nn.Module):
    """Rollout 수 G, 객체 N, 시각 T. 연속 특징은 히스토그램 구간 안 균등 초기화, 이진형은 logits."""

    def __init__(
        self,
        config,
        n_rollouts: int,
        n_objects: int,
        n_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_rollouts = n_rollouts
        self.n_objects = n_objects
        self.n_steps = n_steps

        self._continuous: nn.ParameterDict = nn.ParameterDict()
        ts_configs = [
            ("linear_speed", config.linear_speed),
            ("angular_speed", config.angular_speed),
            ("linear_acceleration", config.linear_acceleration),
            ("angular_acceleration", config.angular_acceleration),
            ("distance_to_nearest_object", config.distance_to_nearest_object),
            ("distance_to_road_edge", config.distance_to_road_edge),
            ("time_to_collision", config.time_to_collision),
        ]
        for name, fc in ts_configs:
            lo, hi = _hist_bounds(fc)
            u = torch.empty(n_rollouts, n_objects, n_steps, device=device, dtype=dtype).uniform_(lo, hi)
            self._continuous[name] = nn.Parameter(u)

        self._logits_binary: nn.ParameterDict = nn.ParameterDict()
        for name in ("collision_per_step", "offroad_per_step", "traffic_light_violation_per_step"):
            self._logits_binary[name] = nn.Parameter(
                torch.randn(n_rollouts, n_objects, n_steps, device=device, dtype=dtype) * 0.8
            )

    def sim_features(self) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {k: v for k, v in self._continuous.items()}
        for k, v in self._logits_binary.items():
            d[k] = torch.sigmoid(v)
        return d


def main() -> None:
    p = argparse.ArgumentParser(description="Maximize soft WOSAC metametric (toy).")
    p.add_argument("--g-rollouts", type=int, default=16, help="Sim rollouts (G)")
    p.add_argument("--n-objects", type=int, default=12, help="Objects (N)")
    p.add_argument("--n-steps", type=int, default=80, help="Time steps (T)")
    p.add_argument("--steps", type=int, default=250, help="Adam steps")
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    config = WOSACMetrics.load_metrics_config()
    log_features = _build_fixed_log(config, args.n_objects, args.n_steps, device=device, dtype=dtype)
    model = ToyLearnableSim(
        config,
        args.g_rollouts,
        args.n_objects,
        args.n_steps,
        device=device,
        dtype=dtype,
        seed=args.seed,
    ).to(device)

    with torch.no_grad():
        m0 = compute_wosac_metametric_soft(config, log_features, model.sim_features()).metametric.item()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    last_met: float | None = None
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        sim = model.sim_features()
        out = compute_wosac_metametric_soft(config, log_features, sim)
        meta = out.metametric
        loss = -meta
        loss.backward()
        opt.step()
        last_met = float(meta.detach())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:4d}  metametric={last_met:.6f}  (-loss objective)")

    assert last_met is not None
    print(f"\ninit metametric (no grad): {m0:.6f}")
    print(f"final metametric:           {last_met:.6f}  (delta {last_met - m0:+.6f})")


if __name__ == "__main__":
    main()
