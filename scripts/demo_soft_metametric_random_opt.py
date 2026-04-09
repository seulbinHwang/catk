#!/usr/bin/env python3
"""
합성 ``log_features``(고정 GT 패턴) 대비, 무작위 초기화된 ``sim_features``를 Adam으로 올려
``compute_wosac_metametric_soft`` 의 metametric 을 최대화하는 toy 실험.

- 실제 Waymo 궤적이 아니라 **이미 metric 공간에 올라간 텐서**를 직접 학습한다.
- Flow/BPTT 대신 “미분 경로가 살아 있으면 최적화가 되는지”만 빠르게 보려는 용도.
- **학습은 soft 만** 사용한다 (``loss = -soft_metametric``, 역전파·``opt.step()`` 만).
- **검증용 “RMM”** 은 기본적으로 ``--log-every`` 와 같은 간격 (``--hard-every`` 로 조절,
  ``0`` 이면 끔). 각 스텝 ``opt.step()`` **이후** ``no_grad`` 로만 계산한다.
- **기본**: 리더보드와 수치 정합용 **PyTorch 이산 포트**
  (``compute_wosac_metametric_from_features_torch``).
- **``--official-tf-eval``**: Waymo **TensorFlow** ``compute_scenario_metrics_for_features_bundle``
  (동일 특징 텐서를 ``MetricFeatures`` 로 넣어 호출). TF·의존성 필요.
- **속도**: 이 스크립트는 **이미 feature 공간의 작은 텐서**만 다룬다. 실제 WOSAC 가 무거운 이유는
  대개 ``compute_scenario_rollouts_features``(맵·상호작용·궤적 추출)이지,
  “features → metametric” 자체는 ``N,T,G`` 가 작으면 매우 가볍다.

기본적으로 **ReduceLROnPlateau**(모니터: ``soft_rmm``, ``mode=max``)로 LR 을 줄이고,
``--converge-patience`` 스텝 연속으로 (/rtol·atol 기준) 개선이 없으면 조기 종료한다.
고정 LR 은 ``--no-adaptive-lr`` .

실행 예::

    python scripts/demo_soft_metametric_random_opt.py --steps 5000 --device cuda
    python scripts/demo_soft_metametric_random_opt.py --steps 5000 --official-tf-eval  # 검증만 TF
    python scripts/demo_soft_metametric_random_opt.py --steps 5000 --hard-every 0   # 검증 스칼라 끔
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waymo_open_dataset.protos import scenario_pb2

from typing import Mapping

from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch
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


_BINARY_KEYS = ("collision_per_step", "offroad_per_step", "traffic_light_violation_per_step")


def _sim_features_for_hard_rmm(sim: Mapping[str, torch.Tensor], *, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """PyTorch hard 포트는 bool per_step 을 기대. float(시그모이드)면 ``threshold`` 로 이진화."""
    out: dict[str, torch.Tensor] = {}
    for k, v in sim.items():
        if k in _BINARY_KEYS:
            if v.dtype == torch.bool:
                out[k] = v
            else:
                out[k] = (v >= threshold).to(torch.bool)
        else:
            out[k] = v
    return out


def _torch_to_tf_metric_features(
    log_d: Mapping[str, torch.Tensor],
    sim_d: Mapping[str, torch.Tensor],
    *,
    n_objects: int,
    g_rollouts: int,
    n_steps: int,
    bool_threshold: float,
):
    """torch feature dict → Waymo ``MetricFeatures`` (TF Tensor). 검증용."""
    import numpy as np
    import tensorflow as tf
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features as mf_wm

    def npv(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _log_bool(x: np.ndarray) -> np.ndarray:
        if x.dtype == np.bool_:
            return x
        return x > 0.5

    def _sim_bool(key: str):
        x = npv(sim_d[key])
        if x.dtype == np.bool_:
            return tf.constant(x)
        return tf.constant(x >= bool_threshold)

    oid = np.arange(n_objects, dtype=np.int32)
    lv = _log_bool(npv(log_d["valid"]))
    ot1 = npv(log_d["object_type"]).astype(np.int32)
    if ot1.ndim == 1:
        ot1 = ot1[np.newaxis, :]

    log_mf = mf_wm.MetricFeatures(
        object_id=tf.constant(oid),
        object_type=tf.constant(ot1),
        valid=tf.constant(lv),
        average_displacement_error=tf.zeros((1, n_objects), tf.float32),
        linear_speed=tf.constant(npv(log_d["linear_speed"]), tf.float32),
        linear_acceleration=tf.constant(npv(log_d["linear_acceleration"]), tf.float32),
        angular_speed=tf.constant(npv(log_d["angular_speed"]), tf.float32),
        angular_acceleration=tf.constant(npv(log_d["angular_acceleration"]), tf.float32),
        distance_to_nearest_object=tf.constant(npv(log_d["distance_to_nearest_object"]), tf.float32),
        collision_per_step=tf.constant(_log_bool(npv(log_d["collision_per_step"]))),
        time_to_collision=tf.constant(npv(log_d["time_to_collision"]), tf.float32),
        distance_to_road_edge=tf.constant(npv(log_d["distance_to_road_edge"]), tf.float32),
        offroad_per_step=tf.constant(_log_bool(npv(log_d["offroad_per_step"]))),
        traffic_light_violation_per_step=tf.constant(
            _log_bool(npv(log_d["traffic_light_violation_per_step"]))
        ),
    )

    lv_sim = np.broadcast_to(lv, (g_rollouts, n_objects, n_steps))
    otg = np.broadcast_to(ot1, (g_rollouts, n_objects))

    sim_mf = mf_wm.MetricFeatures(
        object_id=tf.constant(oid),
        object_type=tf.constant(otg),
        valid=tf.constant(lv_sim),
        average_displacement_error=tf.zeros((g_rollouts, n_objects), tf.float32),
        linear_speed=tf.constant(npv(sim_d["linear_speed"]), tf.float32),
        linear_acceleration=tf.constant(npv(sim_d["linear_acceleration"]), tf.float32),
        angular_speed=tf.constant(npv(sim_d["angular_speed"]), tf.float32),
        angular_acceleration=tf.constant(npv(sim_d["angular_acceleration"]), tf.float32),
        distance_to_nearest_object=tf.constant(npv(sim_d["distance_to_nearest_object"]), tf.float32),
        collision_per_step=_sim_bool("collision_per_step"),
        time_to_collision=tf.constant(npv(sim_d["time_to_collision"]), tf.float32),
        distance_to_road_edge=tf.constant(npv(sim_d["distance_to_road_edge"]), tf.float32),
        offroad_per_step=_sim_bool("offroad_per_step"),
        traffic_light_violation_per_step=_sim_bool("traffic_light_violation_per_step"),
    )
    return log_mf, sim_mf


def _eval_reference_rmm(
    *,
    use_official_tf: bool,
    config,
    log_d: Mapping[str, torch.Tensor],
    sim_d: Mapping[str, torch.Tensor],
    n_objects: int,
    g_rollouts: int,
    n_steps: int,
    bool_threshold: float,
) -> float:
    """리더보드 수식과 맞춘 스칼라: TF 공식 경로 또는 PyTorch 이산 포트."""
    if use_official_tf:
        from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm_metrics

        log_mf, sim_mf = _torch_to_tf_metric_features(
            log_d,
            sim_d,
            n_objects=n_objects,
            g_rollouts=g_rollouts,
            n_steps=n_steps,
            bool_threshold=bool_threshold,
        )
        proto = wm_metrics.compute_scenario_metrics_for_features_bundle(
            config, "toy_demo", log_mf, sim_mf
        )
        return float(proto.metametric)
    sim_pt = _sim_features_for_hard_rmm(sim_d, threshold=bool_threshold)
    return compute_wosac_metametric_from_features_torch(config, log_d, sim_pt).metametric


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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument(
        "--hard-every",
        type=int,
        default=-1,
        help="hard RMM 주기. -1(기본)=log-every 와 동일. 0=끔. k>0=매 k 스텝",
    )
    p.add_argument(
        "--hard-bool-threshold",
        type=float,
        default=0.5,
        help="sim 이진 특징이 float 일 때 hard/official 검증용 임계값",
    )
    p.add_argument(
        "--official-tf-eval",
        action="store_true",
        help="검증 스칼라를 PyTorch 포트 대신 TF compute_scenario_metrics_for_features_bundle 로",
    )
    p.add_argument(
        "--no-adaptive-lr",
        action="store_true",
        help="켜면 학습률 고정 (스케줄러 비활성)",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=20,
        help="ReduceLROnPlateau: soft_rmm 개선 없을 때 기다리는 에폭(step) 수",
    )
    p.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="학습률 감쇠 배수",
    )
    p.add_argument(
        "--lr-threshold",
        type=float,
        default=1e-4,
        help="Plateau 로 인정하는 최소 변화량(모드 max 기준)",
    )
    p.add_argument("--min-lr", type=float, default=1e-7, help="학습률 하한")
    p.add_argument(
        "--converge-patience",
        type=int,
        default=50,
        help="연속으로 거의 개선 없으면 조기 종료 (0이면 비활성)",
    )
    p.add_argument(
        "--converge-rtol",
        type=float,
        default=1e-5,
        help="최고 soft_rmm 대비 상대 개선으로 '수렴 아님' 판정",
    )
    p.add_argument(
        "--converge-atol",
        type=float,
        default=1e-7,
        help="절대 개선 임계 (best + atol 초과 시 개선으로 간주)",
    )
    args = p.parse_args()
    if args.hard_every < 0:
        args.hard_every = max(1, args.log_every)

    ref_tag = "official_rmm(tf)" if args.official_tf_eval else "port_rmm(pt)"

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
        sim0 = model.sim_features()
        m0 = compute_wosac_metametric_soft(config, log_features, sim0).metametric.item()
        m0_ref: float | None = None
        if args.hard_every > 0:
            m0_ref = _eval_reference_rmm(
                use_official_tf=args.official_tf_eval,
                config=config,
                log_d=log_features,
                sim_d=sim0,
                n_objects=args.n_objects,
                g_rollouts=args.g_rollouts,
                n_steps=args.n_steps,
                bool_threshold=args.hard_bool_threshold,
            )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched: lr_scheduler.ReduceLROnPlateau | None = None
    if not args.no_adaptive_lr:
        sched = lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )

    last_met: float | None = None
    best_soft = float("-inf")
    stagnation = 0
    stop_reason = "max_steps"
    # 학습: soft 만. 로그의 soft_rmm = 해당 스텝 forward(갱신 직전 가중치)에서 나온 값.
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        sim = model.sim_features()
        out = compute_wosac_metametric_soft(config, log_features, sim)
        meta = out.metametric
        loss = -meta
        loss.backward()
        opt.step()
        last_met = float(meta.detach())

        if sched is not None:
            sched.step(last_met)

        if not (best_soft > float("-inf")):
            improved = True
        else:
            margin = args.converge_atol + args.converge_rtol * abs(best_soft)
            improved = last_met > best_soft + margin
        if improved:
            best_soft = last_met
            stagnation = 0
        else:
            stagnation += 1

        current_lr = opt.param_groups[0]["lr"]
        do_log = step == 1 or step % args.log_every == 0 or step == args.steps
        do_hard = args.hard_every > 0 and (
            step == 1 or step % args.hard_every == 0 or step == args.steps
        )
        last_ref: float | None = None
        if do_hard:
            with torch.no_grad():
                sim_eval = {k: v.detach() for k, v in model.sim_features().items()}
                last_ref = _eval_reference_rmm(
                    use_official_tf=args.official_tf_eval,
                    config=config,
                    log_d=log_features,
                    sim_d=sim_eval,
                    n_objects=args.n_objects,
                    g_rollouts=args.g_rollouts,
                    n_steps=args.n_steps,
                    bool_threshold=args.hard_bool_threshold,
                )

        if do_log or do_hard:
            line = f"step {step:4d}  soft_rmm={last_met:.6f}  lr={current_lr:.2e}"
            if do_hard and last_ref is not None:
                line += f"  {ref_tag}(after step)={last_ref:.6f}"
            print(line)

        if args.converge_patience > 0 and stagnation >= args.converge_patience:
            stop_reason = "converged(no_improvement)"
            if not (do_log or do_hard):
                print(
                    f"step {step:4d}  soft_rmm={last_met:.6f}  lr={current_lr:.2e}  "
                    f"(converged, stagnation={stagnation})"
                )
            break

        if (
            sched is not None
            and args.min_lr > 0
            and current_lr <= args.min_lr * (1.0 + 1e-6)
            and stagnation >= min(args.converge_patience, args.lr_patience)
            and args.converge_patience > 0
        ):
            stop_reason = "converged(min_lr_and_plateau)"
            if not (do_log or do_hard):
                print(
                    f"step {step:4d}  soft_rmm={last_met:.6f}  lr={current_lr:.2e}  "
                    f"(min_lr floor, stagnation={stagnation})"
                )
            break

    assert last_met is not None
    with torch.no_grad():
        sim_final = model.sim_features()
        final_soft = compute_wosac_metametric_soft(config, log_features, sim_final).metametric.item()
        final_ref: float | None = None
        if args.hard_every > 0:
            final_ref = _eval_reference_rmm(
                use_official_tf=args.official_tf_eval,
                config=config,
                log_d=log_features,
                sim_d=sim_final,
                n_objects=args.n_objects,
                g_rollouts=args.g_rollouts,
                n_steps=args.n_steps,
                bool_threshold=args.hard_bool_threshold,
            )

    print(f"\nstop: {stop_reason}")
    print(f"init soft_rmm (no grad): {m0:.6f}")
    print(f"final soft_rmm:          {final_soft:.6f}  (delta {final_soft - m0:+.6f})")
    print(f"best soft_rmm (track):   {best_soft:.6f}")
    if args.hard_every > 0 and m0_ref is not None and final_ref is not None:
        print(f"init {ref_tag}:          {m0_ref:.6f}")
        print(f"final {ref_tag}:          {final_ref:.6f}  (delta {final_ref - m0_ref:+.6f})")


if __name__ == "__main__":
    main()
