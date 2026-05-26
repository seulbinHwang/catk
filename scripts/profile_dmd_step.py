"""Standalone profiler for DMD `_run_flow_dmd_ft_step` phase timing.

목적:
  Self-Forcing DMD fine-tuning 의 학습 속도 개선 후보를 정량적으로 찾기 위해,
  한 training batch 의 DMD step 안에서 다음 phase 들이 차지하는 GPU 시간을 측정.

  phase 분해 (per training batch):
    - prepare_cache         : encoder.prepare_inference_cache  (per anchor)
    - cl_rollout            : _run_parallel_rollout_chunk full_grad=True  (per anchor)
                              · generator forward + ODE solver activation
    - ref_score_fw          : ref_flow_decoder forward (no_grad)            (per anchor)
    - fake_score_fw_eval    : fake_score_decoder forward, no_grad ctx       (per anchor)
    - fake_score_fw_train   : fake_score_decoder forward, grad enabled      (per anchor)
    - ode_sample            : flow_ode.sample on x_gen.detach()             (per anchor)
    - backward_total        : manual_backward(L_gen) + manual_backward(L_fake) 합
    - other                 : total - 위 합계 (Python loop, host overhead 등)

사용:
  scripts/profile_dmd_step.py
    (launcher 와 동일하게 hydra config 받음.  default 는 launcher default 와 일치.)

  예: TRAIN_B=16 G=4 stride=4 pred=4 으로 측정
    CUDA_VISIBLE_DEVICES=3 python scripts/profile_dmd_step.py

  예: G=1, adjoint off 비교
    CUDA_VISIBLE_DEVICES=3 PROFILE_G=1 PROFILE_ADJOINT=false python scripts/profile_dmd_step.py
"""
from __future__ import annotations

import contextlib
import os
import pickle
import time
from collections import defaultdict
from typing import Callable

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf


# ── CUDA timing helper ──────────────────────────────────────────────────────


class Timings:
    """Per-phase elapsed ms accumulator (GPU sync 기반)."""

    def __init__(self) -> None:
        self._total_ms: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def section(self, name: str):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self._total_ms[name] += (time.perf_counter() - t0) * 1000.0
            self._count[name] += 1

    def add(self, name: str, ms: float, count: int = 1) -> None:
        self._total_ms[name] += ms
        self._count[name] += count

    def report(self, total_ms: float | None = None) -> str:
        lines: list[str] = []
        items = sorted(self._total_ms.items(), key=lambda kv: -kv[1])
        sum_phase = sum(self._total_ms.values())
        if total_ms is None:
            total_ms = sum_phase
        lines.append(f"{'phase':<28} {'total_ms':>10} {'count':>6} {'mean_ms':>10} {'pct':>7}")
        lines.append("-" * 70)
        for name, t in items:
            n = max(1, self._count[name])
            pct = (t / total_ms * 100.0) if total_ms > 0 else 0.0
            lines.append(f"{name:<28} {t:>10.2f} {n:>6d} {t / n:>10.2f} {pct:>6.1f}%")
        lines.append("-" * 70)
        lines.append(f"{'(sum of phases)':<28} {sum_phase:>10.2f}")
        if total_ms is not None and total_ms > sum_phase:
            lines.append(f"{'other (host/py/launch)':<28} {total_ms - sum_phase:>10.2f}")
            lines.append(f"{'TOTAL step':<28} {total_ms:>10.2f}")
        return "\n".join(lines)


def _wrap_module_forward(module: torch.nn.Module, timings: Timings, name_eval: str, name_train: str):
    """nn.Module 의 forward 를 wrap 해 grad ctx 에 따라 phase 분리 기록.

    name_eval  : torch.is_grad_enabled() == False 일 때 (score evaluation 용 호출)
    name_train : grad 켜진 호출 (critic FM loss path)
    """
    orig_forward = module.forward

    def wrapped(*args, **kwargs):
        phase = name_train if torch.is_grad_enabled() else name_eval
        with timings.section(phase):
            return orig_forward(*args, **kwargs)

    module.forward = wrapped  # type: ignore[assignment]
    return orig_forward


def _wrap_callable(owner, attr: str, timings: Timings, name: str) -> Callable:
    """owner.attr 를 timing wrapper 로 교체.  반환은 원본 callable."""
    orig = getattr(owner, attr)

    def wrapped(*args, **kwargs):
        with timings.section(name):
            return orig(*args, **kwargs)

    setattr(owner, attr, wrapped)
    return orig


# ── Main profiling routine ──────────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig) -> None:
    # ── seed / determinism ──────────────────────────────────────────────
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── instantiate datamodule / model / (minimal) trainer ──────────────
    print(f"[profile] instantiating datamodule: {cfg.data._target_}")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    print(f"[profile] instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # ── load finetune ckpt (state_dict 만 — same as run.py finetune path) ──
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        print(f"[profile] loading ckpt state_dict: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[profile] ckpt loaded — missing={len(missing)} unexpected={len(unexpected)}")

    # ── move to cuda + train mode ───────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # on_train_start 가 fake_score_decoder 를 main 으로 sync — 수동 트리거.
    # (Trainer 없이 호출하면 self.trainer 가 없어 fail; 우회로 ref/fake 가
    # __init__ 에서 이미 deepcopy 된 상태라 weight sync 는 사실상 OK.)
    if model.fake_score_decoder is not None:
        model.fake_score_decoder = model.fake_score_decoder.to(device)
    if model.ref_flow_decoder is not None:
        model.ref_flow_decoder = model.ref_flow_decoder.to(device)

    # ── fetch one training batch ────────────────────────────────────────
    train_loader = datamodule.train_dataloader()
    batch_iter = iter(train_loader)
    data = next(batch_iter)
    # move to device
    if hasattr(data, "to"):
        data = data.to(device)
    print(f"[profile] batch: scenarios={len(data['scenario_id'])} agents={data['agent']['valid_mask'].shape[0]}")

    # ── fake trainer attribute (DMD step 이 _batches_that_stepped 참조) ──
    class _FakeEpochLoop:
        _batches_that_stepped = 0

    class _FakeFitLoop:
        epoch_loop = _FakeEpochLoop()

    class _FakeTrainer:
        fit_loop = _FakeFitLoop()
        strategy = None  # no DDP

    model._trainer = _FakeTrainer()  # bypass Lightning trainer

    # ── monkey-patch manual_backward (DMD step is manual-mode) ──────────
    def _noop_backward(loss):  # we keep grad accum but profile its time
        with timings.section("backward_total"):
            loss.backward()

    model.manual_backward = _noop_backward  # type: ignore[assignment]

    # ── token_processor (one-time, outside DMD step) ────────────────────
    timings_warmup = Timings()
    timings = Timings()

    def _run_one_step(t: Timings) -> float:
        with t.section("token_processor"):
            tokenized_map, tokenized_agent = model.token_processor(data)

        # patch the 6 hot phases (per-call)
        flow_decoder = model.encoder.agent_encoder.flow_decoder
        flow_ode = model.encoder.agent_encoder.flow_ode
        fake_score = model.fake_score_decoder
        ref_score = model.ref_flow_decoder
        encoder = model.encoder
        agent_enc = model.encoder.agent_encoder

        orig_cl_rollout = _wrap_callable(model, "_run_parallel_rollout_chunk", t, "cl_rollout")
        orig_prepare_cache_enc = _wrap_callable(encoder, "prepare_inference_cache", t, "prepare_cache")
        orig_prepare_cache_aenc = _wrap_callable(agent_enc, "prepare_inference_cache", t, "prepare_cache")
        orig_ode_sample = _wrap_callable(flow_ode, "sample", t, "ode_sample")
        orig_fake_fw = _wrap_module_forward(fake_score, t, "fake_score_fw_eval", "fake_score_fw_train")
        if ref_score is not None:
            orig_ref_fw = _wrap_module_forward(ref_score, t, "ref_score_fw", "ref_score_fw_grad")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            diag = model._run_flow_dmd_ft_step(tokenized_map, tokenized_agent, data)
            # final DDP dummy backward (DMD step 끝에서 reduce 용으로 묶는 grad=0 path)
            if "loss" in diag:
                with t.section("backward_total"):
                    diag["loss"].backward()
        finally:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # restore
            model._run_parallel_rollout_chunk = orig_cl_rollout
            encoder.prepare_inference_cache = orig_prepare_cache_enc
            agent_enc.prepare_inference_cache = orig_prepare_cache_aenc
            flow_ode.sample = orig_ode_sample
            fake_score.forward = orig_fake_fw
            if ref_score is not None:
                ref_score.forward = orig_ref_fw

        # zero grads (don't actually step optimizer — we only profile)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if fake_score is not None:
            for p in fake_score.parameters():
                if p.grad is not None:
                    p.grad = None

        return elapsed_ms

    n_warmup = int(os.environ.get("PROFILE_WARMUP_STEPS", "1"))
    n_measure = int(os.environ.get("PROFILE_MEASURE_STEPS", "3"))

    print(f"[profile] warmup x{n_warmup}, measure x{n_measure}")
    for i in range(n_warmup):
        elapsed = _run_one_step(timings_warmup)
        print(f"  warmup {i + 1}/{n_warmup}: {elapsed:.1f} ms")

    total_elapsed = 0.0
    for i in range(n_measure):
        elapsed = _run_one_step(timings)
        total_elapsed += elapsed
        print(f"  measure {i + 1}/{n_measure}: {elapsed:.1f} ms")

    print()
    print("=" * 70)
    print("DMD step phase breakdown (averaged over measure runs)")
    print("=" * 70)
    avg_total_ms = total_elapsed / max(1, n_measure)
    # Scale every accumulator by 1/n_measure → per-step ms
    avg = Timings()
    for k, v in timings._total_ms.items():
        avg._total_ms[k] = v / n_measure
        avg._count[k] = timings._count[k] // max(1, n_measure)
    print(avg.report(total_ms=avg_total_ms))
    print()
    print(f"[profile] total per-step (incl. token_processor) ≈ {avg_total_ms:.1f} ms")
    if avg_total_ms > 0:
        print(
            f"[profile] estimated throughput: "
            f"{1000.0 / avg_total_ms:.2f} step/s,  "
            f"{3600.0 * 1000.0 / avg_total_ms:.0f} step/hr"
        )


if __name__ == "__main__":
    main()
