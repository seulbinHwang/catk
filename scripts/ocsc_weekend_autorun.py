#!/usr/bin/env python3
"""Monitor OCSC runs and relaunch bounded fallback experiments.

The script is intentionally conservative:
  * First validation RMM below the guard threshold stops the run.
  * After N validations, the run must show either RMM improvement or a
    downward val_open ADE/FDE trend.
  * Failed runs are replaced by the next preset, all starting from the base
    checkpoint through scripts/train_ocsc_ft.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


RMM_KEY = "val_closed/sim_agents_2025/realism_meta_metric"
ADE_KEYS = ("val_open/ADE2s", "val_open/ADE2s_epoch")
FDE_KEYS = ("val_open/FDE2s", "val_open/FDE2s_epoch")
TRAIN_LOSS_KEYS = ("train/ocsc_ft/loss", "train/loss")
CRASH_PATTERNS = (
    "DataLoader worker",
    "CUDA error",
    "Traceback (most recent call last)",
    "ChildFailedError",
    "RuntimeError:",
    "Exception:",
    "FAILED",
)


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


FALLBACK_VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="ocsc_lr1e6_evalselect_shuffle",
        note="Lower LR only, keeping the fast data path closest to the initial run.",
        env={
            "LR": "1.0e-6",
            "DATA_SHUFFLE": "true",
            "TRAIN_USE_EVAL_AGENT_SELECTION": "true",
            "TRAIN_EPOCH_SAMPLE_FRACTION": "0.5",
            "PRECISION": "32-true",
            "GRADIENT_CLIP_VAL": "0",
            "NUM_WORKERS": "12",
            "PREFETCH_FACTOR": "4",
            "EVAL_NUM_WORKERS": "12",
            "EVAL_PREFETCH_FACTOR": "2",
            "SIM_AGENTS_METRIC_WORKERS": "3",
        },
    ),
    Variant(
        name="ocsc_m12_lr1e6_evalselect_b8",
        note="Increase OL target pool with smaller train batch to avoid the M16 OOM mode.",
        env={
            "LR": "1.0e-6",
            "TRAIN_B": "8",
            "OCSC_N_OL_ROLLOUTS": "12",
            "DATA_SHUFFLE": "true",
            "TRAIN_USE_EVAL_AGENT_SELECTION": "true",
            "TRAIN_EPOCH_SAMPLE_FRACTION": "0.5",
            "PRECISION": "32-true",
            "GRADIENT_CLIP_VAL": "0",
            "NUM_WORKERS": "12",
            "PREFETCH_FACTOR": "4",
            "EVAL_NUM_WORKERS": "12",
            "EVAL_PREFETCH_FACTOR": "2",
            "SIM_AGENTS_METRIC_WORKERS": "3",
        },
    ),
    Variant(
        name="ocsc_steprefiner_lr1e6",
        note="Unfreeze step_refiner plus velocity_head; useful if velocity head only cannot move open ADE/FDE.",
        env={
            "LR": "1.0e-6",
            "OCSC_VELOCITY_HEAD_ONLY": "false",
            "OCSC_FULL_FLOW_DECODER": "false",
            "DATA_SHUFFLE": "true",
            "TRAIN_USE_EVAL_AGENT_SELECTION": "true",
            "TRAIN_EPOCH_SAMPLE_FRACTION": "0.5",
            "PRECISION": "32-true",
            "GRADIENT_CLIP_VAL": "0",
            "NUM_WORKERS": "12",
            "PREFETCH_FACTOR": "4",
            "EVAL_NUM_WORKERS": "12",
            "EVAL_PREFETCH_FACTOR": "2",
            "SIM_AGENTS_METRIC_WORKERS": "3",
        },
    ),
    Variant(
        name="ocsc_gt_target_lr1e6",
        note="Fallback to the GT-target clean mode if OL nearest matching does not learn.",
        env={
            "LR": "1.0e-6",
            "OCSC_GT_TARGET": "true",
            "OCSC_OL_NEAREST_MATCH": "false",
            "DATA_SHUFFLE": "true",
            "TRAIN_USE_EVAL_AGENT_SELECTION": "true",
            "TRAIN_EPOCH_SAMPLE_FRACTION": "0.5",
            "PRECISION": "32-true",
            "GRADIENT_CLIP_VAL": "0",
            "NUM_WORKERS": "12",
            "PREFETCH_FACTOR": "4",
            "EVAL_NUM_WORKERS": "12",
            "EVAL_PREFETCH_FACTOR": "2",
            "SIM_AGENTS_METRIC_WORKERS": "3",
        },
    ),
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def load_wandb_api() -> Any:
    import wandb

    return wandb.Api()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def series_for_key(run: Any, key: str) -> list[float]:
    return [value for _, value in rows_for_key(run, key)]


def rows_for_key(run: Any, key: str) -> list[tuple[int | None, float]]:
    rows: list[tuple[int | None, float]] = []
    try:
        for row in run.scan_history(keys=["_step", key], page_size=1000):
            value = row.get(key)
            if value is not None:
                step = row.get("_step")
                rows.append((int(step) if step is not None else None, float(value)))
    except Exception:
        rows = []

    if rows:
        return rows

    try:
        history_rows = run.history(keys=["_step", key], pandas=False)
        return [
            (
                int(row["_step"]) if row.get("_step") is not None else None,
                float(row[key]),
            )
            for row in history_rows
            if row.get(key) is not None
        ]
    except Exception:
        return []


def collapse_adjacent_duplicate_rows(
    rows: list[tuple[int | None, float]],
    *,
    max_step_gap: int = 2,
    atol: float = 1e-12,
) -> list[tuple[int | None, float]]:
    """Collapse duplicate metric rows emitted around one validation boundary."""
    collapsed: list[tuple[int | None, float]] = []
    for step, value in rows:
        if collapsed:
            prev_step, prev_value = collapsed[-1]
            adjacent_step = (
                step is not None
                and prev_step is not None
                and 0 <= step - prev_step <= max_step_gap
            )
            if adjacent_step and abs(value - prev_value) <= atol:
                collapsed[-1] = (step, value)
                continue
        collapsed.append((step, value))
    return collapsed


def values_for_rows(rows: list[tuple[int | None, float]]) -> list[float]:
    return [value for _, value in rows]


def validation_metric_series(run: Any, key: str) -> list[float]:
    return values_for_rows(collapse_adjacent_duplicate_rows(rows_for_key(run, key)))


def sampled_series_for_key(run: Any, key: str, samples: int = 100) -> list[float]:
    try:
        rows = run.history(keys=[key], samples=int(samples), pandas=False)
        return [float(row[key]) for row in rows if row.get(key) is not None]
    except Exception:
        return []


def first_available_series(run: Any, keys: tuple[str, ...]) -> tuple[str | None, list[float]]:
    for key in keys:
        values = series_for_key(run, key)
        if values:
            return key, values
    return None, []


def first_available_sampled_series(
    run: Any,
    keys: tuple[str, ...],
    samples: int = 100,
) -> tuple[str | None, list[float]]:
    for key in keys:
        values = sampled_series_for_key(run, key, samples=samples)
        if values:
            return key, values
    return None, []


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def decreasing_signal(values: list[float], min_rel_drop: float, min_abs_drop: float) -> bool:
    if len(values) < 3:
        return False
    drop = values[0] - values[-1]
    required = max(min_abs_drop, abs(values[0]) * min_rel_drop)
    non_increasing_steps = sum(
        1 for prev, cur in zip(values, values[1:]) if cur <= prev + min_abs_drop
    )
    return drop >= required and slope(values) < 0.0 and non_increasing_steps >= len(values) - 2


def metric_snapshot(api: Any, project_path: str, run_id: str) -> dict[str, Any]:
    run = api.run(f"{project_path}/{run_id}")
    ade_key, ade = first_available_series(run, ADE_KEYS)
    fde_key, fde = first_available_series(run, FDE_KEYS)
    train_loss_key, train_loss = first_available_sampled_series(run, TRAIN_LOSS_KEYS)
    rmm = validation_metric_series(run, RMM_KEY)
    return {
        "state": run.state,
        "name": run.name,
        "rmm": rmm,
        "ade_key": ade_key,
        "ade": ade,
        "fde_key": fde_key,
        "fde": fde,
        "train_loss_key": train_loss_key,
        "train_loss": train_loss[-200:],
    }


def metric_snapshot_from_path(api: Any, run_path: str) -> dict[str, Any]:
    parts = run_path.strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"run path must be entity/project/run_id, got {run_path!r}")
    snap = metric_snapshot(api, "/".join(parts[:2]), parts[2])
    snap["path"] = run_path
    return snap


def reference_snapshots(api: Any, run_paths: list[str]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for run_path in run_paths:
        try:
            snap = metric_snapshot_from_path(api, run_path)
        except Exception as exc:
            refs[run_path] = {"error": repr(exc)}
            continue
        refs[run_path] = {
            "name": snap.get("name"),
            "state": snap.get("state"),
            "rmm": snap.get("rmm", []),
            "ade_key": snap.get("ade_key"),
            "ade": snap.get("ade", []),
            "fde_key": snap.get("fde_key"),
            "fde": snap.get("fde", []),
            "train_loss_key": snap.get("train_loss_key"),
            "train_loss_tail": snap.get("train_loss", [])[-10:],
        }
    return refs


def decision(
    snap: dict[str, Any],
    *,
    min_validations: int,
    rmm_floor: float,
    rmm_min_gain: float,
    open_min_rel_drop: float,
    open_min_abs_drop: float,
) -> tuple[str, str]:
    rmm = snap["rmm"]
    ade = snap["ade"]
    fde = snap["fde"]
    state = snap["state"]

    if rmm and rmm[0] < rmm_floor:
        return "fail", f"first RMM {rmm[0]:.8f} < floor {rmm_floor:.8f}"

    if state in {"failed", "crashed", "killed"}:
        return "fail", f"wandb state={state}"

    n_val = len(rmm)
    if n_val < min_validations:
        return "wait", f"validations {n_val}/{min_validations}"

    rmm_gain = max(rmm) - rmm[0] if rmm else 0.0
    rmm_signal = rmm_gain >= rmm_min_gain
    ade_signal = decreasing_signal(ade[:n_val], open_min_rel_drop, open_min_abs_drop)
    fde_signal = decreasing_signal(fde[:n_val], open_min_rel_drop, open_min_abs_drop)
    open_signal = ade_signal and fde_signal

    if rmm_signal or open_signal:
        parts = []
        if rmm_signal:
            parts.append(f"RMM gain {rmm_gain:.8f}")
        if open_signal:
            parts.append(
                "open ADE/FDE downward "
                f"({ade[0]:.6f}->{ade[min(n_val, len(ade)) - 1]:.6f}, "
                f"{fde[0]:.6f}->{fde[min(n_val, len(fde)) - 1]:.6f})"
            )
        return "keep", "; ".join(parts)

    ade_text = f"{ade[:n_val]}" if ade else "missing"
    fde_text = f"{fde[:n_val]}" if fde else "missing"
    return (
        "fail",
        f"no learning signal after {n_val} validations: "
        f"rmm={rmm[:n_val]}, ade={ade_text}, fde={fde_text}",
    )


def parse_cutoff_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(
            f"cutoff timestamp must include timezone offset, got {value!r}"
        )
    return dt.timestamp()


def cutoff_reached(cutoff_ts: float | None) -> bool:
    return cutoff_ts is not None and time.time() >= cutoff_ts


def crash_tail(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    try:
        text = log_path.read_bytes()[-300_000:].decode("utf-8", "ignore")
    except OSError:
        return []
    lines = text.replace("\r", "\n").splitlines()
    hits = [
        line[-240:]
        for line in lines
        if any(pattern in line for pattern in CRASH_PATTERNS)
    ]
    return hits[-8:]


def parse_run_id(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_bytes()[-200_000:].decode("utf-8", "ignore")
    except OSError:
        return None
    matches = re.findall(r"wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"wandb/run-\d+_\d+-([A-Za-z0-9_-]+)", text)
    if matches:
        return matches[-1]
    return None


def stop_tmux_target(target: str) -> None:
    if not target:
        return
    log(f"stopping tmux target {target}")
    for _ in range(2):
        subprocess.run(["tmux", "send-keys", "-t", target, "C-c"], check=False)
        time.sleep(10)


def terminate_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    log(f"stopping child process group pid={proc.pid}")
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(12):
        if proc.poll() is not None:
            return
        time.sleep(5)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(6):
        if proc.poll() is not None:
            return
        time.sleep(5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def base_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "NPROC_PER_NODE": str(args.nproc_per_node),
        "MY_EXPERIMENT": "ocsc_ft",
        "TRAIN_B": str(args.train_batch_size),
        "VAL_B": str(args.val_batch_size),
        "TEST_B": str(args.val_batch_size),
        "LIMIT_VAL_BATCHES": str(args.limit_val_batches),
        "VAL_CHECK_INTERVAL": str(args.val_check_interval),
        "MAX_EPOCHS": str(args.max_epochs),
        "OCSC_N_ROLLOUTS": "4",
        "OCSC_N_OL_ROLLOUTS": "8",
        "OCSC_OL_NEAREST_MATCH": "true",
        "OCSC_USE_PRETRAINED_REF": "true",
        "OCSC_STRICT_ACTIVE_MASK": "true",
        "OCSC_POSITION_WEIGHT": "1.0",
        "OCSC_HEADING_WEIGHT": "0.01",
        "EVAL_MULTIPROCESSING_CONTEXT": "spawn",
        "TMUX_LOG_TAIL": "false",
        "WANDB_TAGS": "[ocsc_weekend_autorun]",
    }


def launch_variant(
    variant: Variant,
    args: argparse.Namespace,
    log_dir: Path,
) -> tuple[subprocess.Popen[Any], Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{variant.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{task_name}.log"
    env = os.environ.copy()
    env.update(base_env(args))
    env.update(variant.env)
    env["MY_TASK_NAME"] = task_name
    env["TMUX_LOG_PATH"] = str(log_path)
    command = ["bash", "scripts/train_ocsc_ft.sh"]

    log(f"launching {task_name}: {variant.note}")
    log(f"log file: {log_path}")
    log_file = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(
        command,
        cwd=args.repo_root,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_path


def monitor_run(
    api: Any,
    args: argparse.Namespace,
    *,
    run_id: str,
    run_label: str,
    log_path: Path,
    state_path: Path,
    stop_current: Any,
) -> tuple[str, str]:
    last_summary = ""
    while True:
        if cutoff_reached(args.cutoff_ts):
            reason = f"reached cutoff {args.stop_at_kst}"
            log(reason)
            stop_current()
            return "cutoff", reason

        crashes = crash_tail(log_path)
        if crashes:
            reason = "crash detected: " + " | ".join(crashes[-3:])
            log(reason)
            stop_current()
            return "fail", reason

        try:
            snap = metric_snapshot(api, args.wandb_project_path, run_id)
            status, reason = decision(
                snap,
                min_validations=args.min_validations,
                rmm_floor=args.rmm_baseline - args.rmm_tolerance,
                rmm_min_gain=args.rmm_min_gain,
                open_min_rel_drop=args.open_min_rel_drop,
                open_min_abs_drop=args.open_min_abs_drop,
            )
        except Exception as exc:
            snap = {"state": "api_error", "rmm": [], "ade": [], "fde": []}
            status, reason = "wait", f"wandb api error: {exc!r}"

        rmm = snap.get("rmm", [])
        ade = snap.get("ade", [])
        fde = snap.get("fde", [])
        train_loss = snap.get("train_loss", [])
        summary = (
            f"{run_label} run={run_id} status={status} reason={reason} "
            f"state={snap.get('state')} rmm={rmm[-3:]} ade={ade[-3:]} "
            f"fde={fde[-3:]} train_loss={train_loss[-3:]}"
        )
        if summary != last_summary:
            log(summary)
            last_summary = summary
        write_state(
            state_path,
            {
                "time": now(),
                "run_label": run_label,
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "cutoff": args.stop_at_kst,
                "references": args.reference_snapshots,
                "snapshot": snap,
            },
        )

        if status == "fail":
            stop_current()
            return status, reason

        if snap.get("state") == "finished":
            return "finished", "run finished"

        time.sleep(args.poll_seconds)


def wait_for_run_id(log_path: Path, proc: subprocess.Popen[Any], timeout_s: int) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run_id = parse_run_id(log_path)
        if run_id:
            return run_id
        if proc.poll() is not None:
            return parse_run_id(log_path)
        time.sleep(10)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--wandb-project-path", default="se99an/clsft-catk")
    parser.add_argument("--initial-run-id", default="ebyre3qr")
    parser.add_argument(
        "--initial-log-path",
        default="/tmp/ocsc_pose_2hz_g4_m8_velhead_lr2e6_b16v32_2gpu_v2_refsync_globalnm.log",
    )
    parser.add_argument("--initial-tmux-target", default="kinematic:99")
    parser.add_argument("--state-path", default="logs/ocsc_weekend_autorun_state.json")
    parser.add_argument("--log-dir", default="/tmp/ocsc_weekend_autorun")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument(
        "--stop-at-kst",
        default="2026-06-01T10:00:00+09:00",
        help="Stop current work at this absolute KST timestamp. Empty disables cutoff.",
    )
    parser.add_argument(
        "--reference-run",
        action="append",
        default=[
            "se99an/clsft-catk/hdbfyfn2",
            "se99an/SMART-FLOW/dk3njfnf",
        ],
        help=(
            "Reference W&B run path entity/project/run_id. Defaults include "
            "pretrained val and a known OCSC-clean run."
        ),
    )
    parser.add_argument("--min-validations", type=int, default=3)
    parser.add_argument("--rmm-baseline", type=float, default=0.77927)
    parser.add_argument("--rmm-tolerance", type=float, default=0.0001)
    parser.add_argument("--rmm-min-gain", type=float, default=0.0002)
    parser.add_argument("--open-min-rel-drop", type=float, default=0.0005)
    parser.add_argument("--open-min-abs-drop", type=float, default=0.0001)
    parser.add_argument("--cuda-visible-devices", default="2,3")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--limit-val-batches", default="0.1")
    parser.add_argument("--val-check-interval", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--skip-initial", action="store_true")
    args = parser.parse_args()

    args.repo_root = str(Path(args.repo_root).resolve())
    args.cutoff_ts = parse_cutoff_timestamp(args.stop_at_kst)
    state_path = Path(args.repo_root) / args.state_path
    log_dir = Path(args.log_dir)
    api = load_wandb_api()
    args.reference_snapshots = reference_snapshots(api, args.reference_run)
    log(f"references: {json.dumps(args.reference_snapshots, sort_keys=True)}")

    if not args.skip_initial and args.initial_run_id:
        status, reason = monitor_run(
            api,
            args,
            run_id=args.initial_run_id,
            run_label="initial",
            log_path=Path(args.initial_log_path),
            state_path=state_path,
            stop_current=lambda: stop_tmux_target(args.initial_tmux_target),
        )
        if status in {"finished", "cutoff"}:
            log(f"initial run ended: {reason}")
            return 0
        log(f"initial run failed guard: {reason}; moving to fallback variants")

    for idx, variant in enumerate(FALLBACK_VARIANTS, start=1):
        if cutoff_reached(args.cutoff_ts):
            log(f"cutoff reached before launching {variant.name}; stopping autorun")
            return 0
        proc, variant_log = launch_variant(variant, args, log_dir)
        run_id = wait_for_run_id(variant_log, proc, timeout_s=900)
        if not run_id:
            reason = f"could not find wandb run id for {variant.name}"
            log(reason)
            terminate_process(proc)
            continue

        status, reason = monitor_run(
            api,
            args,
            run_id=run_id,
            run_label=f"fallback_{idx}:{variant.name}",
            log_path=variant_log,
            state_path=state_path,
            stop_current=lambda proc=proc: terminate_process(proc),
        )
        if status in {"finished", "cutoff"}:
            log(f"{variant.name} ended: {reason}")
            return 0
        log(f"{variant.name} failed guard: {reason}; trying next variant")

    failure_path = state_path.with_name("ocsc_weekend_autorun_failed.json")
    write_state(
        failure_path,
        {
            "time": now(),
            "status": "exhausted",
            "message": "All fallback variants failed the guard. Algorithm inspection is required.",
        },
    )
    log(f"all fallback variants exhausted; wrote {failure_path}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise
