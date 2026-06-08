#!/usr/bin/env python3
"""Patch optimizer/scheduler learning-rate fields in a Lightning checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-lr", required=True, type=float)
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _patch_scheduler_state(state: dict[str, Any], new_base_lr: float) -> float | None:
    base_lrs = state.get("base_lrs")
    last_lrs = state.get("_last_lr")
    if not base_lrs:
        return None
    old_base = float(base_lrs[0])
    if last_lrs:
        old_current = float(last_lrs[0])
    else:
        old_current = old_base
    multiplier = old_current / old_base if old_base > 0 else 1.0
    new_current = new_base_lr * multiplier
    state["base_lrs"] = [new_base_lr for _ in base_lrs]
    if last_lrs:
        state["_last_lr"] = [new_current for _ in last_lrs]
    return new_current


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {input_path}")

    checkpoint = torch.load(input_path.as_posix(), map_location="cpu", weights_only=False)
    scheduler_current_lrs: list[float] = []
    for scheduler_state in _as_list(checkpoint.get("lr_schedulers")):
        if isinstance(scheduler_state, dict):
            patched_lr = _patch_scheduler_state(scheduler_state, args.base_lr)
            if patched_lr is not None:
                scheduler_current_lrs.append(patched_lr)

    fallback_current_lr = scheduler_current_lrs[0] if scheduler_current_lrs else args.base_lr
    for optimizer_state in _as_list(checkpoint.get("optimizer_states")):
        if not isinstance(optimizer_state, dict):
            continue
        for group in optimizer_state.get("param_groups", []):
            if not isinstance(group, dict):
                continue
            group["lr"] = fallback_current_lr
            if "initial_lr" in group:
                group["initial_lr"] = args.base_lr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path.as_posix())
    print(f"INPUT={input_path}")
    print(f"OUTPUT={output_path}")
    print(f"BASE_LR={args.base_lr:.10g}")
    print(f"PATCHED_CURRENT_LR={fallback_current_lr:.10g}")


if __name__ == "__main__":
    main()
