from __future__ import annotations

import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class ScalePartial:
    sq_sum: tuple[float, ...]
    count: tuple[int, ...]


def _agent_field(agent_store: Any, key: str) -> Any:
    if isinstance(agent_store, dict):
        return agent_store[key]
    try:
        return agent_store[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(agent_store, key)


def compute_file_partial(
    path: str,
    *,
    num_historical_steps: int,
    num_agent_types: int,
) -> ScalePartial:
    torch.set_num_threads(1)
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    agent = data["agent"]

    position = _agent_field(agent, "position")
    valid_mask = _agent_field(agent, "valid_mask")
    agent_type = _agent_field(agent, "type").to(dtype=torch.long).clamp(0, num_agent_types - 1)

    current_index = max(0, int(num_historical_steps) - 1)
    future_start = int(num_historical_steps)
    if position.shape[0] == 0 or position.shape[1] <= future_start:
        return ScalePartial((0.0,) * num_agent_types, (0,) * num_agent_types)

    current_valid = valid_mask[:, current_index]
    future_valid = valid_mask[:, future_start:] & current_valid[:, None]
    if not bool(future_valid.any()):
        return ScalePartial((0.0,) * num_agent_types, (0,) * num_agent_types)

    current_pos = position[:, current_index, :2]
    future_pos = position[:, future_start:, :2]
    square_distance = (future_pos - current_pos[:, None, :]).square().sum(dim=-1)

    sq_sum: list[float] = []
    count: list[int] = []
    for type_index in range(num_agent_types):
        type_valid = future_valid & (agent_type[:, None] == type_index)
        if bool(type_valid.any()):
            sq_sum.append(float(square_distance[type_valid].sum(dtype=torch.float64).item()))
            count.append(int(type_valid.sum().item()))
        else:
            sq_sum.append(0.0)
            count.append(0)
    return ScalePartial(tuple(sq_sum), tuple(count))


def _compute_file_partial_star(args: tuple[str, int, int]) -> ScalePartial:
    path, num_historical_steps, num_agent_types = args
    return compute_file_partial(
        path,
        num_historical_steps=num_historical_steps,
        num_agent_types=num_agent_types,
    )


def _iter_cache_files(train_dir: Path) -> list[str]:
    return [
        path.as_posix()
        for path in sorted(train_dir.glob("*.pkl"))
        if path.is_file() and not path.name.startswith(".")
    ]


def _merge_partials(partials: Iterable[ScalePartial], *, num_agent_types: int) -> ScalePartial:
    sq_sum = [0.0] * num_agent_types
    count = [0] * num_agent_types
    for partial in partials:
        for type_index in range(num_agent_types):
            sq_sum[type_index] += float(partial.sq_sum[type_index])
            count[type_index] += int(partial.count[type_index])
    return ScalePartial(tuple(sq_sum), tuple(count))


def compute_type_scale(
    train_dir: Path,
    *,
    num_historical_steps: int = 11,
    num_agent_types: int = 3,
    num_workers: int = 0,
) -> dict[str, Any]:
    files = _iter_cache_files(train_dir)
    if not files:
        raise FileNotFoundError(f"No .pkl files found under {train_dir}.")

    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            partial_iter = executor.map(
                _compute_file_partial_star,
                ((path, num_historical_steps, num_agent_types) for path in files),
                chunksize=64,
            )
            merged = _merge_partials(partial_iter, num_agent_types=num_agent_types)
    else:
        merged = _merge_partials(
            (
                compute_file_partial(
                    path,
                    num_historical_steps=num_historical_steps,
                    num_agent_types=num_agent_types,
                )
                for path in files
            ),
            num_agent_types=num_agent_types,
        )

    scale: list[float] = []
    for sq_sum, count in zip(merged.sq_sum, merged.count):
        scale.append(math.sqrt(float(sq_sum) / float(count)) if count > 0 else 1.0)

    return {
        "train_dir": train_dir.as_posix(),
        "num_files": len(files),
        "num_historical_steps": int(num_historical_steps),
        "num_agent_types": int(num_agent_types),
        "sq_sum": list(merged.sq_sum),
        "count": list(merged.count),
        "scale": scale,
        "agent_type_order": ["vehicle", "pedestrian", "cyclist"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fixed WOSAC CPD/CES type scales from SMART training cache."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/workspace/womd_v1_3/SMART_cache/training"),
        help="SMART training cache directory containing scenario .pkl files.",
    )
    parser.add_argument("--num-historical-steps", type=int, default=11)
    parser.add_argument("--num-agent-types", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    result = compute_type_scale(
        args.train_dir,
        num_historical_steps=args.num_historical_steps,
        num_agent_types=args.num_agent_types,
        num_workers=args.num_workers,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
