from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _validate_file(path: Path, *, expected_rollouts: int, expected_steps: int) -> tuple[int, int, int]:
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"{path}: expected dict")
    for key in ("rollout_pose", "agent_id"):
        if key not in cache:
            raise KeyError(f"{path}: missing {key!r}")
    rollout = cache["rollout_pose"]
    if rollout.dim() != 4 or rollout.shape[-1] != 4:
        raise ValueError(f"{path}: rollout_pose must be [R,T,N,4], got {tuple(rollout.shape)}")
    if rollout.shape[0] < expected_rollouts:
        raise ValueError(
            f"{path}: expected at least {expected_rollouts} rollouts, got {rollout.shape[0]}"
        )
    if rollout.shape[1] != expected_steps:
        raise ValueError(f"{path}: expected {expected_steps} future steps, got {rollout.shape[1]}")
    if cache["agent_id"].numel() != rollout.shape[2]:
        raise ValueError(f"{path}: agent_id length and rollout N mismatch")
    if "valid_mask" in cache and cache["valid_mask"].numel() != rollout.shape[2]:
        raise ValueError(f"{path}: valid_mask length and rollout N mismatch")
    if not torch.isfinite(rollout.float()).all():
        raise ValueError(f"{path}: rollout_pose has non-finite values")
    return int(rollout.shape[0]), int(rollout.shape[1]), int(rollout.shape[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate self-forced GAN teacher rollout cache files.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--expected-rollouts", type=int, default=32)
    parser.add_argument("--expected-steps", type=int, default=20)
    args = parser.parse_args()

    files = sorted(args.cache_root.glob("*.pt"))[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .pt cache files found under {args.cache_root}")
    total_agents = 0
    for path in files:
        _, _, n_agent = _validate_file(
            path,
            expected_rollouts=args.expected_rollouts,
            expected_steps=args.expected_steps,
        )
        total_agents += n_agent
    print(
        f"validated_files={len(files)} avg_agents={total_agents / max(len(files), 1):.2f} "
        f"cache_root={args.cache_root}"
    )


if __name__ == "__main__":
    main()
