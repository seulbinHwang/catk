from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unimm_anchors import DEFAULT_NUM_WORKERS, compute_context_thresholds_from_cache
from src.unimm.anchors import AGENT_TYPE_NAMES, load_anchor_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute UniMM posterior error thresholds for an existing anchor bank "
            "over the full training context distribution."
        )
    )
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--anchor-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-steps", type=int, default=5)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-start-step", type=int, default=10)
    parser.add_argument("--threshold-end-step", type=int, default=85)
    parser.add_argument("--threshold-step", type=int, default=5)
    parser.add_argument("--heading-weight", type=float, default=1.0)
    parser.add_argument("--threshold-row-chunk-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--collect-file-chunk-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for threshold computation: auto, cpu, cuda, cuda:0, ...",
    )
    args = parser.parse_args()

    if args.match_steps < 1:
        raise ValueError("--match-steps must be positive")
    if args.threshold_step < 1:
        raise ValueError("--threshold-step must be positive")
    if args.threshold_end_step < args.threshold_start_step:
        raise ValueError("--threshold-end-step must be >= --threshold-start-step")
    if args.threshold_row_chunk_size < 1:
        raise ValueError("--threshold-row-chunk-size must be positive")
    if args.collect_file_chunk_size < 1:
        raise ValueError("--collect-file-chunk-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"using threshold device: {device}")

    anchor_file = Path(args.anchor_file)
    anchors, _, payload = load_anchor_file(anchor_file)
    anchors_by_name = {
        name: anchors[type_idx]
        for type_idx, name in enumerate(AGENT_TYPE_NAMES)
    }
    threshold_context_steps = list(
        range(args.threshold_start_step, args.threshold_end_step + 1, args.threshold_step)
    )
    thresholds, threshold_counts = compute_context_thresholds_from_cache(
        train_cache_dir=Path(args.train_cache_dir),
        anchors_by_name=anchors_by_name,
        match_steps=args.match_steps,
        context_steps=threshold_context_steps,
        quantile=args.threshold_quantile,
        heading_weight=args.heading_weight,
        row_chunk_size=args.threshold_row_chunk_size,
        device=device,
        num_workers=args.num_workers,
        collect_file_chunk_size=args.collect_file_chunk_size,
    )

    if not isinstance(payload, dict):
        raise TypeError(f"anchor payload must be a dict, got {type(payload)!r}")
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "threshold_source": "training_cache_context_starts",
            "threshold_context_steps": threshold_context_steps,
            "threshold_counts": threshold_counts,
            "threshold_quantile": float(args.threshold_quantile),
            "threshold_match_steps": int(args.match_steps),
            "threshold_heading_weight": float(args.heading_weight),
            "threshold_distance_metric": "mean(pos_sq + heading_weight * wrap_angle(heading_diff)^2)",
        }
    )
    payload["posterior_error_threshold"] = thresholds
    payload["metadata"] = metadata

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tmp_output.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp_output.replace(output)
    print(f"saved {output}")
    print(f"posterior_error_threshold={thresholds}")


if __name__ == "__main__":
    main()
