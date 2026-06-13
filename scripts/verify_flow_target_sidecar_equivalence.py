#!/usr/bin/env python3
"""Verify that Flow target sidecars reproduce online token processing."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.datamodules.scalable_datamodule import (
    FlowTargetSidecarCollater,
    FlowTargetSidecarPayloadTransform,
    SequentialTransform,
    build_train_agent_target_builder,
)
from src.smart.datasets import MultiDataset
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--sidecar-dir", default="")
    parser.add_argument("--experiment", default="pre_bc_flow_2x4_h100")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--preload-sidecar", action="store_true")
    return parser.parse_known_args()


def make_processor(cfg, *, sidecar_dir: str, read: bool, write: bool, required: bool) -> FlowTokenProcessor:
    token_processor_cfg = OmegaConf.to_container(
        cfg.model.model_config.token_processor,
        resolve=True,
    )
    token_processor_cfg.update(
        {
            "flow_target_sidecar_dir": sidecar_dir,
            "flow_target_sidecar_read": read,
            "flow_target_sidecar_write": write,
            "flow_target_sidecar_required": required,
        }
    )
    processor = FlowTokenProcessor(**token_processor_cfg)
    processor.train()
    return processor


def assert_tensor_equal(path: str, actual: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{path} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{path} dtype mismatch: {actual.dtype} != {expected.dtype}")
    if actual.dtype.is_floating_point:
        if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
            max_abs = (actual - expected).abs().max().item()
            raise AssertionError(f"{path} values differ: max_abs={max_abs}")
    else:
        if not torch.equal(actual, expected):
            mismatch = int((actual != expected).sum().item())
            raise AssertionError(f"{path} values differ: mismatch_count={mismatch}")


def compare_tree(path: str, actual: Any, expected: Any, atol: float) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        assert_tensor_equal(path, actual.detach().cpu(), expected.detach().cpu(), atol=atol)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            raise AssertionError(
                f"{path} keys mismatch: actual_only={sorted(actual_keys - expected_keys)}, "
                f"expected_only={sorted(expected_keys - actual_keys)}"
            )
        for key in sorted(actual_keys):
            compare_tree(f"{path}.{key}", actual[key], expected[key], atol=atol)
        return
    if actual != expected:
        raise AssertionError(f"{path} mismatch: {actual!r} != {expected!r}")


def main() -> None:
    args, extra_overrides = parse_args()
    config_dir = (Path(__file__).resolve().parents[1] / "configs").as_posix()
    overrides = [
        f"experiment={args.experiment}",
        f"paths.cache_root={args.cache_root}",
    ]
    overrides.extend(extra_overrides)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="run", overrides=overrides)

    base_transform = build_train_agent_target_builder(
        train_max_num=int(cfg.data.train_max_num),
        train_use_eval_agent_selection=bool(cfg.data.train_use_eval_agent_selection),
    )
    raw_dir = Path(args.cache_root) / args.split
    base_dataset = MultiDataset(raw_dir.as_posix(), base_transform)
    num_samples = min(int(args.num_samples), len(base_dataset))

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    with tempfile.TemporaryDirectory(prefix="flow_sidecar_verify_") as tmp_dir:
        sidecar_dir = args.sidecar_dir or tmp_dir
        writer = make_processor(
            cfg,
            sidecar_dir=sidecar_dir,
            read=False,
            write=True,
            required=False,
        ).to(device)
        online = make_processor(
            cfg,
            sidecar_dir=sidecar_dir,
            read=False,
            write=False,
            required=False,
        ).to(device)
        cached = make_processor(
            cfg,
            sidecar_dir=sidecar_dir,
            read=True,
            write=False,
            required=True,
        ).to(device)

        with torch.no_grad():
            for idx in range(num_samples):
                sample = Batch.from_data_list([base_dataset[idx]]).to(device)
                writer(sample)

            reader_dataset = base_dataset
            if args.preload_sidecar:
                reader_transform = SequentialTransform(
                    base_transform,
                    FlowTargetSidecarPayloadTransform(
                        sidecar_root=writer._flow_target_sidecar_root().as_posix(),
                        required=True,
                    ),
                )
                reader_dataset = MultiDataset(raw_dir.as_posix(), reader_transform)

            subset = torch.utils.data.Subset(reader_dataset, list(range(num_samples)))
            if args.preload_sidecar:
                loader = TorchDataLoader(
                    subset,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    collate_fn=FlowTargetSidecarCollater(required=True),
                )
            else:
                loader = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False)
            checked_batches = 0
            for batch in loader:
                batch = batch.to(device)
                expected_map, expected_agent = online._compute_online(batch)
                actual_map, actual_agent = cached(batch)
                compare_tree("tokenized_map", actual_map, expected_map, atol=float(args.atol))
                compare_tree("tokenized_agent", actual_agent, expected_agent, atol=float(args.atol))
                checked_batches += 1

    print(
        f"[sidecar-verify] ok samples={num_samples} batch_size={args.batch_size} "
        f"checked_batches={checked_batches} device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
