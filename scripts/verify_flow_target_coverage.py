#!/usr/bin/env python3
"""Verify that SMART cache splits produce at least one Flow target per sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.smart.datamodules.scalable_datamodule import build_train_agent_target_builder
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.smart.tokens.flow_token_processor import (
    FLOW_CONTEXT_TOKEN_COUNT,
    FLOW_TRAIN_ANCHOR_COUNT,
    FlowTokenProcessor,
)


DEFAULT_EXPERIMENT = "pre_bc_flow_2x4_h100"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
SPLIT_DIRS = {
    "train": "training",
    "validation": "validation",
    "test": "testing",
}

_WORKER_MODE = "train"
_WORKER_SOURCE = "raw"
_WORKER_TRAIN_RAW_MODE = "fast"
_WORKER_SIDECAR_ROOT: Path | None = None
_WORKER_TRANSFORM = None
_WORKER_PROCESSOR: FlowTokenProcessor | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan train/validation/test SMART cache splits and fail if any sample "
            "has no open-loop Flow target under the selected experiment config."
        )
    )
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRS),
        default=["train", "validation", "test"],
    )
    parser.add_argument(
        "--train-source",
        choices=("raw", "sidecar"),
        default="raw",
        help="Use precomputed sidecar payloads for train coverage when available.",
    )
    parser.add_argument(
        "--train-raw-mode",
        choices=("fast", "full"),
        default="fast",
        help=(
            "For raw train scans, 'fast' skips token matching and computes only "
            "the target masks/round-trip filter needed for coverage."
        ),
    )
    parser.add_argument(
        "--train-sidecar-dir",
        default="",
        help=(
            "Base flow_target_sidecars directory. The script appends the token "
            "processor fingerprint automatically."
        ),
    )
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 32)))
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--status-every", type=int, default=5000)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. model.model_config.foo=bar.",
    )
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
    config_dir = (Path(__file__).resolve().parents[1] / "configs").as_posix()
    overrides = [
        f"experiment={args.experiment}",
        f"paths.cache_root={args.cache_root}",
        *args.overrides,
    ]
    if args.train_sidecar_dir:
        overrides.append(
            "model.model_config.token_processor.flow_target_sidecar_dir="
            f"{args.train_sidecar_dir}"
        )
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name="run", overrides=overrides)


def _processor_kwargs(cfg) -> dict[str, Any]:
    kwargs = OmegaConf.to_container(
        cfg.model.model_config.token_processor,
        resolve=True,
    )
    if not isinstance(kwargs, dict):
        raise TypeError("model.model_config.token_processor must resolve to a dict.")
    return dict(kwargs)


def _raw_processor_kwargs(cfg) -> dict[str, Any]:
    kwargs = _processor_kwargs(cfg)
    kwargs["flow_target_sidecar_dir"] = ""
    kwargs["flow_target_sidecar_read"] = False
    kwargs["flow_target_sidecar_write"] = False
    kwargs["flow_target_sidecar_required"] = False
    return kwargs


def _fingerprinted_sidecar_root(cfg, sidecar_dir: str) -> Path:
    kwargs = _processor_kwargs(cfg)
    kwargs["flow_target_sidecar_dir"] = sidecar_dir
    kwargs["flow_target_sidecar_read"] = False
    kwargs["flow_target_sidecar_write"] = False
    processor = FlowTokenProcessor(**kwargs)
    return processor._flow_target_sidecar_root()


def _sample_paths(cache_root: str, split: str, max_samples: int) -> list[str]:
    split_dir = Path(cache_root) / SPLIT_DIRS[split]
    paths = sorted(
        path.as_posix()
        for path in split_dir.glob("*.pkl")
        if path.is_file() and not path.name.startswith(".")
    )
    if not paths:
        raise FileNotFoundError(f"No cached samples found under {split_dir}")
    if max_samples > 0:
        paths = paths[:max_samples]
    return paths


def _scenario_id_from_data(data) -> str:
    scenario_id = getattr(data, "scenario_id", None)
    if scenario_id is None and isinstance(data, dict):
        scenario_id = data.get("scenario_id")
    if isinstance(scenario_id, (list, tuple)):
        if len(scenario_id) != 1:
            raise ValueError(f"Expected one scenario id, got {scenario_id!r}")
        scenario_id = scenario_id[0]
    if scenario_id is None:
        raise KeyError("Sample does not contain scenario_id.")
    return str(scenario_id)


def _sidecar_path(root: Path, scenario_id: str) -> Path:
    safe_hash = hashlib.sha1(str(scenario_id).encode("utf-8")).hexdigest()
    return root / f"{safe_hash}.pt"


def _init_worker(
    *,
    mode: str,
    source: str,
    train_raw_mode: str,
    processor_kwargs: dict[str, Any],
    train_use_eval_agent_selection: bool,
    train_max_num: int,
    sidecar_root: str,
) -> None:
    global _WORKER_MODE
    global _WORKER_SOURCE
    global _WORKER_TRAIN_RAW_MODE
    global _WORKER_SIDECAR_ROOT
    global _WORKER_TRANSFORM
    global _WORKER_PROCESSOR

    torch.set_num_threads(1)
    _WORKER_MODE = mode
    _WORKER_SOURCE = source
    _WORKER_TRAIN_RAW_MODE = train_raw_mode
    _WORKER_SIDECAR_ROOT = Path(sidecar_root) if sidecar_root else None
    if mode == "train":
        _WORKER_TRANSFORM = build_train_agent_target_builder(
            train_max_num=train_max_num,
            train_use_eval_agent_selection=train_use_eval_agent_selection,
        )
    else:
        _WORKER_TRANSFORM = WaymoTargetBuilderVal()
    _WORKER_PROCESSOR = None
    if source == "raw":
        _WORKER_PROCESSOR = FlowTokenProcessor(**processor_kwargs)
        if mode == "train":
            _WORKER_PROCESSOR.train()
        else:
            _WORKER_PROCESSOR.eval()


def _load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _scan_sidecar_sample(path: str) -> dict[str, Any]:
    if _WORKER_SIDECAR_ROOT is None:
        raise RuntimeError("sidecar source requires a sidecar root.")
    data = _load_pickle(path)
    scenario_id = _scenario_id_from_data(data)
    sidecar_path = _sidecar_path(_WORKER_SIDECAR_ROOT, scenario_id)
    if not sidecar_path.exists():
        return {
            "path": path,
            "scenario_id": scenario_id,
            "has_target": False,
            "target_rows": 0,
            "valid_steps": 0,
            "error": f"missing sidecar: {sidecar_path}",
        }
    try:
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(sidecar_path, map_location="cpu")
    agent_payload = payload["agent"]
    loss_mask = agent_payload["flow_train_loss_mask"].to(dtype=torch.bool)
    target_rows = int(loss_mask.shape[0])
    valid_steps = int(loss_mask.long().sum().item())
    return {
        "path": path,
        "scenario_id": scenario_id,
        "has_target": bool(valid_steps > 0),
        "target_rows": target_rows,
        "valid_steps": valid_steps,
        "error": "",
    }


def _scan_raw_sample(path: str) -> dict[str, Any]:
    if _WORKER_TRANSFORM is None or _WORKER_PROCESSOR is None:
        raise RuntimeError("raw source worker was not initialized.")
    if _WORKER_MODE == "train" and _WORKER_TRAIN_RAW_MODE == "fast":
        return _scan_fast_train_raw_sample(path)
    return _scan_full_raw_sample(path)


def _scan_full_raw_sample(path: str) -> dict[str, Any]:
    if _WORKER_TRANSFORM is None or _WORKER_PROCESSOR is None:
        raise RuntimeError("raw source worker was not initialized.")
    with torch.no_grad():
        data = _WORKER_TRANSFORM(_load_pickle(path))
        scenario_id = _scenario_id_from_data(data)
        data = Batch.from_data_list([data])
        _tokenized_map, tokenized_agent = _WORKER_PROCESSOR(data)
        if _WORKER_MODE == "train":
            loss_mask = tokenized_agent["flow_train_loss_mask"].to(dtype=torch.bool)
            target_rows = int(loss_mask.shape[0])
            valid_steps = int(loss_mask.long().sum().item())
            has_target = bool(valid_steps > 0)
        else:
            clean_norm = tokenized_agent["flow_eval_clean_norm"]
            target_rows = int(clean_norm.shape[0])
            valid_steps = target_rows
            has_target = bool(target_rows > 0)
    return {
        "path": path,
        "scenario_id": scenario_id,
        "has_target": has_target,
        "target_rows": target_rows,
        "valid_steps": valid_steps,
        "error": "",
    }


def _coarse_segment_valid(processor: FlowTokenProcessor, valid: torch.Tensor) -> torch.Tensor:
    n_agent, n_step = valid.shape
    device = valid.device
    coarse_end_steps = torch.arange(processor.shift, n_step, processor.shift, device=device)
    if int(coarse_end_steps.numel()) == 0:
        return valid.new_zeros((n_agent, 0))
    coarse_start_steps = coarse_end_steps - processor.shift
    window_offsets = torch.arange(processor.shift + 1, device=device)
    segment_step_index = coarse_start_steps.unsqueeze(1) + window_offsets.unsqueeze(0)
    return valid[:, segment_step_index].all(dim=-1)


def _scan_fast_train_raw_sample(path: str) -> dict[str, Any]:
    if _WORKER_TRANSFORM is None or _WORKER_PROCESSOR is None:
        raise RuntimeError("raw source worker was not initialized.")
    if not bool(_WORKER_PROCESSOR.use_kinematic_control_flow):
        return _scan_full_raw_sample(path)

    with torch.no_grad():
        data = _WORKER_TRANSFORM(_load_pickle(path))
        scenario_id = _scenario_id_from_data(data)
        valid = data["agent"]["valid_mask"].clone()
        pos = data["agent"]["position"][..., :2].contiguous().clone()
        heading = data["agent"]["heading"].clone()
        vel = data["agent"]["velocity"].clone()

        heading = _WORKER_PROCESSOR._clean_heading(valid, heading)
        valid, pos, heading, _vel = _WORKER_PROCESSOR._extrapolate_agent_to_prev_token_step(
            valid,
            pos,
            heading,
            vel,
        )
        train_mask = (
            data["agent"]["train_mask"].bool()
            if "train_mask" in data["agent"]
            else torch.ones(valid.shape[0], device=valid.device, dtype=torch.bool)
        )
        tokenized_agent = {
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
        }
        ctx_valid = _coarse_segment_valid(_WORKER_PROCESSOR, valid)[
            :, :FLOW_CONTEXT_TOKEN_COUNT
        ].contiguous()
        raw_current_steps = [
            _WORKER_PROCESSOR.shift * (anchor_idx + 2)
            for anchor_idx in range(FLOW_TRAIN_ANCHOR_COUNT)
        ]
        tokenized_agent = _WORKER_PROCESSOR._build_kinematic_flow_train_targets_batched(
            tokenized_agent=tokenized_agent,
            pos=pos,
            heading=heading,
            valid=valid,
            train_mask=train_mask,
            ctx_valid=ctx_valid,
            raw_current_steps=raw_current_steps,
            dtype=pos.dtype,
            device=pos.device,
        )
        loss_mask = tokenized_agent["flow_train_loss_mask"].to(dtype=torch.bool)
        target_rows = int(loss_mask.shape[0])
        valid_steps = int(loss_mask.long().sum().item())
    return {
        "path": path,
        "scenario_id": scenario_id,
        "has_target": bool(valid_steps > 0),
        "target_rows": target_rows,
        "valid_steps": valid_steps,
        "error": "",
    }


def _scan_one(path: str) -> dict[str, Any]:
    try:
        if _WORKER_SOURCE == "sidecar":
            return _scan_sidecar_sample(path)
        return _scan_raw_sample(path)
    except Exception as exc:  # noqa: BLE001 - report the failing sample.
        return {
            "path": path,
            "scenario_id": "",
            "has_target": False,
            "target_rows": 0,
            "valid_steps": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _iter_results(paths: list[str], workers: int, chunksize: int) -> Iterable[dict[str, Any]]:
    if workers <= 1:
        for path in paths:
            yield _scan_one(path)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_scan_one, paths, chunksize=max(1, chunksize))


def _scan_split(
    *,
    cfg,
    args: argparse.Namespace,
    split: str,
    source: str,
    sidecar_root: Path | None,
) -> dict[str, Any]:
    paths = _sample_paths(args.cache_root, split, args.max_samples_per_split)
    processor_kwargs = _raw_processor_kwargs(cfg)
    train_use_eval_agent_selection = bool(cfg.data.train_use_eval_agent_selection)
    train_max_num = int(cfg.data.train_max_num)
    started = time.time()

    _init_worker(
        mode="train" if split == "train" else "eval",
        source=source,
        train_raw_mode=args.train_raw_mode,
        processor_kwargs=processor_kwargs,
        train_use_eval_agent_selection=train_use_eval_agent_selection,
        train_max_num=train_max_num,
        sidecar_root=sidecar_root.as_posix() if sidecar_root is not None else "",
    )

    total = 0
    targetless = 0
    errors = 0
    target_rows = 0
    valid_steps = 0
    examples: list[dict[str, str]] = []
    for result in _iter_results(paths, args.workers, args.chunksize):
        total += 1
        if result["error"]:
            errors += 1
        if not result["has_target"]:
            targetless += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "path": result["path"],
                        "scenario_id": result["scenario_id"],
                        "error": result["error"],
                    }
                )
        target_rows += int(result["target_rows"])
        valid_steps += int(result["valid_steps"])
        if args.status_every > 0 and total % int(args.status_every) == 0:
            elapsed = time.time() - started
            rate = total / max(elapsed, 1.0e-6)
            print(
                f"[coverage] split={split} source={source} "
                f"{total}/{len(paths)} targetless={targetless} errors={errors} "
                f"rate={rate:.1f} samples/s",
                flush=True,
            )

    elapsed = time.time() - started
    summary = {
        "split": split,
        "source": source,
        "samples": total,
        "targetless_samples": targetless,
        "errors": errors,
        "target_rows": target_rows,
        "valid_steps": valid_steps,
        "seconds": elapsed,
        "samples_per_second": total / max(elapsed, 1.0e-6),
        "examples": examples,
    }
    print("[coverage] " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    cfg = _compose_cfg(args)
    sidecar_root = None
    if args.train_source == "sidecar":
        sidecar_dir = args.train_sidecar_dir or str(
            cfg.model.model_config.token_processor.flow_target_sidecar_dir or ""
        )
        if not sidecar_dir:
            raise SystemExit("--train-source=sidecar requires --train-sidecar-dir.")
        sidecar_root = _fingerprinted_sidecar_root(cfg, sidecar_dir)
        print(f"[coverage] train sidecar root={sidecar_root}", flush=True)

    summaries = []
    for split in args.splits:
        source = args.train_source if split == "train" else "raw"
        summaries.append(
            _scan_split(
                cfg=cfg,
                args=args,
                split=split,
                source=source,
                sidecar_root=sidecar_root if split == "train" else None,
            )
        )

    payload = {
        "experiment": args.experiment,
        "cache_root": args.cache_root,
        "summaries": summaries,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    targetless_failures = sum(int(item["targetless_samples"]) for item in summaries)
    error_failures = sum(int(item["errors"]) for item in summaries)
    if targetless_failures > 0 or error_failures > 0:
        raise SystemExit(
            "Flow target coverage check failed: "
            f"targetless={targetless_failures}, errors={error_failures}"
        )


if __name__ == "__main__":
    main()
