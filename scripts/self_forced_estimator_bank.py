#!/usr/bin/env python3
"""Manage self-forced generated-estimator warmup bank artifacts.

The bank stores only ``self_forced_generated_estimator`` weights. It never
stores trainer/optimizer state, so loading a bank item skips estimator warmup
without accidentally resuming the generator, epoch, or optimizer schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


GENERATED_ESTIMATOR_PREFIX = "self_forced_generated_estimator."
MANIFEST_NAME = "manifest.json"


def _normalize_lr(value: str | float) -> str:
    lr = float(value)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be a positive finite value, got {value!r}.")
    return f"{lr:.8g}"


def _entry_relpath(*, warmup_epochs: int, lr: str | float) -> str:
    lr_key = _normalize_lr(lr).replace("+", "")
    return f"warmup_{int(warmup_epochs)}/lr_{lr_key}/generated_estimator.pt"


def _metadata_relpath(*, warmup_epochs: int, lr: str | float) -> str:
    lr_key = _normalize_lr(lr).replace("+", "")
    return f"warmup_{int(warmup_epochs)}/lr_{lr_key}/metadata.json"


def _load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_generated_estimator_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict.")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise TypeError("checkpoint state_dict must be a dict.")
    if any(isinstance(key, str) and key.startswith(GENERATED_ESTIMATOR_PREFIX) for key in raw_state):
        return {
            key[len(GENERATED_ESTIMATOR_PREFIX) :]: value.detach().cpu()
            for key, value in raw_state.items()
            if isinstance(key, str)
            and key.startswith(GENERATED_ESTIMATOR_PREFIX)
            and torch.is_tensor(value)
        }
    return {
        key: value.detach().cpu()
        for key, value in raw_state.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }


def _checkpoint_epoch_step(checkpoint: Any) -> tuple[int | None, int | None]:
    if not isinstance(checkpoint, dict):
        return None, None
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    return (
        int(epoch) if epoch is not None else None,
        int(global_step) if global_step is not None else None,
    )


def _write_entry(
    *,
    source_ckpt: Path,
    output: Path,
    warmup_epochs: int,
    lr: str,
    pretrain_artifact: str,
    source_task: str,
    source_pod: str,
) -> None:
    checkpoint = _load_torch(source_ckpt)
    state_dict = _extract_generated_estimator_state(checkpoint)
    if not state_dict:
        raise RuntimeError(f"No generated-estimator tensors found in {source_ckpt}.")
    epoch, global_step = _checkpoint_epoch_step(checkpoint)
    metadata = {
        "format_version": 1,
        "state_dict_kind": "self_forced_generated_estimator",
        "state_dict_prefix": "",
        "warmup_epochs": int(warmup_epochs),
        "lr": _normalize_lr(lr),
        "pretrain_artifact": pretrain_artifact,
        "source_task": source_task,
        "source_pod": source_pod,
        "source_ckpt": source_ckpt.as_posix(),
        "source_epoch": epoch,
        "source_global_step": global_step,
        "num_tensors": len(state_dict),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": metadata}, output)
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return {"format_version": 1, "entries": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _merge_entry_into_bank(
    *,
    bank_root: Path,
    entry_path: Path,
    warmup_epochs: int,
    lr: str,
) -> None:
    rel_entry = _entry_relpath(warmup_epochs=warmup_epochs, lr=lr)
    rel_metadata = _metadata_relpath(warmup_epochs=warmup_epochs, lr=lr)
    target_entry = bank_root / rel_entry
    target_metadata = bank_root / rel_metadata
    target_entry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry_path, target_entry)
    metadata_source = entry_path.with_suffix(entry_path.suffix + ".metadata.json")
    if metadata_source.is_file():
        shutil.copy2(metadata_source, target_metadata)

    manifest = _read_manifest(bank_root)
    entries = [
        entry
        for entry in manifest.get("entries", [])
        if not (
            int(entry.get("warmup_epochs", -1)) == int(warmup_epochs)
            and _normalize_lr(entry.get("lr", "nan")) == _normalize_lr(lr)
        )
    ]
    entry_meta = {
        "warmup_epochs": int(warmup_epochs),
        "lr": _normalize_lr(lr),
        "path": rel_entry,
        "metadata_path": rel_metadata,
    }
    if target_metadata.is_file():
        entry_meta.update(json.loads(target_metadata.read_text(encoding="utf-8")))
    entries.append(entry_meta)
    manifest["format_version"] = 1
    manifest["entries"] = sorted(
        entries,
        key=lambda item: (int(item.get("warmup_epochs", 0)), str(item.get("lr", ""))),
    )
    _write_manifest(bank_root, manifest)


def _artifact_ref(name: str, entity: str, project: str) -> str:
    if "/" in name:
        return name
    if ":" in name:
        return f"{entity}/{project}/{name}"
    return f"{entity}/{project}/{name}:latest"


def _copy_entry_metadata(source: Path, output: Path) -> None:
    """Copy sidecar metadata for both extracted entries and bank entries."""
    metadata_candidates = [
        source.with_suffix(source.suffix + ".metadata.json"),
        source.parent / "metadata.json",
    ]
    for metadata_source in metadata_candidates:
        if metadata_source.is_file():
            shutil.copy2(metadata_source, output.with_suffix(output.suffix + ".metadata.json"))
            return


def command_extract(args: argparse.Namespace) -> None:
    _write_entry(
        source_ckpt=Path(args.source_ckpt),
        output=Path(args.output),
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        pretrain_artifact=args.pretrain_artifact,
        source_task=args.source_task,
        source_pod=args.source_pod,
    )
    print(f"wrote generated-estimator entry: {args.output}")


def command_download(args: argparse.Namespace) -> None:
    import wandb

    artifact_ref = _artifact_ref(args.artifact, args.entity, args.project)
    output = Path(args.output)
    rel_entry = _entry_relpath(warmup_epochs=args.warmup_epochs, lr=args.lr)
    with tempfile.TemporaryDirectory(prefix="catk_estimator_bank_") as tmp_dir:
        artifact = wandb.Api().artifact(artifact_ref, type="generated_estimator_bank")
        root = Path(artifact.download(root=tmp_dir))
        source = root / rel_entry
        if not source.is_file():
            raise FileNotFoundError(
                f"No bank entry for warmup={args.warmup_epochs}, lr={_normalize_lr(args.lr)} "
                f"in {artifact_ref}; expected {rel_entry}."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        _copy_entry_metadata(source, output)
    print(f"downloaded generated-estimator entry: {output}")


def _find_best_entry(root: Path, *, warmup_epochs: int, lr: str | float) -> tuple[int, Path]:
    """Find the exact or nearest lower warmup entry from a downloaded bank."""
    target_warmup = int(warmup_epochs)
    lr_key = _normalize_lr(lr)
    manifest = _read_manifest(root)
    best: tuple[int, Path] | None = None
    for entry in manifest.get("entries", []):
        try:
            entry_warmup = int(entry.get("warmup_epochs"))
            entry_lr = _normalize_lr(entry.get("lr"))
        except Exception:
            continue
        if entry_lr != lr_key or entry_warmup > target_warmup:
            continue
        entry_path = root / str(entry.get("path", ""))
        if not entry_path.is_file():
            continue
        if best is None or entry_warmup > best[0]:
            best = (entry_warmup, entry_path)

    if best is not None:
        return best

    for candidate_warmup in range(target_warmup, -1, -1):
        entry_path = root / _entry_relpath(warmup_epochs=candidate_warmup, lr=lr_key)
        if entry_path.is_file():
            return candidate_warmup, entry_path

    raise FileNotFoundError(
        f"No bank entry for lr={lr_key} with warmup <= {target_warmup}."
    )


def command_resolve(args: argparse.Namespace) -> None:
    import wandb

    requested_warmup = int(args.warmup_epochs)
    artifact_ref = _artifact_ref(args.artifact, args.entity, args.project)
    output = Path(args.output)
    env_output = Path(args.env_output) if args.env_output else None
    with tempfile.TemporaryDirectory(prefix="catk_estimator_bank_") as tmp_dir:
        artifact = wandb.Api().artifact(artifact_ref, type="generated_estimator_bank")
        root = Path(artifact.download(root=tmp_dir))
        resolved_warmup, source = _find_best_entry(
            root,
            warmup_epochs=requested_warmup,
            lr=args.lr,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        _copy_entry_metadata(source, output)

    remaining_warmup = max(0, requested_warmup - resolved_warmup)
    exact = int(resolved_warmup == requested_warmup)
    payload = {
        "requested_warmup_epochs": requested_warmup,
        "resolved_warmup_epochs": resolved_warmup,
        "remaining_warmup_epochs": remaining_warmup,
        "exact": bool(exact),
        "output": output.as_posix(),
    }
    if env_output is not None:
        env_output.parent.mkdir(parents=True, exist_ok=True)
        env_output.write_text(
            "\n".join(
                [
                    "ESTIMATOR_WARMUP_BANK_RESOLVED=1",
                    f"ESTIMATOR_WARMUP_BANK_REQUESTED_WARMUP={requested_warmup}",
                    f"ESTIMATOR_WARMUP_BANK_RESOLVED_WARMUP={resolved_warmup}",
                    f"ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP={remaining_warmup}",
                    f"ESTIMATOR_WARMUP_BANK_EXACT={exact}",
                    f"ESTIMATOR_WARMUP_BANK_OUTPUT={output.as_posix()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))


def _manifest_has_entries(
    manifest: dict[str, Any],
    entries: list[tuple[int, str]],
) -> bool:
    existing: set[tuple[int, str]] = set()
    for entry in manifest.get("entries", []):
        try:
            existing.add((int(entry.get("warmup_epochs", -1)), _normalize_lr(entry.get("lr"))))
        except Exception:
            continue
    return all((warmup, _normalize_lr(lr)) in existing for warmup, lr in entries)


def _download_bank_or_empty(
    *,
    artifact_ref: str,
    bank_root: Path,
) -> None:
    import wandb

    try:
        artifact = wandb.Api().artifact(artifact_ref, type="generated_estimator_bank")
        artifact.download(root=bank_root)
    except Exception as exc:
        message = str(exc).lower()
        missing = any(
            marker in message
            for marker in (
                "not found",
                "does not exist",
                "unable to find artifact",
                "artifact not found",
                "404",
            )
        )
        if not missing:
            raise
        print(f"starting new generated-estimator bank: {artifact_ref} ({exc})")
        bank_root.mkdir(parents=True, exist_ok=True)


def command_upsert(args: argparse.Namespace) -> None:
    import wandb

    artifact_name = args.artifact_name
    artifact_ref = _artifact_ref(artifact_name, args.entity, args.project)
    parsed_entries: list[tuple[int, str, Path]] = []
    for entry_spec in args.entry:
        warmup_text, lr_text, path_text = entry_spec.split(":", 2)
        parsed_entries.append((int(warmup_text), lr_text, Path(path_text)))

    expected_entries = [(warmup, lr) for warmup, lr, _ in parsed_entries]
    for attempt in range(int(args.merge_retries) + 1):
        with tempfile.TemporaryDirectory(prefix="catk_estimator_bank_") as tmp_dir:
            bank_root = Path(tmp_dir) / "bank"
            _download_bank_or_empty(artifact_ref=artifact_ref, bank_root=bank_root)

            for warmup, lr, entry_path in parsed_entries:
                _merge_entry_into_bank(
                    bank_root=bank_root,
                    entry_path=entry_path,
                    warmup_epochs=warmup,
                    lr=lr,
                )

            run = wandb.init(
                entity=args.entity,
                project=args.project,
                name=args.run_name if attempt == 0 else f"{args.run_name}_merge_retry{attempt}",
                job_type="generated_estimator_bank_upload",
            )
            metadata = _read_manifest(bank_root)
            new_artifact = wandb.Artifact(
                name=artifact_name,
                type="generated_estimator_bank",
                metadata=metadata,
            )
            new_artifact.add_dir(bank_root.as_posix())
            aliases = args.alias or ["latest"]
            run.log_artifact(new_artifact, aliases=aliases)
            run.finish()

        if int(args.merge_retries) <= 0:
            break
        time.sleep(float(args.merge_wait_seconds))
        with tempfile.TemporaryDirectory(prefix="catk_estimator_bank_verify_") as verify_tmp:
            verify_root = Path(verify_tmp) / "bank"
            _download_bank_or_empty(artifact_ref=artifact_ref, bank_root=verify_root)
            if _manifest_has_entries(_read_manifest(verify_root), expected_entries):
                break
        print(
            "generated-estimator bank latest manifest missed an upserted entry; "
            f"retrying merge ({attempt + 1}/{args.merge_retries})."
        )

    print(f"uploaded generated-estimator bank artifact: {args.entity}/{args.project}/{artifact_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--source-ckpt", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--warmup-epochs", type=int, required=True)
    extract.add_argument("--lr", required=True)
    extract.add_argument("--pretrain-artifact", default="")
    extract.add_argument("--source-task", default="")
    extract.add_argument("--source-pod", default="")
    extract.set_defaults(func=command_extract)

    download = subparsers.add_parser("download")
    download.add_argument("--artifact", required=True)
    download.add_argument("--warmup-epochs", type=int, required=True)
    download.add_argument("--lr", required=True)
    download.add_argument("--output", required=True)
    download.add_argument("--entity", default="jksg01019-naver-labs")
    download.add_argument("--project", default="SMART-FLOW")
    download.set_defaults(func=command_download)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--artifact", required=True)
    resolve.add_argument("--warmup-epochs", type=int, required=True)
    resolve.add_argument("--lr", required=True)
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--env-output", default="")
    resolve.add_argument("--entity", default="jksg01019-naver-labs")
    resolve.add_argument("--project", default="SMART-FLOW")
    resolve.set_defaults(func=command_resolve)

    upsert = subparsers.add_parser("upsert")
    upsert.add_argument("--artifact-name", required=True)
    upsert.add_argument("--entry", action="append", required=True, help="<warmup>:<lr>:<entry.pt>")
    upsert.add_argument("--entity", default="jksg01019-naver-labs")
    upsert.add_argument("--project", default="SMART-FLOW")
    upsert.add_argument("--run-name", default="generated_estimator_warmup_bank_upload")
    upsert.add_argument("--alias", action="append")
    upsert.add_argument("--merge-retries", type=int, default=2)
    upsert.add_argument("--merge-wait-seconds", type=float, default=2.0)
    upsert.set_defaults(func=command_upsert)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
