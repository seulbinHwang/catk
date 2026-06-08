#!/usr/bin/env python3
"""Download the newest W&B ``epoch_last.ckpt`` artifact for a run name."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="jksg01019-naver-labs")
    parser.add_argument("--project", default="SMART-FLOW")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--artifact-prefix",
        default="epoch-last",
        help="Artifact name prefix used by EpochLastCheckpointCallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=90)
    project_path = f"{args.entity}/{args.project}"
    runs = list(api.runs(project_path, filters={"display_name": args.run_name}))
    if not runs:
        raise SystemExit(f"No W&B run found with display_name={args.run_name!r}.")

    candidates = []
    for run in runs:
        for artifact in run.logged_artifacts():
            aliases = set(artifact.aliases or [])
            if artifact.type != "model":
                continue
            if not artifact.name.startswith(args.artifact_prefix):
                continue
            if "latest" not in aliases and "epoch_last" not in aliases:
                continue
            candidates.append((artifact.created_at, run, artifact))

    if not candidates:
        run_ids = ", ".join(sorted(run.id for run in runs))
        raise SystemExit(
            "No latest epoch_last model artifact found for matching runs: "
            f"{run_ids}"
        )

    _, run, artifact = max(candidates, key=lambda item: item[0])
    output_root = Path(args.output_dir).expanduser().resolve()
    artifact_dir = output_root / f"{run.id}_{artifact.version}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir = Path(artifact.download(root=artifact_dir.as_posix()))
    ckpt_path = downloaded_dir / "epoch_last.ckpt"
    if not ckpt_path.is_file():
        files = ", ".join(sorted(path.name for path in downloaded_dir.iterdir()))
        raise SystemExit(
            f"Downloaded artifact does not contain epoch_last.ckpt. Files: {files}"
        )

    metadata = artifact.metadata or {}
    print(f"RUN_ID={run.id}")
    print(f"RUN_STATE={run.state}")
    print(f"ARTIFACT={artifact.name}")
    print(f"ARTIFACT_VERSION={artifact.version}")
    print(f"ARTIFACT_CREATED_AT={artifact.created_at}")
    print(f"CHECKPOINT_EPOCH={metadata.get('epoch', '')}")
    print(f"CHECKPOINT_GLOBAL_STEP={metadata.get('global_step', '')}")
    print(f"CKPT_PATH={ckpt_path}")


if __name__ == "__main__":
    main()
