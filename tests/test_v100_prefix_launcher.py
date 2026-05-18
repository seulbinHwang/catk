import subprocess
import sys
from pathlib import Path


def test_prefix_v100_launcher_dry_run_enables_metadata_preflight() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py"),
            "--dry-run",
            "--replace",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--memory-metadata-preflight" in result.stdout
    assert "--memory-metadata-cache-path" in result.stdout
    assert "dataset_metadata/womd_training_memory_balance_v1.pt" in result.stdout
    assert "--max-same-bs-oom-retries 3" in result.stdout
