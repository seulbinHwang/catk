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


def test_latest_prefix_v100_launcher_uses_distinct_task_name() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "scripts"
                / "launch_pre_bc_flow_control_v100x47_prefix_default_noslip_latest_static_pods.py"
            ),
            "--dry-run",
            "--replace",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "launch_pre_bc_flow_control_v100x47_prefix_default_noslip_static_pods.py" in result.stdout
    assert "--branch semi_control_stable_original" in result.stdout
    assert (
        "flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_"
        "roundtrip05_lr6e-4_bs4_stable_original_latest"
    ) in result.stdout
    assert (
        "catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix-"
        "stable-original-latest"
    ) in result.stdout


def test_oom_retry_preflight_builds_metadata_on_all_pods() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "h100x4_multinode_pretrain_with_oom_retry.sh"
    ).read_text()

    assert "prebuild_memory_balance_metadata_for_pod()" in script
    assert "copy_memory_balance_metadata_from_master()" in script
    assert "validate_memory_balance_metadata_on_pod()" in script
    assert 'for pod in "${POD_ARRAY[@]}"; do' in script
    assert 'copy_memory_balance_metadata_from_master "$pod"' in script
    assert 'validate_memory_balance_metadata_on_pod "$pod"' in script
    assert "memory-balance metadata preflight ready on all" in script
