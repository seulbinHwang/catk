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
    assert "--branch semi_control_stable" in result.stdout
    assert (
        "flow_control_space_pretrain_v100x47_prefix_default_noslip_tailprefix_"
        "roundtrip05_lr6e-4_bs4_stable_latest"
    ) in result.stdout
    assert "catk-control-pretrain-v100x47-prefix-default-noslip-tailprefix-stable-latest" in result.stdout


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
    assert "CATK_REMOTE_PYTHON" in script
    assert "${remote_python_q} tools/build_memory_balance_metadata.py" in script


def test_h100x4_h100x2_launcher_dry_run_uses_workspace_cache_and_bs20() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "scripts"
                / "launch_pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip_static_pods.py"
            ),
            "--dry-run",
            "--replace",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PODS='hsb-npc-training wo-pvc-2'" in result.stdout
    assert "NPROC_PER_NODE=gpu" in result.stdout
    assert "MANUAL_RANK_OFFSETS=1" in result.stdout
    assert "INITIAL_BS=20" in result.stdout
    assert "OOM_STEP=1" in result.stdout
    assert "hsb-npc-training=/workspace/womd_v1_3/SMART_cache" in result.stdout
    assert "wo-pvc-2=/workspace/womd_v1_3/SMART_cache" in result.stdout
    assert "womd_training_memory_balance_h100x6_hsb_wo_pvc2.pt" in result.stdout
    assert "pre_bc_flow_control_h100x4_h100x2_prefix_default_noslip" in result.stdout


def test_h100x4_h100x2_hsb2_wrapper_targets_wo_pvc1() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "scripts"
                / "launch_pre_bc_flow_control_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_static_pods.py"
            ),
            "--dry-run-wrapper",
            "--replace",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--pods hsb-npc-training-2 wo-pvc-1" in result.stdout
    assert "flow_control_space_pretrain_h100x4_h100x2_hsb2_wo1_prefix_default_noslip_lr6e-4_bs18" in result.stdout
    assert "catk-control-pretrain-h100x4-h100x2-hsb2-wo1-prefix-default-noslip" in result.stdout
    assert "womd_training_memory_balance_h100x6_hsb2_wo1.pt" in result.stdout
