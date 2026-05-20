import importlib.util
import sys
from pathlib import Path

import pytest


def _load_launcher_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = (
        repo_root
        / "scripts"
        / "launch_pre_bc_flow_control_h100x6_hsb2_wo2_execctx_balanced_static_pods.py"
    )
    spec = importlib.util.spec_from_file_location("h100x6_launcher", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h100x6_launcher_pins_safe_sampler_overrides(monkeypatch) -> None:
    launcher = _load_launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--extra-hydra-overrides",
            "model.model_config.lr=1e-4",
            "--check-val-every-n-epoch",
            "16",
        ],
    )

    args = launcher.parse_args()
    overrides = launcher.training_extra_hydra_overrides(args)

    assert "data.train_memory_balanced_batches=true" in overrides
    assert "trainer.use_distributed_sampler=false" in overrides
    assert "trainer.check_val_every_n_epoch=16" in overrides
    assert (
        "trainer.strategy._target_=src.smart.utils.heterogeneous_torchelastic.HeterogeneousDDPStrategy"
        in overrides
    )


@pytest.mark.parametrize(
    "unsafe_override",
    [
        "data.train_memory_balanced_batches=false",
        "trainer.use_distributed_sampler=true",
        "trainer.check_val_every_n_epoch=16",
    ],
)
def test_h100x6_launcher_rejects_unsafe_sampler_overrides(
    monkeypatch, unsafe_override: str
) -> None:
    launcher = _load_launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "--extra-hydra-overrides", unsafe_override],
    )

    with pytest.raises(SystemExit):
        launcher.parse_args()
