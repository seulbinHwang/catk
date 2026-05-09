from __future__ import annotations

from lightning.fabric.plugins.environments.torchelastic import TorchElasticEnvironment


class HeterogeneousTorchElasticEnvironment(TorchElasticEnvironment):
    """TorchElastic environment for static pods with uneven local GPU counts.

    Lightning's default ``TorchElasticEnvironment`` requires
    ``trainer.devices * trainer.num_nodes == WORLD_SIZE``. That is correct for
    homogeneous jobs, but the static V100 fleet has V100x4 and V100x3 pods in
    one torchrun job. In that case torchrun already provides the true
    ``WORLD_SIZE``/``RANK`` values, so the homogeneous product check is the only
    part that needs to be relaxed.
    """

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if num_devices < 1:
            raise ValueError(f"`devices` must resolve to at least one process, got {num_devices}.")
        if num_nodes < 1:
            raise ValueError(f"`num_nodes` must be at least one, got {num_nodes}.")
