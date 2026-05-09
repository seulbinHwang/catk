from __future__ import annotations

from typing import Any

from lightning.fabric.plugins.environments.torchelastic import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only


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


class HeterogeneousDDPStrategy(DDPStrategy):
    """DDP strategy for static pod fleets with uneven GPU counts.

    Lightning's stock DDP sampler kwargs use ``num_nodes * local_processes``.
    That is correct for homogeneous jobs, but a V100x4 + V100x3 fleet needs the
    true ``WORLD_SIZE`` exported by the launcher. Process-group initialization
    already reads that value through ``TorchElasticEnvironment``; the sampler
    must do the same or ranks on the later 3-GPU pods become invalid.
    """

    @property
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    def set_world_ranks(self) -> None:
        # Keep rank bookkeeping aligned with the launcher-provided RANK. The
        # TorchElastic environment ignores setter writes, but this explicit
        # implementation avoids any future homogeneous-product assumption here.
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank
