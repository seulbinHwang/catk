from __future__ import annotations

from typing import Any

from lightning.fabric.plugins.environments.torchelastic import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only


class HeterogeneousTorchElasticEnvironment(TorchElasticEnvironment):
    """TorchElastic environment for static pods with uneven local GPU counts.

    Lightning's default TorchElastic environment validates the homogeneous
    product ``trainer.devices * trainer.num_nodes == WORLD_SIZE``. Static
    multi-node jobs such as H100x4 + H100x2 intentionally violate that product,
    while the launcher still exports the correct ``WORLD_SIZE`` and ``RANK`` for
    every worker.
    """

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if num_devices < 1:
            raise ValueError(f"`devices` must resolve to at least one process, got {num_devices}.")
        if num_nodes < 1:
            raise ValueError(f"`num_nodes` must be at least one, got {num_nodes}.")


class HeterogeneousDDPStrategy(DDPStrategy):
    """DDP strategy for static pod fleets with uneven local GPU counts."""

    @property
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    def set_world_ranks(self) -> None:
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank
