from pathlib import Path
from typing import Optional

from lightning import Callback, LightningModule, Trainer
from omegaconf import DictConfig

from src.smart.road.cache import delete_cache_dir, generate_road_cache


class RoadCacheRefreshCallback(Callback):
    def __init__(
        self,
        original_train_raw_dir: str,
        cache_root_dir: str,
        sampling_scheme: DictConfig,
        num_rollouts_per_scenario: int,
        generation_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        delete_after_use: bool,
    ) -> None:
        """RoaD fine-tuning용 epoch-local cache를 관리한다.

        Args:
            original_train_raw_dir: 원본 WOMD training pickle cache 디렉터리이다.
            cache_root_dir: epoch별 RoaD cache를 만들 상위 디렉터리이다.
            sampling_scheme: RoaD Sample-K rollout 설정이다.
            num_rollouts_per_scenario: scenario당 생성할 rollout 수이다.
            generation_batch_size: cache 생성용 batch size이다. 0 이하면 학습 batch size를 쓴다.
            num_workers: cache 생성용 worker 수이다.
            pin_memory: cache 생성용 DataLoader pin_memory 여부이다.
            delete_after_use: 학습이 끝난 cache를 바로 삭제할지 여부이다.
        """
        super().__init__()
        self.original_train_raw_dir = original_train_raw_dir
        self.cache_root_dir = cache_root_dir
        self.sampling_scheme = sampling_scheme
        self.num_rollouts_per_scenario = num_rollouts_per_scenario
        self.generation_batch_size = generation_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.delete_after_use = delete_after_use
        self.current_cache_dir: Optional[str] = None

    def _cache_dir(self, epoch: int) -> str:
        """epoch 번호에 대응하는 RoaD cache 디렉터리 이름을 만든다.

        Args:
            epoch: cache를 만들 epoch 번호이다.

        Returns:
            epoch별 RoaD cache 디렉터리 경로이다.
        """
        return (Path(self.cache_root_dir) / f"epoch_{epoch:03d}").as_posix()

    def _sync(self, trainer: Trainer, tag: str) -> None:
        """분산 학습 process들이 cache 생성·삭제 시점을 맞춘다.

        Args:
            trainer: Lightning trainer이다.
            tag: 동기화 지점을 구분하기 위한 이름이다.
        """
        if trainer.world_size > 1:
            trainer.strategy.barrier(tag)

    def _generation_batch_size(self, trainer: Trainer) -> int:
        """cache 생성용 batch size를 정한다.

        Args:
            trainer: Lightning trainer이다.

        Returns:
            실제 cache 생성에 사용할 batch size이다.
        """
        if self.generation_batch_size > 0:
            return self.generation_batch_size
        return trainer.datamodule.train_batch_size

    def _generate_and_attach_cache(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        epoch: int,
    ) -> str:
        """RoaD cache를 만들고 datamodule이 그 cache를 읽도록 바꾼다.

        Args:
            trainer: Lightning trainer이다.
            pl_module: 학습 중인 SMART model이다.
            epoch: 만들 cache의 epoch 번호이다.

        Returns:
            생성된 RoaD cache 디렉터리 경로이다.
        """
        cache_dir = self._cache_dir(epoch)
        if trainer.global_rank == 0:
            generate_road_cache(
                model=pl_module,
                original_train_raw_dir=self.original_train_raw_dir,
                output_dir=cache_dir,
                transform=trainer.datamodule.train_transform,
                sampling_scheme=self.sampling_scheme,
                num_rollouts_per_scenario=self.num_rollouts_per_scenario,
                batch_size=self._generation_batch_size(trainer),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                num_historical_steps=pl_module.num_historical_steps,
                device=None,
            )
        self._sync(trainer, f"road_cache_epoch_{epoch:03d}_generated")
        trainer.datamodule.set_train_raw_dir(
            cache_dir,
            road_num_rollouts_per_scenario=self.num_rollouts_per_scenario,
        )
        return cache_dir

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        if stage != "fit":
            return
        Path(self.cache_root_dir).mkdir(parents=True, exist_ok=True)
        self.current_cache_dir = self._generate_and_attach_cache(trainer, pl_module, 0)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        next_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or next_epoch
        if next_epoch >= max_epochs:
            return
        previous_cache_dir = self.current_cache_dir
        self.current_cache_dir = self._generate_and_attach_cache(
            trainer, pl_module, next_epoch
        )
        if self.delete_after_use and trainer.global_rank == 0:
            delete_cache_dir(previous_cache_dir)
        self._sync(trainer, f"road_cache_epoch_{trainer.current_epoch:03d}_deleted")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.delete_after_use and trainer.global_rank == 0:
            delete_cache_dir(self.current_cache_dir)
        self._sync(trainer, "road_cache_final_deleted")
        trainer.datamodule.set_train_raw_dir(
            self.original_train_raw_dir,
            road_num_rollouts_per_scenario=1,
        )
