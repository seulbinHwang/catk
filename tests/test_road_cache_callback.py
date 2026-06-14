from pathlib import Path

from src.smart.road.callback import RoadCacheRefreshCallback


class _DummyDataModule:
    def __init__(self) -> None:
        self.train_raw_dir = None
        self.road_num_rollouts_per_scenario = None

    def set_train_raw_dir(
        self, train_raw_dir: str, road_num_rollouts_per_scenario: int = 1
    ) -> None:
        self.train_raw_dir = train_raw_dir
        self.road_num_rollouts_per_scenario = road_num_rollouts_per_scenario


class _DummyTrainer:
    world_size = 1
    global_rank = 0

    def __init__(self, datamodule: _DummyDataModule) -> None:
        self.datamodule = datamodule


def _make_callback(
    original_train_raw_dir: Path,
    cache_root_dir: Path,
    *,
    delete_after_use: bool,
) -> RoadCacheRefreshCallback:
    return RoadCacheRefreshCallback(
        original_train_raw_dir=original_train_raw_dir.as_posix(),
        cache_root_dir=cache_root_dir.as_posix(),
        sampling_scheme={},
        road_data_use_ratio=1.0,
        num_rollouts_per_scenario=3,
        generation_batch_size=1,
        num_workers=0,
        pin_memory=False,
        delete_after_use=delete_after_use,
    )


def test_road_callback_restores_original_train_dir_after_deleting_cache(tmp_path):
    original_train_raw_dir = tmp_path / "training"
    original_train_raw_dir.mkdir()
    cache_dir = tmp_path / "road_cache" / "epoch_000"
    cache_dir.mkdir(parents=True)
    (cache_dir / "sample.pkl").write_bytes(b"cache")

    datamodule = _DummyDataModule()
    callback = _make_callback(
        original_train_raw_dir,
        tmp_path / "road_cache",
        delete_after_use=True,
    )
    callback.current_cache_dir = cache_dir.as_posix()

    callback.on_train_end(_DummyTrainer(datamodule), pl_module=None)

    assert not cache_dir.exists()
    assert datamodule.train_raw_dir == original_train_raw_dir.as_posix()
    assert datamodule.road_num_rollouts_per_scenario == 1


def test_road_callback_restores_original_train_dir_when_cache_is_kept(tmp_path):
    original_train_raw_dir = tmp_path / "training"
    original_train_raw_dir.mkdir()
    cache_dir = tmp_path / "road_cache" / "epoch_000"
    cache_dir.mkdir(parents=True)

    datamodule = _DummyDataModule()
    callback = _make_callback(
        original_train_raw_dir,
        tmp_path / "road_cache",
        delete_after_use=False,
    )
    callback.current_cache_dir = cache_dir.as_posix()

    callback.on_train_end(_DummyTrainer(datamodule), pl_module=None)

    assert cache_dir.exists()
    assert datamodule.train_raw_dir == original_train_raw_dir.as_posix()
    assert datamodule.road_num_rollouts_per_scenario == 1
