import pickle

from src.smart.datamodules.scalable_datamodule import MultiDataModule
from src.smart.datasets.scalable_dataset import MultiDataset, is_cache_sample_path


def _write_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def test_multi_dataset_ignores_hidden_metadata_pickle(tmp_path):
    _write_pickle(tmp_path / ".catk_memory_balanced_metadata_v1.pkl", {"entries": []})
    _write_pickle(tmp_path / "scenario_b.pkl", {"scenario_id": "scenario_b"})
    _write_pickle(tmp_path / "scenario_a.pkl", {"scenario_id": "scenario_a"})
    (tmp_path / "notes.txt").write_text("not a SMART cache sample")

    dataset = MultiDataset(str(tmp_path), transform=None)

    assert dataset.len() == 2
    assert [path.split("/")[-1] for path in dataset.raw_paths] == [
        "scenario_a.pkl",
        "scenario_b.pkl",
    ]
    assert dataset.get(0)["scenario_id"] == "scenario_a"


def test_multi_dataset_concatenates_multiple_raw_dirs_in_order(tmp_path):
    train_dir = tmp_path / "training"
    val_dir = tmp_path / "validation"
    train_dir.mkdir()
    val_dir.mkdir()
    _write_pickle(train_dir / "scenario_b.pkl", {"scenario_id": "train_b"})
    _write_pickle(train_dir / "scenario_a.pkl", {"scenario_id": "train_a"})
    _write_pickle(val_dir / "scenario_c.pkl", {"scenario_id": "val_c"})

    dataset = MultiDataset([str(train_dir), str(val_dir)], transform=None)

    assert dataset.len() == 3
    assert [path.split("/")[-2:] for path in dataset.raw_paths] == [
        ["training", "scenario_a.pkl"],
        ["training", "scenario_b.pkl"],
        ["validation", "scenario_c.pkl"],
    ]
    assert dataset.get(0)["scenario_id"] == "train_a"
    assert dataset.get(2)["scenario_id"] == "val_c"


def test_datamodule_train_raw_dirs_do_not_change_val_dataset(tmp_path):
    train_dir = tmp_path / "training"
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "testing"
    tfrecord_dir = tmp_path / "validation_tfrecords_splitted"
    for path in (train_dir, val_dir, test_dir, tfrecord_dir):
        path.mkdir()
    _write_pickle(train_dir / "train.pkl", {"scenario_id": "train"})
    _write_pickle(val_dir / "val.pkl", {"scenario_id": "val"})
    _write_pickle(test_dir / "test.pkl", {"scenario_id": "test"})

    datamodule = MultiDataModule(
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        train_raw_dir=str(train_dir),
        train_raw_dirs=[str(train_dir), str(val_dir)],
        val_raw_dir=str(val_dir),
        test_raw_dir=str(test_dir),
        val_tfrecords_splitted=str(tfrecord_dir),
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        persistent_workers=False,
        train_max_num=32,
        train_use_eval_agent_selection=True,
    )

    datamodule.setup("fit")

    assert [path.split("/")[-2:] for path in datamodule.train_dataset.raw_paths] == [
        ["training", "train.pkl"],
        ["validation", "val.pkl"],
    ]
    assert [path.split("/")[-2:] for path in datamodule.val_dataset.raw_paths] == [
        ["validation", "val.pkl"],
    ]


def test_cache_sample_path_filter_requires_visible_pickle_file(tmp_path):
    visible_sample = tmp_path / "scenario.pkl"
    hidden_metadata = tmp_path / ".catk_memory_balanced_metadata_v1.pkl"
    sidecar = tmp_path / "notes.txt"
    subdir = tmp_path / "nested.pkl"

    visible_sample.write_bytes(b"sample")
    hidden_metadata.write_bytes(b"metadata")
    sidecar.write_text("not a pickle")
    subdir.mkdir()

    assert is_cache_sample_path(visible_sample)
    assert not is_cache_sample_path(hidden_metadata)
    assert not is_cache_sample_path(sidecar)
    assert not is_cache_sample_path(subdir)
