import pickle

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
