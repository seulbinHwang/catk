import pickle

from src.smart.datasets.scalable_dataset import MultiDataset


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
