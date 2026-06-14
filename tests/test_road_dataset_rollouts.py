import pickle

from src.smart.datasets.scalable_dataset import MultiDataset


def test_road_dataset_uses_every_rollout_file(tmp_path):
    for scenario_id in ("scenario_a", "scenario_b"):
        for rollout_idx in range(3):
            path = tmp_path / f"{scenario_id}__road_r{rollout_idx:02d}.pkl"
            with open(path, "wb") as handle:
                pickle.dump({"scenario_id": path.stem}, handle)

    dataset = MultiDataset(
        tmp_path.as_posix(),
        transform=None,
        road_num_rollouts_per_scenario=3,
    )

    assert len(dataset) == 6
    scenario_ids = [dataset.get(idx)["scenario_id"] for idx in range(len(dataset))]
    assert scenario_ids == [
        "scenario_a__road_r00",
        "scenario_a__road_r01",
        "scenario_a__road_r02",
        "scenario_b__road_r00",
        "scenario_b__road_r01",
        "scenario_b__road_r02",
    ]
