from __future__ import annotations

import pickle

import torch

from scripts.compute_wosac_cpd_type_scale import compute_type_scale


def test_compute_wosac_cpd_type_scale_from_training_cache(tmp_path) -> None:
    """offline scale 스크립트가 metric fallback과 같은 RMS 수식을 써야 합니다."""
    train_dir = tmp_path / "training"
    train_dir.mkdir()
    data = {
        "agent": {
            "position": torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 8.0, 0.0]],
                    [[1.0, 1.0, 0.0], [1.0, 4.0, 0.0], [1.0, 7.0, 0.0]],
                    [[2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "valid_mask": torch.tensor(
                [
                    [True, True, True],
                    [True, True, False],
                    [False, True, True],
                ]
            ),
            "type": torch.tensor([0, 1, 2], dtype=torch.uint8),
        }
    }
    with open(train_dir / "sample.pkl", "wb") as handle:
        pickle.dump(data, handle)

    result = compute_type_scale(
        train_dir,
        num_historical_steps=1,
        num_agent_types=3,
        num_workers=0,
    )

    assert result["count"] == [2, 1, 0]
    assert torch.allclose(
        torch.tensor(result["scale"]),
        torch.tensor([(25.0 + 100.0) / 2.0, 9.0, 1.0]).sqrt(),
    )
