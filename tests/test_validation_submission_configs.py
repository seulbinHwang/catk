from pathlib import Path

import pytest
from omegaconf import OmegaConf
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.smart.model.smart import SMART


def _load_experiment_config(name: str):
    return OmegaConf.load(Path("configs/experiment") / f"{name}.yaml")


def test_validation_configs_match_pre_bc_num_freq_bands() -> None:
    pre_bc = _load_experiment_config("pre_bc")
    expected_num_freq_bands = pre_bc.model.model_config.decoder.num_freq_bands

    assert (
        _load_experiment_config("local_val").model.model_config.decoder.num_freq_bands
        == expected_num_freq_bands
    )
    assert (
        _load_experiment_config("wosac_sub").model.model_config.decoder.num_freq_bands
        == expected_num_freq_bands
    )


def test_wosac_submission_config_uses_waymo_rollout_count() -> None:
    expected_rollouts = submission_specs.get_submission_config(
        submission_specs.ChallengeType.SIM_AGENTS
    ).n_rollouts

    assert (
        _load_experiment_config("wosac_sub").model.model_config.n_rollout_closed_val
        == expected_rollouts
    )


def test_active_submission_rejects_wrong_rollout_count() -> None:
    expected_rollouts = SMART._required_sim_agents_rollout_count()

    SMART._check_sim_agents_submission_rollout_count(
        is_active=False,
        n_rollout_closed_val=1,
    )
    SMART._check_sim_agents_submission_rollout_count(
        is_active=True,
        n_rollout_closed_val=expected_rollouts,
    )
    with pytest.raises(ValueError, match=f"n_rollout_closed_val={expected_rollouts}"):
        SMART._check_sim_agents_submission_rollout_count(
            is_active=True,
            n_rollout_closed_val=expected_rollouts - 1,
        )
