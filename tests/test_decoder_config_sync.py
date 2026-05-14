from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.smart.model.smart_flow import _build_decoder_config_from_token_processor


def _token_processor_stub(**overrides):
    values = {
        "use_kinematic_control_flow": True,
        "use_holonomic_model_only": False,
        "use_rolling_supervision": True,
        "control_pos_scale_m": 1.0,
        "control_vehicle_no_slip_point_ratio": 0.25,
        "control_cyclist_no_slip_point_ratio": 0.05,
        "control_vehicle_yaw_scale_rad": 0.025,
        "control_pedestrian_yaw_scale_rad": 0.20,
        "control_cyclist_yaw_scale_rad": 0.06,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_decoder_config_uses_token_processor_as_single_source_of_truth() -> None:
    decoder_config = {
        "hidden_dim": 128,
        "control_vehicle_no_slip_point_ratio": 0.25,
        "control_cyclist_no_slip_point_ratio": 0.05,
    }

    synced = _build_decoder_config_from_token_processor(
        decoder_config=decoder_config,
        token_processor=_token_processor_stub(),
    )

    assert synced["hidden_dim"] == 128
    assert synced["control_vehicle_no_slip_point_ratio"] == 0.25
    assert synced["control_cyclist_no_slip_point_ratio"] == 0.05
    assert synced["control_pos_scale_m"] == 1.0
    assert synced["use_kinematic_control_flow"] is True


def test_decoder_config_rejects_control_space_mismatch() -> None:
    decoder_config = {
        "hidden_dim": 128,
        "control_vehicle_no_slip_point_ratio": 0.0,
    }

    with pytest.raises(ValueError, match="token_processor.control_vehicle_no_slip_point_ratio"):
        _build_decoder_config_from_token_processor(
            decoder_config=decoder_config,
            token_processor=_token_processor_stub(control_vehicle_no_slip_point_ratio=0.25),
        )
