from __future__ import annotations

import numpy as np
import pickle
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.data_preprocess import (
    MDG_MAP_SAMPLING_VERSION,
    build_mdg_map_features,
    build_mdg_traffic_signal_features,
    process_dynamic_map,
)
from src.mdg.data import MDGDataset, collate_mdg_samples
from src.mdg.geometry import relation_features, rotate_points, wrap_angle
from src.mdg.model import MDG
from src.mdg.modules import KinematicDynamics, _fourier_relation_features


def _sample(index: int = 0) -> dict:
    num_agents = 4
    total_steps = 91
    future_steps = 80
    num_map = 5
    map_waypoints = 16
    num_signals = 3

    time = torch.arange(total_steps, dtype=torch.float32) * 0.1
    agent_speed = torch.tensor([3.0, 2.0, 1.0, 0.5])
    agent_heading = torch.tensor([0.0, 0.2, -0.3, 1.0])
    agent_origin = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 2.0],
            [-5.0, -3.0],
            [3.0, 8.0],
        ],
        dtype=torch.float32,
    )
    direction = torch.stack((torch.cos(agent_heading), torch.sin(agent_heading)), dim=-1)
    xy = agent_origin[:, None, :] + direction[:, None, :] * agent_speed[:, None, None] * time[None, :, None]
    z = torch.zeros(num_agents, total_steps, 1)
    position = torch.cat((xy, z), dim=-1)
    velocity = direction[:, None, :] * agent_speed[:, None, None]
    velocity = velocity.expand(num_agents, total_steps, 2).clone()
    valid_mask = torch.ones(num_agents, total_steps, dtype=torch.bool)

    map_x = torch.linspace(-20.0, 20.0, map_waypoints)
    map_position = torch.zeros(num_map, map_waypoints, 2)
    for poly_idx in range(num_map):
        map_position[poly_idx, :, 0] = map_x
        map_position[poly_idx, :, 1] = float(poly_idx) * 3.0
    map_heading = torch.zeros(num_map, map_waypoints)

    return {
        "scenario_id": f"100{index}",
        "tfrecord_path": None,
        "agent_id": torch.arange(1, num_agents + 1, dtype=torch.long) + index * 10,
        "agent_type": torch.tensor([0, 0, 1, 2], dtype=torch.long),
        "agent_shape": torch.tensor(
            [[4.5, 1.9, 1.5], [4.0, 1.8, 1.5], [0.8, 0.8, 1.7], [1.8, 0.6, 1.6]],
            dtype=torch.float32,
        ),
        "agent_valid": torch.ones(num_agents, dtype=torch.bool),
        "agent_position": position,
        "agent_heading": agent_heading[:, None].expand(num_agents, total_steps).clone(),
        "agent_velocity": velocity,
        "agent_valid_mask": valid_mask,
        "map_position": map_position,
        "map_heading": map_heading,
        "map_type": torch.zeros(num_map, dtype=torch.long),
        "map_light_type": torch.zeros(num_map, dtype=torch.long),
        "map_valid": torch.ones(num_map, dtype=torch.bool),
        "signal_position": torch.zeros(num_signals, 2),
        "signal_heading": torch.zeros(num_signals),
        "signal_state": torch.arange(num_signals, dtype=torch.long),
        "signal_valid": torch.ones(num_signals, dtype=torch.bool),
    }


def _small_model_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "lr": 2.0e-4,
            "weight_decay": 0.01,
            "lr_warmup_steps": 10,
            "lr_decay_step": 20,
            "lr_decay_factor": 0.98,
            "denoising_loss_weight": 1.0,
            "action_loss_weight": 0.1,
            "aux_loss_weight": 5.0,
            "n_rollout_closed_val": 2,
            "rollout_chunk_size": 2,
            "replanning_interval": 10,
            "closed_loop_denoising_steps": 2,
            "validation_closed_seed": 0,
            "val_closed_loop": True,
            "n_batch_sim_agents_metric": 0,
            "n_vis_batch": 0,
            "n_vis_scenario": 0,
            "n_vis_rollout": 0,
            "delete_local_videos_after_wandb_upload": True,
            "wosac_cpd_reference": None,
            "wosac_distribution_type_scale": [1.0, 1.0, 1.0],
            "backbone": {
                "hidden_dim": 16,
                "history_steps": 11,
                "future_steps": 80,
                "action_chunk": 2,
                "map_waypoints": 16,
                "num_noise_levels": 5,
                "num_mixer_layers": 1,
                "num_encoder_layers": 1,
                "num_denoiser_blocks": 1,
                "num_heads": 4,
                "ffn_dim": 32,
                "dropout": 0.0,
                "predictor_modes": 2,
                "num_relation_freq_bands": 2,
                "action_mean": [0.0, 0.0],
                "action_std": [1.0, 0.5],
            },
            "sim_agents_submission": {
                "is_active": False,
                "method_name": "MDG-test",
                "authors": ["test"],
                "affiliation": "test",
                "description": "test",
                "method_link": "test",
                "account_name": "test",
                "num_model_parameters": "test",
            },
        }
    )


def test_mdg_training_forward_backward_and_rollout_shapes() -> None:
    batch = collate_mdg_samples([_sample(0), _sample(1)])
    model = MDG(_small_model_config())

    out = model._training_forward(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    pred = model.generate_closed_loop_rollouts(batch, num_rollouts=2)
    assert tuple(pred["pred_pos"].shape) == (2, 4, 2, 80, 2)
    assert tuple(pred["pred_heading"].shape) == (2, 4, 2, 80)

    flat = model._flatten_rollouts(batch, pred)
    assert tuple(flat["pred_traj"].shape) == (8, 2, 80, 2)
    assert tuple(flat["pred_head"].shape) == (8, 2, 80)
    assert tuple(flat["pred_z"].shape) == (8, 2, 80)


def test_mdg_collate_pads_variable_eval_agent_count() -> None:
    sample_a = _sample(0)
    sample_b = _sample(1)
    for key in (
        "agent_id",
        "agent_type",
        "agent_shape",
        "agent_valid",
        "agent_position",
        "agent_heading",
        "agent_velocity",
        "agent_valid_mask",
    ):
        sample_b[key] = sample_b[key][:3]

    batch = collate_mdg_samples([sample_a, sample_b])

    assert tuple(batch["agent_id"].shape) == (2, 4)
    assert batch["agent_id"][1, 3].item() == -1
    assert batch["agent_valid"][1, 3].item() is False
    assert batch["agent_valid_mask"][1, 3].any().item() is False
    assert tuple(batch["map_position"].shape) == (2, 5, 16, 2)


def test_mdg_closed_loop_inference_uses_full_noise_n_step_replanning() -> None:
    cfg = _small_model_config()
    cfg.rollout_chunk_size = 2
    cfg.replanning_interval = 10
    cfg.closed_loop_denoising_steps = 3
    batch = collate_mdg_samples([_sample(0)])
    model = MDG(cfg)
    model.eval()

    full_noise_calls = []
    denoise_calls = []
    max_noise_level = model.backbone.num_noise_levels - 1
    expected_schedule = model._closed_loop_mask_schedule(torch.device("cpu")).tolist()

    def fake_full_noise_sample(rollout_batch, generator=None):
        shape = (
            rollout_batch["agent_position"].shape[0],
            rollout_batch["agent_position"].shape[1],
            model.backbone.action_steps,
            2,
        )
        noise = torch.randn(
            shape,
            device=rollout_batch["agent_position"].device,
            generator=generator,
        )
        mask = torch.full(shape[:-1], max_noise_level, dtype=torch.long, device=noise.device)
        full_noise_calls.append((tuple(noise.shape), tuple(mask.shape), int(mask.min()), int(mask.max())))
        return noise, mask

    def fake_denoise_actions(rollout_batch, noised_action, mask_level, scene=None, compute_aux=True):
        call_index = len(denoise_calls)
        denoise_calls.append((tuple(noised_action.shape), tuple(mask_level.shape), int(mask_level.min()), int(mask_level.max()), scene, compute_aux))
        expected_level = expected_schedule[call_index % cfg.closed_loop_denoising_steps]
        assert bool((mask_level == expected_level).all())
        assert scene is not None
        assert compute_aux is False
        bsz, num_agents = mask_level.shape[:2]
        pred_action = torch.zeros(
            bsz,
            num_agents,
            model.backbone.action_steps,
            2,
            device=noised_action.device,
            dtype=noised_action.dtype,
        )
        pred_pos = torch.full(
            (bsz, num_agents, model.num_future_steps, 2),
            float(call_index),
            device=noised_action.device,
            dtype=noised_action.dtype,
        )
        pred_heading = torch.zeros(
            bsz,
            num_agents,
            model.num_future_steps,
            device=noised_action.device,
            dtype=noised_action.dtype,
        )
        pred_speed = torch.zeros_like(pred_heading)
        pred_chunk_state = torch.zeros(
            bsz,
            num_agents,
            model.backbone.action_steps,
            5,
            device=noised_action.device,
            dtype=noised_action.dtype,
        )
        return pred_action, pred_pos, pred_heading, pred_speed, pred_chunk_state, None, None

    model.backbone.full_noise_sample = fake_full_noise_sample
    model.backbone.denoise_actions = fake_denoise_actions

    pred = model.generate_closed_loop_rollouts(batch, num_rollouts=2)

    assert len(full_noise_calls) == model.num_future_steps // cfg.replanning_interval
    assert len(denoise_calls) == len(full_noise_calls) * cfg.closed_loop_denoising_steps
    assert all(call[2:] == (max_noise_level, max_noise_level) for call in full_noise_calls)
    assert all(call[4] is denoise_calls[0][4] for call in denoise_calls[: cfg.closed_loop_denoising_steps])
    assert tuple(pred["pred_pos"].shape) == (1, 4, 2, 80, 2)
    for segment_idx in range(model.num_future_steps // cfg.replanning_interval):
        start = segment_idx * cfg.replanning_interval
        end = start + cfg.replanning_interval
        expected_call_index = (segment_idx + 1) * cfg.closed_loop_denoising_steps - 1
        expected = torch.full_like(pred["pred_pos"][:, :, :, start:end], float(expected_call_index))
        torch.testing.assert_close(pred["pred_pos"][:, :, :, start:end], expected)


def test_mdg_loss_decreases_on_fixed_corruption() -> None:
    batch = collate_mdg_samples([_sample(0), _sample(1)])
    model = MDG(_small_model_config())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    losses = []
    for _ in range(4):
        torch.manual_seed(1234)
        optimizer.zero_grad(set_to_none=True)
        loss = model._training_forward(batch)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


def test_mdg_auxiliary_loss_selects_best_mode_by_xy_l2() -> None:
    batch = collate_mdg_samples([_sample(0)])
    batch["agent_valid"][0, 1:] = False
    batch["agent_valid_mask"][0, 1:] = False
    model = MDG(_small_model_config())

    future_pos = batch["agent_position"][:, :, model.num_historical_steps :, :2]
    future_heading = batch["agent_heading"][:, :, model.num_historical_steps :]
    current_pos = batch["agent_position"][:, :, model.num_historical_steps - 1, :2]
    current_heading = batch["agent_heading"][:, :, model.num_historical_steps - 1]
    local_pos = future_pos - current_pos.unsqueeze(2)
    local_heading = future_heading - current_heading.unsqueeze(-1)
    target = torch.cat((local_pos, local_heading.unsqueeze(-1)), dim=-1)
    aux = target.unsqueeze(2).repeat(1, 1, 2, 1, 1)

    aux[:, 0, 0, :, 0] += 2.0
    aux[:, 0, 1, :, 0] += 0.1
    aux[:, 0, 1, :, 2] += 4.0

    loss = model._auxiliary_loss(aux, batch)
    expected = F.smooth_l1_loss(aux[:, 0, 1], target[:, 0], reduction="none").sum(dim=-1).mean()

    torch.testing.assert_close(loss, expected)


def test_mdg_default_loss_is_state_plus_aux_without_action_loss() -> None:
    cfg = _small_model_config()
    cfg.action_loss_weight = 0.0
    batch = collate_mdg_samples([_sample(0), _sample(1)])
    model = MDG(cfg)

    torch.manual_seed(1234)
    out = model._training_forward(batch)

    torch.testing.assert_close(out["action_loss"], torch.zeros_like(out["action_loss"]))
    expected = out["state_loss"] + cfg.aux_loss_weight * out["aux_loss"]
    torch.testing.assert_close(out["loss"].detach(), expected)


def test_mdg_invalid_future_chunks_do_not_change_valid_denoiser_outputs() -> None:
    batch = collate_mdg_samples([_sample(0)])
    model = MDG(_small_model_config())
    model.eval()

    invalid_start = model.num_historical_steps + 20
    batch["agent_valid_mask"][0, 0, invalid_start : invalid_start + model.backbone.action_chunk] = False
    chunk_valid = model._chunk_valid(batch)
    clean_action, _ = model.backbone.clean_actions_and_chunk_state_from_batch(batch)
    mask_level = torch.full(
        clean_action.shape[:-1],
        model.num_noise_levels - 1,
        dtype=torch.long,
        device=clean_action.device,
    )
    perturbed_action = clean_action.clone()
    perturbed_action[~chunk_valid] = 1000.0

    with torch.no_grad():
        pred_action, _, _, _, pred_chunk_state, _, _ = model.backbone.denoise_actions(
            batch,
            clean_action,
            mask_level,
            future_valid=chunk_valid,
            compute_aux=False,
        )
        perturbed_pred_action, _, _, _, perturbed_chunk_state, _, _ = model.backbone.denoise_actions(
            batch,
            perturbed_action,
            mask_level,
            future_valid=chunk_valid,
            compute_aux=False,
        )

    torch.testing.assert_close(
        pred_action[chunk_valid],
        perturbed_pred_action[chunk_valid],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        pred_chunk_state[chunk_valid],
        perturbed_chunk_state[chunk_valid],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(perturbed_pred_action[~chunk_valid], torch.zeros_like(perturbed_pred_action[~chunk_valid]))


def test_mdg_invalid_future_nan_targets_are_excluded_from_training_loss() -> None:
    sample = _sample(0)
    invalid_start = 11 + 20
    sample["agent_valid_mask"][0, invalid_start:] = False
    sample["agent_position"][0, invalid_start:] = float("nan")
    sample["agent_heading"][0, invalid_start:] = float("nan")
    sample["agent_velocity"][0, invalid_start:] = float("nan")
    batch = collate_mdg_samples([sample, _sample(1)])
    model = MDG(_small_model_config())

    out = model._training_forward(batch)

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["state_loss"])
    assert torch.isfinite(out["aux_loss"])


def test_mdg_lr_is_step_based_without_lightning_scheduler() -> None:
    model = MDG(_small_model_config())
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == model.lr / model.lr_warmup_steps

    model._set_optimizer_lr(optimizer, step=9)
    assert optimizer.param_groups[0]["lr"] == model.lr

    model._set_optimizer_lr(optimizer, step=30)
    assert optimizer.param_groups[0]["lr"] == model.lr * model.lr_decay_factor


def test_mdg_waymo_paper_contract_defaults() -> None:
    model_cfg = OmegaConf.load("configs/model/mdg.yaml").model_config
    data_cfg = OmegaConf.load("configs/data/mdg_waymo.yaml")
    pretrain_cfg = OmegaConf.load("configs/experiment/mdg_pretrain.yaml")
    submission_cfg = OmegaConf.load("configs/experiment/mdg_wosac_sub.yaml")

    assert data_cfg.train_max_agents == 64
    assert "eval_max_agents" not in data_cfg
    assert data_cfg.max_map_polylines == 320
    assert data_cfg.map_waypoints == 16
    assert data_cfg.max_traffic_lights == 16

    assert model_cfg.backbone.hidden_dim == 192
    assert model_cfg.backbone.history_steps == 11
    assert model_cfg.backbone.future_steps == 80
    assert model_cfg.backbone.action_chunk == 2
    assert model_cfg.backbone.num_noise_levels == 5
    assert model_cfg.backbone.num_mixer_layers == 2
    assert model_cfg.backbone.num_encoder_layers == 6
    assert model_cfg.backbone.num_denoiser_blocks == 2
    assert model_cfg.backbone.num_heads == 8
    assert model_cfg.backbone.ffn_dim == 704
    assert model_cfg.backbone.predictor_modes == 6
    assert model_cfg.n_rollout_closed_val == 32
    assert model_cfg.replanning_interval == 10
    assert model_cfg.closed_loop_denoising_steps == 5
    assert model_cfg.action_loss_weight == 0.0
    assert model_cfg.aux_loss_weight == 5.0
    assert model_cfg.sim_agents_submission.num_model_parameters == "7.11M"
    assert submission_cfg.model.model_config.n_rollout_closed_val == 32
    assert submission_cfg.model.model_config.replanning_interval == 10
    assert submission_cfg.model.model_config.closed_loop_denoising_steps == 5
    assert submission_cfg.model.model_config.val_closed_loop is True
    assert submission_cfg.model.model_config.sim_agents_submission.is_active is True
    assert pretrain_cfg.trainer.precision == "16-mixed"
    assert pretrain_cfg.trainer.limit_val_batches == 0.1
    assert pretrain_cfg.trainer.check_val_every_n_epoch == 16
    assert pretrain_cfg.data.val_batch_size == 12
    assert "eval_max_agents" not in pretrain_cfg.data
    assert pretrain_cfg.model.model_config.scorer_scene_num == 1680
    assert "eval_max_agents" not in submission_cfg.data
    assert submission_cfg.trainer.precision == "16-mixed"

    model = MDG(model_cfg)
    assert sum(p.numel() for p in model.parameters()) == 7_111_168


def test_mdg_closed_loop_denoising_steps_match_discrete_noise_levels() -> None:
    cfg = _small_model_config()
    expected = {
        1: [4],
        2: [4, 0],
        3: [4, 2, 0],
        4: [4, 3, 1, 0],
        5: [4, 3, 2, 1, 0],
    }
    for steps, schedule in expected.items():
        cfg.closed_loop_denoising_steps = steps
        model = MDG(cfg)
        assert model._closed_loop_mask_schedule(torch.device("cpu")).tolist() == schedule

    cfg.closed_loop_denoising_steps = 6
    with pytest.raises(ValueError, match="closed_loop_denoising_steps must be <="):
        MDG(cfg)


def test_mdg_noise_schedule_and_fourier_relation_features() -> None:
    model = MDG(_small_model_config())
    alpha = model._alpha_schedule(torch.device("cpu"), torch.float32)
    torch.testing.assert_close(alpha, torch.linspace(0.99, 0.01, 5))

    for block in model.backbone.denoiser.blocks:
        assert block.temporal.attn.rel_emb is None

    batch = collate_mdg_samples([_sample(0), _sample(1)])
    mask = model._sample_mask_levels(batch)
    assert tuple(mask.shape) == (2, 4, 40)
    assert int(mask.min()) >= 0
    assert int(mask.max()) <= 4

    rel = torch.zeros(2, 3, 4, 3)
    encoded = _fourier_relation_features(rel, num_bands=2)
    assert tuple(encoded.shape) == (2, 3, 4, 15)

    pos = torch.zeros(1, 3, 2)
    heading = torch.zeros(1, 3)
    self_rel = relation_features(pos, heading, pos, heading)
    torch.testing.assert_close(
        self_rel[0, torch.arange(3), torch.arange(3)],
        torch.full((3, 3), 1.0e-4),
    )


def test_mdg_masking_deltas_cover_global_ddp_batch(monkeypatch) -> None:
    model = MDG(_small_model_config())
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)

    bins = []
    for rank in range(3):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        deltas = model._masking_deltas(4, torch.device("cpu"), batch_idx=7)
        bins.append(torch.round(deltas * 11).long())

    all_bins = torch.cat(bins).sort().values
    torch.testing.assert_close(all_bins, torch.arange(12))


def test_mdg_temporal_mask_prefix_is_progressively_increasing() -> None:
    model = MDG(_small_model_config())
    batch = {"agent_valid": torch.ones(8, 4, dtype=torch.bool)}
    original_rand = torch.rand

    def force_temporal_masking(*args, **kwargs):
        if len(args) == 1 and args[0] == ():
            return torch.zeros((), device=kwargs.get("device", None))
        return original_rand(*args, **kwargs)

    torch.manual_seed(0)
    torch.rand = force_temporal_masking
    try:
        mask = model._sample_mask_levels(batch)
    finally:
        torch.rand = original_rand

    assert tuple(mask.shape) == (8, 4, 40)
    saw_partial_temporal_mask = False
    saw_agent_specific_prefix = False
    for sample_idx in range(mask.shape[0]):
        for agent_idx in range(mask.shape[1]):
            row = mask[sample_idx, agent_idx]
            full_suffix = int((row == 4).sum().item())
            prefix = row[: row.numel() - full_suffix] if full_suffix > 0 else row
            suffix = row[row.numel() - full_suffix :] if full_suffix > 0 else row.new_empty(0)
            if prefix.numel() > 1:
                assert bool((prefix[1:] >= prefix[:-1]).all())
                assert int(prefix.max().item()) <= 3
            if suffix.numel() > 0:
                assert bool((suffix == 4).all())

        reference = mask[sample_idx, 0]
        if any(not torch.equal(mask[sample_idx, agent_idx], reference) for agent_idx in range(1, mask.shape[1])):
            saw_agent_specific_prefix = True
        full_suffix = int((reference == 4).sum().item())
        if 0 < full_suffix < reference.numel():
            saw_partial_temporal_mask = True

    assert saw_partial_temporal_mask
    assert saw_agent_specific_prefix


def test_mdg_dynamics_chunk_state_is_agent_local() -> None:
    dynamics = KinematicDynamics(action_chunk=2, dt=0.1, action_mean=(0.0, 0.0), action_std=(1.0, 1.0))
    action = torch.tensor(
        [
            [
                [[0.20, 0.10], [-0.10, 0.05], [0.00, -0.20]],
                [[0.05, -0.15], [0.10, 0.20], [-0.05, 0.00]],
            ]
        ],
        dtype=torch.float32,
    )
    current_pos = torch.tensor([[[5.0, -2.0], [1.0, 4.0]]], dtype=torch.float32)
    current_heading = torch.tensor([[0.3, -1.2]], dtype=torch.float32)
    current_speed = torch.tensor([[3.0, 1.5]], dtype=torch.float32)

    full_pos, full_heading, full_speed, chunk_state, _ = dynamics(
        action,
        current_pos,
        current_heading,
        current_speed,
    )

    rotation = torch.tensor(1.1, dtype=torch.float32)
    translation = torch.tensor([10.0, -3.0], dtype=torch.float32)
    rotated_pos = rotate_points(current_pos, rotation) + translation
    rotated_heading = wrap_angle(current_heading + rotation)
    rotated_full_pos, rotated_full_heading, rotated_full_speed, rotated_chunk_state, _ = dynamics(
        action,
        rotated_pos,
        rotated_heading,
        current_speed,
    )

    torch.testing.assert_close(rotated_chunk_state, chunk_state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rotated_full_pos, rotate_points(full_pos, rotation) + translation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        wrap_angle(rotated_full_heading - full_heading - rotation),
        torch.zeros_like(full_heading),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(rotated_full_speed, full_speed, atol=1e-5, rtol=1e-5)


def test_mdg_traffic_signal_stop_points_are_cached() -> None:
    dynamic_map = {
        "lane_id": [np.array([[7, 8]])],
        "state": [np.array([["LANE_STATE_STOP", "LANE_STATE_GO"]])],
        "stop_point": [np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32)],
    }
    lights = process_dynamic_map(dynamic_map)
    mdg_map = {
        "position": torch.zeros(0, 16, 2, dtype=torch.float32),
        "heading": torch.zeros(0, 16, dtype=torch.float32),
        "light_type": torch.zeros(0, dtype=torch.long),
    }
    signal = build_mdg_traffic_signal_features(mdg_map, lights)

    torch.testing.assert_close(signal["position"], torch.tensor([[1.5, 2.5], [3.5, 4.5]]))
    assert signal["state"].tolist() == [2, 3]
    assert signal["valid"].tolist() == [True, True]


def test_mdg_map_cache_preserves_detailed_polyline_types() -> None:
    map_data = {
        "map_polygon": {
            "type": torch.tensor([0, 1], dtype=torch.uint8),
            "light_type": torch.tensor([0, 0], dtype=torch.uint8),
        },
        "map_point": {
            "position": torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            "type": torch.tensor([3, 3, 8, 8], dtype=torch.uint8),
        },
        ("map_point", "to", "map_polygon"): {
            "edge_index": torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        },
    }

    mdg_map = build_mdg_map_features(map_data, num_waypoints=16)

    assert mdg_map["type"].tolist() == [3, 8]
    assert mdg_map["sampling"] == MDG_MAP_SAMPLING_VERSION


def test_mdg_map_cache_uses_arc_length_waypoint_sampling() -> None:
    map_data = {
        "map_polygon": {
            "type": torch.tensor([0], dtype=torch.uint8),
            "light_type": torch.tensor([0], dtype=torch.uint8),
        },
        "map_point": {
            "position": torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [101.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "type": torch.tensor([3, 3, 3], dtype=torch.uint8),
        },
        ("map_point", "to", "map_polygon"): {
            "edge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
        },
    }

    mdg_map = build_mdg_map_features(map_data, num_waypoints=5)

    torch.testing.assert_close(
        mdg_map["position"][0, :, 0],
        torch.linspace(0.0, 101.0, steps=5),
    )
    torch.testing.assert_close(mdg_map["position"][0, :, 1], torch.zeros(5))
    torch.testing.assert_close(mdg_map["heading"][0], torch.zeros(5))


def _write_minimal_mdg_cache(tmp_path, mdg_map: dict) -> None:
    num_agents = 1
    total_steps = 91
    data = {
        "scenario_id": "minimal_mdg_cache",
        "agent": {
            "position": torch.zeros(num_agents, total_steps, 3),
            "heading": torch.zeros(num_agents, total_steps),
            "velocity": torch.zeros(num_agents, total_steps, 2),
            "valid_mask": torch.ones(num_agents, total_steps, dtype=torch.bool),
            "shape": torch.ones(num_agents, 3),
            "type": torch.zeros(num_agents, dtype=torch.long),
            "role": torch.tensor([[True, False, False]]),
            "id": torch.tensor([1], dtype=torch.long),
        },
        "mdg_map": mdg_map,
        "mdg_traffic_signal": {
            "position": torch.zeros(0, 2),
            "heading": torch.zeros(0),
            "state": torch.zeros(0, dtype=torch.long),
            "valid": torch.zeros(0, dtype=torch.bool),
        },
    }
    with (tmp_path / "minimal.pkl").open("wb") as handle:
        pickle.dump(data, handle)


def test_mdg_dataset_requires_arclength_map_cache(tmp_path) -> None:
    mdg_map = {
        "position": torch.zeros(1, 16, 2),
        "heading": torch.zeros(1, 16),
        "type": torch.zeros(1, dtype=torch.long),
        "light_type": torch.zeros(1, dtype=torch.long),
        "valid": torch.ones(1, dtype=torch.bool),
    }
    _write_minimal_mdg_cache(tmp_path, mdg_map)
    dataset = MDGDataset(
        raw_dir=str(tmp_path),
        max_agents=1,
        max_map_polylines=1,
        map_waypoints=16,
        max_traffic_lights=1,
        training=False,
    )

    with pytest.raises(ValueError, match="arclength_v1"):
        _ = dataset[0]


def test_mdg_active_submission_requires_waymo_rollout_count() -> None:
    expected = submission_specs.get_submission_config(
        submission_specs.ChallengeType.SIM_AGENTS
    ).n_rollouts

    MDG._check_sim_agents_submission_rollout_count(False, 1)
    MDG._check_sim_agents_submission_rollout_count(True, expected)

    try:
        MDG._check_sim_agents_submission_rollout_count(True, expected - 1)
    except ValueError as exc:
        assert f"n_rollout_closed_val={expected}" in str(exc)
    else:
        raise AssertionError("active submission accepted an invalid rollout count")
