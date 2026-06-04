import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from scripts.build_unimm_anchors import (
    collect_training_trajectories,
    compute_context_thresholds_from_cache,
    compute_threshold,
    lloyd_refine_kmeans,
    minibatch_kmeans,
    nearest_anchor_assignment,
)
from src.unimm.losses import unimm_classification_loss, unimm_nll_loss
from src.unimm.anchors import match_anchors_by_type
from src.unimm.model.anchor_based_4s import UniMMAnchorBased4s
from src.unimm.modules import UniMMMotionDecoder
from src.unimm.processor import UniMMProcessor


def _make_anchor_payload(num_anchors: int = 8):
    anchors = {}
    for name, speed in {"veh": 0.7, "ped": 0.15, "cyc": 0.35}.items():
        bank = torch.zeros(num_anchors, 80, 3)
        for k in range(num_anchors):
            scale = speed * (k + 1) / num_anchors
            bank[k, :, 0] = torch.arange(1, 81, dtype=torch.float32) * scale
        anchors[name] = bank.numpy()
    return {
        "anchors": anchors,
        "posterior_error_threshold": {"veh": 1e9, "ped": 1e9, "cyc": 1e9},
    }


def _make_data(num_agents: int = 3) -> HeteroData:
    data = HeteroData()
    traj_pos = torch.tensor(
        [
            [[0.0, 0.0], [2.5, 0.0], [5.0, 0.0]],
            [[0.0, 3.0], [2.5, 3.0], [5.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    data["map_save"]["traj_pos"] = traj_pos
    data["map_save"]["traj_theta"] = torch.zeros(2)
    data["pt_token"]["type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["pl_type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["light_type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["num_nodes"] = 2
    data["pt_token"]["batch"] = torch.zeros(2, dtype=torch.long)

    position = torch.zeros(num_agents, 91, 3)
    heading = torch.zeros(num_agents, 91)
    velocity = torch.zeros(num_agents, 91, 2)
    for agent_idx in range(num_agents):
        speed = 0.3 + 0.1 * agent_idx
        position[agent_idx, :, 0] = torch.arange(91, dtype=torch.float32) * speed
        position[agent_idx, :, 1] = float(agent_idx)
        velocity[agent_idx, :, 0] = speed / 0.1

    data["agent"]["num_nodes"] = num_agents
    data["agent"]["valid_mask"] = torch.ones(num_agents, 91, dtype=torch.bool)
    data["agent"]["role"] = torch.zeros(num_agents, 3, dtype=torch.bool)
    data["agent"]["role"][0, 0] = True
    data["agent"]["id"] = torch.arange(num_agents, dtype=torch.long)
    data["agent"]["type"] = torch.tensor([0, 1, 2], dtype=torch.uint8)[:num_agents]
    data["agent"]["position"] = position
    data["agent"]["heading"] = heading
    data["agent"]["velocity"] = velocity
    data["agent"]["shape"] = torch.tensor(
        [[4.8, 2.0, 1.5], [1.0, 1.0, 1.7], [2.0, 1.0, 1.5]],
        dtype=torch.float32,
    )[:num_agents]
    data["agent"]["batch"] = torch.zeros(num_agents, dtype=torch.long)
    data["scenario_id"] = ["synthetic"]
    data["tfrecord_path"] = ["synthetic.tfrecords"]
    return data


def _make_two_scenario_data(order: tuple[str, str] = ("A", "B")) -> HeteroData:
    agents_by_scenario = {
        "A": [
            {"type": 0, "y": 0.0, "speed": 0.30, "shape": [4.8, 2.0, 1.5]},
            {"type": 1, "y": 1.0, "speed": 0.35, "shape": [1.0, 1.0, 1.7]},
        ],
        "B": [
            {"type": 2, "y": 10.0, "speed": 0.40, "shape": [2.0, 1.0, 1.5]},
            {"type": 0, "y": 11.0, "speed": 0.45, "shape": [4.8, 2.0, 1.5]},
        ],
    }
    rows = [agent for scenario_id in order for agent in agents_by_scenario[scenario_id]]
    n_agent = len(rows)

    data = HeteroData()
    data["map_save"]["traj_pos"] = torch.tensor(
        [
            [[0.0, 0.0], [2.5, 0.0], [5.0, 0.0]],
            [[0.0, 10.0], [2.5, 10.0], [5.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    data["map_save"]["traj_theta"] = torch.zeros(2)
    data["pt_token"]["type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["pl_type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["light_type"] = torch.zeros(2, dtype=torch.uint8)
    data["pt_token"]["num_nodes"] = 2
    data["pt_token"]["batch"] = torch.arange(2, dtype=torch.long)

    position = torch.zeros(n_agent, 91, 3)
    heading = torch.zeros(n_agent, 91)
    velocity = torch.zeros(n_agent, 91, 2)
    agent_type = torch.zeros(n_agent, dtype=torch.uint8)
    shape = torch.zeros(n_agent, 3)
    batch = torch.zeros(n_agent, dtype=torch.long)
    for agent_idx, row in enumerate(rows):
        speed = float(row["speed"])
        position[agent_idx, :, 0] = torch.arange(91, dtype=torch.float32) * speed
        position[agent_idx, :, 1] = float(row["y"])
        velocity[agent_idx, :, 0] = speed / 0.1
        agent_type[agent_idx] = int(row["type"])
        shape[agent_idx] = torch.tensor(row["shape"], dtype=torch.float32)
        batch[agent_idx] = agent_idx // 2

    data["agent"]["num_nodes"] = n_agent
    data["agent"]["valid_mask"] = torch.ones(n_agent, 91, dtype=torch.bool)
    data["agent"]["role"] = torch.zeros(n_agent, 3, dtype=torch.bool)
    data["agent"]["role"][0, 0] = True
    data["agent"]["id"] = torch.arange(n_agent, dtype=torch.long)
    data["agent"]["type"] = agent_type
    data["agent"]["position"] = position
    data["agent"]["heading"] = heading
    data["agent"]["velocity"] = velocity
    data["agent"]["shape"] = shape
    data["agent"]["batch"] = batch
    data["scenario_id"] = list(order)
    data["tfrecord_path"] = [f"{scenario_id}.tfrecords" for scenario_id in order]
    return data


def _make_model_cfg(anchor_path: Path, **overrides):
    cfg = {
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "lr_warmup_steps": 0,
        "lr_total_steps": 1,
        "anchor_file": str(anchor_path),
        "num_historical_steps": 11,
        "prediction_horizon_steps": 40,
        "commit_steps": 5,
        "match_steps": 5,
        "first_context_step": 10,
        "last_train_context_step": 85,
        "anchor_heading_weight": 1.0,
        "anchor_match_chunk_size": 16,
        "use_closed_loop_training": True,
        "inference_temperature": 1.0,
        "validation_closed_seed": 0,
        "val_open_loop": False,
        "val_closed_loop": False,
        "n_rollout_closed_val": 1,
        "n_batch_sim_agents_metric": 0,
        "n_vis_batch": 0,
        "n_vis_scenario": 0,
        "n_vis_rollout": 0,
        "loss_weights": {"cls": 1.0, "reg": 1.0},
        "decoder": {
            "hidden_dim": 32,
            "num_freq_bands": 8,
            "num_heads": 2,
            "head_dim": 8,
            "dropout": 0.0,
            "num_map_layers": 1,
            "num_agent_layers": 1,
            "pl2pl_radius": 20.0,
            "pl2a_radius": 50.0,
            "a2a_radius": 50.0,
            "time_span": 8,
            "min_laplace_scale": 0.05,
            "min_von_mises_concentration": 0.001,
            "max_von_mises_concentration": 100.0,
        },
        "sim_agents_submission": {
            "is_active": False,
            "method_name": "UniMM-test",
            "authors": ["test"],
            "affiliation": "test",
            "description": "test",
            "method_link": "test",
            "account_name": "test",
            "num_model_parameters": "4M",
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_unimm_processor_builds_closed_loop_training_batch():
    data = _make_data()
    payload = _make_anchor_payload()
    anchors = torch.stack(
        [torch.as_tensor(payload["anchors"][name], dtype=torch.float32) for name in ("veh", "ped", "cyc")],
        dim=0,
    )
    processor = UniMMProcessor(anchor_match_chunk_size=16)
    batch = processor.build_training_batch(
        data=data,
        anchors_by_type=anchors,
        posterior_threshold=torch.full((3,), 1e9),
        use_closed_loop=True,
    )

    assert batch.target_local.shape == (3, 16, 40, 3)
    assert batch.target_valid.shape == (3, 16, 40)
    assert batch.z_star.shape == (3, 16)
    assert batch.tokenized_agent["state_pos"].shape == (3, 17, 2)
    assert batch.tokenized_agent["tracklet_pos"].shape == (3, 17, 5, 2)
    assert batch.tokenized_agent["tracklet_head"].shape == (3, 17, 5)
    assert batch.tokenized_agent["tracklet_valid"].shape == (3, 17, 5)
    assert torch.equal(batch.context_indices, torch.arange(1, 17))
    assert batch.target_valid[:, -1, :5].all()
    assert not batch.target_valid[:, -1, 5:].any()
    assert torch.isfinite(batch.tokenized_agent["tracklet_pos"]).all()
    assert torch.isclose(batch.posterior_stats["accept_rate"], torch.tensor(1.0))
    assert batch.posterior_stats["accept_rate_by_type"].shape == (3,)
    assert batch.posterior_stats["accept_rate_by_context"].shape == (15,)
    assert torch.equal(
        batch.posterior_stats["context_raw_steps"],
        torch.arange(10, 85, 5),
    )


def test_unimm_classification_loss_uses_positive_matching_horizon_only():
    logits = torch.tensor(
        [
            [[4.0, -4.0], [-4.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    z_star = torch.tensor([[0, 0]], dtype=torch.long)
    valid = torch.zeros(1, 2, 40, dtype=torch.bool)
    valid[:, :, 5:] = True

    loss = unimm_classification_loss(logits, z_star, valid, match_steps=5)

    assert loss.item() == 0.0


def test_unimm_positive_matching_tie_break_uses_prediction_tail_only_for_near_ties():
    anchors = torch.zeros(3, 2, 40, 3)
    anchors[0, 0, 5:, 0] = 100.0
    anchors[0, 1, 5:, 0] = 1.0
    target = torch.zeros(1, 40, 3)
    target[:, 5:, 0] = 1.0
    valid = torch.ones(1, 40, dtype=torch.bool)
    agent_type = torch.zeros(1, dtype=torch.long)

    z_no_tie, _ = match_anchors_by_type(
        anchors,
        agent_type,
        target,
        valid,
        horizon_steps=5,
        row_chunk_size=4,
    )
    z_tie, _ = match_anchors_by_type(
        anchors,
        agent_type,
        target,
        valid,
        horizon_steps=5,
        row_chunk_size=4,
        tie_break_horizon_steps=40,
        tie_break_tolerance=1e-4,
    )

    assert z_no_tie.item() == 0
    assert z_tie.item() == 1

    anchors[0, 1, :5, 0] = 0.1
    z_not_close, err_not_close = match_anchors_by_type(
        anchors,
        agent_type,
        target,
        valid,
        horizon_steps=5,
        row_chunk_size=4,
        tie_break_horizon_steps=40,
        tie_break_tolerance=1e-4,
    )

    assert z_not_close.item() == 0
    assert err_not_close.item() == 0.0


def test_unimm_regression_loss_averages_valid_timesteps():
    pred = {
        "mean_pos": torch.zeros(2, 3, 5, 2),
        "pos_scale": torch.ones(2, 3, 5, 2),
        "mean_head": torch.zeros(2, 3, 5),
        "head_concentration": torch.ones(2, 3, 5),
    }
    target_local = torch.zeros(2, 3, 5, 3)
    target_valid = torch.zeros(2, 3, 5, dtype=torch.bool)
    target_valid[0, 0, :] = True
    target_valid[0, 1, :2] = True

    loss = unimm_nll_loss(pred, target_local, target_valid)
    per_step_loss = unimm_nll_loss(pred, target_local, target_valid[0:1, 0:1])

    assert torch.allclose(loss, per_step_loss)


def test_unimm_regression_loss_scale_is_horizon_invariant():
    pred = {
        "mean_pos": torch.zeros(2, 3, 40, 2),
        "pos_scale": torch.ones(2, 3, 40, 2),
        "mean_head": torch.zeros(2, 3, 40),
        "head_concentration": torch.ones(2, 3, 40),
    }
    target_local = torch.zeros(2, 3, 40, 3)
    target_valid = torch.ones(2, 3, 40, dtype=torch.bool)

    full_horizon_loss = unimm_nll_loss(pred, target_local, target_valid)
    short_pred = {
        "mean_pos": pred["mean_pos"][..., :5, :],
        "pos_scale": pred["pos_scale"][..., :5, :],
        "mean_head": pred["mean_head"][..., :5],
        "head_concentration": pred["head_concentration"][..., :5],
    }
    short_horizon_loss = unimm_nll_loss(short_pred, target_local[..., :5, :], target_valid[..., :5])

    assert torch.allclose(full_horizon_loss, short_horizon_loss)


def test_unimm_motion_decoder_caps_heading_concentration():
    decoder = UniMMMotionDecoder(
        hidden_dim=4,
        num_anchors=2,
        num_prediction_steps=2,
        min_laplace_scale=0.05,
        min_von_mises_concentration=0.001,
        max_von_mises_concentration=100.0,
    )
    with torch.no_grad():
        for param in decoder.parameters():
            param.zero_()
        decoder.regressor[-1].bias[5::6].fill_(1000.0)

    pred = decoder.decode_selected(
        agent_embedding=torch.zeros(1, 4),
        selected_anchor=torch.zeros(1, 2, 3),
    )

    assert torch.isfinite(pred["head_concentration"]).all()
    assert pred["head_concentration"].max().item() == 100.0


def test_unimm_minibatch_kmeans_builds_valid_anchor_bank():
    trajectories = torch.zeros(16, 80, 3)
    trajectories[:8, :, 0] = torch.linspace(0.1, 8.0, 80)
    trajectories[8:, :, 1] = torch.linspace(0.1, 4.0, 80)

    anchors = minibatch_kmeans(
        trajectories=trajectories,
        num_clusters=4,
        num_iters=3,
        batch_size=8,
        heading_weight=1.0,
        seed=0,
    )
    threshold = compute_threshold(
        trajectories=trajectories,
        anchors=anchors,
        match_steps=5,
        quantile=0.95,
        heading_weight=1.0,
        row_chunk_size=8,
    )

    assert anchors.shape == (4, 80, 3)
    assert torch.isfinite(anchors).all()
    assert threshold >= 0.0


def test_unimm_lloyd_refinement_uses_full_dataset_assignment():
    trajectories = torch.zeros(24, 20, 3)
    trajectories[:12, :, 0] = torch.linspace(0.1, 2.0, 20)
    trajectories[12:, :, 1] = torch.linspace(0.1, 3.0, 20)
    init = trajectories[torch.tensor([0, 1, 12, 13])].clone()
    _, before = nearest_anchor_assignment(
        trajectories=trajectories,
        anchors=init,
        heading_weight=1.0,
        row_chunk_size=8,
        anchor_chunk_size=2,
    )

    anchors, history = lloyd_refine_kmeans(
        trajectories=trajectories,
        centroids=init,
        num_iters=3,
        heading_weight=1.0,
        row_chunk_size=8,
        anchor_chunk_size=2,
        tol=0.0,
    )
    _, after = nearest_anchor_assignment(
        trajectories=trajectories,
        anchors=anchors,
        heading_weight=1.0,
        row_chunk_size=8,
        anchor_chunk_size=2,
    )

    assert anchors.shape == (4, 20, 3)
    assert len(history) >= 1
    assert after.mean() <= before.mean()


def test_unimm_anchor_collection_reads_cache_file(tmp_path: Path):
    data = _make_data()
    agent = {
        key: data["agent"][key]
        for key in ("position", "heading", "valid_mask", "type")
    }
    with (tmp_path / "sample.pkl").open("wb") as handle:
        pickle.dump({"agent": agent}, handle)

    trajectories = collect_training_trajectories(
        train_cache_dir=tmp_path,
        max_per_type=None,
        horizon_steps=40,
        start_step=10,
        seed=0,
        num_workers=0,
        collect_file_chunk_size=1,
    )

    assert set(trajectories) == {"veh", "ped", "cyc"}
    assert trajectories["veh"].shape == (1, 40, 3)
    assert trajectories["ped"].shape == (1, 40, 3)
    assert trajectories["cyc"].shape == (1, 40, 3)


def test_unimm_context_thresholds_use_late_context_windows(tmp_path: Path):
    data = _make_data()
    agent = {
        key: data["agent"][key]
        for key in ("position", "heading", "valid_mask", "type")
    }
    with (tmp_path / "sample.pkl").open("wb") as handle:
        pickle.dump({"agent": agent}, handle)

    payload = _make_anchor_payload()
    anchors_by_name = {
        name: torch.as_tensor(payload["anchors"][name], dtype=torch.float32)
        for name in ("veh", "ped", "cyc")
    }
    thresholds, counts = compute_context_thresholds_from_cache(
        train_cache_dir=tmp_path,
        anchors_by_name=anchors_by_name,
        match_steps=5,
        context_steps=[10, 85],
        quantile=0.95,
        heading_weight=1.0,
        row_chunk_size=8,
        device=torch.device("cpu"),
        num_workers=0,
        collect_file_chunk_size=1,
    )

    assert set(thresholds) == {"veh", "ped", "cyc"}
    assert counts == {"veh": 2, "ped": 2, "cyc": 2}
    assert all(value >= 0.0 for value in thresholds.values())


def test_unimm_lightning_training_step_runs(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    cfg = OmegaConf.create(
        {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "lr_warmup_steps": 0,
            "lr_total_steps": 1,
            "anchor_file": str(anchor_path),
            "num_historical_steps": 11,
            "prediction_horizon_steps": 40,
            "commit_steps": 5,
            "match_steps": 5,
            "first_context_step": 10,
            "last_train_context_step": 85,
            "anchor_heading_weight": 1.0,
            "anchor_match_chunk_size": 16,
            "use_closed_loop_training": True,
            "inference_temperature": 1.0,
            "validation_closed_seed": 0,
            "val_open_loop": False,
            "val_closed_loop": False,
            "n_rollout_closed_val": 1,
            "n_batch_sim_agents_metric": 0,
            "n_vis_batch": 0,
            "n_vis_scenario": 0,
            "n_vis_rollout": 0,
            "loss_weights": {"cls": 1.0, "reg": 1.0},
            "decoder": {
                "hidden_dim": 32,
                "num_freq_bands": 8,
                "num_heads": 2,
                "head_dim": 8,
                "dropout": 0.0,
                "num_map_layers": 1,
                "num_agent_layers": 1,
                "pl2pl_radius": 20.0,
                "pl2a_radius": 50.0,
                "a2a_radius": 50.0,
                "time_span": 8,
                "min_laplace_scale": 0.05,
                "min_von_mises_concentration": 0.001,
                "max_von_mises_concentration": 100.0,
            },
            "sim_agents_submission": {
                "is_active": False,
                "method_name": "UniMM-test",
                "authors": ["test"],
                "affiliation": "test",
                "description": "test",
                "method_link": "test",
                "account_name": "test",
                "num_model_parameters": "4M",
            },
        }
    )
    model = UniMMAnchorBased4s(cfg)
    loss, logs = model._forward_loss(_make_data(), use_closed_loop=True)
    assert torch.isfinite(loss)
    assert "posterior_accept_rate" in logs
    assert "posterior_error_p95" in logs
    assert "posterior_accept_rate_veh" in logs
    assert "posterior_accept_rate_ctx_80" in logs


def test_unimm_rollout_shapes(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    cfg = OmegaConf.create(
        {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "lr_warmup_steps": 0,
            "lr_total_steps": 1,
            "anchor_file": str(anchor_path),
            "num_historical_steps": 11,
            "prediction_horizon_steps": 40,
            "commit_steps": 5,
            "match_steps": 5,
            "first_context_step": 10,
            "last_train_context_step": 85,
            "anchor_heading_weight": 1.0,
            "anchor_match_chunk_size": 16,
            "use_closed_loop_training": True,
            "inference_temperature": 1.0,
            "validation_closed_seed": 0,
            "val_open_loop": False,
            "val_closed_loop": False,
            "n_rollout_closed_val": 2,
            "n_batch_sim_agents_metric": 0,
            "n_vis_batch": 0,
            "n_vis_scenario": 0,
            "n_vis_rollout": 0,
            "loss_weights": {"cls": 1.0, "reg": 1.0},
            "decoder": {
                "hidden_dim": 32,
                "num_freq_bands": 8,
                "num_heads": 2,
                "head_dim": 8,
                "dropout": 0.0,
                "num_map_layers": 1,
                "num_agent_layers": 1,
                "pl2pl_radius": 20.0,
                "pl2a_radius": 50.0,
                "a2a_radius": 50.0,
                "time_span": 8,
                "min_laplace_scale": 0.05,
                "min_von_mises_concentration": 0.001,
                "max_von_mises_concentration": 100.0,
            },
            "sim_agents_submission": {
                "is_active": False,
                "method_name": "UniMM-test",
                "authors": ["test"],
                "affiliation": "test",
                "description": "test",
                "method_link": "test",
                "account_name": "test",
                "num_model_parameters": "4M",
            },
        }
    )
    model = UniMMAnchorBased4s(cfg).eval()
    pred_traj, pred_z, pred_head = model._run_closed_loop_rollouts(_make_data(), ["synthetic"])
    assert pred_traj.shape == (3, 2, 80, 2)
    assert pred_z.shape == (3, 2, 80)
    assert pred_head.shape == (3, 2, 80)


def test_unimm_inference_samples_components_instead_of_argmax(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(num_anchors=2), handle)

    model = UniMMAnchorBased4s(_make_model_cfg(anchor_path)).eval()
    logits = torch.zeros(256, 2)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(0)

    sampled = model._sample_component(logits, generator)

    assert sampled.max().item() == 1
    assert model._sample_component(logits, None).shape == (256,)


def test_unimm_rejects_invalid_inference_sampling_config(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    invalid_overrides = [
        {"inference_top_k": -1},
        {"inference_top_p": 0.0},
        {"inference_top_p": -0.1},
        {"inference_top_p": 1.1},
        {"inference_top_p": float("nan")},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError):
            UniMMAnchorBased4s(_make_model_cfg(anchor_path, **overrides))


def test_unimm_top_k_and_top_p_sampling_remap_to_original_anchor_indices(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(num_anchors=5), handle)

    top_k_model = UniMMAnchorBased4s(
        _make_model_cfg(anchor_path, inference_top_k=2, inference_top_p=1.0)
    ).eval()
    logits = torch.tensor([[0.0, 1.0, 10.0, 9.0, 8.0]]).repeat(128, 1)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(0)

    sampled = top_k_model._sample_component(logits, generator)

    assert set(sampled.tolist()).issubset({2, 3})

    top_p_model = UniMMAnchorBased4s(
        _make_model_cfg(anchor_path, inference_top_k=3, inference_top_p=0.01)
    ).eval()
    sampled = top_p_model._sample_component(logits[:16], None)

    assert torch.equal(sampled, torch.full((16,), 2, dtype=torch.long))


def test_unimm_lr_schedule_starts_at_initial_lr_and_decays_to_zero(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    model = UniMMAnchorBased4s(
        _make_model_cfg(
            anchor_path,
            lr=0.001224744871,
            lr_warmup_steps=0,
            lr_total_steps=64,
        )
    )

    assert math.isclose(model._lr_multiplier(0), 1.0)
    assert math.isclose(model._lr_multiplier(32), 0.5)
    assert math.isclose(model._lr_multiplier(64), 0.0, abs_tol=1e-12)
    optimizers, schedulers = model.configure_optimizers()
    assert math.isclose(optimizers[0].param_groups[0]["lr"], 0.001224744871)
    assert schedulers[0]["interval"] == "epoch"


def test_unimm_lr_schedule_warms_up_then_decays_to_zero(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    model = UniMMAnchorBased4s(
        _make_model_cfg(
            anchor_path,
            lr=0.001224744871,
            lr_warmup_steps=4,
            lr_total_steps=64,
        )
    )

    assert math.isclose(model._lr_multiplier(0), 0.25)
    assert math.isclose(model._lr_multiplier(1), 0.5)
    assert math.isclose(model._lr_multiplier(2), 0.75)
    assert math.isclose(model._lr_multiplier(3), 1.0)
    assert math.isclose(model._lr_multiplier(4), 1.0)
    assert math.isclose(model._lr_multiplier(34), 0.5)
    assert math.isclose(model._lr_multiplier(64), 0.0, abs_tol=1e-12)


def test_unimm_scorer_scene_num_sets_metric_batch_count(tmp_path: Path):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    model = UniMMAnchorBased4s(
        _make_model_cfg(
            anchor_path,
            n_batch_sim_agents_metric=10,
            scorer_scene_num=1680,
        )
    )
    model._trainer = SimpleNamespace(
        world_size=6,
        datamodule=SimpleNamespace(val_batch_size=12),
        is_global_zero=False,
    )

    model._apply_scorer_scene_num_overrides()

    assert model.n_batch_sim_agents_metric == 24


def test_unimm_inference_rollout_commits_half_second_chunks(tmp_path: Path, monkeypatch):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(), handle)

    model = UniMMAnchorBased4s(_make_model_cfg(anchor_path)).eval()
    calls = {"count": 0}

    def fake_predict_one_step(tokenized_map, tokenized_agent, current_pos, current_head, generator):
        del tokenized_map, tokenized_agent, generator
        calls["count"] += 1
        offsets = torch.arange(1, 41, device=current_pos.device, dtype=current_pos.dtype)
        pred_pos = current_pos[:, None, :].repeat(1, 40, 1)
        pred_pos[..., 0] += offsets.view(1, 40)
        pred_head = current_head[:, None] + offsets.view(1, 40) * 0.01
        z = torch.full((current_pos.shape[0],), calls["count"] - 1, dtype=torch.long)
        return pred_pos, pred_head, z

    monkeypatch.setattr(model, "_predict_one_step", fake_predict_one_step)
    data = _make_data()
    current_x = data["agent"]["position"][:, 10, 0]

    pred_traj, _, pred_head = model._run_one_rollout(data, rollout_idx=0)

    expected_offsets = torch.arange(1, 81, dtype=pred_traj.dtype).view(1, 80)
    assert calls["count"] == 16
    assert torch.allclose(pred_traj[..., 0], current_x.view(-1, 1) + expected_offsets)
    assert torch.allclose(pred_head[:, :5], torch.arange(1, 6).view(1, 5) * 0.01)


def test_unimm_rollout_sampling_is_stable_when_scenario_batch_order_changes(
    tmp_path: Path,
    monkeypatch,
):
    anchor_path = tmp_path / "anchors.pkl"
    with anchor_path.open("wb") as handle:
        pickle.dump(_make_anchor_payload(num_anchors=4), handle)

    model = UniMMAnchorBased4s(
        _make_model_cfg(
            anchor_path,
            inference_temperature=1.0,
            inference_top_k=0,
            inference_top_p=1.0,
        )
    ).eval()

    def fake_predict_one_step(tokenized_map, tokenized_agent, current_pos, current_head, generator):
        del tokenized_map
        logits = torch.zeros(current_pos.shape[0], 4, device=current_pos.device)
        if isinstance(generator, dict):
            z = model._sample_component_by_agent_batch(
                logits=logits,
                agent_batch=tokenized_agent["batch"],
                generators_by_batch=generator,
            )
        else:
            z = model._sample_component(logits, generator)
        offsets = z.to(dtype=current_pos.dtype).add(1.0)
        pred_pos = current_pos[:, None, :].repeat(1, 40, 1)
        pred_pos[..., 0] += offsets.view(-1, 1)
        pred_head = current_head[:, None].repeat(1, 40)
        return pred_pos, pred_head, z

    monkeypatch.setattr(model, "_predict_one_step", fake_predict_one_step)
    data_ab = _make_two_scenario_data(("A", "B"))
    data_ba = _make_two_scenario_data(("B", "A"))

    pred_ab, _, _ = model._run_one_rollout(data_ab, rollout_idx=3)
    pred_ba, _, _ = model._run_one_rollout(data_ba, rollout_idx=3)

    assert torch.allclose(pred_ab[0:2], pred_ba[2:4])
    assert torch.allclose(pred_ab[2:4], pred_ba[0:2])
