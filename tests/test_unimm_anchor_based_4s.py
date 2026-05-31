import math
import pickle
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from scripts.build_unimm_anchors import collect_training_trajectories, compute_threshold, minibatch_kmeans
from src.unimm.losses import unimm_classification_loss
from src.unimm.model.anchor_based_4s import UniMMAnchorBased4s
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
        "last_train_context_step": 50,
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

    assert batch.target_local.shape == (3, 9, 40, 3)
    assert batch.target_valid.shape == (3, 9, 40)
    assert batch.z_star.shape == (3, 9)
    assert batch.tokenized_agent["state_pos"].shape == (3, 10, 2)
    assert batch.tokenized_agent["tracklet_pos"].shape == (3, 10, 5, 2)
    assert batch.tokenized_agent["tracklet_head"].shape == (3, 10, 5)
    assert batch.tokenized_agent["tracklet_valid"].shape == (3, 10, 5)
    assert torch.isfinite(batch.tokenized_agent["tracklet_pos"]).all()


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
            "last_train_context_step": 50,
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
    loss = model.training_step(_make_data(), 0)
    assert torch.isfinite(loss)


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
            "last_train_context_step": 50,
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
