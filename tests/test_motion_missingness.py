from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.smart.modules.flow_agent_decoder as flow_agent_decoder_module
from src.smart.modules.agent_encoder import SMARTAgentEncoder
from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.model.smart_flow import SMARTFlow


class _ZeroTokenEmbedding:
    def __call__(self, token):
        return torch.zeros((token.shape[0], 1), device=token.device, dtype=torch.float32)


class _ZeroCategoricalEmbedding:
    def __call__(self, value):
        return torch.zeros((value.shape[0], 1), device=value.device, dtype=torch.float32)


class _RecordingEmbedding:
    def __init__(self) -> None:
        self.continuous_inputs: torch.Tensor | None = None
        self.input_dim = 3

    def __call__(self, continuous_inputs, categorical_embs=None):
        self.continuous_inputs = continuous_inputs.detach().clone()
        return torch.zeros((continuous_inputs.shape[0], 1), device=continuous_inputs.device)


class _ZeroFusion:
    def __call__(self, inputs):
        return torch.zeros((*inputs.shape[:-1], 1), device=inputs.device, dtype=inputs.dtype)


class _IdentityAttention:
    def __call__(self, inputs, relation, edge_index):
        if isinstance(inputs, tuple):
            return inputs[1]
        return inputs


class _ZeroFlowODE:
    solver_steps = 1
    solver_method = "euler"

    def __init__(self, decoder: SMARTFlowAgentDecoder) -> None:
        self.decoder = decoder

    def generate(self, x_init, model_fn, steps, method, **kwargs):
        return x_init.new_zeros(
            (x_init.shape[0], self.decoder.flow_window_steps, self.decoder.flow_state_dim)
        )


class _StraightCommitBridge:
    config = SimpleNamespace(history_steps=6)

    def commit(self, y_hat_norm, current_pos, current_head, agent_type=None, agent_length=None):
        offsets = torch.arange(1, 6, device=current_pos.device, dtype=current_pos.dtype).view(1, 5, 1)
        commit_pos = current_pos.unsqueeze(1) + torch.cat([offsets, offsets.new_zeros(offsets.shape)], dim=-1)
        commit_head = current_head.unsqueeze(1).expand(-1, 5)
        return commit_pos, commit_head, None, None

    def retokenize(self, current_pos, current_head, commit_pos, commit_head, *args, **kwargs):
        return torch.zeros((current_pos.shape[0],), device=current_pos.device, dtype=torch.long)

    def build_stop_motion_mask(self, *args, **kwargs):
        current_pos = kwargs["current_pos"]
        return None, torch.zeros((current_pos.shape[0],), device=current_pos.device, dtype=torch.bool)


def _make_flow_decoder() -> SMARTFlowAgentDecoder:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.hidden_dim = 1
    decoder.shift = 5
    decoder.num_historical_steps = 11
    decoder.num_future_steps = 10
    decoder.flow_window_steps = 5
    decoder.flow_state_dim = 4
    decoder.num_layers = 1
    decoder.a2a_radius = 100.0
    decoder.closed_loop_rollout_mode = "raw_fm"
    decoder.use_lqr = False
    decoder.use_stop_motion = False
    decoder.flow_ode = _ZeroFlowODE(decoder)
    decoder.flow_decoder = lambda hidden, x_t, tau: x_t
    decoder.commit_bridge = _StraightCommitBridge()
    decoder.token_emb_veh = _ZeroTokenEmbedding()
    decoder.token_emb_ped = _ZeroTokenEmbedding()
    decoder.token_emb_cyc = _ZeroTokenEmbedding()
    decoder.type_a_emb = _ZeroCategoricalEmbedding()
    decoder.shape_emb = _ZeroCategoricalEmbedding()
    decoder.x_a_emb = _RecordingEmbedding()
    decoder.r_a2a_emb = _RecordingEmbedding()
    decoder.fusion_emb = _ZeroFusion()
    decoder.t_attn_layers = [_IdentityAttention()]
    decoder.pt2a_attn_layers = [_IdentityAttention()]
    decoder.a2a_attn_layers = [_IdentityAttention()]
    decoder.build_temporal_edge = lambda *args, **kwargs: (
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0, 1)),
    )
    decoder.build_map2agent_edge = lambda *args, **kwargs: (
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0, 1)),
    )
    return decoder


def _make_context_inputs():
    pos = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    heading = torch.zeros((3, 3), dtype=torch.float32)
    valid = torch.tensor(
        [
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ],
        dtype=torch.bool,
    )
    tokenized_agent = {
        "trajectory_token_veh": torch.zeros((8, 1), dtype=torch.float32),
        "trajectory_token_ped": torch.zeros((8, 1), dtype=torch.float32),
        "trajectory_token_cyc": torch.zeros((8, 1), dtype=torch.float32),
        "type": torch.zeros(3, dtype=torch.long),
        "shape": torch.ones((3, 3), dtype=torch.float32),
        "batch": torch.zeros(3, dtype=torch.long),
        "num_graphs": 1,
    }
    map_feature = {
        "pt_token": torch.zeros((0, 1), dtype=torch.float32),
        "position": torch.zeros((0, 2), dtype=torch.float32),
        "orientation": torch.zeros(0, dtype=torch.float32),
        "batch": torch.zeros(0, dtype=torch.long),
    }
    return pos, heading, valid, tokenized_agent, map_feature


def _fake_same_step_edges(x, r, batch, loop, max_num_neighbors):
    # build_interaction_edge now filters invalid nodes before radius_graph.
    # These filtered-node indices map back to original step-major edges
    # final-step agent2 -> agent0 and agent2 -> agent1.
    return torch.tensor([[7, 7], [5, 6]], device=x.device, dtype=torch.long)


def _run_encode_context_with_recorded_motion():
    decoder = _make_flow_decoder()
    pos, heading, valid, tokenized_agent, map_feature = _make_context_inputs()
    original_radius_graph = flow_agent_decoder_module.radius_graph
    flow_agent_decoder_module.radius_graph = _fake_same_step_edges
    try:
        decoder._encode_context(
            agent_token_index=torch.zeros((3, 3), dtype=torch.long),
            pos_a=pos,
            head_a=heading,
            mask=valid,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
    finally:
        flow_agent_decoder_module.radius_graph = original_radius_graph
    assert decoder.x_a_emb.continuous_inputs is not None
    assert decoder.r_a2a_emb.continuous_inputs is not None
    return decoder.x_a_emb.continuous_inputs, decoder.r_a2a_emb.continuous_inputs


def test_motion_feature_marks_missing_motion_separately_from_stationary_motion() -> None:
    pos = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [2.0, 0.0]],
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, True],
            [True, True, True, True],
        ]
    )
    head_vector = torch.zeros_like(pos)
    head_vector[..., 0] = 1.0

    motion_vector = SMARTAgentEncoder._build_motion_vector(pos, valid)
    motion_valid = SMARTAgentEncoder._build_motion_valid_mask(pos, valid)
    motion_feature = SMARTAgentEncoder._build_motion_feature(pos, head_vector, valid)

    assert motion_feature.shape == (2, 4, 3)
    assert motion_valid.tolist() == [
        [False, True, False, False],
        [False, True, True, True],
    ]
    assert torch.allclose(motion_vector[0, 2], torch.zeros(2))
    assert torch.allclose(motion_vector[0, 3], torch.zeros(2))

    # Both have zero-valued motion, but the validity bit keeps them separable.
    assert motion_feature[0, 3, 2].item() == 0.0
    assert motion_feature[1, 3, 2].item() == 1.0


def test_motion_feature_requires_validity_mask() -> None:
    pos = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="valid_mask is required"):
        SMARTAgentEncoder._build_motion_valid_mask(pos, None)


def test_agent_token_embedding_requires_validity_mask() -> None:
    decoder = _make_flow_decoder()
    pos = torch.zeros(1, 2, 2)
    head_vector = torch.zeros(1, 2, 2)
    head_vector[..., 0] = 1.0

    with pytest.raises(ValueError, match="valid_mask is required"):
        decoder.agent_token_embedding(
            agent_token_index=torch.zeros((1, 2), dtype=torch.long),
            trajectory_token_veh=torch.zeros((4, 1)),
            trajectory_token_ped=torch.zeros((4, 1)),
            trajectory_token_cyc=torch.zeros((4, 1)),
            pos_a=pos,
            head_vector_a=head_vector,
            agent_type=torch.zeros(1, dtype=torch.long),
            agent_shape=torch.ones((1, 3)),
            valid_mask=None,
        )


def test_open_loop_train_context_preserves_motion_missingness() -> None:
    motion_features, relation_features = _run_encode_context_with_recorded_motion()

    assert motion_features[2, -1].item() == 0.0  # invalid/valid boundary, zero value but missing.
    assert motion_features[5, -1].item() == 1.0  # stationary, zero value and valid.
    torch.testing.assert_close(motion_features[5, 0], torch.tensor(0.0), atol=1.0e-5, rtol=0.0)
    assert relation_features.shape[-1] == 3


def test_open_loop_eval_context_preserves_motion_missingness() -> None:
    motion_features, relation_features = _run_encode_context_with_recorded_motion()

    assert motion_features[0, -1].item() == 0.0  # first context step has no previous motion.
    assert motion_features[2, -1].item() == 0.0
    assert motion_features[8, -1].item() == 1.0
    assert relation_features.shape[-1] == 3


def test_interaction_edge_uses_geometry_only_and_skips_invalid_before_radius() -> None:
    decoder = _make_flow_decoder()
    pos = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
        ]
    )
    head = torch.zeros(3, 1)
    head_vector = torch.stack([head.cos(), head.sin()], dim=-1)
    mask = torch.tensor([[True], [False], [True]])
    batch_s = torch.zeros(3, dtype=torch.long)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_radius_graph(x, r, batch, loop, max_num_neighbors):
        calls.append((x.detach().clone(), batch.detach().clone()))
        return torch.tensor([[0, 1], [1, 0]], device=x.device, dtype=torch.long)

    original_radius_graph = flow_agent_decoder_module.radius_graph
    flow_agent_decoder_module.radius_graph = fake_radius_graph
    try:
        edge_index, relation = decoder.build_interaction_edge(
            pos_a=pos,
            head_a=head,
            head_vector_a=head_vector,
            batch_s=batch_s,
            mask=mask,
        )
    finally:
        flow_agent_decoder_module.radius_graph = original_radius_graph

    assert relation.shape[-1] == 1
    assert decoder.r_a2a_emb.continuous_inputs is not None
    assert decoder.r_a2a_emb.continuous_inputs.shape[-1] == 3
    assert calls[0][0].shape[0] == 2
    assert calls[0][1].shape[0] == 2
    assert 1 not in edge_index.reshape(-1).tolist()


def _make_rollout_tokenized_agent(valid_mask: torch.Tensor, pos: torch.Tensor | None = None):
    n_agent = int(valid_mask.shape[0])
    if pos is None:
        pos = torch.zeros((n_agent, valid_mask.shape[1], 2), dtype=torch.float32)
        pos[2, 1] = torch.tensor([1.0, 0.0])
    heading = torch.zeros((n_agent, valid_mask.shape[1]), dtype=torch.float32)
    token_bank = torch.zeros((8, 1), dtype=torch.float32)
    return {
        "trajectory_token_veh": torch.zeros((8, 1), dtype=torch.float32),
        "trajectory_token_ped": torch.zeros((8, 1), dtype=torch.float32),
        "trajectory_token_cyc": torch.zeros((8, 1), dtype=torch.float32),
        "type": torch.zeros(n_agent, dtype=torch.long),
        "shape": torch.ones((n_agent, 3), dtype=torch.float32),
        "batch": torch.zeros(n_agent, dtype=torch.long),
        "num_graphs": 1,
        "gt_idx": torch.zeros((n_agent, valid_mask.shape[1]), dtype=torch.long),
        "gt_pos": pos,
        "gt_heading": heading,
        "valid_mask": valid_mask,
        "rollout_init_fine_pos_history": torch.zeros((n_agent, 6, 2), dtype=torch.float32),
        "rollout_init_fine_head_history": torch.zeros((n_agent, 6), dtype=torch.float32),
        "rollout_init_fine_valid_history": torch.ones((n_agent, 6), dtype=torch.bool),
        "token_agent_shape": torch.ones((n_agent, 2), dtype=torch.float32),
        "token_bank_all_veh": token_bank,
        "token_bank_all_ped": token_bank,
        "token_bank_all_cyc": token_bank,
        "gt_pos_raw": torch.zeros((n_agent, 12, 2), dtype=torch.float32),
        "gt_head_raw": torch.zeros((n_agent, 12), dtype=torch.float32),
        "gt_valid_raw": torch.ones((n_agent, 12), dtype=torch.bool),
        "gt_z_raw": torch.zeros(n_agent, dtype=torch.float32),
    }


def _empty_map_feature():
    return {
        "pt_token": torch.zeros((0, 1), dtype=torch.float32),
        "position": torch.zeros((0, 2), dtype=torch.float32),
        "orientation": torch.zeros(0, dtype=torch.float32),
        "batch": torch.zeros(0, dtype=torch.long),
    }


def _fake_two_step_edges(x, r, batch, loop, max_num_neighbors):
    return torch.tensor([[4, 4], [2, 3]], device=x.device, dtype=torch.long)


def test_closed_loop_initial_cache_preserves_motion_missingness() -> None:
    decoder = _make_flow_decoder()
    valid = torch.tensor(
        [
            [False, True],
            [True, True],
            [True, True],
        ],
        dtype=torch.bool,
    )
    tokenized_agent = _make_rollout_tokenized_agent(valid)
    original_radius_graph = flow_agent_decoder_module.radius_graph
    flow_agent_decoder_module.radius_graph = _fake_two_step_edges
    try:
        decoder._prepare_rollout_cache_impl(
            tokenized_agent=tokenized_agent,
            map_feature=_empty_map_feature(),
        )
    finally:
        flow_agent_decoder_module.radius_graph = original_radius_graph

    assert decoder.x_a_emb.continuous_inputs is not None
    assert decoder.r_a2a_emb.continuous_inputs is not None
    motion_features = decoder.x_a_emb.continuous_inputs
    relation_features = decoder.r_a2a_emb.continuous_inputs
    assert motion_features[0, -1].item() == 0.0
    assert motion_features[1, -1].item() == 0.0
    assert motion_features[3, -1].item() == 1.0
    assert motion_features[5, -1].item() == 1.0
    assert relation_features.shape[-1] == 3


def _make_rollout_cache_for_update_test():
    valid = torch.ones((3, 2), dtype=torch.bool)
    tokenized_agent = _make_rollout_tokenized_agent(valid)
    return {
        "n_agent": 3,
        "n_step_future_10hz": 10,
        "n_step_future_2hz": 2,
        "max_context_steps": 14,
        "pos_window": tokenized_agent["gt_pos"].clone(),
        "head_window": tokenized_agent["gt_heading"].clone(),
        "head_vector_window": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]]] * 3,
            dtype=torch.float32,
        ),
        "valid_window": tokenized_agent["valid_mask"].clone(),
        "pred_idx_window": tokenized_agent["gt_idx"].clone(),
        "exec_pos_history_10hz": torch.zeros((3, 6, 2), dtype=torch.float32),
        "exec_head_history_10hz": torch.zeros((3, 6), dtype=torch.float32),
        "exec_valid_history_10hz": torch.ones((3, 6), dtype=torch.bool),
        "exec_pos_pair_10hz": torch.zeros((3, 2, 2), dtype=torch.float32),
        "exec_head_pair_10hz": torch.zeros((3, 2), dtype=torch.float32),
        "exec_valid_pair_10hz": torch.ones((3, 2), dtype=torch.bool),
        "feat_a": torch.zeros((3, 2, 1), dtype=torch.float32),
        "agent_token_emb": torch.zeros((3, 2, 1), dtype=torch.float32),
        "agent_token_emb_veh": torch.zeros((8, 1), dtype=torch.float32),
        "agent_token_emb_ped": torch.zeros((8, 1), dtype=torch.float32),
        "agent_token_emb_cyc": torch.zeros((8, 1), dtype=torch.float32),
        "veh_mask": torch.ones(3, dtype=torch.bool),
        "ped_mask": torch.zeros(3, dtype=torch.bool),
        "cyc_mask": torch.zeros(3, dtype=torch.bool),
        "categorical_embs": None,
        "feat_a_now": torch.zeros((3, 1), dtype=torch.float32),
        "feat_a_t_dict": {},
    }, tokenized_agent


def _run_rollout_and_record_interaction_kwargs(self_forced_epoch: int | None):
    decoder = _make_flow_decoder()
    rollout_cache, tokenized_agent = _make_rollout_cache_for_update_test()
    records: list[dict] = []
    original_build_interaction_edge = decoder.build_interaction_edge

    def record_build_interaction_edge(*args, **kwargs):
        records.append(dict(kwargs))
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    decoder.build_interaction_edge = record_build_interaction_edge
    decoder._rollout_from_cache_impl(
        rollout_cache=rollout_cache,
        tokenized_agent=tokenized_agent,
        map_feature=_empty_map_feature(),
        sampling_scheme=SimpleNamespace(noise_scale=0.0, sample_steps=1, sample_method="euler"),
        rollout_steps_2hz=2,
        self_forced_epoch=self_forced_epoch,
    )
    decoder.build_interaction_edge = original_build_interaction_edge
    return records


def test_closed_loop_update_does_not_pass_motion_to_relation_input() -> None:
    records = _run_rollout_and_record_interaction_kwargs(self_forced_epoch=None)

    assert len(records) == 1
    assert "motion_a" not in records[0]
    assert "motion_valid_a" not in records[0]


def test_self_forced_training_rollout_does_not_pass_motion_to_relation_input() -> None:
    records = _run_rollout_and_record_interaction_kwargs(self_forced_epoch=0)

    assert len(records) == 1
    assert "motion_a" not in records[0]
    assert "motion_valid_a" not in records[0]


def test_old_motion_feature_checkpoint_fails_with_clear_message() -> None:
    model = SMARTFlow.__new__(SMARTFlow)
    object.__setattr__(
        model,
        "state_dict",
        lambda: {
            "encoder.agent_encoder.x_a_emb.freqs.weight": torch.zeros((3, 4)),
            "encoder.agent_encoder.r_a2a_emb.freqs.weight": torch.zeros((3, 4)),
        },
    )
    checkpoint = {
        "state_dict": {
            "encoder.agent_encoder.x_a_emb.freqs.weight": torch.zeros((2, 4)),
            "encoder.agent_encoder.r_a2a_emb.freqs.weight": torch.zeros((3, 4)),
        }
    }

    with pytest.raises(RuntimeError, match="requires a fresh pretrain checkpoint"):
        model._assert_motion_missingness_checkpoint_compatible(checkpoint)


def test_six_dim_relation_checkpoint_fails_with_clear_message() -> None:
    model = SMARTFlow.__new__(SMARTFlow)
    object.__setattr__(
        model,
        "state_dict",
        lambda: {
            "encoder.agent_encoder.x_a_emb.freqs.weight": torch.zeros((3, 4)),
            "encoder.agent_encoder.r_a2a_emb.freqs.weight": torch.zeros((3, 4)),
        },
    )
    checkpoint = {
        "state_dict": {
            "encoder.agent_encoder.x_a_emb.freqs.weight": torch.zeros((3, 4)),
            "encoder.agent_encoder.r_a2a_emb.freqs.weight": torch.zeros((6, 4)),
        }
    }

    with pytest.raises(RuntimeError, match="requires a fresh pretrain checkpoint"):
        model._assert_motion_missingness_checkpoint_compatible(checkpoint)
