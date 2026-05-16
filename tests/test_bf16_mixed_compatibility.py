import torch
from torch import nn

from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.ego_gmm_agent_decoder import EgoGMMAgentDecoder
from src.utils.instantiators import _disable_wandb_log_model_when_offline


class _BFloat16TokenEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            continuous_inputs.shape[0],
            self.hidden_dim,
            device=continuous_inputs.device,
            dtype=torch.bfloat16,
        )


class _BFloat16AgentFeatureEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, continuous_inputs, categorical_embs=None) -> torch.Tensor:
        return torch.zeros(
            continuous_inputs.shape[0],
            self.hidden_dim,
            device=continuous_inputs.device,
            dtype=torch.bfloat16,
        )


class _IdentityFusion(nn.Module):
    def forward(self, continuous_inputs, categorical_embs=None) -> torch.Tensor:
        return continuous_inputs


def _agent_inputs():
    return {
        "agent_token_index": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        "trajectory_token_veh": torch.randn(5, 8),
        "trajectory_token_ped": torch.randn(5, 8),
        "trajectory_token_cyc": torch.randn(5, 8),
        "pos_a": torch.randn(3, 2, 2),
        "head_vector_a": torch.randn(3, 2, 2),
        "agent_type": torch.tensor([0, 1, 2]),
        "agent_shape": torch.randn(3, 3),
    }


def _patch_embedding_modules(decoder) -> None:
    hidden_dim = decoder.hidden_dim
    decoder.token_emb_veh = _BFloat16TokenEmbedding(hidden_dim)
    decoder.token_emb_ped = _BFloat16TokenEmbedding(hidden_dim)
    decoder.token_emb_cyc = _BFloat16TokenEmbedding(hidden_dim)
    decoder.x_a_emb = _BFloat16AgentFeatureEmbedding(hidden_dim)
    decoder.fusion_emb = _IdentityFusion()


def test_smart_agent_token_embedding_accepts_bf16_token_embeddings() -> None:
    decoder = SMARTAgentDecoder(
        hidden_dim=8,
        num_historical_steps=2,
        num_future_steps=80,
        time_span=30,
        pl2a_radius=30.0,
        a2a_radius=60.0,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        hist_drop_prob=0.0,
        n_token_agent=5,
    )
    _patch_embedding_modules(decoder)

    out = decoder.agent_token_embedding(
        **_agent_inputs(),
        valid_mask=torch.ones(3, 2, dtype=torch.bool),
    )

    assert out.dtype == torch.bfloat16


def test_ego_gmm_agent_token_embedding_accepts_bf16_token_embeddings() -> None:
    decoder = EgoGMMAgentDecoder(
        hidden_dim=8,
        num_historical_steps=2,
        num_future_steps=80,
        time_span=30,
        pl2a_radius=30.0,
        a2a_radius=60.0,
        num_freq_bands=2,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        hist_drop_prob=0.0,
        k_ego_gmm=3,
        cov_ego_gmm=[1.0, 0.1],
        cov_learnable=False,
    )
    _patch_embedding_modules(decoder)

    out = decoder.agent_token_embedding(**_agent_inputs())

    assert out.dtype == torch.bfloat16


def test_wandb_offline_disables_model_artifact_logging(monkeypatch) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
            "offline": False,
            "log_model": "all",
        }
    )
    monkeypatch.setenv("WANDB_MODE", "offline")

    _disable_wandb_log_model_when_offline(cfg)

    assert cfg.log_model is False
