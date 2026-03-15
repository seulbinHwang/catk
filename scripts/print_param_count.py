from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(repo_root / "configs/model/smart.yaml")
    model_config = config.model_config
    token_processor = TokenProcessor(**model_config.token_processor)
    model = SMARTDecoder(
        **model_config.decoder,
        n_token_agent=token_processor.n_token_agent,
    )
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"SMARTDecoder total params: {total_params:,}")
    print(f"SMARTDecoder trainable params: {trainable_params:,}")


if __name__ == "__main__":
    main()
