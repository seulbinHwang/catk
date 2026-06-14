import torch

from src.smart.utils.finetune import set_model_for_finetuning


class _DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder = torch.nn.Linear(2, 2)
        self.agent_encoder = torch.nn.Module()
        self.agent_encoder.token_predict_head = torch.nn.Linear(2, 2)
        self.agent_encoder.t_attn_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.agent_encoder.pt2a_attn_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.agent_encoder.a2a_attn_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.other = torch.nn.Linear(2, 2)


def test_map_encoder_only_finetune_freezes_only_map_encoder():
    model = _DummyEncoder()

    set_model_for_finetuning(model, finetune=True, freeze_mode="map_encoder_only")

    assert all(not p.requires_grad for p in model.map_encoder.parameters())
    trainable_non_map = [
        p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith("map_encoder.")
    ]
    assert trainable_non_map
    assert all(trainable_non_map)
