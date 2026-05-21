import torch

from src.smart.layers.fourier_embedding import FourierEmbedding


def _make_embedding() -> FourierEmbedding:
    torch.manual_seed(11)
    return FourierEmbedding(input_dim=3, hidden_dim=16, num_freq_bands=5)


def test_accumulated_fourier_embedding_matches_loop_forward() -> None:
    embedding = _make_embedding()
    continuous_inputs = torch.randn(19, 3)
    categorical_embs = [torch.randn(19, 16), torch.randn(19, 16)]

    loop_pre = embedding._embed_continuous_loop(continuous_inputs)
    accumulated_pre = embedding._embed_continuous_accumulated(continuous_inputs)
    loop_out = embedding.to_out(loop_pre + torch.stack(categorical_embs).sum(dim=0))
    accumulated_out = embedding.to_out(
        accumulated_pre + embedding._sum_embeddings(categorical_embs)
    )

    torch.testing.assert_close(accumulated_pre, loop_pre, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(accumulated_out, loop_out, rtol=1e-5, atol=1e-5)


def test_accumulated_fourier_embedding_matches_loop_backward() -> None:
    loop_embedding = _make_embedding()
    accumulated_embedding = _make_embedding()
    accumulated_embedding.load_state_dict(loop_embedding.state_dict())

    loop_inputs = torch.randn(23, 3, requires_grad=True)
    accumulated_inputs = loop_inputs.detach().clone().requires_grad_(True)

    loop_out = loop_embedding.to_out(loop_embedding._embed_continuous_loop(loop_inputs))
    accumulated_out = accumulated_embedding.to_out(
        accumulated_embedding._embed_continuous_accumulated(accumulated_inputs)
    )
    loop_out.square().mean().backward()
    accumulated_out.square().mean().backward()

    torch.testing.assert_close(
        accumulated_inputs.grad, loop_inputs.grad, rtol=1e-5, atol=1e-5
    )
    for (_, loop_param), (_, accumulated_param) in zip(
        loop_embedding.named_parameters(),
        accumulated_embedding.named_parameters(),
    ):
        if loop_param.grad is None:
            assert accumulated_param.grad is None
        else:
            torch.testing.assert_close(
                accumulated_param.grad,
                loop_param.grad,
                rtol=1e-5,
                atol=1e-5,
            )


def test_accumulated_fourier_embedding_keeps_parameter_count() -> None:
    embedding = _make_embedding()
    n_params_before = sum(parameter.numel() for parameter in embedding.parameters())
    _ = embedding(torch.randn(7, 3))
    n_params_after = sum(parameter.numel() for parameter in embedding.parameters())

    assert n_params_after == n_params_before


def test_accumulated_fourier_embedding_handles_empty_inputs() -> None:
    embedding = _make_embedding()
    output = embedding(torch.empty(0, 3))

    assert output.shape == (0, 16)
