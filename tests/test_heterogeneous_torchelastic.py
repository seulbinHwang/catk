import pytest


def _heterogeneous_classes():
    pytest.importorskip("torch")
    module = pytest.importorskip("src.smart.utils.heterogeneous_torchelastic")
    return module.HeterogeneousTorchElasticEnvironment, module.HeterogeneousDDPStrategy


def test_heterogeneous_environment_allows_uneven_local_gpu_counts(monkeypatch) -> None:
    HeterogeneousTorchElasticEnvironment, _ = _heterogeneous_classes()
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    env = HeterogeneousTorchElasticEnvironment()
    env.validate_settings(num_devices=4, num_nodes=2)
    env.validate_settings(num_devices=2, num_nodes=2)


def test_heterogeneous_strategy_uses_launcher_world_size_for_sampler(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    HeterogeneousTorchElasticEnvironment, HeterogeneousDDPStrategy = _heterogeneous_classes()
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    env = HeterogeneousTorchElasticEnvironment()
    strategy = HeterogeneousDDPStrategy(
        cluster_environment=env,
        parallel_devices=[torch.device("cpu"), torch.device("cpu")],
    )

    assert strategy.world_size == 6
    assert strategy.global_rank == 4
    assert strategy.local_rank == 1
    assert strategy.distributed_sampler_kwargs == {"num_replicas": 6, "rank": 4}
