from pathlib import Path

from src.smart.cache_filter import is_smart_cache_sample_file


def test_smart_cache_sample_filter_skips_metadata_and_non_pkl(tmp_path: Path) -> None:
    scenario = tmp_path / "100006d9c3e93b6e.pkl"
    hidden_metadata = tmp_path / ".catk_memory_balanced_metadata_v1.pkl"
    non_pkl = tmp_path / "notes.txt"
    subdir = tmp_path / "nested.pkl"

    scenario.write_bytes(b"scenario")
    hidden_metadata.write_bytes(b"metadata")
    non_pkl.write_text("not a cache sample")
    subdir.mkdir()

    assert is_smart_cache_sample_file(scenario)
    assert not is_smart_cache_sample_file(hidden_metadata)
    assert not is_smart_cache_sample_file(non_pkl)
    assert not is_smart_cache_sample_file(subdir)
