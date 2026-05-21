from pathlib import Path


def is_smart_cache_sample_file(path: Path) -> bool:
    """Return whether ``path`` is a real SMART scenario cache sample."""

    return path.is_file() and path.suffix == ".pkl" and not path.name.startswith(".")
