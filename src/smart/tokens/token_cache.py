import os
from pathlib import Path
from typing import Any

import torch


CACHE_VERSION = 1
ENV_NAME = "SMART_NTP_TOKEN_CACHE"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def token_cache_enabled() -> bool:
    return os.environ.get(ENV_NAME, "").strip().lower() in _TRUE_VALUES


def token_cache_path(raw_path: str) -> Path:
    raw = Path(raw_path)
    return raw.parent / f".catk_smart_ntp_token_cache_v{CACHE_VERSION}" / f"{raw.stem}.pt"


def raw_file_stamp(raw_path: str) -> str:
    raw = Path(raw_path)
    stat = raw.stat()
    return f"{raw.name}:{stat.st_size}:{stat.st_mtime_ns}"


def load_token_cache(raw_path: str) -> dict[str, Any] | None:
    if not token_cache_enabled():
        return None
    cache_path = token_cache_path(raw_path)
    if not cache_path.is_file():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("raw_stamp") != raw_file_stamp(raw_path):
        return None
    return payload


def empty_token_cache(num_agent_token_steps: int) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "fingerprint": "",
        "raw_stamp": "",
        "map": {
            "token_idx": torch.empty(0, dtype=torch.long),
        },
        "agent": {
            "gt_pos_raw": torch.empty(0, num_agent_token_steps, 2),
            "gt_head_raw": torch.empty(0, num_agent_token_steps),
            "gt_valid_raw": torch.empty(0, num_agent_token_steps, dtype=torch.bool),
            "valid_mask": torch.empty(0, num_agent_token_steps, dtype=torch.bool),
            "gt_idx": torch.empty(0, num_agent_token_steps, dtype=torch.long),
            "gt_pos": torch.empty(0, num_agent_token_steps, 2),
            "gt_heading": torch.empty(0, num_agent_token_steps),
            "sampled_idx": torch.empty(0, num_agent_token_steps, dtype=torch.long),
            "sampled_pos": torch.empty(0, num_agent_token_steps, 2),
            "sampled_heading": torch.empty(0, num_agent_token_steps),
        },
    }


def save_token_cache(raw_path: str, payload: dict[str, Any]) -> None:
    if not token_cache_enabled():
        return
    cache_path = token_cache_path(raw_path)
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    try:
        payload = dict(payload)
        payload["raw_stamp"] = raw_file_stamp(raw_path)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
