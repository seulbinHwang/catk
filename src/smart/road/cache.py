from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor


def clone_sample_to_cpu(value: Any) -> Any:
    """캐시 저장 전에 tensor를 안전하게 CPU 값으로 복사합니다.

    Args:
        value: pickle 캐시에 들어갈 값입니다. tensor, dict, list, tuple을 포함할 수 있습니다.

    Returns:
        Any: 원본과 같은 구조이며, tensor는 ``detach().cpu().clone()`` 된 값입니다.
    """
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, MutableMapping):
        return {k: clone_sample_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_sample_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_sample_to_cpu(v) for v in value)
    return value


def safe_scenario_id(sample: Mapping[str, Any], source_path: Path) -> str:
    """원본 scenario id를 안정적으로 가져옵니다.

    Args:
        sample: 원본 `.pkl`에서 읽은 scenario 사전입니다.
        source_path: scenario id가 없을 때 사용할 원본 파일 경로입니다.

    Returns:
        str: scenario id 문자열입니다.
    """
    scenario_id = sample.get("scenario_id")
    if scenario_id is None or str(scenario_id) == "":
        return source_path.stem
    return str(scenario_id)


def estimate_future_velocity(
    future_xy: Tensor,
    future_valid: Tensor,
    history_last_xy: Tensor,
    dt: float = 0.1,
) -> Tensor:
    """RoaD future 위치에서 10Hz 속도를 다시 계산합니다.

    Args:
        future_xy: RoaD future 중심점입니다. shape은 ``[n_agent, 80, 2]`` 입니다.
        future_valid: RoaD future 유효 여부입니다. shape은 ``[n_agent, 80]`` 입니다.
        history_last_xy: history 마지막 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
        dt: 시간 간격입니다. WOMD 10Hz 기준 기본값은 0.1초입니다.

    Returns:
        Tensor: future 속도입니다. shape은 ``[n_agent, 80, 2]`` 입니다.
    """
    if future_xy.dim() != 3 or future_xy.shape[-1] != 2:
        raise ValueError(f"future_xy must have shape [N,80,2], got {tuple(future_xy.shape)}.")
    if future_valid.shape != future_xy.shape[:2]:
        raise ValueError(
            "future_valid shape must match future_xy first two dims: "
            f"future_valid={tuple(future_valid.shape)}, future_xy={tuple(future_xy.shape)}."
        )
    prev_xy = torch.cat([history_last_xy.unsqueeze(1), future_xy[:, :-1]], dim=1)
    velocity = (future_xy - prev_xy) / float(dt)
    velocity = velocity.masked_fill(~future_valid.bool().unsqueeze(-1), 0.0)
    return velocity


def build_road_cache_sample(
    source_sample: Mapping[str, Any],
    rollout_xy: Tensor,
    rollout_heading: Tensor,
    rollout_valid: Tensor,
    rollout_index: int,
    source_path: Path,
) -> dict[str, Any]:
    """원본 WOMD 캐시와 같은 schema의 RoaD 캐시 sample을 만듭니다.

    Args:
        source_sample: 원본 `.pkl` sample입니다.
        rollout_xy: 선택된 RoaD future 중심점입니다. shape은 ``[n_agent, 80, 2]`` 입니다.
        rollout_heading: 선택된 RoaD future 방향입니다. shape은 ``[n_agent, 80]`` 입니다.
        rollout_valid: 선택된 RoaD future 유효 여부입니다. shape은 ``[n_agent, 80]`` 입니다.
        rollout_index: 같은 scenario에서 몇 번째 RoaD rollout인지 나타내는 번호입니다.
        source_path: 원본 `.pkl` 경로입니다.

    Returns:
        dict[str, Any]: 기존 학습 파이프라인이 바로 읽을 수 있는 RoaD `.pkl` sample입니다.
    """
    road_sample = clone_sample_to_cpu(source_sample)
    agent = road_sample["agent"]
    scenario_id = safe_scenario_id(source_sample, source_path)

    position = agent["position"].clone()
    heading = agent["heading"].clone()
    velocity = agent["velocity"].clone()
    valid_mask = agent["valid_mask"].clone()

    rollout_xy = rollout_xy.detach().cpu().to(dtype=position.dtype)
    rollout_heading = rollout_heading.detach().cpu().to(dtype=heading.dtype)
    rollout_valid = rollout_valid.detach().cpu().bool()

    n_agent = int(position.shape[0])
    if tuple(rollout_xy.shape) != (n_agent, 80, 2):
        raise ValueError(
            "rollout_xy must have shape [n_agent,80,2], "
            f"expected={(n_agent, 80, 2)}, actual={tuple(rollout_xy.shape)}."
        )
    if tuple(rollout_heading.shape) != (n_agent, 80):
        raise ValueError(
            "rollout_heading must have shape [n_agent,80], "
            f"expected={(n_agent, 80)}, actual={tuple(rollout_heading.shape)}."
        )

    original_future_valid = valid_mask[:, 11:91].bool()
    # RoaD는 future 위치와 방향만 교체합니다. 유효 mask와 z 값은 원본 WOMD cache를 유지합니다.
    future_valid = original_future_valid

    position[:, 11:91, :2] = rollout_xy
    heading[:, 11:91] = rollout_heading
    velocity[:, 11:91] = estimate_future_velocity(
        future_xy=rollout_xy,
        future_valid=future_valid,
        history_last_xy=position[:, 10, :2],
    )
    valid_mask[:, 11:91] = future_valid

    agent["position"] = position
    agent["heading"] = heading
    agent["velocity"] = velocity
    agent["valid_mask"] = valid_mask
    road_sample["agent"] = agent
    road_sample["scenario_id"] = f"{scenario_id}__road_r{int(rollout_index):02d}"
    road_sample["source_scenario_id"] = scenario_id
    road_sample["road_rollout_index"] = int(rollout_index)
    road_sample["road_source_path"] = source_path.as_posix()
    return road_sample


def write_pickle(sample: Mapping[str, Any], output_path: Path) -> None:
    """sample을 pickle 파일로 저장합니다.

    Args:
        sample: 저장할 sample 사전입니다.
        output_path: 저장할 `.pkl` 경로입니다.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)


def choose_variant_for_epoch(
    scenario_id: str,
    epoch_idx: int,
    num_variants: int,
    seed: int,
) -> int:
    """epoch마다 3개 rollout 중 하나를 균등하게 고릅니다.

    Args:
        scenario_id: 원본 scenario id입니다.
        epoch_idx: 현재 RoaD fine-tuning epoch 번호입니다. 0부터 시작합니다.
        num_variants: scenario당 RoaD rollout 개수입니다.
        seed: 재현성을 위한 기본 seed입니다.

    Returns:
        int: 선택된 rollout 번호입니다.
    """
    if num_variants <= 0:
        raise ValueError(f"num_variants must be positive, got {num_variants}.")
    key = f"{seed}:{epoch_idx}:{scenario_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return value % int(num_variants)


def link_or_copy_variant(source_path: Path, target_path: Path) -> None:
    """선택된 RoaD cache를 hardlink로 만들고, 실패하면 복사합니다.

    Args:
        source_path: 실제 RoaD variant cache 파일입니다.
        target_path: 이번 epoch 학습 loader가 읽을 파일 경로입니다.

    Returns:
        None
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def build_selected_epoch_cache(
    variant_dirs: Sequence[Path],
    selected_dir: Path,
    epoch_idx: int,
    seed: int,
    rank: int,
    world_size: int,
) -> int:
    """생성된 3N cache 중 scenario마다 하나만 골라 학습용 폴더를 만듭니다.

    Args:
        variant_dirs: rollout 번호별 cache 폴더 목록입니다.
        selected_dir: 이번 epoch에서 실제 학습 loader가 읽을 폴더입니다.
        epoch_idx: 현재 RoaD epoch 번호입니다. 0부터 시작합니다.
        seed: 선택 재현성을 위한 seed입니다.
        rank: 현재 분산 학습 process 번호입니다.
        world_size: 전체 process 개수입니다.

    Returns:
        int: 현재 rank가 만든 selected cache 개수입니다.
    """
    if len(variant_dirs) == 0:
        raise ValueError("variant_dirs must not be empty.")
    for variant_dir in variant_dirs:
        if not variant_dir.exists():
            raise FileNotFoundError(f"RoaD variant directory does not exist: {variant_dir}")

    selected_dir.mkdir(parents=True, exist_ok=True)
    base_paths = sorted(p for p in variant_dirs[0].glob("*") if p.is_file())
    rank = int(rank)
    world_size = max(1, int(world_size))
    made = 0
    for source0 in base_paths[rank::world_size]:
        scenario_id = source0.stem
        variant_idx = choose_variant_for_epoch(
            scenario_id=scenario_id,
            epoch_idx=epoch_idx,
            num_variants=len(variant_dirs),
            seed=seed,
        )
        source_path = variant_dirs[variant_idx] / source0.name
        if not source_path.exists():
            raise FileNotFoundError(f"Selected RoaD variant is missing: {source_path}")
        target_path = selected_dir / source0.name
        link_or_copy_variant(source_path, target_path)
        made += 1
    return made
