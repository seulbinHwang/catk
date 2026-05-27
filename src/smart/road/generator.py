from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch

from src.smart.road.cache import build_road_cache_sample, safe_scenario_id, write_pickle
from src.smart.road.geometry import corner_distance_score, wrap_angle


@dataclass(frozen=True)
class RoadGenerationConfig:
    """RoaD cache 생성에 필요한 값만 담는 설정입니다."""

    candidates_per_agent: int = 64
    rollouts_per_scenario: int = 3
    rollout_steps: int = 80
    commit_steps: int = 5
    selection_horizon_steps: int = 20
    temperature: float = 0.8
    sample_steps: int = 16
    sample_method: str = "euler"
    generation_batch_size: int = 1
    candidate_micro_batch_size: int = 4
    seed: int = 817
    source_count_hint: int = 486_995
    overwrite_cache: bool = False


def build_sampling_scheme(config: RoadGenerationConfig) -> SimpleNamespace:
    """Flow decoder에 넘길 RoaD sampling 설정을 만듭니다.

    Args:
        config: RoaD 생성 설정입니다.

    Returns:
        SimpleNamespace: 기존 decoder가 읽는 ``sample_steps``, ``sample_method``,
        ``noise_scale`` 값을 가진 객체입니다.
    """
    return SimpleNamespace(
        sample_steps=int(config.sample_steps),
        sample_method=str(config.sample_method),
        noise_scale=float(config.temperature),
        temperature=float(config.temperature),
    )


def chunked_paths(paths: Sequence[Path], chunk_size: int) -> Iterable[list[Path]]:
    """파일 경로 목록을 작은 묶음으로 나눕니다.

    Args:
        paths: 원본 scenario `.pkl` 경로 목록입니다.
        chunk_size: 한 번에 묶을 파일 수입니다.

    Yields:
        list[Path]: 길이가 최대 ``chunk_size`` 인 경로 묶음입니다.
    """
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(paths), chunk_size):
        yield list(paths[start : start + chunk_size])


def load_source_sample(source_path: Path) -> Mapping[str, Any]:
    """원본 WOMD pkl cache 1개를 읽습니다.

    Args:
        source_path: 원본 scenario `.pkl` 경로입니다.

    Returns:
        Mapping[str, Any]: 원본 scenario cache입니다.
    """
    with source_path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, Mapping):
        raise TypeError(f"source cache must be a mapping, got {type(sample).__name__}")
    return sample


def _copy_tensor(value: Tensor) -> Tensor:
    """tensor를 CPU 기준으로 분리 복사합니다.

    Args:
        value: 복사할 tensor입니다. shape은 제한이 없습니다.

    Returns:
        Tensor: ``detach().cpu().clone()`` 된 tensor입니다.
    """
    return value.detach().cpu().clone()


def initialize_rollout_state(source_sample: Mapping[str, Any]) -> dict[str, Tensor]:
    """RoaD 생성용 agent 상태를 초기화합니다.

    Args:
        source_sample: 원본 scenario cache입니다.

    Returns:
        dict[str, Tensor]: 현재까지 commit한 agent 상태입니다.
            ``position`` shape은 ``[A, 91, 3]`` 입니다.
            ``heading`` shape은 ``[A, 91]`` 입니다.
            ``velocity`` shape은 ``[A, 91, 2]`` 입니다.
            ``valid_mask`` shape은 ``[A, 91]`` 입니다.
    """
    agent = source_sample["agent"]
    state = {
        "position": _copy_tensor(agent["position"]),
        "heading": _copy_tensor(agent["heading"]),
        "velocity": _copy_tensor(agent["velocity"]),
        "valid_mask": _copy_tensor(agent["valid_mask"]).bool(),
    }
    state["position"][:, 11:] = state["position"][:, 10:11]
    state["heading"][:, 11:] = state["heading"][:, 10:11]
    state["velocity"][:, 11:] = 0.0
    state["valid_mask"][:, 11:] = False
    return state


def build_shifted_sample(
    source_sample: Mapping[str, Any],
    rollout_state: Mapping[str, Tensor],
    current_abs_step: int,
) -> dict[str, Any]:
    """현재 closed-loop 시점을 모델 입력의 raw step 10으로 옮깁니다.

    Args:
        source_sample: 원본 scenario cache입니다.
        rollout_state: 현재까지 RoaD가 commit한 agent 상태입니다.
            ``position`` shape은 ``[A, 91, 3]`` 입니다.
        current_abs_step: 원본 91-step 좌표계에서 현재로 볼 step입니다.

    Returns:
        dict[str, Any]: 모델 입력용 scenario cache입니다.
            map 정보는 원본을 유지하고 agent 시간축만 현재 기준으로 바꿉니다.
    """
    shifted = copy.deepcopy(source_sample)
    agent = copy.deepcopy(source_sample["agent"])
    total_steps = int(agent["position"].shape[1])
    source_steps = torch.arange(total_steps, dtype=torch.long) + int(current_abs_step) - 10
    in_range = (source_steps >= 0) & (source_steps < total_steps)
    safe_steps = source_steps.clamp(0, total_steps - 1)

    # 각 tensor shape: position [A, 91, 3], heading [A, 91], velocity [A, 91, 2], valid_mask [A, 91]
    agent["position"] = rollout_state["position"][:, safe_steps].clone()
    agent["heading"] = rollout_state["heading"][:, safe_steps].clone()
    agent["velocity"] = rollout_state["velocity"][:, safe_steps].clone()
    agent["valid_mask"] = rollout_state["valid_mask"][:, safe_steps].clone() & in_range.unsqueeze(0)

    # 모델의 closed-loop rollout cache는 raw step 10 뒤의 2초 window가 유효해야 합니다.
    # 미래 GT를 쓰지 않기 위해 raw step 11 이후는 현재 상태를 반복해 채웁니다.
    # position shape은 [A, 91, 3], heading shape은 [A, 91], valid_mask shape은 [A, 91] 입니다.
    current_position = agent["position"][:, 10:11].clone()
    current_heading = agent["heading"][:, 10:11].clone()
    current_valid = agent["valid_mask"][:, 10:11].clone()
    agent["position"][:, 11:] = current_position
    agent["heading"][:, 11:] = current_heading
    agent["velocity"][:, 11:] = 0.0
    agent["valid_mask"][:, 11:] = current_valid

    shifted["agent"] = agent
    return shifted


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """PyG batch를 지정된 장치로 옮깁니다.

    Args:
        batch: PyG batch입니다.
        device: 모델이 있는 장치입니다.

    Returns:
        Batch: 같은 내용을 가진 device 이동 batch입니다.
    """
    return batch.to(device)


def _to_repeated_batch(
    sample: Mapping[str, Any],
    repeat_count: int,
    transform: Callable[[Any], Any],
    device: torch.device,
) -> Batch:
    """현재 scenario를 후보 개수만큼 복제해 모델 batch로 만듭니다.

    Args:
        sample: 현재 시점 기준 scenario cache입니다.
        repeat_count: 한 번에 생성할 후보 개수입니다.
        transform: validation/추론 기준 HeteroData transform입니다.
        device: batch를 올릴 장치입니다.

    Returns:
        Batch: 모델 입력입니다. agent 총 개수는 ``repeat_count * A`` 입니다.
    """
    data_list = [transform(copy.deepcopy(sample)) for _ in range(int(repeat_count))]
    return _move_batch_to_device(Batch.from_data_list(data_list), device)


def extract_rollout_prediction(prediction: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """기존 closed-loop inference 출력에서 10Hz rollout을 꺼냅니다.

    Args:
        prediction: decoder rollout 결과 사전입니다. 위치는 ``pred_traj_10hz`` 또는
            호환 이름을 사용합니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 위치, 방향, 유효 여부입니다.
            shape은 각각 ``[N, T, 2]``, ``[N, T]``, ``[N, T]`` 입니다.
    """
    xy_key = next((key for key in ["pred_traj_10hz", "pred_pos_10hz", "pred_traj"] if key in prediction), None)
    head_key = next(
        (key for key in ["pred_head_10hz", "pred_heading_10hz", "pred_head"] if key in prediction),
        None,
    )
    valid_key = next(
        (key for key in ["pred_valid_10hz", "pred_traj_valid_10hz", "pred_valid"] if key in prediction),
        None,
    )
    if xy_key is None or head_key is None:
        raise KeyError(
            "RoaD generation requires rollout outputs with 10Hz position and heading. "
            f"Available keys: {sorted(prediction.keys())}"
        )

    xy = prediction[xy_key]
    heading = prediction[head_key]
    valid = prediction[valid_key] if valid_key is not None else None

    if xy.dim() == 4 and xy.shape[1] == 1:
        xy = xy[:, 0]
    elif xy.dim() == 4 and xy.shape[0] == 1:
        xy = xy.squeeze(0)
    if heading.dim() == 3 and heading.shape[1] == 1:
        heading = heading[:, 0]
    elif heading.dim() == 3 and heading.shape[0] == 1:
        heading = heading.squeeze(0)
    if valid is not None:
        if valid.dim() == 3 and valid.shape[1] == 1:
            valid = valid[:, 0]
        elif valid.dim() == 3 and valid.shape[0] == 1:
            valid = valid.squeeze(0)

    if xy.shape[-1] > 2:
        xy = xy[..., :2]
    if xy.dim() != 3 or xy.shape[-1] != 2:
        raise ValueError(f"rollout position must be [N,T,2], got {tuple(xy.shape)}.")
    if heading.dim() != 2:
        raise ValueError(f"rollout heading must be [N,T], got {tuple(heading.shape)}.")
    if valid is None:
        valid = torch.ones(xy.shape[:2], device=xy.device, dtype=torch.bool)
    return xy, heading, valid.bool()


@torch.no_grad()
def sample_candidate_micro_batch(
    model: Any,
    current_sample: Mapping[str, Any],
    transform: Callable[[Any], Any],
    config: RoadGenerationConfig,
    device: torch.device,
    repeat_count: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """현재 scene에서 후보 rollout micro-batch를 생성합니다.

    Args:
        model: RoaD 생성에 사용할 Lightning model입니다.
        current_sample: 현재 시점이 raw step 10이 되도록 옮겨진 sample입니다.
        transform: validation/추론 기준 transform입니다.
        config: RoaD 생성 설정입니다.
        device: model과 batch가 올라갈 장치입니다.
        repeat_count: 이번 호출에서 만들 후보 개수입니다.
        seed: sampling seed입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 후보 위치, 방향, 유효 여부입니다.
            shape은 각각 ``[M, A, 20, 2]``, ``[M, A, 20]``, ``[M, A, 20]`` 입니다.
    """
    was_training = bool(model.training)
    model.eval()
    model.token_processor.eval()
    batch = _to_repeated_batch(current_sample, repeat_count, transform, device)
    tokenized_map, tokenized_agent = model.token_processor(batch)
    map_feature = model.encoder.encode_map(tokenized_map)
    rollout_cache = model.encoder.prepare_inference_cache(
        tokenized_agent,
        map_feature,
    )
    prediction = model.encoder.rollout_from_cache(
        rollout_cache=rollout_cache,
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
        sampling_scheme=build_sampling_scheme(config),
        sampling_seed=int(seed),
        rollout_steps_2hz=math.ceil(config.selection_horizon_steps / config.commit_steps),
    )
    xy, heading, valid = extract_rollout_prediction(prediction)
    horizon = int(config.selection_horizon_steps)
    agent_count = int(current_sample["agent"]["position"].shape[0])
    expected_agent_count = int(repeat_count) * agent_count
    if xy.shape[0] != expected_agent_count:
        raise ValueError(
            "candidate rollout agent count mismatch: "
            f"expected={expected_agent_count}, actual={xy.shape[0]}"
        )
    if was_training:
        model.train()
    return (
        xy[:, :horizon].reshape(int(repeat_count), agent_count, horizon, 2).detach().cpu(),
        wrap_angle(heading[:, :horizon]).reshape(int(repeat_count), agent_count, horizon).detach().cpu(),
        valid[:, :horizon].reshape(int(repeat_count), agent_count, horizon).detach().cpu(),
    )


def sample_candidate_rollouts_for_block(
    model: Any,
    current_sample: Mapping[str, Any],
    transform: Callable[[Any], Any],
    config: RoadGenerationConfig,
    device: torch.device,
    seed_base: int,
    block_idx: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """현재 0.5초 block에서 K개 후보를 생성합니다.

    Args:
        model: RoaD 생성에 사용할 Lightning model입니다.
        current_sample: 현재 시점 기준 sample입니다.
        transform: validation/추론 기준 transform입니다.
        config: RoaD 생성 설정입니다.
        device: model과 batch가 올라갈 장치입니다.
        seed_base: 후보별 seed 기준값입니다.
        block_idx: 전체 RoaD rollout 기준 0.5초 block 번호입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 후보 위치, 방향, 유효 여부입니다.
            shape은 각각 ``[K, A, 20, 2]``, ``[K, A, 20]``, ``[K, A, 20]`` 입니다.
    """
    xy_chunks: list[Tensor] = []
    heading_chunks: list[Tensor] = []
    valid_chunks: list[Tensor] = []
    made_count = 0
    micro_batch = max(1, int(config.candidate_micro_batch_size))
    while made_count < int(config.candidates_per_agent):
        repeat_count = min(micro_batch, int(config.candidates_per_agent) - made_count)
        xy, heading, valid = sample_candidate_micro_batch(
            model=model,
            current_sample=current_sample,
            transform=transform,
            config=config,
            device=device,
            repeat_count=repeat_count,
            seed=int(seed_base) + made_count * 104729,
        )
        xy_chunks.append(xy)
        heading_chunks.append(heading)
        valid_chunks.append(valid)
        made_count += repeat_count
    return torch.cat(xy_chunks, dim=0), torch.cat(heading_chunks, dim=0), torch.cat(valid_chunks, dim=0)


def _pad_future_window(value: Tensor, target_steps: int) -> Tensor:
    """미래 window가 짧으면 마지막 값을 반복해 길이를 맞춥니다.

    Args:
        value: 미래 tensor입니다. shape은 ``[A, T, ...]`` 입니다.
        target_steps: 맞출 step 수입니다.

    Returns:
        Tensor: shape이 ``[A, target_steps, ...]`` 인 tensor입니다.
    """
    if value.shape[1] >= target_steps:
        return value[:, :target_steps]
    pad_count = int(target_steps) - int(value.shape[1])
    if value.shape[1] == 0:
        pad_shape = (value.shape[0], target_steps, *value.shape[2:])
        return value.new_zeros(pad_shape)
    return torch.cat([value, value[:, -1:].expand(-1, pad_count, *value.shape[2:])], dim=1)


def select_one_block_by_corner_distance(
    candidate_xy: Tensor,
    candidate_heading: Tensor,
    gt_xy: Tensor,
    gt_heading: Tensor,
    gt_valid: Tensor,
    shape_lwh: Tensor,
    config: RoadGenerationConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """현재 block에서 agent별 최적 후보를 고르고 앞 0.5초만 반환합니다.

    Args:
        candidate_xy: K개 후보 위치입니다. shape은 ``[K, A, 20, 2]`` 입니다.
        candidate_heading: K개 후보 방향입니다. shape은 ``[K, A, 20]`` 입니다.
        gt_xy: GT future 위치입니다. shape은 ``[A, 20, 2]`` 입니다.
        gt_heading: GT future 방향입니다. shape은 ``[A, 20]`` 입니다.
        gt_valid: GT future 유효 여부입니다. shape은 ``[A, 20]`` 입니다.
        shape_lwh: agent 크기입니다. shape은 ``[A, 3]`` 입니다.
        config: RoaD 생성 설정입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]: commit 위치, 방향, 유효 여부입니다.
            shape은 각각 ``[A, 5, 2]``, ``[A, 5]``, ``[A, 5]`` 입니다.
    """
    score = corner_distance_score(
        pred_xy=candidate_xy,
        pred_heading=candidate_heading,
        gt_xy=gt_xy,
        gt_heading=gt_heading,
        shape_lwh=shape_lwh,
        valid_mask=gt_valid,
    )
    best_candidate = score.argmin(dim=0)
    agent_index = torch.arange(candidate_xy.shape[1], device=candidate_xy.device)
    commit_steps = int(config.commit_steps)
    selected_xy = candidate_xy[best_candidate, agent_index, :commit_steps]
    selected_heading = candidate_heading[best_candidate, agent_index, :commit_steps]
    selected_valid = gt_valid[:, :commit_steps].bool()
    return selected_xy, wrap_angle(selected_heading), selected_valid


def commit_block_to_rollout_state(
    rollout_state: dict[str, Tensor],
    selected_xy: Tensor,
    selected_heading: Tensor,
    selected_valid: Tensor,
    future_xy: Tensor,
    future_heading: Tensor,
    future_valid: Tensor,
    block_idx: int,
    config: RoadGenerationConfig,
) -> None:
    """선택한 0.5초 block을 생성 상태와 최종 future에 반영합니다.

    Args:
        rollout_state: 다음 block 입력을 만들 현재 상태입니다.
            ``position`` shape은 ``[A, 91, 3]`` 입니다.
        selected_xy: 선택한 후보 위치입니다. shape은 ``[A, 5, 2]`` 입니다.
        selected_heading: 선택한 후보 방향입니다. shape은 ``[A, 5]`` 입니다.
        selected_valid: 선택한 block 유효 여부입니다. shape은 ``[A, 5]`` 입니다.
        future_xy: 최종 RoaD future 위치입니다. shape은 ``[A, 80, 2]`` 입니다.
        future_heading: 최종 RoaD future 방향입니다. shape은 ``[A, 80]`` 입니다.
        future_valid: 최종 RoaD future 유효 여부입니다. shape은 ``[A, 80]`` 입니다.
        block_idx: 0.5초 block 번호입니다.
        config: RoaD 생성 설정입니다.

    Returns:
        None
    """
    future_start = int(block_idx) * int(config.commit_steps)
    future_end = min(future_start + int(config.commit_steps), int(config.rollout_steps))
    commit_count = future_end - future_start
    if commit_count <= 0:
        return

    absolute_start = 11 + future_start
    absolute_end = absolute_start + commit_count
    commit_xy = selected_xy[:, :commit_count]
    commit_heading = selected_heading[:, :commit_count]
    commit_valid = selected_valid[:, :commit_count]

    future_xy[:, future_start:future_end] = commit_xy
    future_heading[:, future_start:future_end] = commit_heading
    future_valid[:, future_start:future_end] = commit_valid
    rollout_state["position"][:, absolute_start:absolute_end, :2] = commit_xy
    rollout_state["heading"][:, absolute_start:absolute_end] = commit_heading
    rollout_state["valid_mask"][:, absolute_start:absolute_end] = commit_valid

    velocity = torch.zeros_like(rollout_state["velocity"])
    valid_pair = rollout_state["valid_mask"][:, 1:] & rollout_state["valid_mask"][:, :-1]
    velocity[:, 1:] = (rollout_state["position"][:, 1:, :2] - rollout_state["position"][:, :-1, :2]) / 0.1
    velocity[:, 1:] = velocity[:, 1:].masked_fill(~valid_pair.unsqueeze(-1), 0.0)
    rollout_state["velocity"] = velocity


@torch.no_grad()
def generate_single_road_rollout(
    model: Any,
    source_sample: Mapping[str, Any],
    transform: Callable[[Any], Any],
    config: RoadGenerationConfig,
    epoch_idx: int,
    rollout_idx: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """하나의 scenario에서 RoaD closed-loop future 1개를 만듭니다.

    Args:
        model: 현재 Flow Matching model입니다.
        source_sample: 원본 scenario cache입니다.
        transform: validation/추론 기준 transform입니다.
        config: RoaD 생성 설정입니다.
        epoch_idx: 현재 RoaD fine-tuning epoch 번호입니다.
        rollout_idx: 같은 scenario 안 rollout 번호입니다.
        device: model과 batch가 올라갈 장치입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]: RoaD future 위치, 방향, 유효 여부입니다.
            shape은 각각 ``[A, 80, 2]``, ``[A, 80]``, ``[A, 80]`` 입니다.
    """
    agent = source_sample["agent"]
    rollout_state = initialize_rollout_state(source_sample)
    future_xy = _copy_tensor(agent["position"][:, 11:91, :2])
    future_heading = _copy_tensor(agent["heading"][:, 11:91])
    future_valid = _copy_tensor(agent["valid_mask"][:, 11:91]).bool()
    gt_xy_full = _copy_tensor(agent["position"][:, 11:91, :2])
    gt_heading_full = _copy_tensor(agent["heading"][:, 11:91])
    gt_valid_full = _copy_tensor(agent["valid_mask"][:, 11:91]).bool()
    shape_lwh = _copy_tensor(agent["shape"])

    num_blocks = math.ceil(int(config.rollout_steps) / int(config.commit_steps))
    for block_idx in range(num_blocks):
        current_abs_step = 10 + block_idx * int(config.commit_steps)
        current_sample = build_shifted_sample(source_sample, rollout_state, current_abs_step=current_abs_step)
        seed_base = (
            int(config.seed)
            + int(epoch_idx) * 1_000_003
            + int(rollout_idx) * 100_003
            + int(block_idx) * 10_007
        )
        candidate_xy, candidate_heading, _ = sample_candidate_rollouts_for_block(
            model=model,
            current_sample=current_sample,
            transform=transform,
            config=config,
            device=device,
            seed_base=seed_base,
            block_idx=block_idx,
        )
        future_start = block_idx * int(config.commit_steps)
        future_end = future_start + int(config.selection_horizon_steps)
        gt_xy = _pad_future_window(gt_xy_full[:, future_start:future_end], int(config.selection_horizon_steps))
        gt_heading = _pad_future_window(gt_heading_full[:, future_start:future_end], int(config.selection_horizon_steps))
        gt_valid = _pad_future_window(gt_valid_full[:, future_start:future_end], int(config.selection_horizon_steps)).bool()
        selected_xy, selected_heading, selected_valid = select_one_block_by_corner_distance(
            candidate_xy=candidate_xy,
            candidate_heading=candidate_heading,
            gt_xy=gt_xy,
            gt_heading=gt_heading,
            gt_valid=gt_valid,
            shape_lwh=shape_lwh,
            config=config,
        )
        commit_block_to_rollout_state(
            rollout_state=rollout_state,
            selected_xy=selected_xy,
            selected_heading=selected_heading,
            selected_valid=selected_valid,
            future_xy=future_xy,
            future_heading=future_heading,
            future_valid=future_valid,
            block_idx=block_idx,
            config=config,
        )
    return future_xy, wrap_angle(future_heading), future_valid


def _variant_output_path(variant_dir: Path, source_path: Path) -> Path:
    """variant cache 파일 경로를 정합니다.

    Args:
        variant_dir: rollout 번호별 저장 폴더입니다.
        source_path: 원본 WOMD cache 경로입니다.

    Returns:
        Path: 저장할 RoaD cache 경로입니다.
    """
    return variant_dir / source_path.name


def generate_road_epoch_cache(
    model: Any,
    source_train_raw_dir: Path,
    epoch_dir: Path,
    transform: Callable[[Any], Any],
    config: RoadGenerationConfig,
    epoch_idx: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> int:
    """현재 모델로 한 epoch용 RoaD cache 3N개 중 현재 rank 몫을 생성합니다.

    Args:
        model: pretrained 또는 fine-tuned Flow Matching model입니다.
        source_train_raw_dir: 원본 WOMD training cache 폴더입니다.
        epoch_dir: 이번 epoch RoaD cache 루트입니다.
        transform: validation/추론 기준 transform입니다.
        config: RoaD 생성 설정입니다.
        epoch_idx: 현재 epoch 번호입니다. 0부터 시작합니다.
        device: model과 batch가 올라갈 장치입니다.
        rank: 현재 process 번호입니다.
        world_size: 전체 process 개수입니다.

    Returns:
        int: 현재 rank가 생성한 `.pkl` 파일 개수입니다.
    """
    source_paths = sorted(p for p in Path(source_train_raw_dir).glob("*") if p.is_file())
    if len(source_paths) == 0:
        raise FileNotFoundError(f"No source WOMD cache files found under: {source_train_raw_dir}")

    variant_dirs = [epoch_dir / "all" / f"variant_{idx:02d}" for idx in range(config.rollouts_per_scenario)]
    for variant_dir in variant_dirs:
        variant_dir.mkdir(parents=True, exist_ok=True)

    rank = int(rank)
    world_size = max(1, int(world_size))
    rank_paths = source_paths[rank::world_size]
    generated = 0
    was_training = bool(model.training)
    model.eval()
    for source_path in rank_paths:
        source_sample = load_source_sample(source_path)
        for rollout_idx, variant_dir in enumerate(variant_dirs):
            output_path = _variant_output_path(variant_dir, source_path)
            if output_path.exists() and not config.overwrite_cache:
                continue
            selected_xy, selected_heading, selected_valid = generate_single_road_rollout(
                model=model,
                source_sample=source_sample,
                transform=transform,
                config=config,
                epoch_idx=epoch_idx,
                rollout_idx=rollout_idx,
                device=device,
            )
            road_sample = build_road_cache_sample(
                source_sample=source_sample,
                rollout_xy=selected_xy,
                rollout_heading=selected_heading,
                rollout_valid=selected_valid,
                rollout_index=rollout_idx,
                source_path=source_path,
            )
            road_sample["source_scenario_id"] = safe_scenario_id(source_sample, source_path)
            write_pickle(road_sample, output_path)
            generated += 1
    if was_training:
        model.train()
    return generated
