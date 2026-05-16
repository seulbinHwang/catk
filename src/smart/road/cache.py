import contextlib
import copy
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.loader import DataLoader

from src.smart.datasets import MultiDataset
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _build_autocast_context(
    autocast_dtype: Optional[torch.dtype],
    device: torch.device,
) -> contextlib.AbstractContextManager:
    """학습 step과 같은 precision으로 RoaD cache 생성을 감싸는 context manager를 만든다.

    Args:
        autocast_dtype: ``bf16-mixed``는 ``torch.bfloat16``, ``16-mixed``는
            ``torch.float16``이다. ``None``이면 autocast를 끈다.
        device: 모델 실행 장치이다. CUDA device일 때만 autocast가 적용된다.

    Returns:
        ``torch.autocast`` context이거나 아무 일도 안 하는 nullcontext이다.
    """
    if autocast_dtype is None:
        return contextlib.nullcontext()
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


_PRECISION_TO_AUTOCAST_DTYPE = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "bf16": torch.bfloat16,
    "16": torch.float16,
}


def resolve_autocast_dtype_from_precision(precision: Any) -> Optional[torch.dtype]:
    """trainer.precision 문자열을 RoaD cache 생성용 autocast dtype으로 바꾼다.

    Args:
        precision: Lightning trainer가 노출하는 precision 식별자이다.

    Returns:
        ``torch.bfloat16`` 또는 ``torch.float16``이거나, mixed precision이 아니면 ``None``.
    """
    if precision is None:
        return None
    return _PRECISION_TO_AUTOCAST_DTYPE.get(str(precision).strip().lower())


def clone_to_cpu(value: Any) -> Any:
    """저장 가능한 형태로 tensor를 CPU에 복사한다.

    Args:
        value: pickle에 저장할 값이다. tensor, dict, list, tuple이 섞여 있을 수 있다.

    Returns:
        입력과 같은 구조이되, 모든 tensor가 CPU에 있는 값이다.
    """
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def get_device() -> torch.device:
    """RoaD cache 생성에 사용할 장치를 고른다.

    Returns:
        CUDA가 있으면 현재 process의 local rank GPU이고, 없으면 CPU이다.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def list_pickle_paths(raw_dir: str) -> List[str]:
    """cache 디렉터리에서 pickle 파일 경로를 정렬해서 가져온다.

    Args:
        raw_dir: 기존 WOMD training cache 디렉터리이다.

    Returns:
        정렬된 pickle 파일 경로 목록이다.
    """
    paths = sorted(Path(raw_dir).glob("*.pkl"))
    if not paths:
        paths = sorted(Path(raw_dir).glob("*"))
    return [path.as_posix() for path in paths]


def make_scenario_path_map(raw_dir: str) -> Dict[str, str]:
    """scenario id로 원본 pickle 파일을 찾을 수 있는 표를 만든다.

    Args:
        raw_dir: 기존 WOMD training cache 디렉터리이다.

    Returns:
        key는 scenario id이고, value는 해당 pickle 파일 경로인 dict이다.
    """
    return {Path(raw_path).stem: raw_path for raw_path in list_pickle_paths(raw_dir)}


def normalize_scenario_ids(scenario_ids: Any) -> List[str]:
    """batch 안의 scenario id를 문자열 list로 바꾼다.

    Args:
        scenario_ids: DataLoader가 묶은 scenario id이다. 문자열 하나이거나 list일 수 있다.

    Returns:
        batch 순서와 같은 scenario id list이다.
    """
    if isinstance(scenario_ids, str):
        return [scenario_ids]
    if isinstance(scenario_ids, (list, tuple)):
        return [str(scenario_id) for scenario_id in scenario_ids]
    return [str(scenario_ids)]


def make_future_velocity(position_xy: Tensor, num_historical_steps: int) -> Tensor:
    """RoaD future 위치에서 10Hz 속도를 계산한다.

    Args:
        position_xy: 전체 9초 위치이다. Shape은 ``[n_agent, 91, 2]``이다.
        num_historical_steps: history step 수이다. WOMD 기준 11이다.

    Returns:
        future 속도이다. Shape은 ``[n_agent, 80, 2]``이다.
    """
    future_xy = position_xy[:, num_historical_steps:]
    previous_xy = torch.cat(
        [position_xy[:, num_historical_steps - 1 : num_historical_steps], future_xy[:, :-1]],
        dim=1,
    )
    return (future_xy - previous_xy) / 0.1


def _apply_runaway_rollout_filter(
    valid_mask: Tensor,
    position: Tensor,
    role: Tensor,
    future_slice: slice,
    max_distance_from_ego: float,
) -> Tensor:
    """ego로부터 너무 멀어진 RoaD rollout step을 invalid로 바꾼다.

    Args:
        valid_mask: 이미 ``future_valid`` broadcast가 끝난 ``[n_agent, 91]`` 마스크이다.
        position: RoaD rollout으로 채워진 ``[n_agent, 91, 3]`` 좌표이다.
        role: ``[n_agent, 3]`` boolean role 마스크이며 ``role[:, 0]``이 ego이다.
        future_slice: future step 범위이다.
        max_distance_from_ego: 이 거리(미터)를 초과하는 step은 invalid 처리한다.

    Returns:
        runaway step이 invalid로 바뀐 valid_mask이다.
    """
    if max_distance_from_ego <= 0:
        return valid_mask

    ego_indices = torch.where(role[:, 0])[0]
    if ego_indices.numel() != 1:
        # ego가 정확히 1개가 아니면 다운스트림 transform이 처리하도록 두고 여기서는 건드리지 않는다.
        return valid_mask

    av_idx = int(ego_indices.item())
    ego_pos = position[av_idx, future_slice, :2]  # [n_future, 2]
    agent_pos = position[:, future_slice, :2]  # [n_agent, n_future, 2]
    distance = torch.norm(agent_pos - ego_pos.unsqueeze(0), dim=-1)  # [n_agent, n_future]
    within_range = distance < float(max_distance_from_ego)  # [n_agent, n_future]
    future_valid = valid_mask[:, future_slice] & within_range
    valid_mask = valid_mask.clone()
    valid_mask[:, future_slice] = future_valid
    return valid_mask


def update_raw_data_with_road_rollout(
    raw_data: Mapping[str, Any],
    scenario_id: str,
    rollout_index: int,
    pred_traj_10hz: Tensor,
    pred_head_10hz: Tensor,
    future_valid: Tensor,
    num_historical_steps: int,
    max_distance_from_ego: float = 0.0,
) -> Dict[str, Any]:
    """원본 WOMD pickle의 future를 RoaD rollout으로 교체한다.

    Args:
        raw_data: 원본 WOMD pickle data이다.
        scenario_id: 원본 scenario id이다.
        rollout_index: 같은 scenario에서 몇 번째 RoaD rollout인지 나타낸다.
        pred_traj_10hz: RoaD가 생성한 future 위치이다. Shape은 ``[n_agent, 80, 2]``이다.
        pred_head_10hz: RoaD가 생성한 future 방향이다. Shape은 ``[n_agent, 80]``이다.
        future_valid: future를 학습에 쓸 agent mask이다. Shape은 ``[n_agent]``이다.
        num_historical_steps: history step 수이다. WOMD 기준 11이다.
        max_distance_from_ego: ego로부터 이 거리(미터)를 넘어가는 RoaD rollout step을
            invalid로 마킹한다. 0 이하이면 거리 기반 필터를 끄고 기존처럼
            ``future_valid``만 broadcast한다.

    Returns:
        기존 학습 cache와 같은 schema를 유지하는 RoaD pickle data이다.
    """
    data = clone_to_cpu(raw_data)
    agent = data["agent"]
    position = agent["position"].clone()
    heading = agent["heading"].clone()
    velocity = agent["velocity"].clone()
    valid_mask = agent["valid_mask"].clone()

    pred_traj_10hz = pred_traj_10hz.detach().cpu().float()
    pred_head_10hz = pred_head_10hz.detach().cpu().float()
    future_valid = future_valid.detach().cpu().bool()

    if pred_traj_10hz.shape[0] != position.shape[0]:
        raise ValueError(
            f"Agent count mismatch for {scenario_id}: "
            f"raw={position.shape[0]}, rollout={pred_traj_10hz.shape[0]}"
        )

    future_slice = slice(num_historical_steps, num_historical_steps + pred_traj_10hz.shape[1])
    position[:, future_slice, :2] = pred_traj_10hz
    if position.shape[-1] > 2:
        position[:, future_slice, 2] = position[:, num_historical_steps - 1 : num_historical_steps, 2]
    heading[:, future_slice] = pred_head_10hz
    velocity[:, future_slice] = make_future_velocity(position[..., :2], num_historical_steps)
    valid_mask[:, future_slice] = future_valid.unsqueeze(1).expand(-1, pred_traj_10hz.shape[1])

    if max_distance_from_ego > 0:
        valid_mask = _apply_runaway_rollout_filter(
            valid_mask=valid_mask,
            position=position,
            role=agent["role"],
            future_slice=future_slice,
            max_distance_from_ego=float(max_distance_from_ego),
        )

    agent["position"] = position
    agent["heading"] = heading
    agent["velocity"] = velocity
    agent["valid_mask"] = valid_mask
    data["agent"] = agent
    data["scenario_id"] = f"{scenario_id}__road_r{rollout_index:02d}"
    return data


def save_pickle_atomic(data: Mapping[str, Any], output_path: Path) -> None:
    """pickle 파일을 안전하게 저장한다.

    Args:
        data: 저장할 RoaD cache data이다.
        output_path: 최종 pickle 파일 경로이다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(data, handle)
    tmp_path.replace(output_path)


def generate_road_cache(
    model: torch.nn.Module,
    original_train_raw_dir: str,
    output_dir: str,
    transform: Any,
    sampling_scheme: DictConfig,
    num_rollouts_per_scenario: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_historical_steps: int,
    device: Optional[torch.device] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    max_distance_from_ego: float = 0.0,
) -> None:
    """현재 모델로 RoaD rollout cache를 만든다.

    Args:
        model: SMART LightningModule이다.
        original_train_raw_dir: 원본 WOMD training pickle cache 디렉터리이다.
        output_dir: 새로 만들 RoaD cache 디렉터리이다.
        transform: 기존 training transform이다.
        sampling_scheme: RoaD Sample-K 설정이다.
        num_rollouts_per_scenario: scenario당 생성할 rollout 수이다. 기본값은 3이다.
        batch_size: cache 생성용 batch size이다.
        num_workers: cache 생성용 worker 수이다.
        pin_memory: DataLoader pin_memory 사용 여부이다.
        num_historical_steps: history step 수이다. WOMD 기준 11이다.
        device: model 실행 장치이다. None이면 자동으로 고른다.
        autocast_dtype: ``bf16-mixed``로 학습할 때 ``torch.bfloat16``을 넘기면
            inference 분포가 학습 step의 분포와 동일해진다. ``None``이면 fp32로 돈다.
        max_distance_from_ego: ego로부터 이 거리(미터)를 넘어간 rollout step을
            invalid로 마킹한다. 0 이하이면 거리 기반 필터를 끈다. 학습 transform이
            중간에 거리 clip을 안 거는 설정(예: ``train_use_eval_agent_selection``)에서도
            폭주한 RoaD rollout이 학습에 흘러 들어가지 않도록 캐시 단계에서 막는다.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scenario_path_map = make_scenario_path_map(original_train_raw_dir)
    dataset = MultiDataset(original_train_raw_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    device = device or get_device()
    was_training = model.training
    model.to(device)
    model.eval()

    autocast_ctx = _build_autocast_context(autocast_dtype, device)
    if autocast_dtype is not None:
        log.info(
            f"Generating RoaD cache at {output_path.as_posix()} with "
            f"{num_rollouts_per_scenario} rollouts per scenario "
            f"under autocast dtype={autocast_dtype}, "
            f"max_distance_from_ego={max_distance_from_ego}."
        )
    else:
        log.info(
            f"Generating RoaD cache at {output_path.as_posix()} with "
            f"{num_rollouts_per_scenario} rollouts per scenario, "
            f"max_distance_from_ego={max_distance_from_ego}."
        )

    with torch.no_grad(), autocast_ctx:
        for rollout_index in range(num_rollouts_per_scenario):
            for batch in dataloader:
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                tokenized_map, tokenized_agent = model.token_processor(batch)
                pred = model.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    sampling_scheme=sampling_scheme,
                )
                scenario_ids = normalize_scenario_ids(batch["scenario_id"])
                agent_batch = batch["agent"]["batch"]
                current_valid = batch["agent"]["valid_mask"][:, num_historical_steps - 1]

                for scenario_batch_index, scenario_id in enumerate(scenario_ids):
                    agent_mask = agent_batch == scenario_batch_index
                    raw_path = scenario_path_map[scenario_id]
                    with open(raw_path, "rb") as handle:
                        raw_data = pickle.load(handle)
                    road_data = update_raw_data_with_road_rollout(
                        raw_data=raw_data,
                        scenario_id=scenario_id,
                        rollout_index=rollout_index,
                        pred_traj_10hz=pred["pred_traj_10hz"][agent_mask],
                        pred_head_10hz=pred["pred_head_10hz"][agent_mask],
                        future_valid=current_valid[agent_mask],
                        num_historical_steps=num_historical_steps,
                        max_distance_from_ego=max_distance_from_ego,
                    )
                    save_pickle_atomic(
                        road_data,
                        output_path / f"{scenario_id}__road_r{rollout_index:02d}.pkl",
                    )
    if was_training:
        model.train()


def delete_cache_dir(cache_dir: Optional[str]) -> None:
    """이미 학습에 사용한 RoaD cache 디렉터리를 삭제한다.

    Args:
        cache_dir: 삭제할 cache 디렉터리이다. None이면 아무 작업도 하지 않는다.
    """
    if cache_dir is None:
        return
    path = Path(cache_dir)
    if path.exists():
        shutil.rmtree(path)
