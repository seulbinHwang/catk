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


def update_raw_data_with_road_rollout(
    raw_data: Mapping[str, Any],
    scenario_id: str,
    rollout_index: int,
    pred_traj_10hz: Tensor,
    pred_head_10hz: Tensor,
    future_valid: Tensor,
    num_historical_steps: int,
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

    Returns:
        기존 학습 cache와 같은 schema를 유지하는 RoaD pickle data이다.
    """
    data = clone_to_cpu(raw_data)
    agent = data["agent"]
    position = agent["position"].clone()
    heading = agent["heading"].clone()
    velocity = agent["velocity"].clone()
    valid_mask = agent["valid_mask"].clone()

    pred_traj_10hz = pred_traj_10hz.detach().cpu()
    pred_head_10hz = pred_head_10hz.detach().cpu()
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

    log.info(
        f"Generating RoaD cache at {output_path.as_posix()} "
        f"with {num_rollouts_per_scenario} rollouts per scenario."
    )
    with torch.no_grad():
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
