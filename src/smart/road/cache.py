import contextlib
import copy
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.smart.datasets import MultiDataset
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

ROAD_UNUSED_AGENT_FIELDS = (
    "control_aligned_future_heading",
    "control_aligned_future_pos",
    "control_alignment_cache_key",
    "control_transition_norm_future",
)


def drop_unused_road_agent_fields(data: Any) -> Any:
    """RoaD cache 생성/학습에 쓰지 않는 optional agent field를 제거한다.

    일부 training cache shard에는 control 보조 field가 있고, 일부에는 없다. PyG는
    batch 안의 schema가 다르면 collate 단계에서 실패하므로, RoaD가 실제로 쓰는
    position/heading/velocity/valid_mask/role/type 계열 field만 남긴다.
    """
    try:
        agent = data["agent"]
    except (KeyError, TypeError):
        return data

    for key in ROAD_UNUSED_AGENT_FIELDS:
        try:
            if key in agent:
                del agent[key]
        except (KeyError, TypeError):
            continue
    return data


class RoadCacheInputTransform:
    """기존 train transform 뒤에 RoaD 입력 schema 정규화를 덧붙인다."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, data: Any) -> Any:
        if self.transform is not None:
            data = self.transform(data)
        return drop_unused_road_agent_fields(data)


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


def select_road_dataset_indices(
    dataset_size: int,
    road_data_use_ratio: float,
    seed: Optional[int] = None,
) -> Optional[List[int]]:
    """RoaD cache 생성에 사용할 원본 scenario index subset을 고른다.

    Args:
        dataset_size: 원본 WOMD training cache의 scenario 개수이다.
        road_data_use_ratio: 이번 epoch에 사용할 scenario 비율이다. 범위는 ``(0, 1]``이다.
        seed: 주어지면 독립 generator로 subset을 뽑는다. DDP rank들이 같은
            subset을 각자 재현해야 할 때 사용한다.

    Returns:
        전체를 쓰면 ``None``이고, subset을 쓰면 무작위 index list이다.
    """
    ratio = float(road_data_use_ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"road_data_use_ratio must be in (0, 1], got {ratio}")
    if ratio >= 1.0:
        return None
    num_selected = max(1, math.ceil(dataset_size * ratio))
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.randperm(dataset_size, generator=generator).tolist()[:num_selected]


def shard_dataset_indices(
    dataset_size: int,
    selected_indices: Optional[List[int]],
    distributed_rank: int,
    distributed_world_size: int,
) -> List[int]:
    """RoaD cache 생성을 rank별로 나눌 index 목록을 만든다.

    Args:
        dataset_size: 원본 dataset 크기이다.
        selected_indices: RoaD에 사용할 전체 subset이다. ``None``이면 전체 dataset이다.
        distributed_rank: 현재 process의 global rank이다.
        distributed_world_size: 전체 DDP process 수이다.

    Returns:
        현재 rank가 생성할 원본 dataset index list이다.
    """
    world_size = max(1, int(distributed_world_size))
    rank = int(distributed_rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"distributed_rank must be in [0, {world_size}), got {distributed_rank}"
        )
    if selected_indices is None:
        return list(range(rank, dataset_size, world_size))
    return selected_indices[rank::world_size]


def filter_readable_dataset_indices(
    raw_paths: List[str],
    indices: List[int],
    distributed_rank: int,
) -> List[int]:
    """깨진 pickle을 RoaD 생성 subset에서 제외한다.

    RoaD 생성은 긴 DDP job이므로 원본 cache 한 파일이 truncated이면 전체 rank가
    중단된다. 원본 cache를 수정하지 않고 현재 rank가 맡은 index 중 읽을 수 없는
    scenario만 건너뛰어 나머지 생성 작업을 보존한다.
    """
    valid_indices = []
    skipped = []
    for index in indices:
        raw_path = raw_paths[index]
        try:
            with open(raw_path, "rb") as handle:
                pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            skipped.append((index, raw_path, type(exc).__name__, str(exc)))
            continue
        valid_indices.append(index)

    if skipped:
        preview = "; ".join(
            f"{path} ({error_type}: {message})"
            for _, path, error_type, message in skipped[:5]
        )
        print(
            f"[rank {distributed_rank}] skipped {len(skipped)} unreadable RoaD "
            f"source pickle(s): {preview}",
            flush=True,
        )
    return valid_indices


def prepare_road_cache_output_dir(output_dir: str) -> None:
    """RoaD cache 출력 디렉터리를 비우고 다시 만든다.

    DDP에서는 rank 0만 이 함수를 호출한 뒤 barrier를 통과해야 한다.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


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

    Note:
        RoaD는 모델 자기 자신의 rollout을 그대로 새 정답으로 학습시키는 방식이다.
        ego로부터 멀어진 step이나 폭주한 궤적도 별도 후처리 없이 그대로 학습 신호가
        되도록, 캐시 단계에서는 거리 기반 invalid 처리 같은 추가 방어 로직을 두지 않는다.
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

    agent["position"] = position
    agent["heading"] = heading
    agent["velocity"] = velocity
    agent["valid_mask"] = valid_mask
    data["agent"] = agent
    data["scenario_id"] = f"{scenario_id}__road_r{rollout_index:02d}"
    return drop_unused_road_agent_fields(data)


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


def build_road_inference_runner(
    model: torch.nn.Module,
    tokenized_map: Mapping[str, Tensor],
    tokenized_agent: Mapping[str, Tensor],
    sampling_scheme: DictConfig,
) -> Any:
    """같은 RoaD batch에서 반복 rollout할 inference callable을 만든다.

    RoaD는 같은 source batch에서 여러 rollout replica를 만든다. Map encoder 출력은
    rollout replica마다 달라지지 않으므로 batch당 한 번만 계산하고 agent rollout만
    반복 실행한다. 예상하지 못한 encoder 구조에서는 기존 ``encoder.inference`` 경로로
    되돌아가도록 한다.
    """
    encoder = model.encoder
    if hasattr(encoder, "map_encoder") and hasattr(encoder, "agent_encoder"):
        map_feature = encoder.map_encoder(tokenized_map)

        def run_inference() -> Dict[str, Tensor]:
            return encoder.agent_encoder.inference(
                tokenized_agent,
                map_feature,
                sampling_scheme=sampling_scheme,
            )

        return run_inference

    def run_inference() -> Dict[str, Tensor]:
        return encoder.inference(
            tokenized_map,
            tokenized_agent,
            sampling_scheme=sampling_scheme,
        )

    return run_inference


def is_deterministic_road_sampling(sampling_scheme: DictConfig) -> bool:
    """같은 입력 batch에서 rollout replica들이 동일한지 판단한다."""
    return str(getattr(sampling_scheme, "criterium", "")) == "road_topk_dist"


def resolve_road_num_rollouts_per_scenario(
    sampling_scheme: DictConfig,
    requested_num_rollouts_per_scenario: int,
) -> int:
    """실제로 생성할 RoaD rollout 수를 정한다.

    ``road_topk_dist``는 같은 입력에서 항상 같은 rollout을 만들기 때문에 같은
    scenario를 여러 번 저장해도 학습 sample만 중복된다. 이 경우에는 1개만 만든다.
    """
    requested = int(requested_num_rollouts_per_scenario)
    if requested <= 0:
        raise ValueError(
            "num_rollouts_per_scenario should be positive, "
            f"got {requested_num_rollouts_per_scenario}"
        )
    if is_deterministic_road_sampling(sampling_scheme):
        return 1
    return requested


def generate_road_cache(
    model: torch.nn.Module,
    original_train_raw_dir: str,
    output_dir: str,
    transform: Any,
    sampling_scheme: DictConfig,
    road_data_use_ratio: float,
    num_rollouts_per_scenario: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_historical_steps: int,
    device: Optional[torch.device] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    selection_seed: Optional[int] = None,
    clean_output_dir: bool = True,
) -> None:
    """현재 모델로 RoaD rollout cache를 만든다.

    Args:
        model: SMART LightningModule이다.
        original_train_raw_dir: 원본 WOMD training pickle cache 디렉터리이다.
        output_dir: 새로 만들 RoaD cache 디렉터리이다.
        transform: 기존 training transform이다.
        sampling_scheme: RoaD rollout token 후보 선택 설정이다.
        road_data_use_ratio: 원본 training cache 중 이번 epoch에 사용할 비율이다.
        num_rollouts_per_scenario: scenario당 생성할 rollout 수이다. 기본값은 3이다.
        batch_size: cache 생성용 batch size이다.
        num_workers: cache 생성용 worker 수이다.
        pin_memory: DataLoader pin_memory 사용 여부이다.
        num_historical_steps: history step 수이다. WOMD 기준 11이다.
        device: model 실행 장치이다. None이면 자동으로 고른다.
        autocast_dtype: ``bf16-mixed``로 학습할 때 ``torch.bfloat16``을 넘기면
            inference 분포가 학습 step의 분포와 동일해진다. ``None``이면 fp32로 돈다.
        distributed_rank: DDP에서 현재 process의 global rank이다.
        distributed_world_size: DDP 전체 process 수이다. 1이면 단일 process와 같다.
        selection_seed: DDP rank들이 같은 scenario subset을 재현하기 위한 seed이다.
        clean_output_dir: True이면 시작 전에 output_dir을 비운다. DDP에서는 rank 0이
            별도로 ``prepare_road_cache_output_dir``를 호출하고 False로 넘긴다.

    Note:
        RoaD는 모델 자기 자신의 rollout을 그대로 새 정답으로 삼아 학습시키는 방식이다.
        ego와의 거리, 도로 이탈 같은 휴리스틱으로 rollout step을 invalid 처리하지 않고,
        모델이 생성한 80 step 전체를 학습 신호로 사용한다. precision 정합(autocast)만
        남기고, 후처리 방어 로직은 일부러 두지 않는다.
    """
    output_path = Path(output_dir)
    if clean_output_dir:
        prepare_road_cache_output_dir(output_dir)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    scenario_path_map = make_scenario_path_map(original_train_raw_dir)
    dataset = MultiDataset(original_train_raw_dir, RoadCacheInputTransform(transform))
    selected_indices = select_road_dataset_indices(
        len(dataset),
        road_data_use_ratio,
        seed=selection_seed,
    )
    rank_indices = shard_dataset_indices(
        dataset_size=len(dataset),
        selected_indices=selected_indices,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    rank_indices = filter_readable_dataset_indices(
        raw_paths=dataset.raw_paths,
        indices=rank_indices,
        distributed_rank=distributed_rank,
    )
    generation_dataset = Subset(dataset, rank_indices)
    total_selected = len(dataset) if selected_indices is None else len(selected_indices)
    if selected_indices is not None:
        log.info(
            f"RoaD cache will use {len(selected_indices)}/{len(dataset)} "
            f"training scenarios for this epoch "
            f"(road_data_use_ratio={float(road_data_use_ratio):.6g})."
        )
    if int(distributed_world_size) > 1:
        log.info(
            f"RoaD cache generation is sharded across {int(distributed_world_size)} "
            f"rank(s); total_selected={total_selected}, "
            f"rank0_scenarios={len(rank_indices)}."
        )
    dataloader = DataLoader(
        generation_dataset,
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

    effective_num_rollouts_per_scenario = resolve_road_num_rollouts_per_scenario(
        sampling_scheme=sampling_scheme,
        requested_num_rollouts_per_scenario=num_rollouts_per_scenario,
    )
    autocast_ctx = _build_autocast_context(autocast_dtype, device)
    rollout_msg = f"{effective_num_rollouts_per_scenario} rollout(s) per scenario"
    if effective_num_rollouts_per_scenario != int(num_rollouts_per_scenario):
        rollout_msg += f" (requested {int(num_rollouts_per_scenario)})"
    if autocast_dtype is not None:
        log.info(
            f"Generating RoaD cache at {output_path.as_posix()} with "
            f"{rollout_msg} "
            f"under autocast dtype={autocast_dtype}."
        )
    else:
        log.info(
            f"Generating RoaD cache at {output_path.as_posix()} with "
            f"{rollout_msg}."
        )

    with torch.no_grad(), autocast_ctx:
        for batch in dataloader:
            if hasattr(batch, "to"):
                batch = batch.to(device)
            tokenized_map, tokenized_agent = model.token_processor(batch)
            run_inference = build_road_inference_runner(
                model=model,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                sampling_scheme=sampling_scheme,
            )
            scenario_ids = normalize_scenario_ids(batch["scenario_id"])
            agent_batch = batch["agent"]["batch"]
            current_valid = batch["agent"]["valid_mask"][:, num_historical_steps - 1]
            raw_data_by_scenario = {}
            for scenario_id in scenario_ids:
                raw_path = scenario_path_map[scenario_id]
                with open(raw_path, "rb") as handle:
                    raw_data_by_scenario[scenario_id] = pickle.load(handle)

            for rollout_index in range(effective_num_rollouts_per_scenario):
                pred = run_inference()
                for scenario_batch_index, scenario_id in enumerate(scenario_ids):
                    agent_mask = agent_batch == scenario_batch_index
                    road_data = update_raw_data_with_road_rollout(
                        raw_data=raw_data_by_scenario[scenario_id],
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
