from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from torch import Tensor


def _sanitize_scenario_id(scenario_id: str) -> str:
    """파일명으로 쓸 수 있는 scene id를 만듭니다.

    Args:
        scenario_id: 원본 scenario id입니다.

    Returns:
        str: 안전한 파일명입니다.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_id)).strip("_")
    if safe:
        return safe
    return hashlib.sha1(str(scenario_id).encode("utf-8")).hexdigest()


class TeacherRolloutCache:
    """offline teacher open-loop rollout cache loader입니다.

    Args:
        cache_root: teacher cache root 디렉터리입니다.
        n_teacher_rollout: scene당 저장된 teacher rollout 개수입니다.
        rollout_set_size: 학습 step에서 뽑을 rollout 개수입니다.
        file_extension: scene별 cache 파일 확장자입니다.

    설명:
        각 scene cache는 ``.pt`` 파일로 저장한다고 가정합니다. 파일 안에는 최소한
        ``rollout_pose`` 와 ``agent_id`` 가 있어야 합니다. ``rollout_pose`` shape은
        ``[32, 20, N_cache, 4]`` 이며 채널은 ``x, y, cos(yaw), sin(yaw)`` 입니다.
        ``valid_mask`` 가 있으면 teacher가 실제로 생성한 agent만 real set에 포함합니다.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        n_teacher_rollout: int = 32,
        rollout_set_size: int = 16,
        file_extension: str = ".pt",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.n_teacher_rollout = int(n_teacher_rollout)
        self.rollout_set_size = int(rollout_set_size)
        self.file_extension = file_extension
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """cache index를 읽습니다.

        Returns:
            Dict[str, str]: scenario id에서 파일 경로로 가는 사전입니다.
        """
        index_path = self.cache_root / "index.json"
        if not index_path.exists():
            return {}
        with index_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        if not isinstance(raw, dict):
            raise ValueError(f"teacher cache index must be a dict: {index_path}")
        result: Dict[str, str] = {}
        for scenario_id, entry in raw.items():
            if isinstance(entry, str):
                result[str(scenario_id)] = entry
            elif isinstance(entry, dict) and "path" in entry:
                result[str(scenario_id)] = str(entry["path"])
            else:
                raise ValueError(f"Invalid teacher cache index entry for {scenario_id!r}: {entry!r}")
        return result

    def _resolve_scene_path(self, scenario_id: str) -> Path:
        """scene id에 대응하는 cache 파일 경로를 찾습니다.

        Args:
            scenario_id: Waymo scenario id입니다.

        Returns:
            Path: cache 파일 경로입니다.
        """
        if scenario_id in self.index:
            path = Path(self.index[scenario_id])
            return path if path.is_absolute() else self.cache_root / path
        safe_name = _sanitize_scenario_id(scenario_id) + self.file_extension
        return self.cache_root / safe_name

    def load_scene(self, scenario_id: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Tensor]:
        """scene 하나의 teacher cache를 읽습니다.

        Args:
            scenario_id: Waymo scenario id입니다.
            map_location: torch.load 위치입니다.

        Returns:
            Dict[str, Tensor]: ``rollout_pose`` 와 ``agent_id`` 를 포함한 cache입니다.
        """
        path = self._resolve_scene_path(scenario_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Teacher rollout cache for scenario_id={scenario_id!r} not found at {path}. "
                "Run tools/build_self_forced_gan_teacher_cache.py before GAN fine-tuning."
            )
        try:
            cache = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            cache = torch.load(path, map_location=map_location)
        if not isinstance(cache, dict):
            raise ValueError(f"Teacher cache file must contain a dict: {path}")
        if "rollout_pose" not in cache or "agent_id" not in cache:
            raise KeyError(
                f"Teacher cache {path} must contain 'rollout_pose' and 'agent_id'."
            )
        rollout_pose = cache["rollout_pose"]
        if rollout_pose.dim() != 4 or rollout_pose.shape[-1] != 4:
            raise ValueError(
                f"rollout_pose must have shape [R, T, N, 4], got {tuple(rollout_pose.shape)}."
            )
        if int(rollout_pose.shape[0]) < self.n_teacher_rollout:
            raise ValueError(
                f"Teacher cache has only {rollout_pose.shape[0]} rollouts, "
                f"but n_teacher_rollout={self.n_teacher_rollout}."
            )
        if "valid_mask" in cache:
            valid_mask = cache["valid_mask"]
            if valid_mask.dim() != 1 or int(valid_mask.numel()) != int(rollout_pose.shape[2]):
                raise ValueError(
                    "valid_mask must have shape [N_cache] matching rollout_pose.shape[2], "
                    f"got {tuple(valid_mask.shape)} and N_cache={rollout_pose.shape[2]}."
                )
        return {key: value for key, value in cache.items() if torch.is_tensor(value)}

    def _sample_indices(self, *, device: torch.device, generator: torch.Generator | None = None) -> Tensor:
        """teacher cache 32개 중 학습에 쓸 subset index를 샘플링합니다.

        Args:
            device: 반환 tensor device입니다.
            generator: 선택적 torch generator입니다.

        Returns:
            Tensor: rollout index입니다. shape은 ``[K]`` 입니다.
        """
        perm = torch.randperm(self.n_teacher_rollout, generator=generator, device=device)
        return perm[: self.rollout_set_size]

    @staticmethod
    def _build_id_to_index(agent_id: Tensor) -> Dict[int, int]:
        """agent id에서 cache 내부 index로 가는 사전을 만듭니다.

        Args:
            agent_id: cache의 agent id입니다. shape은 ``[N_cache]`` 입니다.

        Returns:
            Dict[int, int]: id -> index 사전입니다.
        """
        return {int(value): int(index) for index, value in enumerate(agent_id.detach().cpu().tolist())}

    def load_batch(
        self,
        *,
        scenario_ids: Sequence[str],
        batch_agent_id: Tensor,
        batch_agent_type: Tensor,
        batch_agent_batch: Tensor,
        n_max_agent: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """현재 batch에 맞춰 teacher rollout set을 load/alignment합니다.

        Args:
            scenario_ids: batch 안 scenario id 목록입니다. 길이는 ``B`` 입니다.
            batch_agent_id: flattened agent id입니다. shape은 ``[N_total]`` 입니다.
            batch_agent_type: flattened agent type입니다. shape은 ``[N_total]`` 입니다.
            batch_agent_batch: flattened scene index입니다. shape은 ``[N_total]`` 입니다.
            n_max_agent: padding 후 agent 최대 개수입니다.
            device: 반환 device입니다.
            dtype: 반환 dtype입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                teacher rollout pose ``[B, K, 20, N_max, 4]``,
                valid mask ``[B, N_max]``, agent type ``[B, N_max]`` 입니다.
        """
        batch_size = len(scenario_ids)
        k = self.rollout_set_size
        rollout_batch = torch.zeros(
            (batch_size, k, 20, n_max_agent, 4),
            device=device,
            dtype=dtype,
        )
        valid_batch = torch.zeros((batch_size, n_max_agent), device=device, dtype=torch.bool)
        type_batch = torch.zeros((batch_size, n_max_agent), device=device, dtype=torch.long)

        for scene_index, scenario_id in enumerate(scenario_ids):
            scene_mask = batch_agent_batch == scene_index
            scene_agent_ids = batch_agent_id[scene_mask].detach().cpu()
            scene_agent_types = batch_agent_type[scene_mask].detach().to(device=device, dtype=torch.long)
            n_scene_agent = int(scene_agent_ids.numel())
            if n_scene_agent == 0:
                continue
            cache = self.load_scene(str(scenario_id), map_location="cpu")
            cache_rollout = cache["rollout_pose"].to(device=device, dtype=dtype)
            cache_agent_id = cache["agent_id"]
            cache_valid_mask = cache.get("valid_mask")
            id_to_index = self._build_id_to_index(cache_agent_id)
            sample_idx = self._sample_indices(device=device)
            cache_rollout = cache_rollout.index_select(0, sample_idx)
            gather_indices: list[int] = []
            valid_flags: list[bool] = []
            for agent_id in scene_agent_ids.tolist():
                cache_index = id_to_index.get(int(agent_id), -1)
                gather_indices.append(max(cache_index, 0))
                is_valid = cache_index >= 0
                if is_valid and cache_valid_mask is not None:
                    is_valid = bool(cache_valid_mask[cache_index].item())
                valid_flags.append(is_valid)
            gather = torch.tensor(gather_indices, device=device, dtype=torch.long)
            aligned = cache_rollout.index_select(2, gather)
            valid = torch.tensor(valid_flags, device=device, dtype=torch.bool)
            aligned = aligned * valid.view(1, 1, -1, 1).to(dtype=dtype)
            rollout_batch[scene_index, :, :, :n_scene_agent, :] = aligned
            valid_batch[scene_index, :n_scene_agent] = valid
            type_batch[scene_index, :n_scene_agent] = scene_agent_types
        return rollout_batch, valid_batch, type_batch


def pack_flat_agent_tensor(
    flat_tensor: Tensor,
    batch_index: Tensor,
    *,
    batch_size: int,
    n_max_agent: int,
    fill_value: float = 0.0,
) -> Tensor:
    """flattened agent tensor를 scene별 padded tensor로 바꿉니다.

    Args:
        flat_tensor: flattened agent tensor입니다. shape은 ``[N_total, ...]`` 입니다.
        batch_index: 각 agent의 scene index입니다. shape은 ``[N_total]`` 입니다.
        batch_size: scene 개수입니다.
        n_max_agent: padding 후 최대 agent 수입니다.
        fill_value: padding 값입니다.

    Returns:
        Tensor: padded tensor입니다. shape은 ``[B, N_max, ...]`` 입니다.
    """
    output_shape = (batch_size, n_max_agent) + tuple(flat_tensor.shape[1:])
    output = flat_tensor.new_full(output_shape, fill_value)
    for scene_index in range(batch_size):
        mask = batch_index == scene_index
        values = flat_tensor[mask]
        n_value = int(values.shape[0])
        if n_value > 0:
            output[scene_index, :n_value] = values[:n_max_agent]
    return output


def build_current_pose_from_data(data: Any, *, num_historical_steps: int) -> Tensor:
    """data에서 현재 pose ``[x, y, cos(yaw), sin(yaw)]`` 를 만듭니다.

    Args:
        data: dataloader가 준 batch입니다.
        num_historical_steps: 과거 step 개수입니다. 현재 step index는 ``num_historical_steps - 1`` 입니다.

    Returns:
        Tensor: flattened current pose입니다. shape은 ``[N_total, 4]`` 입니다.
    """
    current_index = int(num_historical_steps) - 1
    agent = data["agent"]
    pos = agent["position"][:, current_index, :2]
    heading_key = None
    for candidate in ("heading", "head", "yaw"):
        if candidate in agent:
            heading_key = candidate
            break
    if heading_key is None:
        raise KeyError("data['agent'] must contain one of 'heading', 'head', or 'yaw'.")
    yaw = agent[heading_key][:, current_index]
    return torch.cat([pos, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)], dim=-1)


def pack_rollout_prediction_to_set(
    *,
    pred_traj: Tensor,
    pred_head: Tensor,
    batch_index: Tensor,
    batch_size: int,
    n_max_agent: int,
) -> Tensor:
    """flattened closed-loop prediction을 rollout-set pose tensor로 바꿉니다.

    Args:
        pred_traj: 위치 예측입니다. shape은 ``[N_total, 20, 2]`` 입니다.
        pred_head: yaw 예측입니다. shape은 ``[N_total, 20]`` 입니다.
        batch_index: 각 agent의 scene index입니다. shape은 ``[N_total]`` 입니다.
        batch_size: scene 개수입니다.
        n_max_agent: padding 후 최대 agent 수입니다.

    Returns:
        Tensor: rollout pose입니다. shape은 ``[B, 20, N_max, 4]`` 입니다.
    """
    pose = torch.cat(
        [
            pred_traj,
            torch.cos(pred_head).unsqueeze(-1),
            torch.sin(pred_head).unsqueeze(-1),
        ],
        dim=-1,
    )
    # [N_total, 20, 4] -> [N_total, 20, 4], padded to [B, N, 20, 4]
    packed = pack_flat_agent_tensor(
        pose,
        batch_index,
        batch_size=batch_size,
        n_max_agent=n_max_agent,
    )
    return packed.permute(0, 2, 1, 3).contiguous()
