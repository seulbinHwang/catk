from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
    DEFAULT_CONTROL_POS_SCALE_M,
    DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M,
    DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
    POSE_FLOW_DIM,
    build_rolling_control_target,
    build_rolling_control_target_with_round_trip_error,
    validate_control_no_slip_ratio_config,
    validate_control_yaw_scale_config,
)
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, validate_flow_window_steps


FLOW_CONTEXT_TOKEN_COUNT = 18
FLOW_TRAIN_ANCHOR_COUNT = 16
FLOW_TARGET_SIDECAR_ROW_ORDER_ANCHOR_MAJOR = "anchor_major_v1"


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습용 anchor 목표와 평가용 메타데이터를 만듭니다."""

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        flow_window_steps: int = 20,
        use_prefix_valid_future_loss_mask: bool = False,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        use_rolling_supervision: bool = True,
        control_pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
        control_vehicle_no_slip_point_ratio: float = DEFAULT_CONTROL_VEHICLE_NO_SLIP_POINT_RATIO,
        control_cyclist_no_slip_point_ratio: float = DEFAULT_CONTROL_CYCLIST_NO_SLIP_POINT_RATIO,
        control_round_trip_max_position_error_m: float = DEFAULT_CONTROL_ROUND_TRIP_MAX_POSITION_ERROR_M,
        flow_target_sidecar_dir: str | None = None,
        flow_target_sidecar_read: bool = True,
        flow_target_sidecar_write: bool = False,
        flow_target_sidecar_required: bool = False,
    ) -> None:
        super().__init__(
            map_token_file=map_token_file,
            agent_token_file=agent_token_file,
        )
        self.flow_window_steps = validate_flow_window_steps(
            flow_window_steps=flow_window_steps,
            commit_steps=self.shift,
        )
        self.use_prefix_valid_future_loss_mask = bool(use_prefix_valid_future_loss_mask)
        self.use_kinematic_control_flow = bool(use_kinematic_control_flow)
        self.use_holonomic_model_only = bool(use_holonomic_model_only)
        self.use_rolling_supervision = bool(use_rolling_supervision)
        self.control_pos_scale_m = float(control_pos_scale_m)
        self.control_vehicle_yaw_scale_rad = control_vehicle_yaw_scale_rad
        self.control_pedestrian_yaw_scale_rad = control_pedestrian_yaw_scale_rad
        self.control_cyclist_yaw_scale_rad = control_cyclist_yaw_scale_rad
        (
            self.control_vehicle_no_slip_point_ratio,
            self.control_cyclist_no_slip_point_ratio,
        ) = validate_control_no_slip_ratio_config(
            vehicle_no_slip_point_ratio=control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=control_cyclist_no_slip_point_ratio,
        )
        if self.use_kinematic_control_flow:
            (
                self.control_vehicle_yaw_scale_rad,
                self.control_pedestrian_yaw_scale_rad,
                self.control_cyclist_yaw_scale_rad,
            ) = validate_control_yaw_scale_config(
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            )
        self.control_round_trip_max_position_error_m = float(
            control_round_trip_max_position_error_m
        )
        if self.control_round_trip_max_position_error_m <= 0.0:
            raise ValueError(
                "control_round_trip_max_position_error_m must be positive, "
                f"got {self.control_round_trip_max_position_error_m}."
            )
        self.flow_target_dim = CONTROL_FLOW_DIM if self.use_kinematic_control_flow else POSE_FLOW_DIM
        self.flow_target_sidecar_dir = (
            str(flow_target_sidecar_dir) if flow_target_sidecar_dir not in (None, "") else ""
        )
        self.flow_target_sidecar_read = bool(flow_target_sidecar_read)
        self.flow_target_sidecar_write = bool(flow_target_sidecar_write)
        self.flow_target_sidecar_required = bool(flow_target_sidecar_required)
        self._flow_target_sidecar_fingerprint = self._build_flow_target_sidecar_fingerprint()

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """지도 토큰과 에이전트 토큰을 만들고 flow 목표를 붙입니다.

        Args:
            data: 원본 장면 배치입니다.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                지도 토큰 사전과 에이전트 토큰 사전입니다.
        """
        if self.training and self._flow_target_sidecar_read_enabled():
            loaded = self._load_training_sidecar_batch(data)
            if loaded is not None:
                return loaded

        tokenized_map, tokenized_agent = self._compute_online(data)
        if self.training and self._flow_target_sidecar_write_enabled() and self._is_single_graph_data(data):
            self._write_training_sidecar_from_tokenized(
                data=data,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
            )
        return tokenized_map, tokenized_agent

    def _compute_online(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self.tokenize_map(data)
        tokenized_agent, processed_agent = self.tokenize_agent(
            data,
            return_preprocessed=True,
        )
        tokenized_agent = self._build_flow_targets(
            data=data,
            tokenized_agent=tokenized_agent,
            processed_agent=processed_agent,
        )
        return tokenized_map, tokenized_agent

    def _flow_target_sidecar_read_enabled(self) -> bool:
        return bool(self.flow_target_sidecar_dir and self.flow_target_sidecar_read)

    def _flow_target_sidecar_write_enabled(self) -> bool:
        return bool(self.flow_target_sidecar_dir and self.flow_target_sidecar_write)

    def _file_sha1(self, path: str) -> str:
        digest = hashlib.sha1()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_flow_target_sidecar_fingerprint(self) -> str:
        payload = {
            "version": 1,
            "map_token_sha1": self._file_sha1(self.map_token_file_path),
            "agent_token_sha1": self._file_sha1(self.agent_token_file_path),
            "shift": int(self.shift),
            "flow_window_steps": int(self.flow_window_steps),
            "flow_train_anchor_count": int(FLOW_TRAIN_ANCHOR_COUNT),
            "flow_context_token_count": int(FLOW_CONTEXT_TOKEN_COUNT),
            "use_prefix_valid_future_loss_mask": bool(self.use_prefix_valid_future_loss_mask),
            "use_kinematic_control_flow": bool(self.use_kinematic_control_flow),
            "use_holonomic_model_only": bool(self.use_holonomic_model_only),
            "use_rolling_supervision": bool(self.use_rolling_supervision),
            "control_pos_scale_m": float(self.control_pos_scale_m),
            "control_vehicle_yaw_scale_rad": (
                None
                if self.control_vehicle_yaw_scale_rad is None
                else float(self.control_vehicle_yaw_scale_rad)
            ),
            "control_pedestrian_yaw_scale_rad": (
                None
                if self.control_pedestrian_yaw_scale_rad is None
                else float(self.control_pedestrian_yaw_scale_rad)
            ),
            "control_cyclist_yaw_scale_rad": (
                None
                if self.control_cyclist_yaw_scale_rad is None
                else float(self.control_cyclist_yaw_scale_rad)
            ),
            "control_vehicle_no_slip_point_ratio": float(self.control_vehicle_no_slip_point_ratio),
            "control_cyclist_no_slip_point_ratio": float(self.control_cyclist_no_slip_point_ratio),
            "control_round_trip_max_position_error_m": float(
                self.control_round_trip_max_position_error_m
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:16]

    def _flow_target_sidecar_root(self) -> Path:
        return Path(self.flow_target_sidecar_dir) / self._flow_target_sidecar_fingerprint

    def _scenario_ids_from_data(self, data: HeteroData) -> List[str]:
        scenario_ids = getattr(data, "scenario_id", None)
        if scenario_ids is None and isinstance(data, dict):
            scenario_ids = data.get("scenario_id")
        if scenario_ids is None:
            raise KeyError("flow target sidecar requires data.scenario_id.")
        if isinstance(scenario_ids, str):
            return [scenario_ids]
        if isinstance(scenario_ids, Sequence):
            return [str(item) for item in scenario_ids]
        return [str(scenario_ids)]

    def _sidecar_path_for_scenario(self, scenario_id: str) -> Path:
        safe_hash = hashlib.sha1(str(scenario_id).encode("utf-8")).hexdigest()
        return self._flow_target_sidecar_root() / f"{safe_hash}.pt"

    def _is_single_graph_data(self, data: HeteroData) -> bool:
        try:
            return len(self._scenario_ids_from_data(data)) == 1
        except Exception:
            return int(getattr(data, "num_graphs", 1) or 1) == 1

    def _metadata_for_sidecar(self, scenario_id: str) -> Dict[str, Any]:
        return {
            "version": 1,
            "scenario_id": str(scenario_id),
            "fingerprint": self._flow_target_sidecar_fingerprint,
        }

    @staticmethod
    def _cpu_detach_tree(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {key: FlowTokenProcessor._cpu_detach_tree(item) for key, item in value.items()}
        return value

    def _training_sidecar_payload(
        self,
        *,
        scenario_id: str,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        map_keys = [
            "position",
            "orientation",
            "token_idx",
            "type",
            "pl_type",
            "light_type",
        ]
        agent_keys = [
            "type",
            "shape",
            "ego_mask",
            "token_agent_shape",
            "ctx_sampled_idx",
            "ctx_sampled_pos",
            "ctx_sampled_heading",
            "ctx_valid",
            "flow_train_mask",
            "flow_train_clean_norm",
            "flow_train_clean_metric_norm",
            "flow_train_loss_mask",
            "flow_train_agent_type",
            "flow_train_agent_length",
        ]
        return {
            "metadata": self._metadata_for_sidecar(scenario_id),
            "map": {
                key: self._cpu_detach_tree(tokenized_map[key])
                for key in map_keys
                if key in tokenized_map
            },
            "agent": {
                key: self._cpu_detach_tree(tokenized_agent[key])
                for key in agent_keys
                if key in tokenized_agent
            },
        }

    def _write_training_sidecar_from_tokenized(
        self,
        *,
        data: HeteroData,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> None:
        scenario_ids = self._scenario_ids_from_data(data)
        if len(scenario_ids) != 1:
            return
        scenario_id = scenario_ids[0]
        path = self._sidecar_path_for_scenario(scenario_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._training_sidecar_payload(
            scenario_id=scenario_id,
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
        )
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _load_one_training_sidecar(self, scenario_id: str) -> Dict[str, Any] | None:
        path = self._sidecar_path_for_scenario(scenario_id)
        if not path.exists():
            if self.flow_target_sidecar_required:
                raise FileNotFoundError(f"Missing flow target sidecar: {path}")
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("fingerprint") != self._flow_target_sidecar_fingerprint:
            if self.flow_target_sidecar_required:
                raise ValueError(
                    "Flow target sidecar fingerprint mismatch for "
                    f"{scenario_id}: expected={self._flow_target_sidecar_fingerprint}, "
                    f"actual={metadata.get('fingerprint')}"
                )
            return None
        if str(metadata.get("scenario_id")) != str(scenario_id):
            if self.flow_target_sidecar_required:
                raise ValueError(
                    f"Flow target sidecar scenario mismatch: expected={scenario_id}, "
                    f"actual={metadata.get('scenario_id')}"
                )
            return None
        return payload

    @staticmethod
    def _to_device(value: Tensor, device: torch.device) -> Tensor:
        return value.to(device=device, non_blocking=True)

    def _load_training_sidecar_batch(
        self,
        data: HeteroData,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]] | None:
        if "train_mask" in data["agent"]:
            if self.flow_target_sidecar_required:
                raise RuntimeError(
                    "Flow target sidecar is disabled when data contains train_mask, "
                    "because random train target selection would no longer be equivalent."
                )
            return None
        preloaded = self._load_preloaded_training_sidecar_batch(data)
        if preloaded is not None:
            return preloaded
        scenario_ids = self._scenario_ids_from_data(data)
        payloads: List[Dict[str, Any]] = []
        for scenario_id in scenario_ids:
            payload = self._load_one_training_sidecar(scenario_id)
            if payload is None:
                return None
            payloads.append(payload)
        if len(payloads) == 0:
            return None
        device = data["agent"]["position"].device
        return self._collate_training_sidecars(payloads=payloads, device=device)

    @staticmethod
    def _split_tensor_by_counts(value: Tensor, counts: List[int]) -> List[Tensor]:
        chunks: List[Tensor] = []
        cursor = 0
        for count in counts:
            next_cursor = cursor + int(count)
            chunks.append(value[cursor:next_cursor])
            cursor = next_cursor
        if cursor != int(value.shape[0]):
            raise ValueError(
                f"Sidecar split count mismatch: expected {cursor}, got {int(value.shape[0])}."
            )
        return chunks

    @staticmethod
    def _counts_from_batch(batch: Tensor, num_graphs: int) -> List[int]:
        if batch.numel() == 0:
            return [0 for _ in range(num_graphs)]
        counts = torch.bincount(batch.detach().cpu(), minlength=num_graphs)
        return [int(item) for item in counts.tolist()]

    def _load_preloaded_training_sidecar_batch(
        self,
        data: HeteroData,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]] | None:
        payload = getattr(data, "flow_target_sidecar_payload", None)
        if payload is None:
            return None
        scenario_ids = self._scenario_ids_from_data(data)
        num_graphs = len(scenario_ids)
        if num_graphs <= 0:
            return None

        metadata = payload.get("metadata", {})
        fingerprints = metadata.get("fingerprint")
        if isinstance(fingerprints, str):
            fingerprints = [fingerprints]
        if fingerprints is not None:
            if len(fingerprints) != num_graphs or any(
                str(item) != self._flow_target_sidecar_fingerprint for item in fingerprints
            ):
                if self.flow_target_sidecar_required:
                    raise ValueError(
                        "Preloaded flow target sidecar fingerprint mismatch: "
                        f"expected={self._flow_target_sidecar_fingerprint}, actual={fingerprints}"
                    )
                return None

        metadata_scenario_ids = metadata.get("scenario_id")
        if isinstance(metadata_scenario_ids, str):
            metadata_scenario_ids = [metadata_scenario_ids]
        if metadata_scenario_ids is not None:
            if len(metadata_scenario_ids) != num_graphs or any(
                str(actual) != str(expected)
                for actual, expected in zip(metadata_scenario_ids, scenario_ids)
            ):
                if self.flow_target_sidecar_required:
                    raise ValueError(
                        "Preloaded flow target sidecar scenario mismatch: "
                        f"expected={scenario_ids}, actual={metadata_scenario_ids}"
                    )
                return None

        map_payload = payload["map"]
        agent_payload = payload["agent"]
        map_keys = ["position", "orientation", "token_idx", "type", "pl_type", "light_type"]
        per_agent_keys = [
            "type",
            "shape",
            "ego_mask",
            "token_agent_shape",
            "ctx_sampled_idx",
            "ctx_sampled_pos",
            "ctx_sampled_heading",
            "ctx_valid",
            "flow_train_mask",
        ]
        row_keys = [
            "flow_train_clean_norm",
            "flow_train_clean_metric_norm",
            "flow_train_loss_mask",
            "flow_train_agent_type",
            "flow_train_agent_length",
        ]

        device = data["agent"]["position"].device
        tokenized_map: Dict[str, Tensor] = {
            key: self._to_device(map_payload[key], device)
            for key in map_keys
        }
        tokenized_map["batch"] = data["pt_token"]["batch"]
        tokenized_map["token_traj_src"] = self.map_token_traj_src

        tokenized_agent: Dict[str, Tensor] = {
            key: self._to_device(agent_payload[key], device)
            for key in per_agent_keys
        }
        tokenized_agent["batch"] = data["agent"]["batch"]
        tokenized_agent["num_graphs"] = num_graphs
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            ).flatten(1, 3)
            tokenized_agent[f"token_bank_all_{k}"] = getattr(self, f"agent_token_all_{k}")

        expected_row_count = int(tokenized_agent["flow_train_mask"].long().sum().item())
        if metadata.get("flow_row_order") == FLOW_TARGET_SIDECAR_ROW_ORDER_ANCHOR_MAJOR:
            for key in row_keys:
                value = self._to_device(agent_payload[key], device)
                if expected_row_count != int(value.shape[0]):
                    raise ValueError(
                        f"Preloaded flow target sidecar row count mismatch for {key}: "
                        f"mask_count={expected_row_count}, value_rows={int(value.shape[0])}."
                    )
                tokenized_agent[key] = value
            return tokenized_map, tokenized_agent

        row_slices: List[List[tuple[int, int]]] = []
        cursor = 0
        flow_train_mask = tokenized_agent["flow_train_mask"]
        agent_batch = tokenized_agent["batch"]
        num_anchor = int(flow_train_mask.shape[1])
        for sample_idx in range(num_graphs):
            sample_counts = flow_train_mask[agent_batch == sample_idx].long().sum(dim=0)
            sample_slices: List[tuple[int, int]] = []
            for anchor_idx in range(num_anchor):
                count = int(sample_counts[anchor_idx].item())
                next_cursor = cursor + count
                sample_slices.append((cursor, next_cursor))
                cursor = next_cursor
            row_slices.append(sample_slices)

        for key in row_keys:
            value = self._to_device(agent_payload[key], device)
            if cursor != int(value.shape[0]):
                raise ValueError(
                    f"Preloaded flow target sidecar row count mismatch for {key}: "
                    f"mask_count={cursor}, value_rows={int(value.shape[0])}."
                )
            ordered_parts: List[Tensor] = []
            for anchor_idx in range(num_anchor):
                for sample_idx in range(num_graphs):
                    start, end = row_slices[sample_idx][anchor_idx]
                    if end > start:
                        ordered_parts.append(value[start:end])
            if ordered_parts:
                tokenized_agent[key] = torch.cat(ordered_parts, dim=0)
            else:
                tokenized_agent[key] = value.new_zeros((0,) + tuple(value.shape[1:]))
        return tokenized_map, tokenized_agent

    def _collate_training_sidecars(
        self,
        *,
        payloads: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self._collate_sidecar_maps(payloads=payloads, device=device)
        tokenized_agent = self._collate_sidecar_agents(payloads=payloads, device=device)
        return tokenized_map, tokenized_agent

    def _collate_sidecar_maps(
        self,
        *,
        payloads: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        map_payloads = [payload["map"] for payload in payloads]
        map_keys = ["position", "orientation", "token_idx", "type", "pl_type", "light_type"]
        tokenized_map = {
            key: torch.cat([self._to_device(item[key], device) for item in map_payloads], dim=0)
            for key in map_keys
        }
        batch_parts = []
        for scenario_idx, item in enumerate(map_payloads):
            count = int(item["position"].shape[0])
            batch_parts.append(torch.full((count,), scenario_idx, device=device, dtype=torch.long))
        tokenized_map["batch"] = torch.cat(batch_parts, dim=0)
        tokenized_map["token_traj_src"] = self.map_token_traj_src
        return tokenized_map

    def _split_sidecar_flow_rows(self, agent_payload: Dict[str, Tensor], key: str) -> List[Tensor]:
        flow_train_mask = agent_payload["flow_train_mask"]
        counts = flow_train_mask.long().sum(dim=0).tolist()
        value = agent_payload[key]
        chunks: List[Tensor] = []
        cursor = 0
        for count in counts:
            next_cursor = cursor + int(count)
            chunks.append(value[cursor:next_cursor])
            cursor = next_cursor
        if cursor != int(value.shape[0]):
            raise ValueError(
                f"Flow target sidecar row count mismatch for {key}: "
                f"mask_count={cursor}, value_rows={int(value.shape[0])}."
            )
        return chunks

    def _collate_sidecar_agents(
        self,
        *,
        payloads: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        agent_payloads = [payload["agent"] for payload in payloads]
        per_agent_keys = [
            "type",
            "shape",
            "ego_mask",
            "token_agent_shape",
            "ctx_sampled_idx",
            "ctx_sampled_pos",
            "ctx_sampled_heading",
            "ctx_valid",
            "flow_train_mask",
        ]
        tokenized_agent: Dict[str, Tensor] = {
            key: torch.cat([self._to_device(item[key], device) for item in agent_payloads], dim=0)
            for key in per_agent_keys
        }
        batch_parts = []
        for scenario_idx, item in enumerate(agent_payloads):
            count = int(item["type"].shape[0])
            batch_parts.append(torch.full((count,), scenario_idx, device=device, dtype=torch.long))
        tokenized_agent["batch"] = torch.cat(batch_parts, dim=0)
        tokenized_agent["num_graphs"] = len(payloads)
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            ).flatten(1, 3)
            tokenized_agent[f"token_bank_all_{k}"] = getattr(self, f"agent_token_all_{k}")

        row_keys = [
            "flow_train_clean_norm",
            "flow_train_clean_metric_norm",
            "flow_train_loss_mask",
            "flow_train_agent_type",
            "flow_train_agent_length",
        ]
        split_rows = {
            key: [self._split_sidecar_flow_rows(item, key) for item in agent_payloads]
            for key in row_keys
        }
        num_anchor = int(tokenized_agent["flow_train_mask"].shape[1])
        for key in row_keys:
            ordered_parts: List[Tensor] = []
            for anchor_idx in range(num_anchor):
                for sample_idx in range(len(agent_payloads)):
                    part = split_rows[key][sample_idx][anchor_idx]
                    if int(part.shape[0]) > 0:
                        ordered_parts.append(self._to_device(part, device))
            if ordered_parts:
                tokenized_agent[key] = torch.cat(ordered_parts, dim=0)
            else:
                example = agent_payloads[0][key]
                shape = (0,) + tuple(example.shape[1:])
                tokenized_agent[key] = torch.zeros(
                    shape,
                    device=device,
                    dtype=example.dtype,
                )
        return tokenized_agent

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
        processed_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습/평가에 필요한 anchor별 미래와 메타데이터를 만듭니다.

        Args:
            data: 원본 장면 배치입니다.
            tokenized_agent: coarse token 기반 에이전트 토큰 사전입니다.
            processed_agent: 전처리된 실제 좌표와 방향 사전입니다.

        Returns:
            Dict[str, Tensor]:
                flow 관련 필드가 추가된 에이전트 토큰 사전입니다.
        """
        valid = processed_agent["valid"]
        pos = processed_agent["pos"]
        heading = processed_agent["heading"]

        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :FLOW_CONTEXT_TOKEN_COUNT].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        num_anchor = FLOW_TRAIN_ANCHOR_COUNT
        raw_current_steps = [
            self.shift * (anchor_idx + 2)
            for anchor_idx in range(num_anchor)
        ]

        if "train_mask" in data["agent"]:
            train_mask = data["agent"]["train_mask"].bool()
        else:
            train_mask = torch.ones(num_agent, device=device, dtype=torch.bool)

        tokenized_agent.update(
            {
                "ctx_sampled_idx": ctx_sampled_idx,
                "ctx_sampled_pos": ctx_sampled_pos,
                "ctx_sampled_heading": ctx_sampled_heading,
                "ctx_valid": ctx_valid,
            }
        )

        if self.training:
            if self.use_kinematic_control_flow:
                tokenized_agent = self._build_kinematic_flow_train_targets_batched(
                    tokenized_agent=tokenized_agent,
                    pos=pos,
                    heading=heading,
                    valid=valid,
                    train_mask=train_mask,
                    ctx_valid=ctx_valid,
                    raw_current_steps=raw_current_steps,
                    dtype=dtype,
                    device=device,
                )
                for key in [
                    "valid_mask",
                    "gt_idx",
                    "gt_pos",
                    "gt_heading",
                    "sampled_idx",
                    "sampled_pos",
                    "sampled_heading",
                ]:
                    tokenized_agent.pop(key, None)
                return tokenized_agent

            flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
            flow_train_chunks: List[Tensor] = []
            flow_train_metric_chunks: List[Tensor] = []
            flow_train_loss_mask_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_agent_length_chunks: List[Tensor] = []

            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = valid[:, raw_step]
                future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
                anchor_mask = current_valid & future_loss_mask.any(dim=1)
                train_anchor_mask = anchor_mask & train_mask
                if not train_anchor_mask.any():
                    continue

                current_pos = pos[:, raw_step]
                current_head = heading[:, raw_step]
                selected_future_loss_mask = future_loss_mask[train_anchor_mask]
                flow_clean_result = self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=current_pos,
                    current_head=current_head,
                    agent_type=tokenized_agent["type"],
                    agent_length=tokenized_agent["shape"][:, 0],
                    anchor_mask=train_anchor_mask,
                    raw_step=raw_step,
                    future_loss_mask=selected_future_loss_mask,
                    return_round_trip_error=self.use_kinematic_control_flow,
                )
                if self.use_kinematic_control_flow:
                    flow_train_clean_norm, round_trip_error_m = flow_clean_result
                    keep_mask = self._build_control_round_trip_keep_mask(
                        round_trip_error_m=round_trip_error_m,
                        future_loss_mask=selected_future_loss_mask,
                    )
                    if not bool(keep_mask.all().item()):
                        selected_agent_index = train_anchor_mask.nonzero(as_tuple=False).flatten()
                        kept_agent_index = selected_agent_index[keep_mask]
                        filtered_train_anchor_mask = torch.zeros_like(train_anchor_mask)
                        filtered_train_anchor_mask[kept_agent_index] = True
                        train_anchor_mask = filtered_train_anchor_mask
                        flow_train_clean_norm = flow_train_clean_norm[keep_mask]
                        selected_future_loss_mask = selected_future_loss_mask[keep_mask]
                else:
                    flow_train_clean_norm = flow_clean_result

                flow_train_mask[:, anchor_offset] = train_anchor_mask
                if not train_anchor_mask.any():
                    continue

                flow_train_metric_norm = (
                    self._build_anchor_clean_norm(
                        pos=pos,
                        heading=heading,
                        current_pos=current_pos,
                        current_head=current_head,
                        agent_type=tokenized_agent["type"],
                        agent_length=tokenized_agent["shape"][:, 0],
                        anchor_mask=train_anchor_mask,
                        raw_step=raw_step,
                        future_loss_mask=selected_future_loss_mask,
                        force_pose_space=True,
                    )
                    if self.use_kinematic_control_flow
                    else flow_train_clean_norm
                )
                flow_train_chunks.append(flow_train_clean_norm)
                flow_train_metric_chunks.append(flow_train_metric_norm)
                flow_train_loss_mask_chunks.append(selected_future_loss_mask)
                flow_train_agent_type_chunks.append(tokenized_agent["type"][train_anchor_mask])
                flow_train_agent_length_chunks.append(tokenized_agent["shape"][train_anchor_mask, 0])

            self._assert_flow_train_anchor_context_valid(
                flow_train_mask=flow_train_mask,
                ctx_valid=ctx_valid,
            )
            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": self._concat_flow_chunks(
                        chunks=flow_train_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_clean_metric_norm": self._concat_flow_chunks(
                        chunks=flow_train_metric_chunks,
                        dtype=dtype,
                        device=device,
                        target_dim=POSE_FLOW_DIM,
                    ),
                    "flow_train_loss_mask": self._concat_mask_chunks(
                        chunks=flow_train_loss_mask_chunks,
                        device=device,
                    ),
                    "flow_train_agent_type": self._concat_vector_chunks(
                        chunks=flow_train_agent_type_chunks,
                        dtype=tokenized_agent["type"].dtype,
                        device=device,
                    ),
                    "flow_train_agent_length": self._concat_vector_chunks(
                        chunks=flow_train_agent_length_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )
            for key in [
                "valid_mask",
                "gt_idx",
                "gt_pos",
                "gt_heading",
                "sampled_idx",
                "sampled_pos",
                "sampled_heading",
            ]:
                tokenized_agent.pop(key, None)
            return tokenized_agent

        flow_eval_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
        flow_eval_chunks: List[Tensor] = []
        flow_eval_metric_chunks: List[Tensor] = []
        flow_eval_agent_type_chunks: List[Tensor] = []
        flow_eval_agent_length_chunks: List[Tensor] = []
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = valid[:, raw_step]
            future_valid = self._build_anchor_future_valid(valid=valid, raw_step=raw_step)
            anchor_mask = current_valid & future_valid
            flow_eval_mask[:, anchor_offset] = anchor_mask
            if not anchor_mask.any():
                continue

            flow_eval_agent_type_chunks.append(tokenized_agent["type"][anchor_mask])
            flow_eval_agent_length_chunks.append(tokenized_agent["shape"][anchor_mask, 0])
            flow_eval_clean_norm = self._build_anchor_clean_norm(
                pos=pos,
                heading=heading,
                current_pos=pos[:, raw_step],
                current_head=heading[:, raw_step],
                agent_type=tokenized_agent["type"],
                agent_length=tokenized_agent["shape"][:, 0],
                anchor_mask=anchor_mask,
                raw_step=raw_step,
            )
            flow_eval_chunks.append(flow_eval_clean_norm)
            flow_eval_metric_chunks.append(
                self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=pos[:, raw_step],
                    current_head=heading[:, raw_step],
                    agent_type=tokenized_agent["type"],
                    agent_length=tokenized_agent["shape"][:, 0],
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                    force_pose_space=True,
                )
                if self.use_kinematic_control_flow
                else flow_eval_clean_norm
            )

        tokenized_agent.update(
            {
                "flow_eval_mask": flow_eval_mask,
                "flow_eval_clean_norm": self._concat_flow_chunks(
                    chunks=flow_eval_chunks,
                    dtype=dtype,
                    device=device,
                ),
                "flow_eval_clean_metric_norm": self._concat_flow_chunks(
                    chunks=flow_eval_metric_chunks,
                    dtype=dtype,
                    device=device,
                    target_dim=POSE_FLOW_DIM,
                ),
                "flow_eval_agent_type": self._concat_vector_chunks(
                    chunks=flow_eval_agent_type_chunks,
                    dtype=tokenized_agent["type"].dtype,
                    device=device,
                ),
                "flow_eval_agent_length": self._concat_vector_chunks(
                    chunks=flow_eval_agent_length_chunks,
                    dtype=dtype,
                    device=device,
                ),
            }
        )
        return tokenized_agent

    def _build_kinematic_flow_train_targets_batched(
        self,
        tokenized_agent: Dict[str, Tensor],
        pos: Tensor,
        heading: Tensor,
        valid: Tensor,
        train_mask: Tensor,
        ctx_valid: Tensor,
        raw_current_steps: List[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """control-space train anchors를 한 번에 모아 target을 만듭니다.

        각 anchor의 future prefix 길이는 ``future_loss_mask`` row로 보존하고,
        round-trip keep mask 적용 뒤에도 anchor 순서와 agent 순서를 유지합니다.
        """
        num_agent = pos.shape[0]
        num_anchor = len(raw_current_steps)
        flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)

        candidate_anchor_offsets: List[Tensor] = []
        candidate_agent_indices: List[Tensor] = []
        candidate_current_pos: List[Tensor] = []
        candidate_current_head: List[Tensor] = []
        candidate_future_pos: List[Tensor] = []
        candidate_future_head: List[Tensor] = []
        candidate_future_loss_mask: List[Tensor] = []

        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = valid[:, raw_step]
            future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
            train_anchor_mask = current_valid & future_loss_mask.any(dim=1) & train_mask
            if not train_anchor_mask.any():
                continue

            selected_agent_index = train_anchor_mask.nonzero(as_tuple=False).flatten()
            selected_current_pos = pos[selected_agent_index, raw_step]
            selected_current_head = heading[selected_agent_index, raw_step]
            selected_future_loss_mask = future_loss_mask[selected_agent_index]
            future_pos, future_head = self._build_selected_anchor_future_window(
                pos=pos,
                heading=heading,
                selected_agent_index=selected_agent_index,
                selected_current_pos=selected_current_pos,
                selected_current_head=selected_current_head,
                raw_step=raw_step,
                future_loss_mask=selected_future_loss_mask,
            )

            candidate_anchor_offsets.append(
                torch.full(
                    (selected_agent_index.shape[0],),
                    anchor_offset,
                    device=device,
                    dtype=torch.long,
                )
            )
            candidate_agent_indices.append(selected_agent_index)
            candidate_current_pos.append(selected_current_pos)
            candidate_current_head.append(selected_current_head)
            candidate_future_pos.append(future_pos)
            candidate_future_head.append(future_head)
            candidate_future_loss_mask.append(selected_future_loss_mask)

        if len(candidate_agent_indices) == 0:
            self._assert_flow_train_anchor_context_valid(
                flow_train_mask=flow_train_mask,
                ctx_valid=ctx_valid,
            )
            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": torch.zeros(
                        (0, self.flow_window_steps, self.flow_target_dim),
                        device=device,
                        dtype=dtype,
                    ),
                    "flow_train_clean_metric_norm": torch.zeros(
                        (0, self.flow_window_steps, POSE_FLOW_DIM),
                        device=device,
                        dtype=dtype,
                    ),
                    "flow_train_loss_mask": torch.zeros(
                        (0, self.flow_window_steps),
                        device=device,
                        dtype=torch.bool,
                    ),
                    "flow_train_agent_type": torch.zeros(
                        (0,),
                        device=device,
                        dtype=tokenized_agent["type"].dtype,
                    ),
                    "flow_train_agent_length": torch.zeros((0,), device=device, dtype=dtype),
                }
            )
            return tokenized_agent

        anchor_offsets = torch.cat(candidate_anchor_offsets, dim=0)
        agent_indices = torch.cat(candidate_agent_indices, dim=0)
        current_pos = torch.cat(candidate_current_pos, dim=0)
        current_head = torch.cat(candidate_current_head, dim=0)
        future_pos = torch.cat(candidate_future_pos, dim=0)
        future_head = torch.cat(candidate_future_head, dim=0)
        future_loss_mask = torch.cat(candidate_future_loss_mask, dim=0)
        agent_type = tokenized_agent["type"][agent_indices]
        agent_length = tokenized_agent["shape"][agent_indices, 0]

        flow_train_clean_norm, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=current_pos,
            current_head=current_head,
            agent_type=agent_type,
            agent_length=agent_length,
            pos_scale_m=self.control_pos_scale_m,
            vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
            use_holonomic_model_only=self.use_holonomic_model_only,
            use_rolling_supervision=self.use_rolling_supervision,
            vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
        )
        keep_mask = self._build_control_round_trip_keep_mask(
            round_trip_error_m=round_trip_error_m,
            future_loss_mask=future_loss_mask,
        )

        kept_agent_indices = agent_indices[keep_mask]
        kept_anchor_offsets = anchor_offsets[keep_mask]
        flow_train_mask[kept_agent_indices, kept_anchor_offsets] = True
        self._assert_flow_train_anchor_context_valid(
            flow_train_mask=flow_train_mask,
            ctx_valid=ctx_valid,
        )

        flow_train_clean_norm = flow_train_clean_norm[keep_mask]
        future_loss_mask = future_loss_mask[keep_mask]
        current_pos = current_pos[keep_mask]
        current_head = current_head[keep_mask]
        future_pos = future_pos[keep_mask]
        future_head = future_head[keep_mask]
        agent_type = agent_type[keep_mask]
        agent_length = agent_length[keep_mask]
        flow_train_metric_norm = self._build_pose_space_target_from_future_window(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=current_pos,
            current_head=current_head,
        )

        tokenized_agent.update(
            {
                "flow_train_mask": flow_train_mask,
                "flow_train_clean_norm": flow_train_clean_norm,
                "flow_train_clean_metric_norm": flow_train_metric_norm,
                "flow_train_loss_mask": future_loss_mask,
                "flow_train_agent_type": agent_type,
                "flow_train_agent_length": agent_length,
            }
        )
        return tokenized_agent

    def _build_selected_anchor_future_window(
        self,
        pos: Tensor,
        heading: Tensor,
        selected_agent_index: Tensor,
        selected_current_pos: Tensor,
        selected_current_head: Tensor,
        raw_step: int,
        future_loss_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """선택된 anchor-agent row의 valid prefix 미래 window를 만듭니다."""
        num_selected = selected_agent_index.shape[0]
        expected_shape = (num_selected, self.flow_window_steps)
        if tuple(future_loss_mask.shape) != expected_shape:
            raise ValueError(
                "future_loss_mask shape must match selected anchors and flow_window_steps: "
                f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
            )
        future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
        valid_step_count = future_loss_mask.long().sum(dim=1)
        if bool((valid_step_count <= 0).any().item()):
            raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

        future_start = raw_step + 1
        future_pos = selected_current_pos.unsqueeze(1).expand(-1, self.flow_window_steps, -1).clone()
        future_head = selected_current_head.unsqueeze(1).expand(-1, self.flow_window_steps).clone()

        available_len = min(self.flow_window_steps, max(0, pos.shape[1] - future_start))
        if available_len > 0:
            step_slice = slice(future_start, future_start + available_len)
            future_pos[:, :available_len] = pos[selected_agent_index, step_slice]
            future_head[:, :available_len] = heading[selected_agent_index, step_slice]

        last_valid_index = valid_step_count - 1
        last_valid_pos = future_pos.gather(
            dim=1,
            index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
        ).squeeze(1)
        last_valid_head = future_head.gather(
            dim=1,
            index=last_valid_index.view(-1, 1),
        ).squeeze(1)
        invalid_future_mask = ~future_loss_mask
        future_pos = torch.where(
            invalid_future_mask.unsqueeze(-1),
            last_valid_pos.unsqueeze(1),
            future_pos,
        )
        future_head = torch.where(
            invalid_future_mask,
            last_valid_head.unsqueeze(1),
            future_head,
        )
        return future_pos, future_head

    def _build_pose_space_target_from_future_window(
        self,
        future_pos: Tensor,
        future_head: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
    ) -> Tensor:
        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=current_pos,
            head_now=current_head,
        )
        return torch.stack(
            [
                future_pos_local[..., 0] / 20.0,
                future_pos_local[..., 1] / 20.0,
                future_head_local.cos(),
                future_head_local.sin(),
            ],
            dim=-1,
        )

    def _build_control_round_trip_keep_mask(
        self,
        round_trip_error_m: Tensor,
        future_loss_mask: Tensor,
    ) -> Tensor:
        """control 복원 위치 오차가 설정값 이하인 anchor만 남깁니다."""
        if round_trip_error_m.ndim != 2:
            raise ValueError(
                "round_trip_error_m must have shape [n_valid_anchor, flow_window_steps], "
                f"got {tuple(round_trip_error_m.shape)}."
            )
        if tuple(future_loss_mask.shape) != tuple(round_trip_error_m.shape):
            raise ValueError(
                "future_loss_mask shape must match round_trip_error_m: "
                f"expected={tuple(round_trip_error_m.shape)}, actual={tuple(future_loss_mask.shape)}."
            )
        if round_trip_error_m.shape[0] == 0:
            return torch.zeros((0,), device=round_trip_error_m.device, dtype=torch.bool)

        mask = future_loss_mask.to(device=round_trip_error_m.device, dtype=torch.bool)
        masked_error_m = torch.where(
            mask,
            round_trip_error_m,
            torch.zeros_like(round_trip_error_m),
        )
        max_position_error_m = masked_error_m.max(dim=1).values
        return max_position_error_m <= self.control_round_trip_max_position_error_m

    def _assert_flow_train_anchor_context_valid(
        self,
        flow_train_mask: Tensor,
        ctx_valid: Tensor,
    ) -> None:
        """선택된 flow 학습 anchor의 현재 0.5초 context token 유효성을 확인합니다."""
        if flow_train_mask.numel() == 0:
            return

        required_ctx_steps = flow_train_mask.shape[1] + 1
        if ctx_valid.shape[1] < required_ctx_steps:
            raise ValueError(
                "Flow train context validity check requires one leading context token "
                f"plus all anchors: required={required_ctx_steps}, actual={ctx_valid.shape[1]}."
            )

        anchor_ctx_valid = ctx_valid[:, 1:required_ctx_steps]
        invalid_anchor_mask = flow_train_mask & ~anchor_ctx_valid
        if invalid_anchor_mask.any():
            invalid_count = int(invalid_anchor_mask.sum().item())
            selected_count = int(flow_train_mask.sum().item())
            raise ValueError(
                "Flow train invariant violated: selected training anchors include invalid "
                "current 0.5s context tokens. "
                f"invalid_count={invalid_count}, selected_count={selected_count}."
            )

    def _build_anchor_future_valid(self, valid: Tensor, raw_step: int) -> Tensor:
        future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
        return future_loss_mask.all(dim=1)

    def _build_anchor_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """현재 설정에 맞는 미래 loss mask를 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
        """
        if self.use_prefix_valid_future_loss_mask:
            return self._build_prefix_valid_future_loss_mask(valid=valid, raw_step=raw_step)
        return self._build_full_window_future_loss_mask(valid=valid, raw_step=raw_step)

    def _build_full_window_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """기존 방식처럼 전체 미래 window가 유효한 경우에만 loss mask를 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
                미래 전체가 유효한 agent만 모든 step이 ``True`` 입니다.
        """
        future_start = raw_step + 1
        # future_loss_mask: [n_agent, flow_window_steps]
        future_loss_mask = torch.zeros(
            (valid.shape[0], self.flow_window_steps),
            device=valid.device,
            dtype=torch.bool,
        )
        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))
        if available_len != self.flow_window_steps:
            return future_loss_mask

        # available_future_valid: [n_agent, flow_window_steps]
        available_future_valid = valid[:, future_start : future_start + available_len].bool()
        full_future_valid = available_future_valid.all(dim=1)
        future_loss_mask[full_future_valid] = True
        return future_loss_mask

    def _build_prefix_valid_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """가까운 미래부터 연속으로 유효한 구간만 loss mask로 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
                ``raw_step + 1``부터 처음 유효하지 않은 step 직전까지만
                ``True`` 입니다. 첫 미래 step이 유효하지 않으면 전부 ``False`` 입니다.
        """
        future_start = raw_step + 1
        # future_loss_mask: [n_agent, flow_window_steps]
        future_loss_mask = torch.zeros(
            (valid.shape[0], self.flow_window_steps),
            device=valid.device,
            dtype=torch.bool,
        )
        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))
        if available_len <= 0:
            return future_loss_mask

        # available_future_valid: [n_agent, available_len]
        available_future_valid = valid[:, future_start : future_start + available_len].bool()
        # prefix_valid: [n_agent, available_len]
        prefix_valid = available_future_valid.to(dtype=torch.long).cumprod(dim=1).bool()
        future_loss_mask[:, :available_len] = prefix_valid
        return future_loss_mask

    def _build_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        agent_type: Tensor,
        agent_length: Tensor | None,
        anchor_mask: Tensor,
        raw_step: int,
        future_loss_mask: Tensor | None = None,
        return_round_trip_error: bool = False,
        force_pose_space: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """한 anchor에서 실제로 쓰는 agent만 골라 미래 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            agent_type: agent 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_length: WOMD box length입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.
            future_loss_mask: loss에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 window를 모두 사용합니다.
            return_round_trip_error: control-space label의 복원 위치 오차도 함께 돌려줄지 정합니다.
            force_pose_space: control-space 학습 중에도 raw GT 기준 pose-space target을
                만들어 open-loop metric 정답으로 쓸 때 켭니다.

        Returns:
            Tensor | Tuple[Tensor, Tensor]:
                정규화된 미래 목표입니다.
                pose-space에서는 ``[n_valid_anchor, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_anchor, flow_window_steps, 3]`` 입니다.
                ``return_round_trip_error=True`` 이면 두 번째 값으로 meter 단위 복원 오차
                ``[n_valid_anchor, flow_window_steps]`` 를 함께 돌려줍니다.
        """
        if force_pose_space and return_round_trip_error:
            raise ValueError("force_pose_space cannot be combined with return_round_trip_error.")
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            target_dim = POSE_FLOW_DIM if force_pose_space else self.flow_target_dim
            empty_target = pos.new_zeros((0, self.flow_window_steps, target_dim))
            if return_round_trip_error:
                return empty_target, pos.new_zeros((0, self.flow_window_steps))
            return empty_target

        selected_current_pos = current_pos[anchor_mask]
        selected_current_head = current_head[anchor_mask]
        selected_agent_type = agent_type[anchor_mask]
        selected_agent_length = agent_length[anchor_mask] if agent_length is not None else None
        future_start = raw_step + 1
        future_end = future_start + self.flow_window_steps

        if future_loss_mask is None:
            if future_end > pos.shape[1]:
                raise ValueError(
                    "Requested flow future window exceeds the available sequence length: "
                    f"raw_step={raw_step}, flow_window_steps={self.flow_window_steps}, "
                    f"n_step={pos.shape[1]}."
                )
            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = pos[anchor_mask, future_start:future_end]
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = heading[anchor_mask, future_start:future_end]
        else:
            expected_shape = (num_valid_anchor, self.flow_window_steps)
            if tuple(future_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "future_loss_mask shape must match selected anchors and flow_window_steps: "
                    f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                )
            future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
            valid_step_count = future_loss_mask.long().sum(dim=1)
            if bool((valid_step_count <= 0).any().item()):
                raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = selected_current_pos.unsqueeze(1).expand(-1, self.flow_window_steps, -1).clone()
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = selected_current_head.unsqueeze(1).expand(-1, self.flow_window_steps).clone()

            available_len = min(self.flow_window_steps, max(0, pos.shape[1] - future_start))
            if available_len > 0:
                future_pos[:, :available_len] = pos[anchor_mask, future_start : future_start + available_len]
                future_head[:, :available_len] = heading[anchor_mask, future_start : future_start + available_len]

            last_valid_index = valid_step_count - 1
            # last_valid_pos: [n_valid_anchor, 2]
            last_valid_pos = future_pos.gather(
                dim=1,
                index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
            ).squeeze(1)
            # last_valid_head: [n_valid_anchor]
            last_valid_head = future_head.gather(
                dim=1,
                index=last_valid_index.view(-1, 1),
            ).squeeze(1)
            invalid_future_mask = ~future_loss_mask
            future_pos = torch.where(
                invalid_future_mask.unsqueeze(-1),
                last_valid_pos.unsqueeze(1),
                future_pos,
            )
            future_head = torch.where(
                invalid_future_mask,
                last_valid_head.unsqueeze(1),
                future_head,
            )

        if self.use_kinematic_control_flow and not force_pose_space:
            if return_round_trip_error:
                return build_rolling_control_target_with_round_trip_error(
                    future_pos=future_pos,
                    future_head=future_head,
                    current_pos=selected_current_pos,
                    current_head=selected_current_head,
                    agent_type=selected_agent_type,
                    agent_length=selected_agent_length,
                    pos_scale_m=self.control_pos_scale_m,
                    vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                    pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                    cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                    use_holonomic_model_only=self.use_holonomic_model_only,
                    use_rolling_supervision=self.use_rolling_supervision,
                    vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                    cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
                )
            return build_rolling_control_target(
                future_pos=future_pos,
                future_head=future_head,
                current_pos=selected_current_pos,
                current_head=selected_current_head,
                agent_type=selected_agent_type,
                agent_length=selected_agent_length,
                pos_scale_m=self.control_pos_scale_m,
                vehicle_yaw_scale_rad=self.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=self.use_holonomic_model_only,
                use_rolling_supervision=self.use_rolling_supervision,
                vehicle_no_slip_point_ratio=self.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=self.control_cyclist_no_slip_point_ratio,
            )

        if return_round_trip_error:
            raise ValueError("return_round_trip_error is only supported for control-space flow targets.")

        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=selected_current_pos,
            head_now=selected_current_head,
        )
        return torch.stack(
            [
                future_pos_local[..., 0] / 20.0,
                future_pos_local[..., 1] / 20.0,
                future_head_local.cos(),
                future_head_local.sin(),
            ],
            dim=-1,
        )

    def _concat_flow_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
        target_dim: int | None = None,
    ) -> Tensor:
        """빈 경우까지 포함해서 flow 목표 조각을 하나로 합칩니다.

        Args:
            chunks: 각 anchor에서 만든 목표 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 목표입니다. shape은 ``[n_total_valid_anchor, 20, 4]`` 입니다.
                유효한 anchor가 없으면 ``[0, 20, 4]`` 빈 텐서를 돌려줍니다.
        """
        if target_dim is None:
            target_dim = self.flow_target_dim
        if len(chunks) == 0:
            return torch.zeros((0, self.flow_window_steps, target_dim), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    def _concat_mask_chunks(
        self,
        chunks: List[Tensor],
        device: torch.device,
    ) -> Tensor:
        """미래 step별 loss mask 조각을 하나로 잇습니다.

        Args:
            chunks: 각 anchor에서 고른 mask 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 mask입니다.
                shape은 ``[n_total_valid_anchor, flow_window_steps]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, self.flow_window_steps), device=device, dtype=torch.bool)
        return torch.cat([chunk.to(device=device, dtype=torch.bool) for chunk in chunks], dim=0)

    def _concat_vector_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """1차원 조각 목록을 하나의 벡터로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 벡터입니다. shape은 ``[n_total_valid_anchor]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0,), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _concat_matrix_chunks(
        self,
        chunks: List[Tensor],
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """2차원 조각 목록을 하나의 행렬로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor, width]`` 입니다.
            width: 마지막 축 너비입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 행렬입니다. shape은 ``[n_total_valid_anchor, width]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, width), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
