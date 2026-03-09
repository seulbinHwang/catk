# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import math
import multiprocessing
import threading
import time
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map

tf.get_logger().setLevel("ERROR")

# agent_types = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
# agent_roles = {0: "ego_vehicle", 1: "interest", 2: "predict"}
# polyline_type = {
#     # for lane
#     "TYPE_FREEWAY": 0,
#     "TYPE_SURFACE_STREET": 1,
#     "TYPE_STOP_SIGN": 2,
#     "TYPE_BIKE_LANE": 3,
#     # for roadedge
#     "TYPE_ROAD_EDGE_BOUNDARY": 4,
#     "TYPE_ROAD_EDGE_MEDIAN": 5,
#     # for roadline
#     "BROKEN": 6,
#     "SOLID_SINGLE": 7,
#     "DOUBLE": 8,
#     # for crosswalk, speed bump and drive way
#     "TYPE_CROSSWALK": 9,
# }
_polygon_types = ["lane", "road_edge", "road_line", "crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]
_signal_state_to_polygon_light_type = {
    0: 1,  # UNKNOWN
    1: 2,  # ARROW_STOP
    2: 4,  # ARROW_CAUTION
    3: 3,  # ARROW_GO
    4: 2,  # STOP
    5: 4,  # CAUTION
    6: 3,  # GO
    7: 2,  # FLASHING_STOP
    8: 4,  # FLASHING_CAUTION
}

_PROGRESS_COUNTER = None


def _configure_runtime_threads(threads_per_worker: int) -> None:
    threads_per_worker = max(1, int(threads_per_worker))
    torch.set_num_threads(threads_per_worker)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(threads_per_worker)
        except RuntimeError:
            pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(threads_per_worker)
    except RuntimeError:
        pass


def _init_worker(progress_counter, threads_per_worker: int) -> None:
    global _PROGRESS_COUNTER
    _PROGRESS_COUNTER = progress_counter
    _configure_runtime_threads(threads_per_worker)


def _increment_progress(delta: int = 1) -> None:
    if _PROGRESS_COUNTER is None:
        return
    with _PROGRESS_COUNTER.get_lock():
        _PROGRESS_COUNTER.value += delta


def _format_hours_minutes(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m"


def _print_progress(
    split: str,
    total_expected: int,
    current_count: int,
    start_count: int,
    start_time: float,
) -> None:
    percent = 100.0 * current_count / total_expected if total_expected > 0 else 0.0
    elapsed_seconds = time.time() - start_time
    processed_this_run = max(current_count - start_count, 0)
    rate = processed_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(total_expected - current_count, 0)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    print(
        "[preprocess-progress] "
        f"split={split} total_expected={total_expected} "
        f"generated={current_count} percent={percent:.2f}% "
        f"elapsed={_format_hours_minutes(elapsed_seconds)} "
        f"eta={_format_hours_minutes(eta_seconds)}",
        flush=True,
    )


def _progress_monitor(
    split: str,
    total_expected: int,
    progress_counter,
    start_count: int,
    start_time: float,
    interval_sec: int,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(interval_sec):
        with progress_counter.get_lock():
            current_count = progress_counter.value
        _print_progress(split, total_expected, current_count, start_count, start_time)


def _iter_tfrecord_numpy(file_path: str):
    dataset = tf.data.TFRecordDataset([file_path], compression_type="")
    return dataset.as_numpy_iterator()


def _count_records_in_tfrecord(file_path: str) -> int:
    return sum(1 for _ in _iter_tfrecord_numpy(file_path))


def _count_cache_path(state_dir: Path, file_path: str) -> Path:
    return state_dir / (Path(file_path).name + ".count")


def _read_cached_int(file_path: Path) -> Optional[int]:
    if not file_path.exists():
        return None
    try:
        return int(file_path.read_text().strip())
    except (OSError, ValueError):
        return None


def _count_records_job(file_path: str):
    return file_path, _count_records_in_tfrecord(file_path)


def _count_total_expected(packages: List[str], num_workers: int, state_dir: Path) -> int:
    cached_total = 0
    missing_packages = []
    for file_path in packages:
        cached_count = _read_cached_int(_count_cache_path(state_dir, file_path))
        if cached_count is None:
            missing_packages.append(file_path)
        else:
            cached_total += cached_count

    if not missing_packages:
        return cached_total

    count_workers = max(1, min(num_workers, multiprocessing.cpu_count()))
    with multiprocessing.Pool(count_workers) as pool:
        counts = pool.imap_unordered(_count_records_job, missing_packages, chunksize=8)
        for file_path, count in tqdm(
            counts,
            total=len(missing_packages),
            desc="counting scenarios",
        ):
            cached_total += count
            _write_done_marker(_count_cache_path(state_dir, file_path), count)
    return cached_total


def _count_existing_pickles(output_dir: Path) -> int:
    return sum(1 for _ in output_dir.glob("*.pkl"))


def _done_marker_path(state_dir: Path, file_path: str) -> Path:
    return state_dir / (Path(file_path).name + ".done")


def _write_done_marker(marker_path: Path, processed_count: int) -> None:
    marker_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = marker_path.with_suffix(marker_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(str(processed_count))
    tmp_path.replace(marker_path)


def get_agent_features(
    track_infos: Dict[str, np.ndarray], split, num_historical_steps, num_steps
) -> Dict[str, Any]:
    """
    track_infos:
    object_id (100,) int64
    object_type (100,) uint8
    states (100, 91, 9) float32
    valid (100, 91) bool
    role (100, 3) bool
    """

    idx_agents_to_add = []
    for i in range(len(track_infos["object_id"])):
        add_agent = track_infos["valid"][i, num_historical_steps - 1]

        if add_agent:
            idx_agents_to_add.append(i)

    num_agents = len(idx_agents_to_add)
    out_dict = {
        "num_nodes": num_agents,
        "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
        "role": torch.zeros([num_agents, 3], dtype=torch.bool),
        "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
        "type": torch.zeros(num_agents, dtype=torch.uint8),
        "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
        "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
        "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
        "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
    }

    for i, idx in enumerate(idx_agents_to_add):

        out_dict["role"][i] = torch.from_numpy(track_infos["role"][idx])
        out_dict["id"][i] = track_infos["object_id"][idx]
        out_dict["type"][i] = track_infos["object_type"][idx]

        valid = track_infos["valid"][idx]  # [n_step]
        states = track_infos["states"][idx]

        object_shape = states[:, 3:6]  # [n_step, 3], length, width, height
        object_shape = object_shape[valid].mean(axis=0)  # [3]
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        valid_steps = np.where(valid)[0]
        position = states[:, :3]  # [n_step, dim], x, y, z
        velocity = states[:, 7:9]  # [n_step, 2], vx, vy
        heading = states[:, 6]  # [n_step], heading
        if valid.sum() > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            t_in = np.arange(t_start, t_end + 1)
            position_valid = position[valid]
            velocity_valid = velocity[valid]
            heading_valid = np.unwrap(heading[valid], axis=0)
            out_dict["valid_mask"][i, t_start : t_end + 1] = True
            out_dict["position"][i, t_start : t_end + 1, 0] = torch.from_numpy(
                np.interp(t_in, valid_steps, position_valid[:, 0])
            )
            out_dict["position"][i, t_start : t_end + 1, 1] = torch.from_numpy(
                np.interp(t_in, valid_steps, position_valid[:, 1])
            )
            out_dict["position"][i, t_start : t_end + 1, 2] = torch.from_numpy(
                np.interp(t_in, valid_steps, position_valid[:, 2])
            )
            out_dict["velocity"][i, t_start : t_end + 1, 0] = torch.from_numpy(
                np.interp(t_in, valid_steps, velocity_valid[:, 0])
            )
            out_dict["velocity"][i, t_start : t_end + 1, 1] = torch.from_numpy(
                np.interp(t_in, valid_steps, velocity_valid[:, 1])
            )
            out_dict["heading"][i, t_start : t_end + 1] = torch.from_numpy(
                np.interp(t_in, valid_steps, heading_valid)
            )
        else:
            t = valid_steps[0]
            out_dict["valid_mask"][i, t] = True
            out_dict["position"][i, t] = torch.from_numpy(position[t])
            out_dict["velocity"][i, t] = torch.from_numpy(velocity[t])
            out_dict["heading"][i, t] = torch.tensor(heading[t])

    return out_dict


def get_map_features(map_infos, current_light_by_lane_id, dim=2):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]
    num_polygons = len(polygon_ids)
    polygon_id_to_idx = {polygon_id: i for i, polygon_id in enumerate(polygon_ids)}

    # initialization
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    # point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    for _key in _polygon_types:
        for _seg in map_infos[_key]:
            _idx = polygon_id_to_idx[_seg["id"]]
            centerline = map_infos["all_polylines"][
                _seg["polyline_index"][0] : _seg["polyline_index"][1]
            ]
            centerline = torch.from_numpy(centerline).float()
            polygon_type[_idx] = _polygon_types.index(_key)

            point_position[_idx] = centerline[:-1, :dim]
            center_vectors = centerline[1:] - centerline[:-1]
            # point_orientation[_idx] = torch.cat(
            #     [torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0
            # )
            point_type[_idx] = torch.full(
                (len(center_vectors),), _seg["type"], dtype=torch.uint8
            )

            if _key == "lane":
                polygon_light_type[_idx] = current_light_by_lane_id.get(_seg["id"], 0)

    num_points = torch.tensor(
        [point.size(0) for point in point_position], dtype=torch.long
    )
    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )

    map_data = {
        "map_polygon": {},
        "map_point": {},
        ("map_point", "to", "map_polygon"): {},
    }
    map_data["map_polygon"]["num_nodes"] = num_polygons
    map_data["map_polygon"]["type"] = polygon_type
    map_data["map_polygon"]["light_type"] = polygon_light_type
    if len(num_points) == 0:
        map_data["map_point"]["num_nodes"] = 0
        map_data["map_point"]["position"] = torch.tensor([], dtype=torch.float)
        # map_data["map_point"]["orientation"] = torch.tensor([], dtype=torch.float)
        map_data["map_point"]["type"] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data["map_point"]["num_nodes"] = num_points.sum().item()
        map_data["map_point"]["position"] = torch.cat(point_position, dim=0)
        # map_data["map_point"]["orientation"] = wrap_angle(
        #     torch.cat(point_orientation, dim=0)
        # )
        map_data["map_point"]["type"] = torch.cat(point_type, dim=0)
    map_data["map_point", "to", "map_polygon"][
        "edge_index"
    ] = point_to_polygon_edge_index
    return map_data


def decode_tracks_from_proto(scenario):
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = [i.track_index for i in scenario.tracks_to_predict]
    object_id_interest = [i for i in scenario.objects_of_interest]

    track_infos = {
        "object_id": [],
        "object_type": [],
        "states": [],
        "valid": [],
        "role": [],
    }
    for i, cur_data in enumerate(scenario.tracks):  # number of objects

        step_state = []
        step_valid = []
        for s in cur_data.states:
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                ]
            )
            step_valid.append(s.valid)
            # This angle is normalized to [-pi, pi). The velocity vector in m/s

        track_infos["object_id"].append(cur_data.id)
        track_infos["object_type"].append(cur_data.object_type - 1)
        track_infos["states"].append(np.array(step_state, dtype=np.float32))
        track_infos["valid"].append(np.array(step_valid))

        track_infos["role"].append([False, False, False])
        if i in track_index_predict:
            track_infos["role"][-1][2] = True  # predict=2
        if cur_data.id in object_id_interest:
            track_infos["role"][-1][1] = True  # interest=1
        if i == sdc_track_index:  # ego_vehicle=0
            track_infos["role"][-1][0] = True

    track_infos["states"] = np.array(track_infos["states"], dtype=np.float32)
    track_infos["valid"] = np.array(track_infos["valid"], dtype=bool)
    track_infos["role"] = np.array(track_infos["role"], dtype=bool)
    track_infos["object_id"] = np.array(track_infos["object_id"], dtype=np.int64)
    track_infos["object_type"] = np.array(track_infos["object_type"], dtype=np.uint8)
    return track_infos


def decode_map_features_from_proto(map_features):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    point_cnt = 0
    lane_id_to_index = {}
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
        if feature_data_type is None:
            continue

        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                if feature.type == 0:  # UNDEFINED
                    cur_info["type"] = 1
                elif feature.type == 1:  # FREEWAY
                    cur_info["type"] = 0
                elif feature.type == 2:  # SURFACE_STREET
                    cur_info["type"] = 1
                elif feature.type == 3:  # BIKE_LANE
                    cur_info["type"] = 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["lane"].append(cur_info)
                lane_id_to_index[mf.id] = len(map_infos["lane"]) - 1
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_edge":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # assert feature.type > 0
                cur_info["type"] = feature.type + 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_edge"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_line":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # there is no UNKNOWN = 0
                # BROKEN_SINGLE_WHITE = 1
                # SOLID_SINGLE_WHITE = 2
                # SOLID_DOUBLE_WHITE = 3
                # BROKEN_SINGLE_YELLOW = 4
                # BROKEN_DOUBLE_YELLOW = 5
                # SOLID_SINGLE_YELLOW = 6
                # SOLID_DOUBLE_YELLOW = 7
                # PASSING_DOUBLE_YELLOW = 8
                # assert feature.type > 0  # no UNKNOWN = 0
                if feature.type in [1, 4, 5]:
                    cur_info["type"] = 6  # BROKEN
                elif feature.type in [2, 6]:
                    cur_info["type"] = 7  # SOLID_SINGLE
                else:
                    cur_info["type"] = 8  # DOUBLE

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_line"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type in ["speed_bump", "driveway", "crosswalk"]:
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
            pl_polygon = get_polylines_from_polygon(xyz[polygon_idx])
            cur_info = {"id": mf.id, "type": 9}

            cur_polyline = np.stack(
                [
                    np.array([p[0], p[1], p[2], cur_info["type"], cur_info["id"]])
                    for p in pl_polygon
                ],
                axis=0,
            )

            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            map_infos["crosswalk"].append(cur_info)
            polylines.append(cur_polyline)
            point_cnt += len(cur_polyline)

    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        if feature_data_type == "stop_sign":
            feature = mf.stop_sign
            for l_id in feature.lane:
                # override FREEWAY/SURFACE_STREET with stop sign lane
                # BIKE_LANE remains unchanged
                lane_idx = lane_id_to_index.get(l_id)
                if lane_idx is not None and map_infos["lane"][lane_idx]["type"] < 2:
                    map_infos["lane"][lane_idx]["type"] = 2

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos

def decode_current_dynamic_map_state(dynamic_map_states, current_time_index):
    current_light_by_lane_id = {}
    for cur_signal in dynamic_map_states[current_time_index].lane_states:
        current_light_by_lane_id[cur_signal.lane] = _signal_state_to_polygon_light_type[
            cur_signal.state
        ]
    return current_light_by_lane_id


def wm2argo(
    file_path,
    split,
    output_dir,
    output_dir_tfrecords_splitted,
    state_dir,
    force_reprocess,
):
    marker_path = _done_marker_path(state_dir, file_path)
    if marker_path.exists() and not force_reprocess:
        return 0

    processed_count = 0
    for tf_data in _iter_tfrecord_numpy(file_path):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        track_infos = decode_tracks_from_proto(scenario)
        map_infos = decode_map_features_from_proto(scenario.map_features)
        current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id
        current_light_by_lane_id = decode_current_dynamic_map_state(
            scenario.dynamic_map_states, current_time_index
        )
        map_data = get_map_features(map_infos, current_light_by_lane_id)

        data = preprocess_map(map_data)
        data["agent"] = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
        )

        data["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if output_dir_tfrecords_splitted is not None:
            file_name = output_dir_tfrecords_splitted / f"{scenario_id}.tfrecords"
            with tf.io.TFRecordWriter(file_name.as_posix()) as file_writer:
                file_writer.write(tf_data)

        processed_count += 1
        _increment_progress(1)

    _write_done_marker(marker_path, processed_count)
    return processed_count


def batch_process9s_transformer(
    input_dir,
    output_dir,
    split,
    num_workers,
    threads_per_worker=1,
    progress_interval_sec=60,
    force_reprocess=False,
):
    _configure_runtime_threads(threads_per_worker)
    output_dir = Path(output_dir)
    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = output_dir / "validation_tfrecords_splitted"
        output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)
    state_dir = output_dir.parent / ".preprocess_state" / split
    state_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    total_expected = _count_total_expected(packages, num_workers, state_dir)
    existing_count = _count_existing_pickles(output_dir)
    progress_counter = multiprocessing.Value("Q", existing_count)
    start_time = time.time()

    print(
        f"[preprocess-start] split={split} total_expected={total_expected} "
        f"existing={existing_count} progress_interval_sec={progress_interval_sec}",
        flush=True,
    )
    _print_progress(split, total_expected, existing_count, existing_count, start_time)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(
            split,
            total_expected,
            progress_counter,
            existing_count,
            start_time,
            progress_interval_sec,
            stop_event,
        ),
        daemon=True,
    )
    monitor_thread.start()

    func = partial(
        wm2argo,
        split=split,
        output_dir=output_dir,
        output_dir_tfrecords_splitted=output_dir_tfrecords_splitted,
        state_dir=state_dir,
        force_reprocess=force_reprocess,
    )
    try:
        with multiprocessing.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(progress_counter, threads_per_worker),
        ) as p:
            list(
                tqdm(
                    p.imap_unordered(func, packages, chunksize=8),
                    total=len(packages),
                    desc=f"packages:{split}",
                )
            )
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)

    with progress_counter.get_lock():
        final_count = progress_counter.value
    _print_progress(split, total_expected, final_count, existing_count, start_time)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/workspace/data/womd/uncompressed/scenario",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/root/workspace/data/SMART_new"
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Maximum CPU threads each preprocessing worker may use internally.",
    )
    parser.add_argument("--progress_interval_sec", type=int, default=60)
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Ignore completed tfrecord markers and rebuild all scenarios.",
    )
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir,
        args.output_dir,
        args.split,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        progress_interval_sec=args.progress_interval_sec,
        force_reprocess=args.force_reprocess,
    )
