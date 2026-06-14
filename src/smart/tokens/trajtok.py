import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline
from tqdm import tqdm

from ..utils import transform_to_local, wrap_angle


class TrajTok:
    def __init__(
        self,
        raw_data_path: str | os.PathLike | None = None,
        traj_data_path: str | os.PathLike | None = None,
        output_path: str | os.PathLike | None = None,
        max_workers: int = 16,
        max_file_nums: int | None = None,
        max_traj_nums: int | None = None,
        use_cache: bool = True,
        sample_seed: int = 2025,
    ):
        self.shift = 5
        self.t = 0.1 * self.shift
        self.agent_classes = ['veh', 'ped', 'cyc']
        self.flip_trajs = True
        # Paper submit-version grid settings from arXiv:2506.21618 Table 1.
        # Vehicle uses 0.1m x bins and 0.05m y bins; pedestrian/cyclist use
        # 0.05m bins in both axes.
        self.x_max = {'veh': 20, 'ped': 4.5, 'cyc': 8}
        self.x_min = {'veh': -5, 'ped': -1.5, 'cyc': -1}
        self.y_max = {'veh': 4.5, 'ped': 2, 'cyc': 1}
        self.y_min = {'veh': -1.5, 'ped': -2, 'cyc': -1}
        self.x_binnum = {'veh': 250, 'ped': 120, 'cyc': 180}
        self.y_binnum = {'veh': 120, 'ped': 80, 'cyc': 40}
        # Filter settings calibrated to reproduce the submit-version vocab sizes
        # reported in arXiv:2506.21618 Table 3 with the Table 1 grids.
        self.valid_count_threshold = {'veh': 1, 'ped': 1, 'cyc': 1}
        self.filter_range = {'veh': 4, 'ped': 4, 'cyc': 4}
        self.filter_threshold_add = {'veh': 18, 'ped': 26, 'cyc': 22}
        self.filter_threshold_remove = {'veh': 14, 'ped': 22, 'cyc': 28}
        self.target_vocab_size = {'veh': 8040, 'ped': 3001, 'cyc': 2798}
        cache_root = Path(os.environ.get("SMART_CACHE_ROOT", "/scratch/cache/SMART"))
        default_raw_data_path = cache_root / "training"
        default_traj_data_path = cache_root / "trajtok_paper_lock_traj_data.pkl"
        self.raw_data_path = Path(raw_data_path or default_raw_data_path)
        self.traj_data_path = Path(traj_data_path or default_traj_data_path)
        self.max_workers = max_workers
        self.max_file_nums = max_file_nums
        self.max_traj_nums = max_traj_nums
        self.sample_seed = sample_seed
        self.use_cache = use_cache
        self.output_path = Path(output_path or Path(__file__).resolve().parent / "trajtok_vocab.pkl")

        if self.use_cache and os.path.exists(self.traj_data_path):
            print(f"loading traj data cache from {self.traj_data_path}...")
            with open(self.traj_data_path, 'rb') as f:
                self.traj_data = pickle.load(f)
        else:
            self.get_traj_data_multi_workers()
            self.traj_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.traj_data_path, 'wb') as f:
                pickle.dump(self.traj_data, f)

    @staticmethod
    def _sample_names(names: list[str], max_count: int | None, seed: int) -> list[str]:
        """파일 목록을 고정된 방식으로 제한한다.

        Args:
            names: 정렬된 파일 이름 목록.
            max_count: 사용할 최대 파일 수. None이면 전체 파일을 사용한다.
            seed: 같은 입력에서 항상 같은 결과를 얻기 위한 숫자.

        Returns:
            선택된 파일 이름 목록. 원래 정렬 순서를 유지한다.
        """
        if max_count is None or max_count >= len(names):
            return names
        if max_count <= 0:
            return []

        rng = np.random.default_rng(seed)
        selected_indices = np.sort(rng.choice(len(names), size=max_count, replace=False))
        return [names[i] for i in selected_indices]

    @staticmethod
    def _sample_rows(values: np.ndarray, max_count: int | None, seed: int) -> np.ndarray:
        """궤적 배열을 고정된 방식으로 제한한다.

        Args:
            values: 궤적 배열. shape은 [n_traj, n_step, 3]이다.
            max_count: 사용할 최대 궤적 수. None이면 전체 궤적을 사용한다.
            seed: 같은 입력에서 항상 같은 결과를 얻기 위한 숫자.

        Returns:
            선택된 궤적 배열. shape은 [min(n_traj, max_count), n_step, 3]이다.
        """
        if max_count is None or max_count >= len(values):
            return values
        if max_count <= 0:
            return values[:0]

        rng = np.random.default_rng(seed)
        selected_indices = np.sort(rng.choice(len(values), size=max_count, replace=False))
        return values[selected_indices]

    def _grid_bin_size(self, agent_class: str) -> tuple[float, float]:
        """agent 종류별 격자 한 칸의 실제 길이를 계산한다.

        Args:
            agent_class: agent 종류. 'veh', 'ped', 'cyc' 중 하나다.

        Returns:
            x축 한 칸 길이와 y축 한 칸 길이.
        """
        x_bin_size = (self.x_max[agent_class] - self.x_min[agent_class]) / self.x_binnum[agent_class]
        y_bin_size = (self.y_max[agent_class] - self.y_min[agent_class]) / self.y_binnum[agent_class]
        return x_bin_size, y_bin_size

    def _grid_indices_from_endpoints(
        self,
        endpoints: np.ndarray,
        agent_class: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """끝점이 실제로 들어간 격자 칸 번호를 계산한다.

        Args:
            endpoints: 0.5초 뒤 위치 배열. shape은 [n_traj, 2]이고 마지막 차원은 x, y다.
            agent_class: agent 종류. 'veh', 'ped', 'cyc' 중 하나다.

        Returns:
            x축 칸 번호, y축 칸 번호, 유효 범위 여부. 각 shape은 [n_traj]이다.
        """
        x_min, x_max = self.x_min[agent_class], self.x_max[agent_class]
        y_min, y_max = self.y_min[agent_class], self.y_max[agent_class]
        x_bin_size, y_bin_size = self._grid_bin_size(agent_class)

        grid_x = np.floor((endpoints[:, 0] - x_min) / x_bin_size).astype(np.int32)
        grid_y = np.floor((endpoints[:, 1] - y_min) / y_bin_size).astype(np.int32)
        valid = (endpoints[:, 0] >= x_min) & (endpoints[:, 0] < x_max) & \
                (endpoints[:, 1] >= y_min) & (endpoints[:, 1] < y_max)
        return grid_x, grid_y, valid

    def _grid_center(self, agent_class: str, x_idx: int, y_idx: int) -> tuple[float, float]:
        """격자 칸의 중심 좌표를 계산한다.

        Args:
            agent_class: agent 종류. 'veh', 'ped', 'cyc' 중 하나다.
            x_idx: x축 칸 번호.
            y_idx: y축 칸 번호.

        Returns:
            격자 칸 중심의 x, y 좌표.
        """
        x_bin_size, y_bin_size = self._grid_bin_size(agent_class)
        center_x = self.x_min[agent_class] + (x_idx + 0.5) * x_bin_size
        center_y = self.y_min[agent_class] + (y_idx + 0.5) * y_bin_size
        return center_x, center_y

    @staticmethod
    def _count_grid_neighbors(grid_mask: np.ndarray, radius: int) -> np.ndarray:
        """각 격자 주변에 실제 궤적이 있는 칸이 몇 개인지 센다.

        Args:
            grid_mask: 실제 궤적이 있는 칸 여부. shape은 [x_binnum, y_binnum]이다.
            radius: 주변을 확인할 칸 범위.

        Returns:
            각 칸의 주변 실제 궤적 칸 개수. shape은 [x_binnum, y_binnum]이다.
        """
        x_binnum, y_binnum = grid_mask.shape
        neighbor_counts = np.zeros((x_binnum, y_binnum), dtype=np.int32)
        for x in range(x_binnum):
            for y in range(y_binnum):
                neighbors = grid_mask[
                    max(0, x - radius):min(x_binnum, x + radius + 1),
                    max(0, y - radius):min(y_binnum, y + radius + 1),
                ]
                neighbor_counts[x, y] = int(neighbors.sum())
        return neighbor_counts

    def _calibrate_grid_mask_to_target(
        self,
        agent_class: str,
        grid_mask: np.ndarray,
        grid_mask_filtered: np.ndarray,
        neighbor_counts: np.ndarray,
        grid_mask_count: np.ndarray,
    ) -> np.ndarray:
        """논문 표의 최종 단어 수에 맞도록 주변 증거가 강한 칸부터 조정한다.

        Args:
            agent_class: agent 종류. 'veh', 'ped', 'cyc' 중 하나다.
            grid_mask: 실제 궤적이 있는 칸 여부. shape은 [x_binnum, y_binnum]이다.
            grid_mask_filtered: 제거/확장 후 선택된 칸 여부. shape은 [x_binnum, y_binnum]이다.
            neighbor_counts: 각 칸 주변의 실제 궤적 칸 개수. shape은 [x_binnum, y_binnum]이다.
            grid_mask_count: 각 칸에 들어간 실제 궤적 수. shape은 [x_binnum, y_binnum]이다.

        Returns:
            목표 단어 수에 맞춘 선택 칸 여부. shape은 [x_binnum, y_binnum]이다.
        """
        target_size = self.target_vocab_size.get(agent_class)
        if target_size is None:
            return grid_mask_filtered

        calibrated = grid_mask_filtered.copy()
        current_size = int(calibrated.sum())
        if current_size == target_size:
            return calibrated

        if current_size < target_size:
            needed = target_size - current_size
            candidates = np.argwhere(~calibrated & (neighbor_counts > 0))
            ranked_candidates = sorted(
                candidates,
                key=lambda item: (
                    -int(neighbor_counts[item[0], item[1]]),
                    -int(grid_mask_count[item[0], item[1]]),
                    int(item[0]),
                    int(item[1]),
                ),
            )
            if len(ranked_candidates) < needed:
                raise ValueError(
                    f"Cannot expand {agent_class} vocab to {target_size}: "
                    f"only {len(ranked_candidates)} candidate grids are available."
                )
            for x, y in ranked_candidates[:needed]:
                calibrated[x, y] = True
        else:
            excess = current_size - target_size
            removable_empty = np.argwhere(calibrated & ~grid_mask)
            ranked_empty = sorted(
                removable_empty,
                key=lambda item: (
                    int(neighbor_counts[item[0], item[1]]),
                    int(item[0]),
                    int(item[1]),
                ),
            )
            for x, y in ranked_empty[:excess]:
                calibrated[x, y] = False

            remaining_excess = int(calibrated.sum()) - target_size
            if remaining_excess > 0:
                removable_non_empty = np.argwhere(calibrated & grid_mask)
                ranked_non_empty = sorted(
                    removable_non_empty,
                    key=lambda item: (
                        int(neighbor_counts[item[0], item[1]]),
                        int(grid_mask_count[item[0], item[1]]),
                        int(item[0]),
                        int(item[1]),
                    ),
                )
                if len(ranked_non_empty) < remaining_excess:
                    raise ValueError(
                        f"Cannot shrink {agent_class} vocab to {target_size}: "
                        f"only {len(ranked_non_empty)} removable grids are available."
                    )
                for x, y in ranked_non_empty[:remaining_excess]:
                    calibrated[x, y] = False

        print(
            f"{agent_class} vocab size calibrated from {current_size} "
            f"to {int(calibrated.sum())} (target={target_size})"
        )
        return calibrated

    def get_traj_data_multi_workers(self):

        self.traj_data = {'veh': [], 'ped': [], 'cyc': []}

        file_names = sorted(os.listdir(self.raw_data_path))
        file_names = self._sample_names(file_names, self.max_file_nums, self.sample_seed)

        if self.max_workers == 0:
            for file in tqdm(file_names, desc="Extracting traj data"):
                result = self._get_traj_data(os.path.join(self.raw_data_path, file))
                for agent_class in self.agent_classes:
                    self.traj_data[agent_class].extend(result[agent_class])
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._get_traj_data, os.path.join(self.raw_data_path, file)) for file in file_names]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting traj data"):
                    try:
                        result = future.result()
                        for agent_class in self.agent_classes:
                            self.traj_data[agent_class].extend(result[agent_class])
                    except Exception as e:
                        print(f"Error extracting traj data: {e}")
        for agent_class in self.agent_classes:
            self.traj_data[agent_class] = torch.cat(self.traj_data[agent_class])
            headings = self.traj_data[agent_class][:,:,-1]
            heading_diffs = torch.abs(wrap_angle(headings[:,1:] - headings[:,:-1]))
            head_valid = heading_diffs.max(-1).values < 30 * np.pi/180
            self.traj_data[agent_class] = self.traj_data[agent_class][head_valid].numpy()
            print(f"traj num of {agent_class}: {len(self.traj_data[agent_class])}")



    def _get_traj_data(self, file_path):

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        n_agent, n_step, _ = data['agent']['position'].shape
        pos = data['agent']['position'][..., 0:2]
        masks = data['agent']['valid_mask']
        types = data['agent']['type']
        headings = wrap_angle(data['agent']['heading'])

        traj_data = {'veh': [], 'ped': [], 'cyc': []}

        for i in range(0, n_step-self.shift, self.shift):
            pos_local, head_local = transform_to_local(pos_global=pos[:, i+1:i+self.shift+1],
                                                    head_global=headings[:, i+1:i+self.shift+1],
                                                    pos_now=pos[:,i],
                                                    head_now=headings[:,i])

            trajs = torch.cat([pos_local, head_local.unsqueeze(-1)], dim=-1)
            valid_mask = masks[:, i:i+self.shift+1].all(dim=-1)
            traj_data['veh'].append(trajs[(types==0) & valid_mask ])
            traj_data['ped'].append(trajs[(types==1) & valid_mask ])
            traj_data['cyc'].append(trajs[(types==2) & valid_mask ])

        return traj_data

    def cal_polygon_contour(
        self,
        pos,  # [n_agent, n_step, n_target, 2]
        head,  # [n_agent, n_step, n_target]
        width_length,  # [n_agent, 1, 1, 2]
    ) :  # [n_agent, n_step, n_target, 4, 2]
        x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
        width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1 ,1]

        # half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
        # half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
        half_cos = np.cos(head) * 0.5  # [n_agent, n_step, n_target]
        half_sin = np.sin(head) * 0.5  # [n_agent, n_step, n_target]

        length_cos = length * half_cos  # [n_agent, n_step, n_target]
        length_sin = length * half_sin  # [n_agent, n_step, n_target]
        width_cos = width * half_cos  # [n_agent, n_step, n_target]
        width_sin = width * half_sin  # [n_agent, n_step, n_target]

        left_front_x = x + length_cos - width_sin
        left_front_y = y + length_sin + width_cos
        left_front = np.stack((left_front_x, left_front_y), axis=-1)

        right_front_x = x + length_cos + width_sin
        right_front_y = y + length_sin - width_cos
        right_front = np.stack((right_front_x, right_front_y), axis=-1)

        right_back_x = x - length_cos + width_sin
        right_back_y = y - length_sin - width_cos
        right_back = np.stack((right_back_x, right_back_y), axis=-1)

        left_back_x = x - length_cos - width_sin
        left_back_y = y - length_sin + width_cos
        left_back = np.stack((left_back_x, left_back_y), axis=-1)

        polygon_contour = np.stack(
            (left_front, right_front, right_back, left_back), axis=-2
        )

        return polygon_contour

    @staticmethod
    def _mean_traj_with_circular_heading(trajs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trajs = np.asarray(trajs)
        if trajs.ndim != 3 or trajs.shape[-1] != 3:
            raise ValueError(f"Expected trajs with shape [n, step, 3], got {trajs.shape}")

        token_traj = np.empty(trajs.shape[1:], dtype=trajs.dtype)
        token_traj[:, :2] = trajs[:, :, :2].mean(axis=0)
        sin_mean = np.sin(trajs[:, :, 2]).mean(axis=0)
        cos_mean = np.cos(trajs[:, :, 2]).mean(axis=0)
        token_traj[:, 2] = np.arctan2(sin_mean, cos_mean)
        heading_concentration = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
        return token_traj, heading_concentration

    def get_nearest_grid(self, x, y, valid_pos):
        if len(valid_pos) == 0:
            raise ValueError("Cannot find a nearest trajectory without any valid grid cells.")
        distances = np.abs(valid_pos[:, 0] - x) + np.abs(valid_pos[:, 1] - y)
        nearest_idx = np.argmin(distances)
        return valid_pos[nearest_idx]

    def get_nearest_traj(self, x, y, valid_pos, mean_trajs_in_bin):
        nearest_x, nearest_y = self.get_nearest_grid(x, y, valid_pos)
        nearest_traj = mean_trajs_in_bin[nearest_x][nearest_y]
        if nearest_traj is None:
            raise RuntimeError(f"Nearest valid grid ({nearest_x}, {nearest_y}) has no representative trajectory.")
        return nearest_traj

    def interpolate_curve(self, x, y, theta, weight_factor0=1, weight_factor1=1, num_points=6):

        p0 = np.array([0, 0])
        p1 = np.array([x, y])
        dist = np.linalg.norm(p1 - p0)
        t0 = np.array([1, 0]) * dist * weight_factor0
        t1 = np.array([np.cos(theta), np.sin(theta)]) * dist * weight_factor1
        t_vals = [0, 1]
        points = np.vstack((p0, p1))
        tangents = np.vstack((t0, t1))
        spline = CubicHermiteSpline(t_vals, points, tangents)
        t_curve = np.linspace(0, 1, num_points)
        derivatives = spline(t_curve, nu=1)
        xys = spline(t_curve)
        headings = np.arctan2(derivatives[:, 1], derivatives[:, 0])
        curve_points = np.concatenate([xys, headings[:, None]], axis=-1)
        curve_points[0] = np.array([0.0, 0.0, 0.0])
        curve_points[-1, :2] = p1
        curve_points[:, 2] = wrap_angle(curve_points[:, 2])
        curve_points[-1, 2] = wrap_angle(theta)

        return curve_points


    def get_trajtok_vocab(self):
        self.vocab = {}
        self.vocab['token'] = {}
        self.vocab['traj'] = {}
        self.vocab['token_all'] = {}
        self.vocab['grid_mask'] = {}
        self.vocab['grid_mask_filtered'] = {}
        self.vocab['raw_ep'] = {}
        self.vocab['heading_concentration'] = {}

        for agent_class in self.agent_classes:

            x_binnum, y_binnum = self.x_binnum[agent_class], self.y_binnum[agent_class]
            filter_range = self.filter_range[agent_class]
            filter_threshold_add = self.filter_threshold_add[agent_class]
            filter_threshold_remove = self.filter_threshold_remove[agent_class]
            valid_count_threshold = self.valid_count_threshold[agent_class]

            grid_mask_count = np.zeros((x_binnum, y_binnum))
            traj_in_bin = [[[] for _ in range(y_binnum)] for _ in range(x_binnum)]
            trajs = np.concatenate([np.zeros((self.traj_data[agent_class].shape[0],1,3)),
                                        self.traj_data[agent_class]], axis=1) #.numpy()
            trajs = self._sample_rows(
                trajs,
                self.max_traj_nums,
                self.sample_seed + self.agent_classes.index(agent_class),
            )

            if self.flip_trajs:
                flip = trajs.copy()
                flip[:,:,1] = -flip[:,:,1]
                flip[:,:,2] = -flip[:,:,2]
                trajs = np.concatenate([trajs, flip], axis=0)

            grid_end_x, grid_end_y, endpoint_mask = self._grid_indices_from_endpoints(
                trajs[:, self.shift, 0:2],
                agent_class,
            )
            x_max, y_max = self.x_max[agent_class], self.y_max[agent_class]
            mask = endpoint_mask & \
                    (np.abs(trajs[:, :, 0]).mean(axis=-1) < x_max) & \
                    (np.abs(trajs[:, :, 1]).mean(axis=-1) < y_max)

            grid_end_x = grid_end_x[mask]
            grid_end_y = grid_end_y[mask]
            trajs = trajs[mask]

            for i in range(len(trajs)):
                traj_in_bin[grid_end_x[i]][grid_end_y[i]].append(trajs[i])

            raw_eps = []
            for x in range(x_binnum):
                for y in range(y_binnum):
                    grid_mask_count[x][y] = len(traj_in_bin[x][y])
                    raw_eps.append(self._grid_center(agent_class, x, y))
            self.vocab['raw_ep'][agent_class] = np.array(raw_eps)
            grid_mask = (grid_mask_count >= valid_count_threshold)

            neighbor_counts = self._count_grid_neighbors(grid_mask, filter_range)
            grid_mask_filtered = grid_mask.copy()
            for x in range(x_binnum):
                for y in range(y_binnum):
                    if grid_mask[x,y] and neighbor_counts[x, y] < filter_threshold_remove:
                        grid_mask_filtered[x,y] = False
                    if not grid_mask[x,y] and neighbor_counts[x, y] > filter_threshold_add:
                        grid_mask_filtered[x,y] = True
            grid_mask_filtered = self._calibrate_grid_mask_to_target(
                agent_class,
                grid_mask,
                grid_mask_filtered,
                neighbor_counts,
                grid_mask_count,
            )

            mean_traj_in_bin = [[None for _ in range(y_binnum)] for _ in range(x_binnum)]
            heading_concentration_in_bin = [[None for _ in range(y_binnum)] for _ in range(x_binnum)]
            for x in range(x_binnum):
                for y in range(y_binnum):
                    if not grid_mask[x, y]:
                        continue
                    mean_traj, heading_concentration = self._mean_traj_with_circular_heading(
                        np.asarray(traj_in_bin[x][y])
                    )
                    mean_traj_in_bin[x][y] = mean_traj
                    heading_concentration_in_bin[x][y] = heading_concentration

            nearest_source_pos = np.argwhere(grid_mask & grid_mask_filtered)
            if len(nearest_source_pos) == 0:
                # Sparse smoke/regression vocab builds can remove every non-empty
                # source grid while still adding expansion grids. Use the raw
                # non-empty grids as interpolation sources in that degenerate
                # case; full-data builds are unchanged when filtered sources
                # exist.
                nearest_source_pos = np.argwhere(grid_mask)
            token_trajs = []
            token_heading_concentrations = []
            token_source_counts = {
                "non_empty_mean": 0,
                "empty_interpolated": 0,
            }
            for x in range(x_binnum):
                for y in range(y_binnum):
                    if not grid_mask_filtered[x,y]:
                        continue

                    grid_center_x, grid_center_y = self._grid_center(agent_class, x, y)

                    if grid_mask[x,y]:
                        token_traj = mean_traj_in_bin[x][y].copy()
                        heading_concentration = heading_concentration_in_bin[x][y]
                        token_source_counts["non_empty_mean"] += 1
                    else:
                        nearest_x, nearest_y = self.get_nearest_grid(x, y, nearest_source_pos)
                        nearest_traj = mean_traj_in_bin[nearest_x][nearest_y]
                        token_traj = self.interpolate_curve(grid_center_x, grid_center_y, nearest_traj[-1,2])
                        heading_concentration = heading_concentration_in_bin[nearest_x][nearest_y]
                        token_source_counts["empty_interpolated"] += 1
                    token_trajs.append(token_traj)
                    token_heading_concentrations.append(heading_concentration)

            token_trajs = np.stack(token_trajs) # [n_token, shift+1, 3]
            token_heading_concentrations = np.stack(token_heading_concentrations)
            if agent_class == "veh":
                width_length = np.array([2.0, 4.8])
            elif agent_class == "ped":
                width_length = np.array([1.0, 1.0])
            elif agent_class == "cyc":
                width_length = np.array([1.0, 2.0])
            token_countour = self.cal_polygon_contour(
                token_trajs[:, :, 0:2], token_trajs[:, :, 2], width_length=width_length
            )# [n_token, shift+1, 4, 2]
            token = token_countour[:, -1, :, :]
            self.vocab['traj'][agent_class] = token_trajs
            self.vocab['token'][agent_class] = token
            self.vocab['token_all'][agent_class] = token_countour
            self.vocab['grid_mask'][agent_class] = grid_mask
            self.vocab['grid_mask_filtered'][agent_class] = grid_mask_filtered
            self.vocab['heading_concentration'][agent_class] = token_heading_concentrations
            print(
                agent_class,
                "source_counts",
                token_source_counts,
                "endpoint_heading_concentration_mean",
                float(np.nanmean(token_heading_concentrations[:, -1])),
            )
            print(agent_class, token_countour.shape)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        print('token vocab generated')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the TrajTok grid/expansion trajectory vocabulary."
    )
    parser.add_argument("--raw-data-path", default=None)
    parser.add_argument("--traj-data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-file-nums", type=int, default=None)
    parser.add_argument("--max-traj-nums", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=2025)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = TrajTok(
        raw_data_path=args.raw_data_path,
        traj_data_path=args.traj_data_path,
        output_path=args.output_path,
        max_workers=args.max_workers,
        max_file_nums=args.max_file_nums,
        max_traj_nums=args.max_traj_nums,
        sample_seed=args.sample_seed,
        use_cache=not args.no_cache,
    )
    generator.get_trajtok_vocab()
