"""Generated scenario의 행동(intent) 수준 다양성을 측정하는 metric.

단순 trajectory variance는 "intent 분기로 생긴 다양성"과 "같은 intent 안의
위치 noise"를 구분하지 못합니다. 이 모듈은 각 closed-loop rollout을 4초 윈도우
(2초 stride: 0~4s / 2~6s / 4~8s) 단위의 행동 토큰으로 이산화한 뒤 두 종류의
지표를 계산합니다.

  (A) categorical diversity — 윈도우별 토큰 분포의 정규화 entropy / coverage /
      mode-fraction. 토큰 분포 자체가 얼마나 퍼져 있는지.
  (B) variance 분해 (eta-squared) — 원본 trajectory variance를 토큰-간(between)과
      토큰-내(within)으로 분해. ``eta2 = V_between / V_total`` 는 전체 다양성 중
      intent 분기가 설명하는 비율이며, 사용자가 묻는 "intent 다양성인가 아닌가"에
      직접 답합니다.

행동 토큰은 3x3 격자입니다.

  종방향 (longitudinal): 윈도우 평균 속도 기반.
    - mean_speed < stop_speed       -> 1 (정지)
    - 그 외 mean(v . heading) > 0   -> 2 (전진)
    - 그 외                         -> 0 (후진)
  횡방향 (lateral): 윈도우-시작 agent frame에서의 횡변위 ``dy'`` 기반.
    - dy' >  lat_threshold -> 2 (좌)
    - dy' < -lat_threshold -> 0 (우)
    - 그 외                -> 1 (직진)
  token = lon * 3 + lat   (0..8)

variance 분해는 rigid 변환에 불변이므로 world frame에서 그대로 계산합니다.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


_N_LON = 3
_N_LAT = 3
_N_TOKEN = _N_LON * _N_LAT
_DEFAULT_EPS = 1.0e-6

# token = lon * 3 + lat. lon: 0=back, 1=stop, 2=fwd. lat: 0=right, 1=straight, 2=left.
_LON_NAMES = ("back", "stop", "fwd")
_LAT_NAMES = ("right", "straight", "left")
_CELL_NAMES = tuple(f"{lon}_{lat}" for lon in _LON_NAMES for lat in _LAT_NAMES)


class ScenarioDiversityMetrics(Metric):
    """closed-loop rollout의 행동(intent) 다양성을 누적 계산합니다.

    CPD(:class:`WOSACDistributionMetrics`)와 같은 방식으로 validation epoch 동안
    배치별 ``update`` 를 호출하고 epoch 종료 시 ``compute`` 로 집계합니다.
    """

    full_state_update = False

    def __init__(
        self,
        prefix: str,
        lat_threshold_m: float = 1.75,
        stop_speed_mps: float = 0.5,
        window_seconds: float = 4.0,
        stride_seconds: float = 2.0,
        dt_seconds: float = 0.1,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        """metric 누적 상태를 만듭니다.

        Args:
            prefix: W&B / Lightning log key의 앞부분 이름입니다.
            lat_threshold_m: 좌/우 판정 횡변위 임계값(m)입니다. 기본 1.75는
                차선 반폭에 해당합니다.
            stop_speed_mps: 정지 판정 평균 속도 임계값(m/s)입니다.
            window_seconds: 한 윈도우의 길이(초)입니다.
            stride_seconds: 윈도우 사이 간격(초)입니다.
            dt_seconds: rollout 한 스텝의 시간 간격(초)입니다. 10Hz면 0.1입니다.
            eps: 0으로 나누는 일을 막는 작은 값입니다.
        """
        super().__init__(sync_on_compute=True)
        if lat_threshold_m <= 0.0:
            raise ValueError(f"lat_threshold_m must be positive, got {lat_threshold_m}.")
        if stop_speed_mps < 0.0:
            raise ValueError(f"stop_speed_mps must be non-negative, got {stop_speed_mps}.")
        if dt_seconds <= 0.0:
            raise ValueError(f"dt_seconds must be positive, got {dt_seconds}.")

        self.prefix = str(prefix).rstrip("/")
        self.lat_threshold = float(lat_threshold_m)
        self.stop_speed = float(stop_speed_mps)
        self.dt = float(dt_seconds)
        self.eps = float(eps)

        # 윈도우 경계(스텝 인덱스)를 미리 계산합니다. 입력은 "현재 시점 + 미래"가
        # prepend된 배열이라 스텝 0이 현재(t=0)입니다.
        window_len = int(round(window_seconds / dt_seconds))
        stride_len = int(round(stride_seconds / dt_seconds))
        future_len = 80  # WOSAC: 8초 @ 10Hz
        total_len = future_len + 1
        starts = list(range(0, total_len - window_len, stride_len))
        if not starts:
            starts = [0]
        self._windows = [(s, s + window_len) for s in starts]
        self.num_windows = len(self._windows)

        w = self.num_windows
        self.add_state("token_hist", default=torch.zeros(w, _N_TOKEN, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("eta_btw_sum", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("eta_wth_sum", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("entropy_sum", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("coverage_sum", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("mode_sum", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("aw_count", default=torch.zeros(w, dtype=torch.float64), dist_reduce_fx="sum")

    def update(
        self,
        pred_traj: Tensor,
        pred_head: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        agent_valid: Optional[Tensor] = None,
    ) -> None:
        """한 validation 배치의 closed-loop rollout을 누적합니다.

        Args:
            pred_traj: closed-loop 미래 위치입니다. shape은 ``[n_agent, n_rollout,
                n_step, 2]`` 이며 world XY (10Hz) 입니다.
            pred_head: closed-loop 미래 heading(rad)입니다.
                shape은 ``[n_agent, n_rollout, n_step]`` 입니다.
            current_pos: simulation 시작 시점 위치입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: simulation 시작 시점 heading(rad)입니다.
                shape은 ``[n_agent]`` 입니다.
            agent_valid: 시작 시점에 유효한 agent인지 나타냅니다. 값이 있으면
                shape은 ``[n_agent]`` 입니다.
        """
        if pred_traj.ndim != 4 or int(pred_traj.shape[-1]) != 2:
            raise ValueError(
                f"pred_traj must have shape [n_agent, n_rollout, n_step, 2], got {tuple(pred_traj.shape)}."
            )
        if pred_head.ndim != 3:
            raise ValueError(
                f"pred_head must have shape [n_agent, n_rollout, n_step], got {tuple(pred_head.shape)}."
            )

        device = self.token_hist.device
        pred_traj = pred_traj.detach().to(device=device, dtype=torch.float32)
        pred_head = pred_head.detach().to(device=device, dtype=torch.float32)
        current_pos = current_pos.detach().to(device=device, dtype=torch.float32)
        current_head = current_head.detach().to(device=device, dtype=torch.float32)

        n_agent, n_rollout = int(pred_traj.shape[0]), int(pred_traj.shape[1])
        # rollout이 1개뿐이면 다양성을 정의할 수 없습니다.
        if n_agent == 0 or n_rollout < 2:
            return

        if agent_valid is None:
            agent_valid = torch.ones(n_agent, dtype=torch.bool, device=device)
        else:
            agent_valid = agent_valid.detach().to(device=device, dtype=torch.bool)

        pred_traj = pred_traj[agent_valid]
        pred_head = pred_head[agent_valid]
        current_pos = current_pos[agent_valid]
        current_head = current_head[agent_valid]
        n_valid = int(pred_traj.shape[0])
        if n_valid == 0:
            return

        # 현재 시점을 스텝 0으로 prepend → [n_valid, G, n_step+1, *].
        cur_pos_b = current_pos[:, None, None, :].expand(n_valid, n_rollout, 1, 2)
        cur_head_b = current_head[:, None, None].expand(n_valid, n_rollout, 1)
        full_pos = torch.cat([cur_pos_b, pred_traj], dim=2)
        full_head = torch.cat([cur_head_b, pred_head], dim=2)

        n_total = int(full_pos.shape[2])
        if n_total < self._windows[-1][1] + 1:
            # 윈도우를 채울 만큼 스텝이 없으면 이 배치는 건너뜁니다.
            return

        # float32 정밀도 확보를 위해 agent별 현재 위치로 recenter (rigid 변환 → 불변).
        full_pos = full_pos - current_pos[:, None, None, :]

        log_max = math.log(min(_N_TOKEN, n_rollout))

        for w_idx, (s, e) in enumerate(self._windows):
            token = self._classify_window(full_pos, full_head, s, e)  # [n_valid, G]
            onehot = F.one_hot(token, _N_TOKEN).to(torch.float32)     # [n_valid, G, 9]
            count_k = onehot.sum(dim=1)                               # [n_valid, 9]

            v_btw, v_wth = self._variance_decomposition(full_pos[:, :, s : e + 1, :], onehot, count_k)
            ent_norm, coverage, mode = self._categorical_stats(count_k, n_rollout, log_max)

            self.eta_btw_sum[w_idx] += v_btw.sum(dtype=torch.float64)
            self.eta_wth_sum[w_idx] += v_wth.sum(dtype=torch.float64)
            self.entropy_sum[w_idx] += ent_norm.sum(dtype=torch.float64)
            self.coverage_sum[w_idx] += coverage.sum(dtype=torch.float64)
            self.mode_sum[w_idx] += mode.sum(dtype=torch.float64)
            self.aw_count[w_idx] += float(n_valid)
            self.token_hist[w_idx] += count_k.sum(dim=0).to(torch.float64)

    def _classify_window(self, full_pos: Tensor, full_head: Tensor, s: int, e: int) -> Tensor:
        """한 윈도우의 모든 (agent, rollout)에 행동 토큰을 매깁니다.

        Args:
            full_pos: 현재 위치로 recenter된 위치입니다.
                shape은 ``[n_valid, n_rollout, n_total, 2]`` 입니다.
            full_head: heading(rad)입니다. shape은 ``[n_valid, n_rollout, n_total]`` 입니다.
            s: 윈도우 시작 스텝 인덱스입니다.
            e: 윈도우 끝 스텝 인덱스입니다.

        Returns:
            Tensor: 행동 토큰입니다. shape은 ``[n_valid, n_rollout]`` 이며 값은 0..8 입니다.
        """
        # 종방향: 윈도우 내 각 스텝의 (velocity . heading) 평균과 평균 속력.
        velocity = (full_pos[:, :, s + 1 : e + 1, :] - full_pos[:, :, s:e, :]) / self.dt
        head_cos = torch.cos(full_head[:, :, s:e])
        head_sin = torch.sin(full_head[:, :, s:e])
        v_lon = velocity[..., 0] * head_cos + velocity[..., 1] * head_sin
        mean_v_lon = v_lon.mean(dim=-1)
        mean_speed = velocity.norm(dim=-1).mean(dim=-1)

        is_stop = mean_speed < self.stop_speed
        lon = torch.where(
            is_stop,
            torch.ones_like(mean_v_lon),
            torch.where(mean_v_lon > 0.0, torch.full_like(mean_v_lon, 2.0), torch.zeros_like(mean_v_lon)),
        ).long()

        # 횡방향: 윈도우-시작 frame에서 끝점의 횡변위 dy'.
        origin = full_pos[:, :, s, :]
        psi0 = full_head[:, :, s]
        rel = full_pos[:, :, e, :] - origin
        dy = -rel[..., 0] * torch.sin(psi0) + rel[..., 1] * torch.cos(psi0)
        lat = torch.where(
            dy > self.lat_threshold,
            torch.full_like(dy, 2.0),
            torch.where(dy < -self.lat_threshold, torch.zeros_like(dy), torch.ones_like(dy)),
        ).long()

        return lon * _N_LAT + lat

    @staticmethod
    def _variance_decomposition(
        segment: Tensor,
        onehot: Tensor,
        count_k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """한 윈도우 trajectory variance를 토큰-간 / 토큰-내로 분해합니다.

        ``V_total = V_between + V_within`` (전분산 법칙)이 정확히 성립합니다.
        rigid 변환에 불변하므로 좌표 frame은 무관합니다.

        Args:
            segment: 윈도우 trajectory 구간입니다.
                shape은 ``[n_valid, n_rollout, window_len + 1, 2]`` 입니다.
            onehot: 토큰 one-hot입니다. shape은 ``[n_valid, n_rollout, 9]`` 입니다.
            count_k: 토큰별 rollout 개수입니다. shape은 ``[n_valid, 9]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: agent별 토큰-간 분산과 토큰-내 분산입니다.
                각 shape은 ``[n_valid]`` 입니다.
        """
        grand_mean = segment.mean(dim=1, keepdim=True)  # [n_valid, 1, Lw, 2]
        group_sum = torch.einsum("agk,agtd->aktd", onehot, segment)  # [n_valid, 9, Lw, 2]
        group_mean = group_sum / count_k.clamp_min(1.0)[:, :, None, None]
        # 각 rollout이 속한 토큰의 그룹 평균을 다시 모읍니다 (one-hot 선택).
        member_mean = torch.einsum("agk,aktd->agtd", onehot, group_mean)  # [n_valid, G, Lw, 2]

        v_between = (member_mean - grand_mean).pow(2).sum(dim=-1).mean(dim=(1, 2))
        v_within = (segment - member_mean).pow(2).sum(dim=-1).mean(dim=(1, 2))
        return v_between, v_within

    def _categorical_stats(
        self,
        count_k: Tensor,
        n_rollout: int,
        log_max: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """윈도우 토큰 분포의 categorical 다양성 지표를 계산합니다.

        Args:
            count_k: 토큰별 rollout 개수입니다. shape은 ``[n_valid, 9]`` 입니다.
            n_rollout: scenario당 rollout 개수 G입니다.
            log_max: entropy 정규화 분모 ``log(min(9, G))`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]: agent별 정규화 entropy, coverage,
                mode-fraction입니다. 각 shape은 ``[n_valid]`` 입니다.
        """
        prob = count_k / float(n_rollout)
        entropy = -(prob * torch.log(prob.clamp_min(self.eps))).sum(dim=-1)
        entropy_norm = entropy / log_max
        coverage = (count_k > 0).sum(dim=-1).to(torch.float32) / float(_N_TOKEN)
        mode = count_k.max(dim=-1).values / float(n_rollout)
        return entropy_norm, coverage, mode

    def compute(self) -> Dict[str, Tensor]:
        """누적된 행동 다양성 지표를 계산합니다.

        Returns:
            Dict[str, Tensor]: Lightning / W&B에 넘길 스칼라 metric 사전입니다.
                누적된 값이 없으면 빈 사전입니다.
        """
        if float(self.aw_count.sum()) <= 0.0:
            return {}

        metric_dict: Dict[str, Tensor] = {}
        aw = self.aw_count.clamp_min(1.0)

        for w_idx in range(self.num_windows):
            denom = (self.eta_btw_sum[w_idx] + self.eta_wth_sum[w_idx]).clamp_min(self.eps)
            metric_dict[f"{self.prefix}/diversity/eta_intent_w{w_idx}"] = (
                self.eta_btw_sum[w_idx] / denom
            ).to(torch.float32)
            metric_dict[f"{self.prefix}/diversity/token_entropy_w{w_idx}"] = (
                self.entropy_sum[w_idx] / aw[w_idx]
            ).to(torch.float32)

        btw = self.eta_btw_sum.sum()
        wth = self.eta_wth_sum.sum()
        total_aw = self.aw_count.sum().clamp_min(1.0)
        metric_dict[f"{self.prefix}/diversity/eta_intent"] = (
            btw / (btw + wth).clamp_min(self.eps)
        ).to(torch.float32)
        metric_dict[f"{self.prefix}/diversity/token_entropy"] = (
            self.entropy_sum.sum() / total_aw
        ).to(torch.float32)
        metric_dict[f"{self.prefix}/diversity/coverage"] = (
            self.coverage_sum.sum() / total_aw
        ).to(torch.float32)
        metric_dict[f"{self.prefix}/diversity/mode_fraction"] = (
            self.mode_sum.sum() / total_aw
        ).to(torch.float32)

        # 전체 agent를 모은 population 토큰 분포와 그 entropy.
        hist = self.token_hist.sum(dim=0)
        prob_pop = hist / hist.sum().clamp_min(self.eps)
        pop_entropy = -(prob_pop * torch.log(prob_pop.clamp_min(self.eps))).sum() / math.log(_N_TOKEN)
        metric_dict[f"{self.prefix}/diversity/pop_entropy"] = pop_entropy.to(torch.float32)
        for token_index, cell_name in enumerate(_CELL_NAMES):
            metric_dict[f"{self.prefix}/diversity/cell/{cell_name}"] = prob_pop[token_index].to(
                torch.float32
            )

        metric_dict[f"{self.prefix}/diversity/n_agent"] = self.aw_count[0].to(torch.float32)
        return metric_dict


def update_scenario_diversity_metric_from_model(
    model: object,
    data: object,
    pred_traj: Tensor,
    pred_head: Tensor,
) -> None:
    """PyG batch와 closed-loop rollout으로 행동 다양성 metric을 갱신합니다.

    Args:
        model: ``ScenarioDiversityMetrics`` 를 ``scenario_diversity_metrics`` 속성으로
            가진 LightningModule 객체입니다.
        data: validation batch입니다.
        pred_traj: closed-loop 미래 위치입니다.
            shape은 ``[n_agent, n_rollout, n_step, 2]`` 입니다.
        pred_head: closed-loop 미래 heading입니다.
            shape은 ``[n_agent, n_rollout, n_step]`` 입니다.
    """
    if pred_traj.ndim != 4 or int(pred_traj.shape[-1]) != 2 or pred_head.ndim != 3:
        return
    metric: Optional[ScenarioDiversityMetrics] = getattr(model, "scenario_diversity_metrics", None)
    if metric is None:
        return

    agent_store = data["agent"]
    position = agent_store["position"]
    heading = agent_store["heading"]
    valid_mask = agent_store["valid_mask"]

    current_index = max(0, int(getattr(model, "num_historical_steps")) - 1)
    current_pos = position[:, current_index, :2]
    current_head = heading[:, current_index]
    agent_valid = valid_mask[:, current_index] if valid_mask.ndim == 2 else None

    metric.update(
        pred_traj=pred_traj,
        pred_head=pred_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_valid=agent_valid,
    )


def log_and_reset_scenario_diversity_metric(
    model: object,
    metric: ScenarioDiversityMetrics,
) -> None:
    """누적된 행동 다양성 metric을 log에 남기고 상태를 초기화합니다.

    Args:
        model: LightningModule 객체입니다.
        metric: 계산과 초기화를 수행할 metric입니다.
    """
    metric_dict = metric.compute()
    if metric_dict:
        model.log_dict(
            metric_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
    metric.reset()
