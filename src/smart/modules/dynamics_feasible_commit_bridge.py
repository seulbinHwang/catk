from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.smart.modules.draft_physics import DEFAULT_LIMITS
from src.smart.utils import transform_to_global, wrap_angle


class DynamicsAwareFeasibleCommitBridge:
    """생성된 2초 미래를 빠른 배치 추종으로 0.5초 실행 상태로 바꿉니다.

    이 모듈은 차량과 자전거처럼 진행 방향 제약이 있는 에이전트에만
    yaw-rate 형태의 간단한 bicycle 계열 모델을 적용합니다.
    보행자는 기존 raw FM commit 경로를 그대로 유지합니다.

    구현 목표는 아래 세 가지입니다.

    1. 생성된 2초 미래를 바로 덮어쓰지 않고, 다음 0.5초 구간만 실행 가능한
       상태로 바꿉니다.
    2. WOMD에 wheelbase가 없으므로 steering angle 대신 speed / yaw-rate 형태로
       추적합니다.
    3. 에이전트 축에 대한 계산은 전부 배치 병렬로 처리하고, 시간축의 짧은
       고정 길이(미리보기 20 step, 실행 5 step)만 작은 반복으로 풉니다.

    Args:
        dt: 내부 10Hz 적분 간격입니다. 기본값은 ``0.1`` 초입니다.
        pos_scale_m: flow decoder가 낸 정규화 좌표를 meter로 되돌릴 배율입니다.
            기본값은 ``20.0`` 입니다.
        preview_steps: 제어 기준으로 볼 미래 길이입니다. 기본값은 ``20`` step,
            즉 2초입니다.
        commit_steps: 실제로 실행해 context에 반영할 길이입니다. 기본값은
            ``5`` step, 즉 0.5초입니다.
        smoothing_alpha: 기준 속도 시퀀스와 yaw-rate 시퀀스를 한 번 부드럽게
            만들 때 공통으로 쓰는 계수입니다.
        q_terminal_speed: 종방향 종단 속도 오차 가중치입니다.
        r_accel: 종방향 가속도 크기 가중치입니다.
        q_terminal_lateral: 횡방향 종단 위치 오차 가중치입니다.
        q_terminal_heading: 횡방향 종단 방향 오차 가중치입니다.
        r_yaw_rate: 목표 yaw-rate 크기 가중치입니다.
        low_speed_threshold_mps: 정지 근처 special handling 기준 속도입니다.
        low_speed_gain: 저속 모드에서 current 0.5초 motion intent에 맞출 때
            쓰는 비례 이득입니다.
        use_stationary_refinement: ``True`` 이면 eab8dd38에서 추가된
            stationary hold, commit-window intent 기반 저속 보정,
            limit-aware propagation을 함께 적용합니다. ``False`` 이면
            그 이전 legacy bridge 동작을 유지합니다.
        stationary_speed_enter_mps: 정지 hold 모드 진입용 현재 속도 임계값입니다.
        stationary_speed_exit_mps: 정지 hold 모드 유지용 현재 속도 임계값입니다.
        stationary_yaw_rate_enter_radps: 정지 hold 모드 진입용 현재 yaw-rate 임계값입니다.
        stationary_yaw_rate_exit_radps: 정지 hold 모드 유지용 현재 yaw-rate 임계값입니다.
        stationary_displacement_enter_m: 정지 hold 모드 진입용 0.5초 net displacement 임계값입니다.
        stationary_displacement_exit_m: 정지 hold 모드 유지용 0.5초 net displacement 임계값입니다.
        stationary_path_length_enter_m: 정지 hold 모드 진입용 0.5초 path length 임계값입니다.
        stationary_path_length_exit_m: 정지 hold 모드 유지용 0.5초 path length 임계값입니다.
        stationary_heading_enter_rad: 정지 hold 모드 진입용 0.5초 heading envelope 임계값입니다.
        stationary_heading_exit_rad: 정지 hold 모드 유지용 0.5초 heading envelope 임계값입니다.
        longitudinal_intent_deadzone_m: 0.5초 block의 종방향 motion intent를
            판정할 때 쓰는 dead-zone입니다.
        v_floor_mps: 곡률 계열 제한을 계산할 때 0으로 나누지 않도록 쓰는 작은 값입니다.
    """

    def __init__(
        self,
        dt: float = 0.1,
        pos_scale_m: float = 20.0,
        preview_steps: int = 10,
        commit_steps: int = 5,
        smoothing_alpha: float = 0.65,
        q_terminal_speed: float = 10.0,
        r_accel: float = 1.0,
        q_terminal_lateral: float = 1.0,
        q_terminal_heading: float = 10.0,
        r_yaw_rate: float = 1.0,
        low_speed_threshold_mps: float = 0.2,
        low_speed_gain: float = 0.5,
        use_stationary_refinement: bool = False,
        stationary_speed_enter_mps: float = 0.05,
        stationary_speed_exit_mps: float = 0.1,
        stationary_yaw_rate_enter_radps: float = 0.1,
        stationary_yaw_rate_exit_radps: float = 0.2,
        stationary_displacement_enter_m: float = 0.04,
        stationary_displacement_exit_m: float = 0.08,
        stationary_path_length_enter_m: float = 0.08,
        stationary_path_length_exit_m: float = 0.16,
        stationary_heading_enter_rad: float = 0.05,
        stationary_heading_exit_rad: float = 0.1,
        stationary_pair_reuse_displacement_epsilon_m: float = 1e-3,
        stationary_pair_reuse_heading_epsilon_rad: float = 1e-3,
        longitudinal_intent_deadzone_m: float = 0.05,
        v_floor_mps: float = 0.1,
    ) -> None:
        self.dt = float(dt)
        self.pos_scale_m = float(pos_scale_m)
        self.preview_steps = int(preview_steps)
        self.commit_steps = int(commit_steps)
        self.smoothing_alpha = float(smoothing_alpha)
        self.q_terminal_speed = float(q_terminal_speed)
        self.r_accel = float(r_accel)
        self.q_terminal_lateral = float(q_terminal_lateral)
        self.q_terminal_heading = float(q_terminal_heading)
        self.r_yaw_rate = float(r_yaw_rate)
        self.low_speed_threshold_mps = float(low_speed_threshold_mps)
        self.low_speed_gain = float(low_speed_gain)
        self.use_stationary_refinement = bool(use_stationary_refinement)
        self.stationary_speed_enter_mps = float(stationary_speed_enter_mps)
        self.stationary_speed_exit_mps = float(stationary_speed_exit_mps)
        self.stationary_yaw_rate_enter_radps = float(stationary_yaw_rate_enter_radps)
        self.stationary_yaw_rate_exit_radps = float(stationary_yaw_rate_exit_radps)
        self.stationary_displacement_enter_m = float(stationary_displacement_enter_m)
        self.stationary_displacement_exit_m = float(stationary_displacement_exit_m)
        self.stationary_path_length_enter_m = float(stationary_path_length_enter_m)
        self.stationary_path_length_exit_m = float(stationary_path_length_exit_m)
        self.stationary_heading_enter_rad = float(stationary_heading_enter_rad)
        self.stationary_heading_exit_rad = float(stationary_heading_exit_rad)
        self.stationary_pair_reuse_displacement_epsilon_m = float(
            stationary_pair_reuse_displacement_epsilon_m
        )
        self.stationary_pair_reuse_heading_epsilon_rad = float(
            stationary_pair_reuse_heading_epsilon_rad
        )
        self.longitudinal_intent_deadzone_m = float(longitudinal_intent_deadzone_m)
        self.v_floor_mps = float(v_floor_mps)

    def commit(
        self,
        y_hat_norm: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        agent_type: Tensor,
        agent_shape: Tensor,
        exec_pos_pair: Tensor,
        exec_head_pair: Tensor,
        exec_valid_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """현재 coarse 상태와 생성 미래를 받아 실행할 0.5초 chunk를 만듭니다.

        Args:
            y_hat_norm: flow decoder가 낸 정규화 2초 미래입니다.
                shape은 ``[n_agent, 20, 4]`` 입니다.
                마지막 축은 ``[x, y, cos, sin]`` 입니다.
            current_pos: 현재 coarse 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse 방향입니다. shape은 ``[n_agent]`` 입니다.
            agent_type: 차종 인덱스입니다. shape은 ``[n_agent]`` 입니다.
                ``0=vehicle, 1=pedestrian, 2=bicycle`` 입니다.
            agent_shape: 실제 데이터셋 에이전트 크기입니다.
                shape은 ``[n_agent, 3]`` 또는 최소 ``[n_agent, 2]`` 입니다.
                앞 두 값은 수평 footprint로 가정합니다.
            exec_pos_pair: 최근 실행된 fine 중심점 2개입니다.
                shape은 ``[n_agent, 2, 2]`` 입니다.
            exec_head_pair: 최근 실행된 fine 방향 2개입니다.
                shape은 ``[n_agent, 2]`` 입니다.
            exec_valid_pair: 최근 실행된 fine 상태 2개의 유효 여부입니다.
                shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                - commit_pos: 실행할 0.5초 중심점 5개. shape은 ``[n_agent, 5, 2]`` 입니다.
                - commit_head: 실행할 0.5초 방향 5개. shape은 ``[n_agent, 5]`` 입니다.
                - next_pos: 다음 coarse 상태 중심점. shape은 ``[n_agent, 2]`` 입니다.
                - next_head: 다음 coarse 상태 방향. shape은 ``[n_agent]`` 입니다.
        """
        commit_pos, commit_head, next_pos, next_head = self._raw_commit_from_flow(
            y_hat_norm=y_hat_norm,
            current_pos=current_pos,
            current_head=current_head,
        )

        nonholonomic = agent_type.long() != 1
        if not nonholonomic.any():
            return commit_pos, commit_head, next_pos, next_head

        dyn_mask = nonholonomic
        dyn_commit_pos, dyn_commit_head = self._commit_nonholonomic_agents(
            y_hat_norm=y_hat_norm[dyn_mask],
            current_pos=current_pos[dyn_mask],
            current_head=current_head[dyn_mask],
            agent_type=agent_type[dyn_mask],
            agent_shape=agent_shape[dyn_mask],
            exec_pos_pair=exec_pos_pair[dyn_mask],
            exec_head_pair=exec_head_pair[dyn_mask],
            exec_valid_pair=exec_valid_pair[dyn_mask],
        )
        commit_pos[dyn_mask] = dyn_commit_pos
        commit_head[dyn_mask] = dyn_commit_head
        next_pos = commit_pos[:, -1]
        next_head = commit_head[:, -1]
        return commit_pos, commit_head, next_pos, next_head

    def _raw_commit_from_flow(
        self,
        y_hat_norm: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """기존 raw FM commit 규칙을 그대로 재현합니다.

        Args:
            y_hat_norm: 정규화 미래입니다. shape은 ``[n_agent, 20, 4]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: 기존 raw commit 결과입니다.
        """
        first_chunk_pos_local, first_chunk_head_local = self._decode_future_norm(
            y_hat_norm=y_hat_norm,
            num_steps=self.commit_steps,
        )
        commit_pos, _ = transform_to_global(
            pos_local=first_chunk_pos_local,
            head_local=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        commit_head = wrap_angle(current_head.unsqueeze(1) + first_chunk_head_local)
        return commit_pos, commit_head, commit_pos[:, -1], commit_head[:, -1]

    def _commit_nonholonomic_agents(
        self,
        y_hat_norm: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        agent_type: Tensor,
        agent_shape: Tensor,
        exec_pos_pair: Tensor,
        exec_head_pair: Tensor,
        exec_valid_pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """차량/자전거에만 dynamics-aware commit을 적용합니다.

        Args:
            y_hat_norm: 정규화 미래입니다. shape은 ``[n_nonhol, 20, 4]`` 입니다.
            current_pos: 현재 중심점입니다. shape은 ``[n_nonhol, 2]`` 입니다.
            current_head: 현재 방향입니다. shape은 ``[n_nonhol]`` 입니다.
            agent_type: 비보행 에이전트 종류입니다. shape은 ``[n_nonhol]`` 입니다.
            agent_shape: 실제 footprint 크기입니다. shape은 ``[n_nonhol, 3]`` 또는 ``[n_nonhol, 2]`` 입니다.
            exec_pos_pair: 최근 fine 중심점 2개입니다. shape은 ``[n_nonhol, 2, 2]`` 입니다.
            exec_head_pair: 최근 fine 방향 2개입니다. shape은 ``[n_nonhol, 2]`` 입니다.
            exec_valid_pair: 최근 fine 상태 유효 여부입니다. shape은 ``[n_nonhol, 2]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: 전역 좌표의 실행 중심점과 방향입니다.
        """
        preview_pos_local, preview_head_local = self._decode_future_norm(
            y_hat_norm=y_hat_norm,
            num_steps=self.preview_steps,
        )
        ref_speed, ref_yaw_rate = self._build_reference_controls(
            preview_pos_local=preview_pos_local,
            preview_head_local=preview_head_local,
        )
        limits = self._gather_limits(
            agent_type=agent_type,
            agent_shape=agent_shape,
            device=y_hat_norm.device,
            dtype=y_hat_norm.dtype,
        )
        speed_0, yaw_rate_0 = self._estimate_initial_controls(
            exec_pos_pair=exec_pos_pair,
            exec_head_pair=exec_head_pair,
            exec_valid_pair=exec_valid_pair,
            ref_speed=ref_speed,
            ref_yaw_rate=ref_yaw_rate,
            v_max=limits["v_max_mps"],
            yaw_rate_max_abs=limits["omega_max_abs_radps"],
        )
        stationary_hold_mask = torch.zeros_like(speed_0, dtype=torch.bool)
        accel_target = self._solve_longitudinal_command(
            speed_0=speed_0,
            ref_speed=ref_speed,
            a_max=limits["a_max_mps2"],
        )
        if self.use_stationary_refinement:
            commit_window_motion = self._build_commit_window_motion(
                preview_pos_local=preview_pos_local,
                preview_head_local=preview_head_local,
            )
            stationary_hold_mask = self._build_stationary_hold_mask(
                speed_0=speed_0,
                yaw_rate_0=yaw_rate_0,
                exec_pos_pair=exec_pos_pair,
                exec_head_pair=exec_head_pair,
                exec_valid_pair=exec_valid_pair,
                commit_window_motion=commit_window_motion,
            )
            window_dt = max(float(self.commit_steps) * self.dt, 1e-6)
            coherent_longitudinal_motion = (
                commit_window_motion["longitudinal_displacement_m"].abs()
                >= self.longitudinal_intent_deadzone_m
            )
            window_signed_speed = torch.where(
                coherent_longitudinal_motion,
                commit_window_motion["longitudinal_displacement_m"] / window_dt,
                torch.zeros_like(speed_0),
            )
            low_speed_mask = (
                speed_0.abs() <= self.low_speed_threshold_mps
            ) & (~stationary_hold_mask)
            low_speed_accel = torch.clamp(
                self.low_speed_gain * (window_signed_speed - speed_0),
                min=-limits["a_max_mps2"],
                max=limits["a_max_mps2"],
            )
        else:
            preview_speed_mean = ref_speed.mean(dim=-1)
            preview_speed_mean_abs = ref_speed.abs().mean(dim=-1)
            low_speed_mask = (
                speed_0.abs() <= self.low_speed_threshold_mps
            ) & (
                preview_speed_mean_abs <= self.low_speed_threshold_mps
            )
            low_speed_accel = torch.clamp(
                self.low_speed_gain * (preview_speed_mean - speed_0),
                min=-limits["a_max_mps2"],
                max=limits["a_max_mps2"],
            )
        accel_target = torch.where(low_speed_mask, low_speed_accel, accel_target)
        speed_profile = self._build_speed_profile(
            speed_0=speed_0,
            accel_target=accel_target,
            v_max=limits["v_max_mps"],
            num_steps=self.preview_steps,
        )
        yaw_rate_target = self._solve_lateral_command(
            speed_profile=speed_profile,
            ref_yaw_rate=ref_yaw_rate,
            yaw_rate_max_abs=limits["omega_max_abs_radps"],
        )
        if not self.use_stationary_refinement:
            yaw_rate_target = torch.where(low_speed_mask, torch.zeros_like(yaw_rate_target), yaw_rate_target)

        commit_pos_local, commit_head_local = self._propagate_commit(
            speed_0=speed_0,
            yaw_rate_0=yaw_rate_0,
            accel_target=accel_target,
            yaw_rate_target=yaw_rate_target,
            limits=limits,
            use_limits=self.use_stationary_refinement,
        )
        if self.use_stationary_refinement and stationary_hold_mask.any():
            commit_pos_local[stationary_hold_mask] = 0.0
            commit_head_local[stationary_hold_mask] = 0.0
        commit_pos_global, _ = transform_to_global(
            pos_local=commit_pos_local,
            head_local=None,
            pos_now=current_pos,
            head_now=current_head,
        )
        commit_head_global = wrap_angle(current_head.unsqueeze(1) + commit_head_local)
        return commit_pos_global, commit_head_global

    def _decode_future_norm(
        self,
        y_hat_norm: Tensor,
        num_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """정규화 미래 일부를 local meter 좌표와 local heading으로 바꿉니다.

        Args:
            y_hat_norm: 정규화 미래입니다. shape은 ``[n_agent, 20, 4]`` 입니다.
            num_steps: 앞에서 몇 step을 쓸지 정합니다.

        Returns:
            tuple[Tensor, Tensor]:
                - local 중심점 ``[n_agent, num_steps, 2]``
                - local heading ``[n_agent, num_steps]``
        """
        used_steps = min(int(num_steps), int(y_hat_norm.shape[1]))
        future_slice = y_hat_norm[:, :used_steps]
        pos_local = future_slice[..., :2] * self.pos_scale_m
        cos_sin = F.normalize(future_slice[..., 2:4], dim=-1)
        head_local = torch.atan2(cos_sin[..., 1], cos_sin[..., 0])
        return pos_local, head_local

    def _build_reference_controls(
        self,
        preview_pos_local: Tensor,
        preview_head_local: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """preview pose를 body-frame 기준 속도와 yaw-rate 기준선으로 바꿉니다.

        Args:
            preview_pos_local: local 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            preview_head_local: local 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:
                - ref_speed: body-frame 앞방향 속도 기준선 ``[n_agent, n_step]``
                - ref_yaw_rate: yaw-rate 기준선 ``[n_agent, n_step]``
        """
        num_agent = preview_pos_local.shape[0]
        zero_pos = preview_pos_local.new_zeros((num_agent, 1, 2))
        zero_head = preview_head_local.new_zeros((num_agent, 1))
        pos_seq = torch.cat([zero_pos, preview_pos_local], dim=1)
        head_seq = torch.cat([zero_head, preview_head_local], dim=1)

        delta_pos = pos_seq[:, 1:] - pos_seq[:, :-1]
        head_start = head_seq[:, :-1]
        delta_head = wrap_angle(head_seq[:, 1:] - head_seq[:, :-1])

        cos_head = head_start.cos()
        sin_head = head_start.sin()
        ref_speed = (delta_pos[..., 0] * cos_head + delta_pos[..., 1] * sin_head) / self.dt
        ref_yaw_rate = delta_head / self.dt

        ref_speed = self._smooth_sequence(ref_speed, alpha=self.smoothing_alpha)
        ref_yaw_rate = self._smooth_sequence(ref_yaw_rate, alpha=self.smoothing_alpha)
        return ref_speed, ref_yaw_rate

    def _build_commit_window_motion(
        self,
        preview_pos_local: Tensor,
        preview_head_local: Tensor,
    ) -> Dict[str, Tensor]:
        """다음 0.5초 commit window의 정지/출발 증거를 요약합니다."""
        window_pos = preview_pos_local[:, : self.commit_steps]
        window_head = preview_head_local[:, : self.commit_steps]
        num_agent = preview_pos_local.shape[0]
        zero_pos = preview_pos_local.new_zeros((num_agent, 1, 2))
        pos_seq = torch.cat([zero_pos, window_pos], dim=1)
        delta_pos = pos_seq[:, 1:] - pos_seq[:, :-1]
        delta_norm = torch.linalg.norm(delta_pos, dim=-1)

        if window_pos.shape[1] == 0:
            zeros = preview_pos_local.new_zeros(num_agent)
            return {
                "net_displacement_m": zeros,
                "path_length_m": zeros,
                "heading_envelope_rad": zeros,
                "longitudinal_displacement_m": zeros,
            }

        return {
            "net_displacement_m": torch.linalg.norm(window_pos[:, -1], dim=-1),
            "path_length_m": delta_norm.sum(dim=-1),
            "heading_envelope_rad": wrap_angle(window_head).abs().amax(dim=-1),
            "longitudinal_displacement_m": delta_pos[..., 0].sum(dim=-1),
        }

    def _build_stationary_hold_mask(
        self,
        speed_0: Tensor,
        yaw_rate_0: Tensor,
        exec_pos_pair: Tensor,
        exec_head_pair: Tensor,
        exec_valid_pair: Tensor,
        commit_window_motion: Dict[str, Tensor],
    ) -> Tensor:
        """현재 상태와 다음 0.5초 preview가 모두 정지 tube 안에 있으면 hold합니다."""
        pair_valid = exec_valid_pair.all(dim=-1)
        pair_delta_pos = exec_pos_pair[:, 1] - exec_pos_pair[:, 0]
        pair_delta_head = wrap_angle(exec_head_pair[:, 1] - exec_head_pair[:, 0]).abs()
        prev_hold_like = pair_valid & (
            torch.linalg.norm(pair_delta_pos, dim=-1)
            <= self.stationary_pair_reuse_displacement_epsilon_m
        ) & (
            pair_delta_head <= self.stationary_pair_reuse_heading_epsilon_rad
        )

        speed_threshold = torch.where(
            prev_hold_like,
            speed_0.new_full(speed_0.shape, self.stationary_speed_exit_mps),
            speed_0.new_full(speed_0.shape, self.stationary_speed_enter_mps),
        )
        yaw_rate_threshold = torch.where(
            prev_hold_like,
            yaw_rate_0.new_full(yaw_rate_0.shape, self.stationary_yaw_rate_exit_radps),
            yaw_rate_0.new_full(yaw_rate_0.shape, self.stationary_yaw_rate_enter_radps),
        )
        displacement_threshold = torch.where(
            prev_hold_like,
            speed_0.new_full(speed_0.shape, self.stationary_displacement_exit_m),
            speed_0.new_full(speed_0.shape, self.stationary_displacement_enter_m),
        )
        path_length_threshold = torch.where(
            prev_hold_like,
            speed_0.new_full(speed_0.shape, self.stationary_path_length_exit_m),
            speed_0.new_full(speed_0.shape, self.stationary_path_length_enter_m),
        )
        heading_threshold = torch.where(
            prev_hold_like,
            speed_0.new_full(speed_0.shape, self.stationary_heading_exit_rad),
            speed_0.new_full(speed_0.shape, self.stationary_heading_enter_rad),
        )

        current_speed_abs = torch.where(pair_valid, speed_0.abs(), torch.zeros_like(speed_0))
        current_yaw_rate_abs = torch.where(pair_valid, yaw_rate_0.abs(), torch.zeros_like(yaw_rate_0))
        current_near_rest = (
            (current_speed_abs <= speed_threshold)
            & (current_yaw_rate_abs <= yaw_rate_threshold)
        )
        preview_in_stationary_tube = (
            (commit_window_motion["net_displacement_m"] <= displacement_threshold)
            & (commit_window_motion["path_length_m"] <= path_length_threshold)
            & (commit_window_motion["heading_envelope_rad"] <= heading_threshold)
        )
        return current_near_rest & preview_in_stationary_tube

    def _smooth_sequence(
        self,
        sequence: Tensor,
        alpha: float,
    ) -> Tensor:
        """짧은 시간축 시퀀스를 한 번만 부드럽게 만듭니다.

        Args:
            sequence: 입력 시퀀스입니다. shape은 ``[n_agent, n_step]`` 입니다.
            alpha: 현재 값을 얼마나 유지할지 정하는 계수입니다.

        Returns:
            Tensor: 같은 shape의 부드러워진 시퀀스입니다.
        """
        if sequence.shape[1] <= 1:
            return sequence
        smoothed = sequence.clone()
        for step_idx in range(1, sequence.shape[1]):
            smoothed[:, step_idx] = (
                alpha * sequence[:, step_idx]
                + (1.0 - alpha) * smoothed[:, step_idx - 1]
            )
        return smoothed

    def _gather_limits(
        self,
        agent_type: Tensor,
        agent_shape: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """에이전트 종류와 실제 footprint를 반영한 제한값을 만듭니다.

        Args:
            agent_type: 차종 인덱스입니다. shape은 ``[n_agent]`` 입니다.
            agent_shape: 실제 shape입니다. shape은 ``[n_agent, 3]`` 또는 ``[n_agent, 2]`` 입니다.
            device: 반환 텐서를 둘 장치입니다.
            dtype: 반환 텐서 자료형입니다.

        Returns:
            Dict[str, Tensor]: agent별 제한값 사전입니다. 각 값의 shape은 ``[n_agent]`` 입니다.
        """
        agent_type = agent_type.to(device=device, dtype=torch.long).clamp(min=0, max=2)

        def _select(values: Tuple[float, float, float]) -> Tensor:
            table = torch.tensor(values, device=device, dtype=dtype)
            return table[agent_type]

        r_min_m = _select(DEFAULT_LIMITS.r_min_m)
        if agent_shape.shape[-1] >= 2:
            footprint_major = agent_shape[..., :2].to(device=device, dtype=dtype).abs().amax(dim=-1)
            r_min_m = torch.maximum(r_min_m, 0.5 * footprint_major)

        return {
            "v_max_mps": _select(DEFAULT_LIMITS.v_max_mps),
            "a_max_mps2": _select(DEFAULT_LIMITS.a_max_mps2),
            "alpha_max_radps2": _select(DEFAULT_LIMITS.alpha_max_radps2),
            "a_lat_max_mps2": _select(DEFAULT_LIMITS.a_lat_max_mps2),
            "r_min_m": r_min_m,
            "omega_max_abs_radps": _select(DEFAULT_LIMITS.omega_max_abs_radps),
        }

    def _estimate_initial_controls(
        self,
        exec_pos_pair: Tensor,
        exec_head_pair: Tensor,
        exec_valid_pair: Tensor,
        ref_speed: Tensor,
        ref_yaw_rate: Tensor,
        v_max: Tensor,
        yaw_rate_max_abs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """최근 실행된 fine state 두 개로 현재 speed와 yaw-rate를 추정합니다.

        Args:
            exec_pos_pair: 최근 fine 중심점 2개입니다. shape은 ``[n_agent, 2, 2]`` 입니다.
            exec_head_pair: 최근 fine 방향 2개입니다. shape은 ``[n_agent, 2]`` 입니다.
            exec_valid_pair: 최근 fine 상태 유효 여부입니다. shape은 ``[n_agent, 2]`` 입니다.
            ref_speed: preview 기준 속도입니다. shape은 ``[n_agent, n_step]`` 입니다.
            ref_yaw_rate: preview 기준 yaw-rate입니다. shape은 ``[n_agent, n_step]`` 입니다.
            v_max: 최고 속도 제한입니다. shape은 ``[n_agent]`` 입니다.
            yaw_rate_max_abs: 절대 yaw-rate 제한입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:
                - speed_0: 현재 앞방향 속도 ``[n_agent]``
                - yaw_rate_0: 현재 yaw-rate ``[n_agent]``
        """
        prev_pos = exec_pos_pair[:, 0]
        curr_pos = exec_pos_pair[:, 1]
        prev_head = exec_head_pair[:, 0]
        curr_head = exec_head_pair[:, 1]
        pair_valid = exec_valid_pair.all(dim=-1)

        delta_pos = curr_pos - prev_pos
        cos_prev = prev_head.cos()
        sin_prev = prev_head.sin()
        speed_0 = (delta_pos[:, 0] * cos_prev + delta_pos[:, 1] * sin_prev) / self.dt
        yaw_rate_0 = wrap_angle(curr_head - prev_head) / self.dt

        speed_0 = torch.where(pair_valid, speed_0, ref_speed[:, 0])
        yaw_rate_0 = torch.where(pair_valid, yaw_rate_0, ref_yaw_rate[:, 0])
        speed_0 = torch.clamp(speed_0, min=-v_max, max=v_max)
        yaw_rate_0 = torch.clamp(yaw_rate_0, min=-yaw_rate_max_abs, max=yaw_rate_max_abs)
        return speed_0, yaw_rate_0

    def _solve_longitudinal_command(
        self,
        speed_0: Tensor,
        ref_speed: Tensor,
        a_max: Tensor,
    ) -> Tensor:
        """종단 속도 하나를 맞추는 상수 가속도를 closed-form으로 풉니다.

        Args:
            speed_0: 현재 속도입니다. shape은 ``[n_agent]`` 입니다.
            ref_speed: preview 기준 속도입니다. shape은 ``[n_agent, n_step]`` 입니다.
            a_max: agent별 최대 가감속 절대값입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: 상수 가속도 명령입니다. shape은 ``[n_agent]`` 입니다.
        """
        horizon_dt = ref_speed.shape[1] * self.dt
        ref_terminal_speed = ref_speed[:, -1]
        numerator = self.q_terminal_speed * horizon_dt * (ref_terminal_speed - speed_0)
        denominator = self.r_accel + self.q_terminal_speed * (horizon_dt ** 2)
        accel = numerator / max(denominator, 1e-6)
        return torch.clamp(accel, min=-a_max, max=a_max)

    def _build_speed_profile(
        self,
        speed_0: Tensor,
        accel_target: Tensor,
        v_max: Tensor,
        num_steps: int,
    ) -> Tensor:
        """상수 가속도 가정으로 preview 속도 프로파일을 만듭니다.

        Args:
            speed_0: 현재 속도입니다. shape은 ``[n_agent]`` 입니다.
            accel_target: 상수 가속도 명령입니다. shape은 ``[n_agent]`` 입니다.
            v_max: 최고 속도 절대값입니다. shape은 ``[n_agent]`` 입니다.
            num_steps: preview 길이입니다.

        Returns:
            Tensor: step별 속도입니다. shape은 ``[n_agent, num_steps]`` 입니다.
        """
        time_idx = torch.arange(num_steps, device=speed_0.device, dtype=speed_0.dtype)
        speed_profile = speed_0.unsqueeze(1) + accel_target.unsqueeze(1) * self.dt * time_idx.unsqueeze(0)
        return torch.clamp(speed_profile, min=-v_max.unsqueeze(1), max=v_max.unsqueeze(1))

    def _solve_lateral_command(
        self,
        speed_profile: Tensor,
        ref_yaw_rate: Tensor,
        yaw_rate_max_abs: Tensor,
    ) -> Tensor:
        """종단 횡오차와 방향오차를 줄이는 상수 yaw-rate를 closed-form으로 풉니다.

        Args:
            speed_profile: preview 속도입니다. shape은 ``[n_agent, n_step]`` 입니다.
            ref_yaw_rate: preview 기준 yaw-rate입니다. shape은 ``[n_agent, n_step]`` 입니다.
            yaw_rate_max_abs: agent별 절대 yaw-rate 제한입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: 목표 yaw-rate입니다. shape은 ``[n_agent]`` 입니다.
        """
        coeff_y = speed_profile.new_zeros(speed_profile.shape[0])
        const_y = speed_profile.new_zeros(speed_profile.shape[0])
        coeff_head = speed_profile.new_zeros(speed_profile.shape[0])
        const_head = speed_profile.new_zeros(speed_profile.shape[0])

        for step_idx in range(ref_yaw_rate.shape[1]):
            coeff_y = coeff_y + self.dt * speed_profile[:, step_idx] * coeff_head
            const_y = const_y + self.dt * speed_profile[:, step_idx] * const_head
            coeff_head = coeff_head + self.dt
            const_head = const_head - self.dt * ref_yaw_rate[:, step_idx]

        denominator = (
            self.q_terminal_lateral * coeff_y.square()
            + self.q_terminal_heading * coeff_head.square()
            + self.r_yaw_rate
        ).clamp_min(1e-6)
        numerator = -(
            self.q_terminal_lateral * coeff_y * const_y
            + self.q_terminal_heading * coeff_head * const_head
        )
        yaw_rate = numerator / denominator
        return torch.clamp(yaw_rate, min=-yaw_rate_max_abs, max=yaw_rate_max_abs)

    def _propagate_commit(
        self,
        speed_0: Tensor,
        yaw_rate_0: Tensor,
        accel_target: Tensor,
        yaw_rate_target: Tensor,
        limits: Dict[str, Tensor],
        use_limits: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """상수 목표 명령을 5개의 10Hz 실행 상태로 적분합니다.

        Args:
            speed_0: 현재 speed입니다. shape은 ``[n_agent]`` 입니다.
            yaw_rate_0: 현재 yaw-rate입니다. shape은 ``[n_agent]`` 입니다.
            accel_target: 목표 가속도입니다. shape은 ``[n_agent]`` 입니다.
            yaw_rate_target: 목표 yaw-rate입니다. shape은 ``[n_agent]`` 입니다.
            limits: agent별 제한값 사전입니다. 각 값 shape은 ``[n_agent]`` 입니다.
            use_limits: ``True`` 이면 물리 제한(속도, yaw-rate, 횡가속도,
                최소 회전반경 등)을 적용합니다. ``False`` 이면 제한 없이
                순수 적분만 수행합니다. 기본값은 ``False`` 입니다.

        Returns:
            tuple[Tensor, Tensor]:
                - commit_pos_local: local 중심점 5개. shape은 ``[n_agent, 5, 2]`` 입니다.
                - commit_head_local: local 방향 5개. shape은 ``[n_agent, 5]`` 입니다.
        """
        num_agent = speed_0.shape[0]
        commit_pos_local = speed_0.new_zeros((num_agent, self.commit_steps, 2))
        commit_head_local = speed_0.new_zeros((num_agent, self.commit_steps))

        pos_x = speed_0.new_zeros(num_agent)
        pos_y = speed_0.new_zeros(num_agent)
        head = speed_0.new_zeros(num_agent)

        if use_limits:
            speed = torch.clamp(speed_0, min=-limits["v_max_mps"], max=limits["v_max_mps"])
            yaw_rate = torch.clamp(
                yaw_rate_0,
                min=-limits["omega_max_abs_radps"],
                max=limits["omega_max_abs_radps"],
            )
            yaw_accel_step_limit = limits["alpha_max_radps2"] * self.dt
        else:
            speed = speed_0.clone()
            yaw_rate = yaw_rate_0.clone()

        for step_idx in range(self.commit_steps):
            if use_limits:
                yaw_rate_candidate = yaw_rate + torch.clamp(
                    yaw_rate_target - yaw_rate,
                    min=-yaw_accel_step_limit,
                    max=yaw_accel_step_limit,
                )
                speed_next = torch.clamp(
                    speed + accel_target * self.dt,
                    min=-limits["v_max_mps"],
                    max=limits["v_max_mps"],
                )
                speed_bound = torch.maximum(speed.abs(), speed_next.abs())
                omega_from_lat_acc = limits["a_lat_max_mps2"] / speed_bound.clamp_min(self.v_floor_mps)
                omega_from_radius = speed_bound / limits["r_min_m"].clamp_min(self.v_floor_mps)
                yaw_rate_step_limit = torch.minimum(
                    limits["omega_max_abs_radps"],
                    torch.minimum(omega_from_lat_acc, omega_from_radius),
                )
                yaw_rate = torch.clamp(
                    yaw_rate_candidate,
                    min=-yaw_rate_step_limit,
                    max=yaw_rate_step_limit,
                )
            else:
                yaw_rate = yaw_rate_target.clone()
                speed_next = speed + accel_target * self.dt

            speed_mid = 0.5 * (speed + speed_next)
            head_mid = head + 0.5 * yaw_rate * self.dt
            pos_x = pos_x + speed_mid * head_mid.cos() * self.dt
            pos_y = pos_y + speed_mid * head_mid.sin() * self.dt
            head = wrap_angle(head + yaw_rate * self.dt)
            speed = speed_next

            commit_pos_local[:, step_idx, 0] = pos_x
            commit_pos_local[:, step_idx, 1] = pos_y
            commit_head_local[:, step_idx] = head

        return commit_pos_local, commit_head_local
