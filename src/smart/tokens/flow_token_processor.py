from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, wrap_angle


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습에 필요한 anchor 단위 목표와 metadata를 만듭니다."""

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
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

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
        processed_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # processed_agent에서 flow 학습/평가에 필요한 연속 상태를 꺼냅니다.
        # valid: [n_agent, n_step] (각 10Hz step의 유효 여부)
        # pos:    [n_agent, n_step, 2] (x,y)
        # heading:[n_agent, n_step] (rad)
        valid = processed_agent["valid"]
        pos = processed_agent["pos"]
        heading = processed_agent["heading"]

        # 모델이 사용하는 context(이전 anchor 묶음) 메타데이터를 구성합니다.
        # tokenized_agent는 TokenProcessor.tokenize_agent() 결과라서 sampled_*가 존재합니다.
        # 여기서는 "앞부분 14개 anchor"만 context로 사용합니다. (shape는 [n_agent, 14, ...] 기대)
        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :14].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :14].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :14].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :14].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        # coarse anchor 개수(현재 구현에서는 13개를 고정으로 사용)
        num_anchor = 13
        # anchor가 가리키는 현재 시점 step 인덱스(10Hz step 기준).
        # self.shift=5 이므로 raw_current_steps는 [10,15,...,70] 형태가 됩니다.
        raw_current_steps = list(range(10, 71, self.shift))

        # 학습 시 어떤 agent를 학습에 포함할지(또는 제외할지) 마스크를 받습니다.
        # 캐시에 따라 train_mask가 없을 수 있으니 기본값은 전부 학습으로 둡니다.
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
            # ======================
            # 학습 모드: flow_train_* 타겟들을 anchor 단위로 구성
            # ======================
            flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
            # anchor별로 만들어진 타겟 조각들을 모아서 마지막에 cat으로 합칩니다.
            # 각 원소는 대략 다음 shape을 가집니다:
            # - flow_train_clean_norm 조각: [n_valid_anchor, 20, 4]
            # - flow_train_agent_type 조각: [n_valid_anchor]
            # - flow_train_current_control 조각: [n_valid_anchor, 3]
            # - flow_train_current_control_valid 조각: [n_valid_anchor]
            flow_train_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_current_control_chunks: List[Tensor] = []
            flow_train_current_control_valid_chunks: List[Tensor] = []
            for anchor_offset, raw_step in enumerate(raw_current_steps):
                # current_valid: 현재 anchor 시점(raw_step)에서 유효한 agent
                current_valid = valid[:, raw_step]
                # future_valid: 현재 anchor 다음 20 step(=2초) 모두 유효한 agent
                # (clean_norm은 raw_step+1 ~ raw_step+20 범위를 쓰므로, 전 구간이 유효해야 함)
                future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
                # anchor_mask: 이번 anchor 학습/평가에 쓸지 여부
                anchor_mask = current_valid & future_valid
                # train_anchor_mask: anchor_mask에 더해, agent 단위 train_mask까지 적용
                train_anchor_mask = anchor_mask & train_mask
                flow_train_mask[:, anchor_offset] = train_anchor_mask

                # 이 anchor에서 학습할 agent가 하나도 없으면, 타겟 조각 생성/concat을 스킵합니다.
                if not train_anchor_mask.any():
                    continue

                # (1) clean_norm:
                # raw_step 이후 2초 미래 궤적을 "해당 anchor의 coarse frame"
                # (sampled_pos/head 기준)으로 local 변환한 뒤 정규화합니다.
                anchor_clean_norm = self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=tokenized_agent["sampled_pos"][:, anchor_offset + 1],
                    current_head=tokenized_agent["sampled_heading"][:, anchor_offset + 1],
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                )

                # (2) current_control:
                # anchor 직전 0.1초 구간(raw_step-1 -> raw_step)에서 body-frame control을 만듭니다.
                anchor_current_control, anchor_current_control_valid = self._build_anchor_current_control(
                    pos=pos,
                    heading=heading,
                    valid=valid,
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                )

                # flow_train_mask는 train_anchor_mask로 채워지지만,
                # 실제로 append할 조각들은 anchor_mask 기준으로 만든 텐서를 "train_mask로 최종 필터링"합니다.
                # train_keep_local은 anchor_mask에서 살아남은 로컬 인덱스에 해당합니다.
                train_keep_local = train_mask[anchor_mask]
                flow_train_chunks.append(anchor_clean_norm[train_keep_local])
                flow_train_agent_type_chunks.append(tokenized_agent["type"][anchor_mask][train_keep_local])
                flow_train_current_control_chunks.append(
                    anchor_current_control[train_keep_local]
                )
                flow_train_current_control_valid_chunks.append(
                    anchor_current_control_valid[train_keep_local]
                )

            # 모든 anchor를 돌며 모은 조각들을 하나의 텐서로 이어 붙입니다.
            # 이렇게 하면 downstream에서 "valid anchor들만 모아서" 배치 연산하기 쉽습니다.
            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": self._concat_flow_chunks(
                        chunks=flow_train_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_agent_type": self._concat_metadata_chunks(
                        chunks=flow_train_agent_type_chunks,
                        empty_shape=(0,),
                        dtype=tokenized_agent["type"].dtype,
                        device=device,
                    ),
                    "flow_train_current_control": self._concat_metadata_chunks(
                        chunks=flow_train_current_control_chunks,
                        empty_shape=(0, 3),
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_current_control_valid": self._concat_metadata_chunks(
                        chunks=flow_train_current_control_valid_chunks,
                        empty_shape=(0,),
                        dtype=torch.bool,
                        device=device,
                    ),
                }
            )

            # closed-loop fine-tuning 롤아웃 캐시 구성에서는 valid_mask/gt_pos/gt_heading/gt_idx가 필요합니다.
            # 하지만 sampled_*는 (학습 타겟을 만들 때) 이미 사용했으므로 불필요하게 남겨두지 않기 위해 제거합니다.
            # closed-loop fine-tuning은 rollout 캐시 구성 시 `valid_mask/gt_pos/gt_heading/gt_idx`를 필요로 하므로
            # 이를 제거하지 않습니다.
            # (샘플링용 noisy state인 sampled_*는 flow_train target 구성에 이미 사용했으므로 정리합니다.)
            for key in ["sampled_idx", "sampled_pos", "sampled_heading"]:
                tokenized_agent.pop(key, None)
            return tokenized_agent

        # ======================
        # 평가(추론) 모드: flow_eval_* 타겟들을 anchor 단위로 구성
        # ======================
        flow_eval_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
        flow_eval_chunks: List[Tensor] = []
        flow_eval_agent_type_chunks: List[Tensor] = []
        flow_eval_current_control_chunks: List[Tensor] = []
        flow_eval_current_control_valid_chunks: List[Tensor] = []
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            # 평가 모드에서는 train_mask 필터링 없이, valid 조건만으로 anchor 사용 여부를 결정합니다.
            current_valid = valid[:, raw_step]
            future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
            anchor_mask = current_valid & future_valid
            flow_eval_mask[:, anchor_offset] = anchor_mask

            # 이 anchor에서 사용할 agent가 없다면 스킵
            if not anchor_mask.any():
                continue

            # 평가 모드에서도 동일한 방식으로 clean_norm / current_control을 생성합니다.
            anchor_clean_norm = self._build_anchor_clean_norm(
                pos=pos,
                heading=heading,
                current_pos=tokenized_agent["sampled_pos"][:, anchor_offset + 1],
                current_head=tokenized_agent["sampled_heading"][:, anchor_offset + 1],
                anchor_mask=anchor_mask,
                raw_step=raw_step,
            )
            anchor_current_control, anchor_current_control_valid = self._build_anchor_current_control(
                pos=pos,
                heading=heading,
                valid=valid,
                anchor_mask=anchor_mask,
                raw_step=raw_step,
            )

            # 평가 모드에서는 anchor_mask에 해당하는 agent들 전체를 concat 대상으로 넣습니다.
            flow_eval_chunks.append(anchor_clean_norm)
            flow_eval_agent_type_chunks.append(tokenized_agent["type"][anchor_mask])
            flow_eval_current_control_chunks.append(anchor_current_control)
            flow_eval_current_control_valid_chunks.append(anchor_current_control_valid)

        # 평가 모드에서도 anchor별 조각을 하나로 이어 붙여서 반환합니다.
        tokenized_agent.update(
            {
                "flow_eval_mask": flow_eval_mask,
                "flow_eval_clean_norm": self._concat_flow_chunks(
                    chunks=flow_eval_chunks,
                    dtype=dtype,
                    device=device,
                ),
                "flow_eval_agent_type": self._concat_metadata_chunks(
                    chunks=flow_eval_agent_type_chunks,
                    empty_shape=(0,),
                    dtype=tokenized_agent["type"].dtype,
                    device=device,
                ),
                "flow_eval_current_control": self._concat_metadata_chunks(
                    chunks=flow_eval_current_control_chunks,
                    empty_shape=(0, 3),
                    dtype=dtype,
                    device=device,
                ),
                "flow_eval_current_control_valid": self._concat_metadata_chunks(
                    chunks=flow_eval_current_control_valid_chunks,
                    empty_shape=(0,),
                    dtype=torch.bool,
                    device=device,
                ),
            }
        )
        return tokenized_agent

    def _build_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
    ) -> Tensor:
        """한 anchor에서 실제로 쓰는 에이전트만 골라 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                정규화된 2초 미래 목표입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
                마지막 차원은 ``[x, y, cos, sin]`` 순서입니다.
        """
        future_pos = pos[anchor_mask, raw_step + 1 : raw_step + 21]
        future_head = heading[anchor_mask, raw_step + 1 : raw_step + 21]
        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=current_pos[anchor_mask],
            head_now=current_head[anchor_mask],
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

    def _build_anchor_current_control(
        self,
        pos: Tensor,
        heading: Tensor,
        valid: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
    ) -> tuple[Tensor, Tensor]:
        """anchor 직전 0.1초 구간의 body-control을 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            valid: 시점별 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 쓰는 에이전트입니다. shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            tuple[Tensor, Tensor]:
                - ``current_control``: ``[vx_b, vy_b, omega]`` 입니다.
                  shape은 ``[n_valid_anchor, 3]`` 입니다.
                - ``current_control_valid``: 위 control을 믿을 수 있는지 나타냅니다.
                  shape은 ``[n_valid_anchor]`` 입니다.
        """
        dt = 0.1
        prev_pos = pos[:, raw_step - 1]
        curr_pos = pos[:, raw_step]
        prev_head = heading[:, raw_step - 1]
        curr_head = heading[:, raw_step]
        control_valid = valid[:, raw_step - 1] & valid[:, raw_step]

        delta_pos_world = (curr_pos - prev_pos) / dt
        delta_head = wrap_angle(curr_head - prev_head)
        head_mid = prev_head + 0.5 * delta_head
        cos_mid = head_mid.cos()
        sin_mid = head_mid.sin()

        vx_b = delta_pos_world[:, 0] * cos_mid + delta_pos_world[:, 1] * sin_mid
        vy_b = -delta_pos_world[:, 0] * sin_mid + delta_pos_world[:, 1] * cos_mid
        omega = delta_head / dt
        current_control = torch.stack([vx_b, vy_b, omega], dim=-1)
        current_control = current_control.masked_fill(~control_valid.unsqueeze(-1), 0.0)
        return current_control[anchor_mask], control_valid[anchor_mask]

    def _concat_flow_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
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
        if len(chunks) == 0:
            return torch.zeros((0, 20, 4), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    def _concat_metadata_chunks(
        self,
        chunks: List[Tensor],
        empty_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """anchor 단위 metadata 조각을 같은 순서로 이어 붙입니다.

        Args:
            chunks: anchor별 metadata 조각 목록입니다.
                각 원소의 첫 축은 ``n_valid_anchor`` 입니다.
            empty_shape: 빈 텐서를 만들 때 쓸 모양입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 metadata 입니다.
                shape은 ``empty_shape`` 또는 ``[n_total_valid_anchor, ...]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros(empty_shape, device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)
