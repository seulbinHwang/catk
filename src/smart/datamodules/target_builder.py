from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class WaymoTargetBuilderTrain(BaseTransform):
    """SMART-flow 학습에 필요한 agent train mask를 만든다."""

    def __init__(
        self,
        max_num: int,
        step_current: int = 10,
        flow_anchor_stride: int = 5,
        flow_num_anchors: int = 13,
        flow_num_future_steps: int = 20,
    ) -> None:
        super().__init__()
        self.step_current = step_current
        self.max_num = max_num
        self.flow_anchor_stride = flow_anchor_stride
        self.flow_num_anchors = flow_num_anchors
        self.flow_num_future_steps = flow_num_future_steps

    def _has_full_flow_anchor(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """2초 GT가 끝까지 있는 anchor가 하나라도 있는지 확인한다.

        Args:
            valid_mask: [n_agent, n_step] 모양의 유효 마스크이다.

        Returns:
            [n_agent] 모양의 bool 텐서를 돌려준다.
        """
        anchor_steps = self.step_current + torch.arange(
            self.flow_num_anchors,
            device=valid_mask.device,
        ) * self.flow_anchor_stride
        full_anchor = torch.zeros(valid_mask.shape[0], dtype=torch.bool, device=valid_mask.device)
        for step in anchor_steps.tolist():
            full_anchor = full_anchor | valid_mask[:, step : step + self.flow_num_future_steps + 1].all(dim=-1)
        return full_anchor

    def __call__(self, data) -> HeteroData:
        pos = data["agent"]["position"]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)
        role_train_mask = data["agent"]["role"].any(-1)
        has_full_anchor = self._has_full_flow_anchor(data["agent"]["valid_mask"])
        extra_train_mask = (distance[:, self.step_current] < 100) & has_full_anchor

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:
            indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = indices[
                torch.randperm(indices.size(0))[: self.max_num - role_train_mask.sum()]
            ]
            data["agent"]["train_mask"] = role_train_mask
            data["agent"]["train_mask"][selected_indices] = True
        else:
            data["agent"]["train_mask"] = train_mask
        return HeteroData(data)


class WaymoTargetBuilderVal(BaseTransform):
    """검증/테스트는 입력을 그대로 통과시킨다."""

    def __call__(self, data) -> HeteroData:
        return HeteroData(data)
