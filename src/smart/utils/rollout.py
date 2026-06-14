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

from typing import Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


def _sample_categorical_logits(
    logits: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if generator is None:
        return Categorical(logits=logits).sample()

    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(
        probs,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(-1)


def _sample_categorical_logits_k(
    logits: Tensor,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if generator is None:
        return (
            Categorical(logits=logits)
            .sample((num_samples,))
            .transpose(0, 1)
            .contiguous()
        )

    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(
        probs,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def _sample_categorical_logits_by_batch(
    logits: Tensor,
    batch_index: Optional[Tensor] = None,
    generators_by_batch: Optional[Sequence[torch.Generator]] = None,
) -> Tensor:
    if generators_by_batch is None:
        return _sample_categorical_logits(logits)
    if batch_index is None:
        raise ValueError("batch_index is required when generators_by_batch is provided.")
    if tuple(batch_index.shape) != (logits.shape[0],):
        raise ValueError(
            "batch_index must have one entry per categorical row, "
            f"got {tuple(batch_index.shape)} for logits shape {tuple(logits.shape)}."
        )

    samples = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    if batch_index.numel() == 0:
        return samples
    min_batch = int(batch_index.min().item())
    if min_batch < 0:
        raise ValueError(f"batch_index must be non-negative, got min batch {min_batch}.")
    max_batch = int(batch_index.max().item())
    if max_batch >= len(generators_by_batch):
        raise ValueError(
            "generators_by_batch must cover every batch id, "
            f"got max batch {max_batch} and {len(generators_by_batch)} generators."
        )

    for batch_id, generator in enumerate(generators_by_batch):
        mask = batch_index == batch_id
        if not bool(mask.any()):
            continue
        samples[mask] = _sample_categorical_logits(
            logits=logits[mask],
            generator=generator,
        )
    return samples


def _sample_categorical_logits_k_by_batch(
    logits: Tensor,
    num_samples: int,
    batch_index: Optional[Tensor] = None,
    generators_by_batch: Optional[Sequence[torch.Generator]] = None,
) -> Tensor:
    if generators_by_batch is None:
        return _sample_categorical_logits_k(logits, num_samples=num_samples)
    if batch_index is None:
        raise ValueError("batch_index is required when generators_by_batch is provided.")
    if tuple(batch_index.shape) != (logits.shape[0],):
        raise ValueError(
            "batch_index must have one entry per categorical row, "
            f"got {tuple(batch_index.shape)} for logits shape {tuple(logits.shape)}."
        )

    samples = torch.empty(
        (logits.shape[0], num_samples),
        dtype=torch.long,
        device=logits.device,
    )
    if batch_index.numel() == 0:
        return samples
    min_batch = int(batch_index.min().item())
    if min_batch < 0:
        raise ValueError(f"batch_index must be non-negative, got min batch {min_batch}.")
    max_batch = int(batch_index.max().item())
    if max_batch >= len(generators_by_batch):
        raise ValueError(
            "generators_by_batch must cover every batch id, "
            f"got max batch {max_batch} and {len(generators_by_batch)} generators."
        )

    for batch_id, generator in enumerate(generators_by_batch):
        mask = batch_index == batch_id
        if not bool(mask.any()):
            continue
        samples[mask] = _sample_categorical_logits_k(
            logits=logits[mask],
            num_samples=num_samples,
            generator=generator,
        )
    return samples


def _clamp_num_k_to_candidate_count(num_k: int, n_candidates: int) -> int:
    if num_k <= 0:
        raise ValueError(f"num_k should be positive for rollout sampling, got {num_k}")
    if n_candidates <= 0:
        raise ValueError(f"rollout sampling requires at least one candidate, got {n_candidates}")
    return min(int(num_k), int(n_candidates))


@torch.no_grad()
def cal_polygon_contour(
    pos: Tensor,
    head: Tensor,
    width_length: Tensor,
) -> Tensor:
    x, y = pos[..., 0], pos[..., 1]
    width, length = width_length[..., 0], width_length[..., 1]

    half_cos = 0.5 * head.cos()
    half_sin = 0.5 * head.sin()
    length_cos = length * half_cos
    length_sin = length * half_sin
    width_cos = width * half_cos
    width_sin = width * half_sin

    left_front = torch.stack((x + length_cos - width_sin, y + length_sin + width_cos), dim=-1)
    right_front = torch.stack((x + length_cos + width_sin, y + length_sin - width_cos), dim=-1)
    right_back = torch.stack((x - length_cos + width_sin, y - length_sin - width_cos), dim=-1)
    left_back = torch.stack((x - length_cos - width_sin, y - length_sin + width_cos), dim=-1)
    return torch.stack((left_front, right_front, right_back, left_back), dim=-2)


def transform_to_global(
    pos_local: Tensor,
    head_local: Optional[Tensor],
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    pos_global = torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)
    if head_local is None:
        head_global = None
    else:
        head_global = head_local + head_now.unsqueeze(1)
    return pos_global, head_global


def transform_to_local(
    pos_global: Tensor,
    head_global: Optional[Tensor],
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    pos_local = torch.bmm(pos_global - pos_now.unsqueeze(1), rot_mat)
    if head_global is None:
        head_local = None
    else:
        head_local = head_global - head_now.unsqueeze(1)
    return pos_local, head_local


def sample_policy_token_candidates(
    next_token_logits: Tensor,
    num_k: int,
    temperature: float,
    sampling_generators_by_batch: Optional[Sequence[torch.Generator]] = None,
    sampling_batch: Optional[Tensor] = None,
) -> Tensor:
    """RoaD 방식으로 모델 확률에서 후보 token을 뽑는다.

    Args:
        next_token_logits: 다음 token 점수이다. Shape은 ``[n_agent, n_token]``이다.
        num_k: agent마다 뽑을 후보 개수이다.
        temperature: 후보를 뽑을 때 확률 분포를 얼마나 넓게 볼지 정하는 값이다.

    Returns:
        agent마다 독립적으로 뽑은 후보 token 번호이다. Shape은 ``[n_agent, num_k]``이다.
    """
    num_k = _clamp_num_k_to_candidate_count(num_k, next_token_logits.shape[-1])
    if temperature <= 0:
        raise ValueError(f"temperature should be positive for RoaD sampling, got {temperature}")
    sampling_logits = next_token_logits.detach() / temperature
    return _sample_categorical_logits_k_by_batch(
        logits=sampling_logits,
        num_samples=num_k,
        batch_index=sampling_batch,
        generators_by_batch=sampling_generators_by_batch,
    )


def select_road_samplek_candidate(
    token_traj: Tensor,
    candidate_indices: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
    pos_next_gt: Tensor,
    head_next_gt: Tensor,
    valid_next_gt: Tensor,
    token_agent_shape: Tensor,
) -> Tensor:
    """Sample-K 후보 중 expert 다음 상태에 가장 가까운 token을 고른다.

    Args:
        token_traj: agent별 token contour이다. Shape은 ``[n_agent, n_token, 4, 2]``이다.
        candidate_indices: 후보 token 번호이다. Shape은 ``[n_agent, num_k]``이다.
        pos_now: 현재 위치이다. Shape은 ``[n_agent, 2]``이다.
        head_now: 현재 방향이다. Shape은 ``[n_agent]``이다.
        pos_next_gt: expert의 다음 위치이다. Shape은 ``[n_agent, 2]``이다.
        head_next_gt: expert의 다음 방향이다. Shape은 ``[n_agent]``이다.
        valid_next_gt: expert 다음 상태의 유효 mask이다. Shape은 ``[n_agent]``이다.
        token_agent_shape: agent별 폭과 길이이다. Shape은 ``[n_agent, 2]``이다.

    Returns:
        RoaD가 실제 실행할 token 번호이다. Shape은 ``[n_agent]``이다.
    """
    n_agent = token_traj.shape[0]
    range_a = torch.arange(n_agent, device=token_traj.device)
    candidate_contour_local = token_traj[range_a.unsqueeze(1), candidate_indices]
    candidate_contour_global = transform_to_global(
        pos_local=candidate_contour_local.flatten(1, 2),
        head_local=None,
        pos_now=pos_now,
        head_now=head_now,
    )[0].view(*candidate_contour_local.shape)

    gt_contour = cal_polygon_contour(pos_next_gt, head_next_gt, token_agent_shape).unsqueeze(1)
    distance = torch.norm(candidate_contour_global - gt_contour, dim=-1).mean(-1)
    closest_sample_idx = distance.argmin(dim=-1)
    selected_indices = candidate_indices[range_a, closest_sample_idx]
    return torch.where(valid_next_gt, selected_indices, candidate_indices[:, 0])


def sample_next_token_traj(
    token_traj: Tensor,
    token_traj_all: Tensor,
    sampling_scheme: DictConfig,
    next_token_logits: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
    pos_next_gt: Tensor,
    head_next_gt: Tensor,
    valid_next_gt: Tensor,
    token_agent_shape: Tensor,
    sampling_generators_by_batch: Optional[Sequence[torch.Generator]] = None,
    sampling_batch: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
        next_token_traj_all: [n_agent, 6, 4, 2], local coord
        next_token_idx: [n_agent], without grad
    """
    device = next_token_logits.device
    range_a = torch.arange(next_token_logits.shape[0], device=device)
    next_token_logits = next_token_logits.detach()
    num_k = _clamp_num_k_to_candidate_count(sampling_scheme.num_k, next_token_logits.shape[-1])

    if sampling_scheme.criterium == "road_samplek_dist":
        candidate_indices = sample_policy_token_candidates(
            next_token_logits=next_token_logits,
            num_k=num_k,
            temperature=sampling_scheme.temp,
            sampling_generators_by_batch=sampling_generators_by_batch,
            sampling_batch=sampling_batch,
        )
        next_token_idx = select_road_samplek_candidate(
            token_traj=token_traj,
            candidate_indices=candidate_indices,
            pos_now=pos_now,
            head_now=head_now,
            pos_next_gt=pos_next_gt,
            head_next_gt=head_next_gt,
            valid_next_gt=valid_next_gt,
            token_agent_shape=token_agent_shape,
        )
        return next_token_idx, token_traj_all[range_a, next_token_idx]

    if (
        sampling_scheme.criterium == "topk_prob"
        or sampling_scheme.criterium == "topk_prob_sampled_with_dist"
    ):
        topk_logits, topk_indices = torch.topk(
            next_token_logits, num_k, dim=-1, sorted=False
        )
        if sampling_scheme.criterium == "topk_prob_sampled_with_dist":
            gt_contour = cal_polygon_contour(pos_next_gt, head_next_gt, token_agent_shape).unsqueeze(1)
            token_world_sample = token_traj[range_a.unsqueeze(1), topk_indices]
            token_world_sample = transform_to_global(
                pos_local=token_world_sample.flatten(1, 2),
                head_local=None,
                pos_now=pos_now,
                head_now=head_now,
            )[0].view(*token_world_sample.shape)
            dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
            topk_logits = topk_logits.masked_fill(valid_next_gt.unsqueeze(1), 0.0) - dist.masked_fill(
                ~valid_next_gt.unsqueeze(1), 0.0
            )
    elif sampling_scheme.criterium == "topk_dist_sampled_with_prob":
        gt_contour = cal_polygon_contour(pos_next_gt, head_next_gt, token_agent_shape).unsqueeze(1)
        token_world_sample = transform_to_global(
            pos_local=token_traj.flatten(1, 2),
            head_local=None,
            pos_now=pos_now,
            head_now=head_now,
        )[0].view(*token_traj.shape)
        _invalid = ~valid_next_gt
        dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
        _logits = -1.0 * dist.masked_fill(_invalid.unsqueeze(1), 0.0)
        if _invalid.any():
            _logits[_invalid] = next_token_logits[_invalid]
        _, topk_indices = torch.topk(_logits, num_k, dim=-1, sorted=False)
        topk_logits = next_token_logits[range_a.unsqueeze(1), topk_indices]
    else:
        raise ValueError(f"Invalid criterium: {sampling_scheme.criterium}")

    topk_logits = topk_logits / sampling_scheme.temp
    samples = _sample_categorical_logits_by_batch(
        logits=topk_logits,
        batch_index=sampling_batch,
        generators_by_batch=sampling_generators_by_batch,
    )
    next_token_idx = topk_indices[range_a, samples]
    return next_token_idx, token_traj_all[range_a, next_token_idx]


def sample_next_gmm_traj(
    token_traj: Tensor,
    token_traj_all: Tensor,
    sampling_scheme: DictConfig,
    ego_mask: Tensor,
    ego_next_logits: Tensor,
    ego_next_poses: Tensor,
    ego_next_cov: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
    pos_next_gt: Tensor,
    head_next_gt: Tensor,
    valid_next_gt: Tensor,
    token_agent_shape: Tensor,
    next_token_idx: Tensor,
) -> Tuple[Tensor, Tensor]:
    n_agent = token_traj.shape[0]
    n_batch = ego_next_logits.shape[0]
    next_token_traj_all = token_traj_all[
        torch.arange(n_agent, device=token_traj.device), next_token_idx
    ]

    assert sampling_scheme.criterium in {"topk_prob", "topk_prob_sampled_with_dist"}
    num_k = _clamp_num_k_to_candidate_count(sampling_scheme.num_k, ego_next_logits.shape[-1])
    topk_logits, topk_indices = torch.topk(
        ego_next_logits, num_k, dim=-1, sorted=False
    )
    ego_pose_topk = ego_next_poses[
        torch.arange(n_batch, device=ego_next_logits.device).unsqueeze(1), topk_indices
    ]

    if sampling_scheme.criterium == "topk_prob_sampled_with_dist":
        gt_contour = cal_polygon_contour(
            pos_next_gt[ego_mask], head_next_gt[ego_mask], token_agent_shape[ego_mask]
        ).unsqueeze(1)
        ego_pos_global, ego_head_global = transform_to_global(
            pos_local=ego_pose_topk[:, :, :2],
            head_local=ego_pose_topk[:, :, -1],
            pos_now=pos_now[ego_mask],
            head_now=head_now[ego_mask],
        )
        ego_contour = cal_polygon_contour(
            ego_pos_global, ego_head_global, token_agent_shape[ego_mask].unsqueeze(1)
        )
        dist = torch.norm(ego_contour - gt_contour, dim=-1).mean(-1)
        topk_logits = topk_logits.masked_fill(valid_next_gt[ego_mask].unsqueeze(1), 0.0) - dist.masked_fill(
            ~valid_next_gt[ego_mask].unsqueeze(1), 0.0
        )

    topk_logits = topk_logits / sampling_scheme.temp_mode
    ego_pose_topk = torch.cat(
        [ego_pose_topk[..., :2], ego_pose_topk[..., [-1]].cos(), ego_pose_topk[..., [-1]].sin()],
        dim=-1,
    )
    cov = (ego_next_cov * sampling_scheme.temp_cov).repeat_interleave(2)[None, None, :].expand(
        *ego_pose_topk.shape
    )
    gmm = MixtureSameFamily(
        Categorical(logits=topk_logits), Independent(Normal(ego_pose_topk, cov), 1)
    )
    ego_sample = gmm.sample()

    ego_contour_local = cal_polygon_contour(
        ego_sample[:, :2], torch.arctan2(ego_sample[:, -1], ego_sample[:, -2]), token_agent_shape[ego_mask]
    )
    ego_token_local = token_traj[ego_mask]
    dist = torch.norm(ego_contour_local.unsqueeze(1) - ego_token_local, dim=-1).mean(-1)
    next_token_idx[ego_mask] = dist.argmin(-1)

    ego_countour_start = next_token_traj_all[ego_mask][:, 0]
    n_step = next_token_traj_all.shape[1]
    diff = (ego_contour_local - ego_countour_start) / (n_step - 1)
    ego_token_interp = [ego_countour_start + diff * i for i in range(n_step)]
    next_token_traj_all[ego_mask] = torch.stack(ego_token_interp, dim=1)
    return next_token_idx, next_token_traj_all
