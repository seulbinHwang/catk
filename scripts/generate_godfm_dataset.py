"""GOD-FM offline dataset generation script.

Runs closed-loop rollout with a pretrained model to collect c_shift states,
then runs Teacher goal-guided inpainting to produce recovery trajectories.
Saves (anchor_hidden_c_shift, tau_target) pairs to disk as .pt files.

Usage:
    python -m scripts.generate_godfm_dataset \
        checkpoint=<path/to/checkpoint.ckpt> \
        output_dir=<path/to/output_dir> \
        n_rollout_collect=4 \
        goal_weight=5.0 \
        inpaint_steps=10 \
        +datamodule=<your_datamodule_config>
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.smart.model.smart_flow import SMARTFlow
from src.smart.modules.flow_godfm_inpainting import GoalGuidedODESampler
from src.smart.utils import transform_to_local, wrap_angle


def _move_to_device(obj, device: torch.device):
    """Recursively move tensors in nested containers to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _compute_goal_in_local_frame(
    gt_pos_raw: torch.Tensor,   # [n_agent, n_2hz, 2]
    gt_head_raw: torch.Tensor,  # [n_agent, n_2hz]
    gt_valid_raw: torch.Tensor, # [n_agent, n_2hz]
    active_mask: torch.Tensor,  # [n_agent]
    c_shift_pos: torch.Tensor,  # [n_active, 2]
    c_shift_head: torch.Tensor, # [n_active]
    goal_step: int,             # index into gt_pos_raw second dim
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return goal [n_active, 4] in c_shift local frame and its valid mask.

    The goal is the GT position and heading at goal_step, expressed in the
    local coordinate frame of the c_shift anchor (c_shift_pos, c_shift_head).

    Encoding: [x/20, y/20, cos(dhead), sin(dhead)]
    """
    n_2hz = gt_pos_raw.shape[1]
    goal_step = min(goal_step, n_2hz - 1)

    gt_pos_g = gt_pos_raw[active_mask, goal_step]     # [n_active, 2]
    gt_head_g = gt_head_raw[active_mask, goal_step]   # [n_active]
    goal_valid = gt_valid_raw[active_mask, goal_step] # [n_active]

    # Transform global GT endpoint → c_shift local frame
    gt_pos_g_exp = gt_pos_g.unsqueeze(1)  # [n_active, 1, 2]
    goal_pos_local, _ = transform_to_local(
        pos_global=gt_pos_g_exp,
        head_global=None,
        pos_now=c_shift_pos,
        head_now=c_shift_head,
    )
    goal_pos_local = goal_pos_local.squeeze(1)  # [n_active, 2]

    dhead = wrap_angle(gt_head_g - c_shift_head)  # [n_active]
    goal = torch.cat(
        [
            goal_pos_local / 20.0,
            dhead.cos().unsqueeze(-1),
            dhead.sin().unsqueeze(-1),
        ],
        dim=-1,
    )  # [n_active, 4]
    return goal, goal_valid


@torch.no_grad()
def generate_for_batch(
    model: SMARTFlow,
    data,
    sampler: GoalGuidedODESampler,
    n_rollout_collect: int,
    device: torch.device,
) -> list[dict]:
    """Process one batch and return a list of (anchor_hidden, tau_target) pairs."""
    data = data.to(device)
    model.token_processor.eval()
    tokenized_map, tokenized_agent = model.token_processor(data)

    # gt_pos_raw is available only in eval mode (set above)
    gt_pos_raw = tokenized_agent["gt_pos_raw"].to(device)    # [n_agent, n_2hz, 2]
    gt_head_raw = tokenized_agent["gt_head_raw"].to(device)  # [n_agent, n_2hz]
    gt_valid_raw = tokenized_agent["gt_valid_raw"].to(device) # [n_agent, n_2hz]

    # Move all tensors (including nested containers) to device.
    tokenized_map = _move_to_device(tokenized_map, device)
    tokenized_agent = _move_to_device(tokenized_agent, device)

    map_feature = model.encoder.encode_map(tokenized_map)
    rollout_cache = model.encoder.agent_encoder.prepare_inference_cache(
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
    )

    # step_current_2hz: how many 2Hz steps of GT context exist before rollout
    step_current_2hz = (model.num_historical_steps - 1) // 5  # shift = 5

    c_shift_states = model.encoder.agent_encoder.collect_godfm_c_shift_states(
        rollout_cache=rollout_cache,
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
        n_rollout_collect=n_rollout_collect,
    )

    flow_decoder = model.encoder.agent_encoder.flow_decoder
    flow_ode = model.encoder.agent_encoder.flow_ode

    pairs: list[dict] = []

    for state in c_shift_states:
        t = state["rollout_step"]
        active_mask = state["active_mask"]  # [n_agent] on CPU
        anchor_hidden_cpu = state["anchor_hidden"]     # [n_active, hidden_dim] CPU
        c_shift_pos_cpu = state["c_shift_pos"]         # [n_active, 2] CPU
        c_shift_head_cpu = state["c_shift_head"]       # [n_active] CPU

        # goal_step: GT endpoint 2s (4 coarse steps at 2Hz) ahead of current step
        goal_step = step_current_2hz + t + 4

        goal, goal_valid = _compute_goal_in_local_frame(
            gt_pos_raw=gt_pos_raw.cpu(),
            gt_head_raw=gt_head_raw.cpu(),
            gt_valid_raw=gt_valid_raw.cpu(),
            active_mask=active_mask,
            c_shift_pos=c_shift_pos_cpu,
            c_shift_head=c_shift_head_cpu,
            goal_step=goal_step,
        )

        # Only keep agents where the goal timestep is valid
        if not goal_valid.any():
            continue

        anchor_hidden_dev = anchor_hidden_cpu[goal_valid].to(device)
        goal_dev = goal[goal_valid].to(device)

        tau_target = sampler.sample(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden=anchor_hidden_dev,
            goal=goal_dev,
        )

        pairs.append({
            "anchor_hidden": anchor_hidden_dev.cpu(),
            "tau_target": tau_target.cpu(),
        })

    return pairs


@hydra.main(config_path="../configs", config_name="generate_godfm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model: SMARTFlow = SMARTFlow.load_from_checkpoint(cfg.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # Build datamodule (uses the same cfg.data structure as src/run.py)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loader: DataLoader = datamodule.train_dataloader()

    sampler = GoalGuidedODESampler(
        inpaint_steps=int(cfg.get("inpaint_steps", 10)),
        goal_weight=float(cfg.get("goal_weight", 5.0)),
    )

    n_rollout_collect = int(cfg.get("n_rollout_collect", 4))

    all_anchor_hiddens: list[torch.Tensor] = []
    all_tau_targets: list[torch.Tensor] = []
    chunk_idx = 0
    chunk_size = int(cfg.get("chunk_size", 5000))

    for batch_idx, data in enumerate(tqdm(loader, desc="Collecting GOD-FM pairs")):
        try:
            pairs = generate_for_batch(
                model=model,
                data=data,
                sampler=sampler,
                n_rollout_collect=n_rollout_collect,
                device=device,
            )
        except Exception as e:
            print(f"[batch {batch_idx}] skipped: {e}")
            if os.environ.get("GODFM_DEBUG_TRACE", "0") == "1":
                traceback.print_exc()
            continue

        for pair in pairs:
            all_anchor_hiddens.append(pair["anchor_hidden"])
            all_tau_targets.append(pair["tau_target"])

        if len(all_anchor_hiddens) >= chunk_size:
            chunk_path = output_dir / f"pairs_{chunk_idx:05d}.pt"
            torch.save(
                {
                    "anchor_hidden": torch.cat(all_anchor_hiddens, dim=0),
                    "tau_target": torch.cat(all_tau_targets, dim=0),
                },
                chunk_path,
            )
            print(f"Saved chunk {chunk_idx} → {chunk_path} "
                  f"({len(all_anchor_hiddens)} pairs)")
            all_anchor_hiddens = []
            all_tau_targets = []
            chunk_idx += 1

    # Save remaining pairs
    if all_anchor_hiddens:
        chunk_path = output_dir / f"pairs_{chunk_idx:05d}.pt"
        torch.save(
            {
                "anchor_hidden": torch.cat(all_anchor_hiddens, dim=0),
                "tau_target": torch.cat(all_tau_targets, dim=0),
            },
            chunk_path,
        )
        print(f"Saved final chunk {chunk_idx} → {chunk_path} "
              f"({len(all_anchor_hiddens)} pairs)")

    print(f"Done. Total chunks: {chunk_idx + 1}")


if __name__ == "__main__":
    main()
