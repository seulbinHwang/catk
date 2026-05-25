from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf, open_dict
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from src.smart.modules.kinematic_control import POSE_NORM_POS_SCALE_M  # noqa: E402
from src.smart.modules.self_forced_gan_cache import _sanitize_scenario_id  # noqa: E402
from src.smart.utils import transform_to_global  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline teacher open-loop rollout cache for self-forced GAN fine-tuning. "
            "Each scene is saved as one .pt file containing rollout_pose [R,20,N,4]."
        )
    )
    parser.add_argument("--ckpt-path", required=True, type=Path, help="Pretrained teacher Lightning checkpoint.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory where scene .pt cache files are saved.")
    parser.add_argument("--split", default="train", choices=("train", "val", "validation", "test"))
    parser.add_argument("--rollouts-per-scene", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=None, help="Debug limit. Omit to build the full split.")
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for teacher sampling.",
    )
    parser.add_argument(
        "--storage-dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Dtype used when writing rollout_pose to disk.",
    )
    parser.add_argument(
        "--amp-dtype",
        default="none",
        choices=("none", "float16", "bfloat16"),
        help="Optional CUDA autocast dtype used during teacher sampling.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Do not rebuild scene files that already exist.")
    parser.add_argument("--config-name", default="run", help="Hydra config name under configs/.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Extra Hydra override. Repeat this flag, e.g. "
            "--override paths.cache_root=$CACHE_ROOT --override data.train_raw_dir=/path/training"
        ),
    )
    return parser.parse_args()


def _default_overrides() -> list[str]:
    return [
        "experiment=self_forced_gan_h100_6",
        "action=validate",
        "model.model_config.self_forced.enabled=false",
        "model.model_config.self_forced_gan.enabled=false",
        "model.model_config.val_open_loop=false",
        "model.model_config.val_closed_loop=false",
        "model.model_config.n_vis_batch=0",
        "model.model_config.n_vis_scenario=0",
        "model.model_config.n_vis_rollout=0",
        "data.train_batch_size=1",
        "data.val_batch_size=1",
        "data.test_batch_size=1",
        "data.shuffle=false",
        "data.num_workers=0",
        "data.prefetch_factor=null",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        "data.train_use_eval_agent_selection=true",
        "data.train_epoch_sample_fraction=1.0",
        "data.train_memory_balanced_batches=false",
        "trainer.devices=1",
        "trainer.accelerator=cpu",
    ]


def _compose_config(args: argparse.Namespace):
    overrides = _default_overrides() + list(args.override)
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides, return_hydra_config=True)
    with open_dict(cfg):
        cfg.hydra.runtime.cwd = str(REPO_ROOT)
        cfg.hydra.runtime.output_dir = str(REPO_ROOT / "outputs" / "teacher_cache_builder")
    if not HydraConfig.initialized():
        HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    return cfg


def _load_teacher_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint must be a Lightning checkpoint with state_dict: {ckpt_path}")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint state_dict must be a mapping: {ckpt_path}")
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(
            "[teacher_cache] missing checkpoint keys ignored: "
            + ", ".join(load_result.missing_keys[:20])
            + (" ..." if len(load_result.missing_keys) > 20 else ""),
            flush=True,
        )
    if load_result.unexpected_keys:
        print(
            "[teacher_cache] unexpected checkpoint keys ignored: "
            + ", ".join(load_result.unexpected_keys[:20])
            + (" ..." if len(load_result.unexpected_keys) > 20 else ""),
            flush=True,
        )


def _select_dataloader(datamodule: Any, split: str):
    normalized = "val" if split == "validation" else split
    stage = {"train": "fit", "val": "validate", "test": "test"}[normalized]
    datamodule.setup(stage)
    if normalized == "train":
        return datamodule.train_dataloader()
    if normalized == "val":
        return datamodule.val_dataloader()
    return datamodule.test_dataloader()


def _to_device(batch: Any, device: torch.device) -> Any:
    to_method = getattr(batch, "to", None)
    if callable(to_method):
        return to_method(device)
    raise TypeError(f"Batch object does not support .to(device): {type(batch).__name__}")


def _scenario_id_list(raw: Any) -> list[str]:
    if isinstance(raw, (str, bytes)):
        return [str(raw)]
    if isinstance(raw, Iterable):
        return [str(item) for item in raw]
    return [str(raw)]


def _stable_seed(base_seed: int, scenario_key: str, rollout_index: int) -> int:
    payload = f"{int(base_seed)}:{scenario_key}:{int(rollout_index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _storage_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_current_anchor_mask(data: Any, tokenized_agent: dict[str, Tensor], num_historical_steps: int) -> Tensor:
    if "ctx_valid" in tokenized_agent:
        ctx_valid = tokenized_agent["ctx_valid"]
        if ctx_valid.dim() == 2 and ctx_valid.shape[1] > 1:
            return ctx_valid[:, 1].bool()
    return data["agent"]["valid_mask"][:, int(num_historical_steps) - 1].bool()


def _build_anchor0_context(
    model: Any,
    tokenized_agent: dict[str, Tensor],
    map_feature: dict[str, Tensor],
    anchor_mask: Tensor,
) -> dict[str, Tensor]:
    n_valid = int(anchor_mask.sum().item())
    flow_state_dim = int(model.encoder.agent_encoder.flow_state_dim)
    flow_window_steps = int(model.flow_window_steps)
    dtype = map_feature["pt_token"].dtype
    device = anchor_mask.device
    dummy_clean = torch.zeros(
        (n_valid, flow_window_steps, flow_state_dim),
        device=device,
        dtype=dtype,
    )
    return model.encoder.agent_encoder.build_anchor_context(
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
        anchor_mask=anchor_mask.view(-1, 1),
        flow_clean_norm=dummy_clean,
        flow_agent_type=tokenized_agent.get("type", None)[anchor_mask] if "type" in tokenized_agent else None,
        flow_agent_length=(
            tokenized_agent["shape"][anchor_mask, 0]
            if "shape" in tokenized_agent
            else None
        ),
    )


def _sample_teacher_rollout_pose(
    *,
    model: Any,
    anchor_context: dict[str, Tensor],
    tokenized_agent: dict[str, Tensor],
    anchor_mask: Tensor,
    sampling_seed: int,
) -> Tensor:
    n_agent = int(anchor_mask.shape[0])
    flow_window_steps = int(model.flow_window_steps)
    device = anchor_mask.device
    dtype = anchor_context["anchor_hidden"].dtype
    pose = torch.zeros((n_agent, flow_window_steps, 4), device=device, dtype=dtype)
    if not bool(anchor_mask.any()):
        return pose

    sample_norm = model.encoder.sample_open_loop_future(
        anchor_hidden=anchor_context["anchor_hidden"],
        anchor_mask=anchor_context["anchor_mask"],
        sampling_scheme=model.validation_rollout_sampling,
        sampling_seed=int(sampling_seed),
        backprop_last_k=None,
    )
    agent_type = tokenized_agent["type"][anchor_mask]
    agent_length = tokenized_agent["shape"][anchor_mask, 0] if "shape" in tokenized_agent else None
    sample_pose_norm = model.encoder.flow_norm_to_pose_metric_norm(
        value=sample_norm,
        agent_type=agent_type,
        agent_length=agent_length,
    )
    pos_local = sample_pose_norm[..., :2] * float(POSE_NORM_POS_SCALE_M)
    head_local = torch.atan2(sample_pose_norm[..., 3], sample_pose_norm[..., 2])
    current_pos = tokenized_agent["ctx_sampled_pos"][anchor_mask, 1]
    current_head = tokenized_agent["ctx_sampled_heading"][anchor_mask, 1]
    pos_global, head_global = transform_to_global(
        pos_local=pos_local,
        head_local=head_local,
        pos_now=current_pos,
        head_now=current_head,
    )
    pose_valid = torch.cat(
        [
            pos_global,
            torch.cos(head_global).unsqueeze(-1),
            torch.sin(head_global).unsqueeze(-1),
        ],
        dim=-1,
    )
    pose[anchor_mask] = pose_valid
    return pose


@torch.no_grad()
def _generate_batch_rollout_set(
    *,
    model: Any,
    data: Any,
    rollouts_per_scene: int,
    base_seed: int,
    amp_dtype: str,
) -> tuple[Tensor, Tensor]:
    tokenized_map, tokenized_agent = model._build_eval_tokenized_inputs(data)
    map_feature = model.encoder.encode_map(tokenized_map)
    anchor_mask = _build_current_anchor_mask(
        data=data,
        tokenized_agent=tokenized_agent,
        num_historical_steps=int(model.num_historical_steps),
    )
    anchor_context = _build_anchor0_context(
        model=model,
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
        anchor_mask=anchor_mask,
    )
    scenario_key = "|".join(_scenario_id_list(data["scenario_id"]))
    rollouts: list[Tensor] = []
    device = next(model.parameters()).device
    with _amp_context(device, amp_dtype):
        for rollout_index in range(int(rollouts_per_scene)):
            seed = _stable_seed(base_seed, scenario_key, rollout_index)
            rollouts.append(
                _sample_teacher_rollout_pose(
                    model=model,
                    anchor_context=anchor_context,
                    tokenized_agent=tokenized_agent,
                    anchor_mask=anchor_mask,
                    sampling_seed=seed,
                )
            )
    return torch.stack(rollouts, dim=0).contiguous(), anchor_mask


def _read_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"Existing index must be a dict: {index_path}")
    return {
        str(key): str(value["path"] if isinstance(value, dict) and "path" in value else value)
        for key, value in raw.items()
    }


def _write_index(index_path: Path, index: dict[str, str]) -> None:
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2, sort_keys=True)
        fp.write("\n")
    os.replace(tmp_path, index_path)


def _save_scene_cache(
    *,
    output_root: Path,
    index: dict[str, str],
    scenario_id: str,
    rollout_set: Tensor,
    data: Any,
    anchor_mask: Tensor,
    scene_index: int,
    storage_dtype: torch.dtype,
    base_seed: int,
    rollouts_per_scene: int,
    overwrite: bool,
) -> bool:
    file_name = _sanitize_scenario_id(scenario_id) + ".pt"
    output_path = output_root / file_name
    if output_path.exists() and not overwrite:
        index[str(scenario_id)] = file_name
        return False

    agent_batch = data["agent"]["batch"]
    scene_mask = agent_batch == int(scene_index)
    scene_rollout = rollout_set[:, scene_mask].permute(0, 2, 1, 3).contiguous()
    scene_valid = anchor_mask[scene_mask].detach().cpu()
    payload = {
        "rollout_pose": scene_rollout.detach().cpu().to(dtype=storage_dtype),
        "agent_id": data["agent"]["id"][scene_mask].detach().cpu().long(),
        "agent_type": data["agent"]["type"][scene_mask].detach().cpu().long(),
        "valid_mask": scene_valid.bool(),
        "seed": torch.tensor(
            [_stable_seed(base_seed, str(scenario_id), rollout_idx) for rollout_idx in range(rollouts_per_scene)],
            dtype=torch.long,
        ),
        "scenario_id_hash": torch.tensor(
            [_stable_seed(0, str(scenario_id), 0)],
            dtype=torch.long,
        ),
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    index[str(scenario_id)] = file_name
    return True


def main() -> None:
    args = _parse_args()
    if args.rollouts_per_scene <= 0:
        raise ValueError("--rollouts-per-scene must be positive.")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    index = _read_index(index_path)

    cfg = _compose_config(args)
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model, _recursive_=False)
    _load_teacher_checkpoint(model, args.ckpt_path.expanduser().resolve())

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    dataloader = _select_dataloader(datamodule, args.split)

    storage_dtype = _storage_dtype(args.storage_dtype)
    processed = 0
    written = 0
    started = time.monotonic()
    for batch_index, data in enumerate(dataloader):
        data = _to_device(data, device)
        scenario_ids = _scenario_id_list(data["scenario_id"])
        if len(scenario_ids) != 1:
            raise ValueError(
                "Teacher cache builder expects batch size 1 so each scene has independent "
                f"sampling seeds. Got {len(scenario_ids)} scenes. Do not override data.*_batch_size."
            )
        if args.skip_existing and all((output_root / (_sanitize_scenario_id(sid) + ".pt")).exists() for sid in scenario_ids):
            for sid in scenario_ids:
                index[str(sid)] = _sanitize_scenario_id(sid) + ".pt"
            processed += len(scenario_ids)
            if args.max_scenes is not None and processed >= args.max_scenes:
                break
            continue

        rollout_set, anchor_mask = _generate_batch_rollout_set(
            model=model,
            data=data,
            rollouts_per_scene=int(args.rollouts_per_scene),
            base_seed=int(args.seed),
            amp_dtype=str(args.amp_dtype),
        )
        for scene_index, scenario_id in enumerate(scenario_ids):
            if args.max_scenes is not None and processed >= args.max_scenes:
                break
            did_write = _save_scene_cache(
                output_root=output_root,
                index=index,
                scenario_id=str(scenario_id),
                rollout_set=rollout_set,
                data=data,
                anchor_mask=anchor_mask,
                scene_index=scene_index,
                storage_dtype=storage_dtype,
                base_seed=int(args.seed),
                rollouts_per_scene=int(args.rollouts_per_scene),
                overwrite=not args.skip_existing,
            )
            written += int(did_write)
            processed += 1
        if batch_index == 0 or processed % 100 == 0:
            elapsed = time.monotonic() - started
            print(
                f"[teacher_cache] processed={processed} written={written} "
                f"elapsed_sec={elapsed:.1f} output_root={output_root}",
                flush=True,
            )
        if args.max_scenes is not None and processed >= args.max_scenes:
            break

    _write_index(index_path, index)
    elapsed = time.monotonic() - started
    print(
        f"[teacher_cache] done processed={processed} written={written} "
        f"index={index_path} elapsed_sec={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
