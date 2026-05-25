from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
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

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "teacher_cache_manifest.json"
EXPECTED_MANIFEST_NAME = "teacher_cache_manifest.expected.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline teacher open-loop rollout cache for self-forced GAN fine-tuning. "
            "Each scene is saved as one .pt file containing rollout_pose [R,20,N,4]."
        )
    )
    parser.add_argument("--ckpt-path", type=Path, help="Pretrained teacher Lightning checkpoint.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory where scene .pt cache files are saved.")
    parser.add_argument("--split", default="train", choices=("train", "val", "validation", "test"))
    parser.add_argument("--rollouts-per-scene", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=None, help="Debug limit. Omit to build the full split.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of scenes to encode together on one GPU worker.",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=None,
        help=(
            "Number of teacher rollouts to sample in one flow call. "
            "Defaults to --rollouts-per-scene."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of cache builder shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="This builder shard index.")
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
    parser.add_argument(
        "--check-manifest",
        action="store_true",
        help="Exit successfully only when teacher_cache_manifest.json matches this build request.",
    )
    parser.add_argument(
        "--merge-shard-indexes",
        action="store_true",
        help="Only merge index.shard_* files under --output-root into index.json.",
    )
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _repo_metadata() -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_file(ckpt_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint must be a Lightning checkpoint with state_dict: {ckpt_path}")
    if not isinstance(checkpoint["state_dict"], dict):
        raise ValueError(f"Checkpoint state_dict must be a mapping: {ckpt_path}")
    return checkpoint


def _checkpoint_metadata(ckpt_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    marker_path = ckpt_path.with_suffix(ckpt_path.suffix + ".wandb.json")
    marker: dict[str, Any] = {}
    if marker_path.exists():
        try:
            raw_marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if isinstance(raw_marker, dict):
                marker = raw_marker
        except Exception:
            marker = {}
    return {
        "path": str(ckpt_path),
        "sha256": _sha256_file(ckpt_path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "wandb": marker,
    }


def _dataset_scene_count(dataloader: Any) -> int | None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except Exception:
        return None


def _cache_identity(
    *,
    args: argparse.Namespace,
    checkpoint_meta: dict[str, Any],
    split_scene_count: int | None,
    flow_window_steps: int,
) -> dict[str, Any]:
    wandb_meta = checkpoint_meta.get("wandb") or {}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint": {
            "sha256": checkpoint_meta.get("sha256"),
            "epoch": checkpoint_meta.get("epoch"),
            "global_step": checkpoint_meta.get("global_step"),
            "artifact": wandb_meta.get("artifact"),
            "version": wandb_meta.get("version"),
            "qualified_artifact": wandb_meta.get("qualified_artifact"),
        },
        "cache": {
            "split": "val" if args.split == "validation" else str(args.split),
            "rollouts_per_scene": int(args.rollouts_per_scene),
            "seed": int(args.seed),
            "storage_dtype": str(args.storage_dtype),
            "flow_window_steps": int(flow_window_steps),
            "max_scenes": int(args.max_scenes) if args.max_scenes is not None else None,
        },
        "dataset": {
            "split_scene_count": split_scene_count,
        },
    }


def _manifest_payload(
    *,
    status: str,
    identity: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    args: argparse.Namespace,
    split_scene_count: int | None,
    processed: int | None = None,
    written: int | None = None,
    index_entries: int | None = None,
    pt_file_count: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "cache_identity": identity,
        "checkpoint": checkpoint_meta,
        "repo": _repo_metadata(),
        "builder": {
            "script": "tools/build_self_forced_gan_teacher_cache.py",
            "batch_size": int(args.batch_size),
            "rollout_batch_size": int(args.rollout_batch_size or args.rollouts_per_scene),
            "amp_dtype": str(args.amp_dtype),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "dataset": {
            "split": "val" if args.split == "validation" else str(args.split),
            "split_scene_count": split_scene_count,
        },
        "files": {
            "expected_file_count": processed,
            "index_entries": index_entries,
            "pt_file_count": pt_file_count,
            "written": written,
        },
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a dict: {path}")
    return manifest


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(_json_safe(payload), fp, indent=2, sort_keys=True)
        fp.write("\n")
    os.replace(tmp_path, path)


def _manifest_identity_matches(output_root: Path, identity: dict[str, Any], *, allow_expected: bool) -> bool:
    candidates = [output_root / MANIFEST_NAME]
    if allow_expected:
        candidates.append(output_root / EXPECTED_MANIFEST_NAME)
    for path in candidates:
        manifest = _read_manifest(path)
        if manifest is None:
            continue
        if manifest.get("cache_identity") == identity:
            return True
    return False


def _check_complete_manifest(output_root: Path, identity: dict[str, Any]) -> None:
    manifest_path = output_root / MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path} is not complete: status={manifest.get('status')!r}")
    if manifest.get("cache_identity") != identity:
        raise ValueError(f"{manifest_path} does not match this checkpoint/cache request")
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    expected = files.get("expected_file_count")
    index_entries = files.get("index_entries")
    index = _read_index(output_root / "index.json")
    if expected is not None and len(index) != int(expected):
        raise ValueError(f"index.json entries={len(index)} expected={expected}")
    if index_entries is not None and len(index) != int(index_entries):
        raise ValueError(f"index.json entries={len(index)} manifest_index_entries={index_entries}")
    missing = [
        rel_path
        for rel_path in index.values()
        if not (output_root / rel_path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing cache files listed by index.json: {missing[:5]}")
    print(f"[teacher_cache] manifest_ok path={manifest_path} entries={len(index)}", flush=True)


def _default_overrides(args: argparse.Namespace) -> list[str]:
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
        f"data.train_batch_size={int(args.batch_size)}",
        f"data.val_batch_size={int(args.batch_size)}",
        f"data.test_batch_size={int(args.batch_size)}",
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
    overrides = _default_overrides(args) + list(args.override)
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


def _load_teacher_checkpoint(model: torch.nn.Module, ckpt_path: Path, checkpoint: dict[str, Any] | None = None) -> None:
    if checkpoint is None:
        checkpoint = _load_checkpoint_file(ckpt_path)
    state_dict = checkpoint["state_dict"]
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


def _attach_shard_trainer(datamodule: Any, *, num_shards: int, shard_index: int) -> None:
    if int(num_shards) <= 1:
        return
    object.__setattr__(
        datamodule,
        "trainer",
        SimpleNamespace(world_size=int(num_shards), global_rank=int(shard_index)),
    )


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


def _sample_teacher_rollout_pose_chunk(
    *,
    model: Any,
    anchor_context: dict[str, Tensor],
    tokenized_agent: dict[str, Tensor],
    anchor_mask: Tensor,
    scenario_ids: list[str],
    rollout_indices: list[int],
    base_seed: int,
) -> Tensor:
    n_agent = int(anchor_mask.shape[0])
    n_rollout = len(rollout_indices)
    flow_window_steps = int(model.flow_window_steps)
    flow_state_dim = int(model.encoder.agent_encoder.flow_state_dim)
    device = anchor_mask.device
    dtype = anchor_context["anchor_hidden"].dtype
    pose = torch.zeros((n_rollout, n_agent, flow_window_steps, 4), device=device, dtype=dtype)
    if n_rollout == 0 or not bool(anchor_mask.any()):
        return pose

    agent_encoder = model.encoder.agent_encoder
    anchor_hidden_valid = agent_encoder._pack_anchor_hidden(
        anchor_context["anchor_hidden"],
        anchor_context["anchor_mask"],
    )
    n_valid = int(anchor_hidden_valid.shape[0])
    valid_agent_batch = tokenized_agent["batch"][anchor_mask]
    x_init = torch.empty(
        (n_rollout, n_valid, flow_window_steps, flow_state_dim),
        device=device,
        dtype=dtype,
    )
    noise_scale = float(getattr(model.validation_rollout_sampling, "noise_scale", 1.0))
    for out_idx, rollout_index in enumerate(rollout_indices):
        for scene_index, scenario_id in enumerate(scenario_ids):
            scene_valid_mask = valid_agent_batch == int(scene_index)
            if not bool(scene_valid_mask.any()):
                continue
            generator = torch.Generator(device=device)
            generator.manual_seed(_stable_seed(base_seed, str(scenario_id), int(rollout_index)))
            x_init[out_idx, scene_valid_mask] = torch.randn(
                int(scene_valid_mask.sum().item()),
                flow_window_steps,
                flow_state_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            ) * noise_scale

    hidden_batched = (
        anchor_hidden_valid.unsqueeze(0)
        .expand(n_rollout, n_valid, anchor_hidden_valid.shape[-1])
        .reshape(n_rollout * n_valid, anchor_hidden_valid.shape[-1])
    )
    x_init_flat = x_init.reshape(n_rollout * n_valid, flow_window_steps, flow_state_dim)
    flow_sample_steps = getattr(
        model.validation_rollout_sampling,
        "sample_steps",
        agent_encoder.flow_ode.solver_steps,
    )
    flow_sample_method = getattr(
        model.validation_rollout_sampling,
        "sample_method",
        agent_encoder.flow_ode.solver_method,
    )
    sample_norm_flat = agent_encoder.flow_ode.generate(
        x_init=x_init_flat,
        model_fn=lambda x_t, tau: agent_encoder.flow_decoder(hidden_batched, x_t, tau),
        steps=flow_sample_steps,
        method=flow_sample_method,
        backprop_last_k=None,
    )
    agent_type = tokenized_agent["type"][anchor_mask]
    agent_length = tokenized_agent["shape"][anchor_mask, 0] if "shape" in tokenized_agent else None
    sample_pose_norm = model.encoder.flow_norm_to_pose_metric_norm(
        value=sample_norm_flat,
        agent_type=agent_type.repeat(n_rollout),
        agent_length=agent_length.repeat(n_rollout) if agent_length is not None else None,
    ).reshape(n_rollout, n_valid, flow_window_steps, -1)
    pos_local = sample_pose_norm[..., :2] * float(POSE_NORM_POS_SCALE_M)
    head_local = torch.atan2(sample_pose_norm[..., 3], sample_pose_norm[..., 2])
    current_pos = tokenized_agent["ctx_sampled_pos"][anchor_mask, 1]
    current_head = tokenized_agent["ctx_sampled_heading"][anchor_mask, 1]
    pos_global, head_global = transform_to_global(
        pos_local=pos_local.reshape(n_rollout * n_valid, flow_window_steps, 2),
        head_local=head_local.reshape(n_rollout * n_valid, flow_window_steps),
        pos_now=current_pos.repeat(n_rollout, 1),
        head_now=current_head.repeat(n_rollout),
    )
    pose_valid = torch.cat(
        [
            pos_global.reshape(n_rollout, n_valid, flow_window_steps, 2),
            torch.cos(head_global).reshape(n_rollout, n_valid, flow_window_steps).unsqueeze(-1),
            torch.sin(head_global).reshape(n_rollout, n_valid, flow_window_steps).unsqueeze(-1),
        ],
        dim=-1,
    )
    pose[:, anchor_mask] = pose_valid
    return pose


@torch.no_grad()
def _generate_batch_rollout_set(
    *,
    model: Any,
    data: Any,
    rollouts_per_scene: int,
    base_seed: int,
    amp_dtype: str,
    rollout_batch_size: int,
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
    scenario_ids = _scenario_id_list(data["scenario_id"])
    rollout_chunks: list[Tensor] = []
    device = next(model.parameters()).device
    with _amp_context(device, amp_dtype):
        for start in range(0, int(rollouts_per_scene), int(rollout_batch_size)):
            end = min(start + int(rollout_batch_size), int(rollouts_per_scene))
            rollout_chunks.append(
                _sample_teacher_rollout_pose_chunk(
                    model=model,
                    anchor_context=anchor_context,
                    tokenized_agent=tokenized_agent,
                    anchor_mask=anchor_mask,
                    scenario_ids=scenario_ids,
                    rollout_indices=list(range(start, end)),
                    base_seed=int(base_seed),
                )
            )
    return torch.cat(rollout_chunks, dim=0).contiguous(), anchor_mask


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


def _merge_shard_indexes(output_root: Path, *, num_shards: int | None = None) -> dict[str, str]:
    if num_shards is None or int(num_shards) <= 1:
        shard_paths = sorted(output_root.glob("index.shard_*.json"))
    else:
        shard_paths = [
            output_root / f"index.shard_{shard_index:05d}_of_{int(num_shards):05d}.json"
            for shard_index in range(int(num_shards))
        ]
    if not shard_paths:
        raise FileNotFoundError(f"No shard index files found under {output_root}")
    merged: dict[str, str] = {}
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing shard index files: " + ", ".join(str(path) for path in missing[:20])
        )
    for path in shard_paths:
        shard_index = _read_index(path)
        overlap = set(merged).intersection(shard_index)
        if overlap:
            raise ValueError(f"Duplicate scenario ids while merging {path}: {sorted(overlap)[:5]}")
        merged.update(shard_index)
    _write_index(output_root / "index.json", merged)
    print(
        f"[teacher_cache] merged_shards={len(shard_paths)} entries={len(merged)} "
        f"index={output_root / 'index.json'}",
        flush=True,
    )
    return merged


def _merge_shard_manifests(output_root: Path, *, num_shards: int, index_entries: int) -> None:
    expected_manifest = _read_manifest(output_root / EXPECTED_MANIFEST_NAME)
    shard_paths = [
        output_root / f"teacher_cache_manifest.shard_{shard_index:05d}_of_{int(num_shards):05d}.json"
        for shard_index in range(int(num_shards))
    ]
    existing_shard_paths = [path for path in shard_paths if path.exists()]
    shard_manifests = [_read_manifest(path) for path in existing_shard_paths]
    if shard_manifests:
        if len(shard_manifests) != int(num_shards):
            missing = [path for path in shard_paths if not path.exists()]
            raise FileNotFoundError(
                "Missing shard manifest files: " + ", ".join(str(path) for path in missing[:20])
            )
        first_identity = shard_manifests[0].get("cache_identity")
        mismatched = [
            path
            for path, manifest in zip(existing_shard_paths, shard_manifests)
            if manifest.get("cache_identity") != first_identity
        ]
        if mismatched:
            raise ValueError("Shard manifest identity mismatch: " + ", ".join(str(path) for path in mismatched[:5]))
        source = shard_manifests[0]
    elif expected_manifest is not None:
        source = expected_manifest
    else:
        print(
            f"[teacher_cache] no shard manifests found under {output_root}; skipping final manifest write",
            flush=True,
        )
        return

    files = source.get("files") if isinstance(source.get("files"), dict) else {}
    total_written = sum(
        int((manifest.get("files") or {}).get("written") or 0)
        for manifest in shard_manifests
        if isinstance(manifest, dict)
    )
    payload = dict(source)
    payload["status"] = "complete"
    payload["created_at_utc"] = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
    payload["files"] = {
        **files,
        "expected_file_count": int(index_entries),
        "index_entries": int(index_entries),
        "pt_file_count": len(list(output_root.glob("*.pt"))),
        "written": total_written if shard_manifests else files.get("written"),
    }
    payload["shards"] = {
        "num_shards": int(num_shards),
        "manifest_count": len(shard_manifests),
    }
    _write_manifest(output_root / MANIFEST_NAME, payload)
    print(
        f"[teacher_cache] wrote_manifest={output_root / MANIFEST_NAME} entries={index_entries}",
        flush=True,
    )


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
    if args.merge_shard_indexes:
        output_root = args.output_root.expanduser().resolve()
        merged = _merge_shard_indexes(output_root, num_shards=int(args.num_shards))
        _merge_shard_manifests(output_root, num_shards=int(args.num_shards), index_entries=len(merged))
        return
    if args.ckpt_path is None:
        raise ValueError("--ckpt-path is required unless --merge-shard-indexes is set.")
    if args.rollouts_per_scene <= 0:
        raise ValueError("--rollouts-per-scene must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    rollout_batch_size = int(args.rollout_batch_size or args.rollouts_per_scene)
    if rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive.")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    if args.num_shards > 1:
        index_path = output_root / f"index.shard_{int(args.shard_index):05d}_of_{int(args.num_shards):05d}.json"
    index = _read_index(index_path)

    ckpt_path = args.ckpt_path.expanduser().resolve()
    checkpoint = _load_checkpoint_file(ckpt_path)
    checkpoint_meta = _checkpoint_metadata(ckpt_path, checkpoint)
    cfg = _compose_config(args)
    datamodule = instantiate(cfg.data)
    _attach_shard_trainer(
        datamodule,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    model = instantiate(cfg.model, _recursive_=False)
    split_scene_count = None

    device = torch.device(args.device)
    dataloader = _select_dataloader(datamodule, args.split)
    split_scene_count = _dataset_scene_count(dataloader)
    identity = _cache_identity(
        args=args,
        checkpoint_meta=checkpoint_meta,
        split_scene_count=split_scene_count,
        flow_window_steps=int(getattr(model, "flow_window_steps", 20)),
    )
    if args.check_manifest:
        try:
            _check_complete_manifest(output_root, identity)
        except Exception as exc:
            print(f"[teacher_cache] manifest_miss reason={exc}", file=sys.stderr, flush=True)
            raise SystemExit(2) from None
        return

    skip_existing = bool(args.skip_existing)
    if skip_existing and not _manifest_identity_matches(output_root, identity, allow_expected=True):
        print(
            "[teacher_cache] --skip-existing requested but no matching complete/expected "
            "manifest was found; existing files will be rebuilt instead of trusted.",
            flush=True,
        )
        skip_existing = False
    _write_manifest(
        output_root / EXPECTED_MANIFEST_NAME,
        _manifest_payload(
            status="expected",
            identity=identity,
            checkpoint_meta=checkpoint_meta,
            args=args,
            split_scene_count=split_scene_count,
        ),
    )

    _load_teacher_checkpoint(model, ckpt_path, checkpoint)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    storage_dtype = _storage_dtype(args.storage_dtype)
    processed = 0
    written = 0
    shard_max_scenes = args.max_scenes
    if args.max_scenes is not None and args.num_shards > 1:
        shard_max_scenes = args.max_scenes // args.num_shards + int(
            args.shard_index < (args.max_scenes % args.num_shards)
        )
    started = time.monotonic()
    for batch_index, data in enumerate(dataloader):
        data = _to_device(data, device)
        scenario_ids = _scenario_id_list(data["scenario_id"])
        if skip_existing and all((output_root / (_sanitize_scenario_id(sid) + ".pt")).exists() for sid in scenario_ids):
            for sid in scenario_ids:
                index[str(sid)] = _sanitize_scenario_id(sid) + ".pt"
            processed += len(scenario_ids)
            if shard_max_scenes is not None and processed >= shard_max_scenes:
                break
            continue

        rollout_set, anchor_mask = _generate_batch_rollout_set(
            model=model,
            data=data,
            rollouts_per_scene=int(args.rollouts_per_scene),
            base_seed=int(args.seed),
            amp_dtype=str(args.amp_dtype),
            rollout_batch_size=rollout_batch_size,
        )
        for scene_index, scenario_id in enumerate(scenario_ids):
            if shard_max_scenes is not None and processed >= shard_max_scenes:
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
                overwrite=not skip_existing,
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
        if shard_max_scenes is not None and processed >= shard_max_scenes:
            break

    _write_index(index_path, index)
    manifest_path = output_root / MANIFEST_NAME
    if args.num_shards > 1:
        manifest_path = output_root / (
            f"teacher_cache_manifest.shard_{int(args.shard_index):05d}_of_{int(args.num_shards):05d}.json"
        )
    _write_manifest(
        manifest_path,
        _manifest_payload(
            status="complete",
            identity=identity,
            checkpoint_meta=checkpoint_meta,
            args=args,
            split_scene_count=split_scene_count,
            processed=processed,
            written=written,
            index_entries=len(index),
            pt_file_count=len(list(output_root.glob("*.pt"))) if args.num_shards <= 1 else None,
        ),
    )
    elapsed = time.monotonic() - started
    print(
        f"[teacher_cache] done processed={processed} written={written} "
        f"index={index_path} manifest={manifest_path} elapsed_sec={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
