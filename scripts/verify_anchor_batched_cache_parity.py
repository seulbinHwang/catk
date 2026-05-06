"""anchor-batched prepare_inference_cache 의 per-anchor 정합성 검증 스크립트.

배경:
  OCSC 학습 step 에서 anchor 별로 ``prepare_inference_cache`` 가 매번 호출돼
  encoder forward 가 N_anchor 회 반복되는 부분 (~33.8% wallclock) 을 한 번의
  batched forward 로 압축한 신규 메서드 ``prepare_inference_cache_anchor_batched``
  의 정합성을 단일 anchor reference 와 비교해 검증한다.

사용:
  CUDA_VISIBLE_DEVICES=2 python scripts/verify_anchor_batched_cache_parity.py \
      --batch-size 4 --num-anchors 6

성공 기준:
  per-anchor cache 와 batched cache 의 ``feat_a_now`` / ``feat_a`` /
  ``feat_a_t_dict`` / ``pos_window`` / ``valid_window`` 등 핵심 텐서의 max
  delta 가 1e-5 이하.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def _compare_cache_dicts(per_anchor: dict, batched: dict, anchor_idx: int) -> dict:
    """anchor 한 개의 per-anchor vs batched cache 비교.  핵심 텐서별 max delta."""
    deltas: dict[str, float] = {}
    for k, v in per_anchor.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k not in batched or not isinstance(batched[k], torch.Tensor):
            continue
        if v.shape != batched[k].shape:
            deltas[k] = float("inf")
            continue
        deltas[k] = _max_abs_delta(v.float(), batched[k].float())
    # feat_a_t_dict — dict-of-tensor.
    if "feat_a_t_dict" in per_anchor and isinstance(per_anchor["feat_a_t_dict"], dict):
        for layer_idx, t in per_anchor["feat_a_t_dict"].items():
            if not isinstance(t, torch.Tensor):
                continue
            b_t = batched.get("feat_a_t_dict", {}).get(layer_idx, None)
            if not isinstance(b_t, torch.Tensor):
                continue
            if t.shape != b_t.shape:
                deltas[f"feat_a_t_dict[{layer_idx}]"] = float("inf")
            else:
                deltas[f"feat_a_t_dict[{layer_idx}]"] = _max_abs_delta(t.float(), b_t.float())
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="logs/pretrained/epoch_last.ckpt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "configs"
    if not cfg_dir.exists():
        raise FileNotFoundError(f"configs/ not found at {cfg_dir}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 1. Hydra config 로드 + Lightning model 인스턴스화 ──────────────────
    with initialize_config_dir(config_dir=str(cfg_dir.absolute()), version_base=None):
        cfg = compose(
            config_name="run.yaml",
            overrides=[
                "experiment=flow_consistency_bptt",
                f"data.train_batch_size={args.batch_size}",
                f"data.val_batch_size={args.batch_size}",
                "data.shuffle=false",
                "trainer.precision=32-true",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=0",
            ],
        )
    print("[parity] config loaded.")

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval().to(device)
    print(f"[parity] model loaded from {args.ckpt}, on {device}.")

    # ── 2. Sample batch 가져오기 ───────────────────────────────────────────
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    batch = batch.to(device)
    print(f"[parity] sample batch obtained.")

    # ── 3. tokenized agent / map_feature 준비 ──────────────────────────────
    tokenized_map, tokenized_agent = model.token_processor(batch)
    with torch.no_grad():
        map_feature = model.encoder.encode_map(tokenized_map)
    agent_enc = model.encoder.agent_encoder

    shift = int(getattr(agent_enc, "shift", 5))
    step_current_2hz = max(1, (int(getattr(agent_enc, "num_historical_steps", 11)) - 1) // shift)
    total_2hz = int(tokenized_agent["gt_pos"].shape[1])
    pred_steps = 2
    valid_anchor_end = max(1, total_2hz - pred_steps)
    anchor_indices = list(range(min(args.num_anchors, valid_anchor_end)))
    print(
        f"[parity] step_current_2hz={step_current_2hz} total_2hz={total_2hz} "
        f"anchor_indices={anchor_indices}"
    )

    seq_keys = ("gt_pos", "gt_heading", "valid_mask", "gt_idx")

    # ── 4. Per-anchor reference cache 빌드 ─────────────────────────────────
    per_anchor_caches: list[dict] = []
    for anchor_idx in anchor_indices:
        hist_start = max(0, anchor_idx + 1 - step_current_2hz)
        ta_anchor: dict = {}
        for k, v in tokenized_agent.items():
            if (
                k in seq_keys
                and isinstance(v, torch.Tensor)
                and v.dim() >= 2
                and v.shape[1] >= anchor_idx + 1
            ):
                ta_anchor[k] = v[:, hist_start : anchor_idx + 1]
            else:
                ta_anchor[k] = v
        with torch.no_grad():
            cache = agent_enc.prepare_inference_cache(
                tokenized_agent=ta_anchor,
                map_feature=map_feature,
            )
        per_anchor_caches.append(cache)
    print(f"[parity] per-anchor cache built ({len(per_anchor_caches)} anchors).")

    # ── 5. Anchor-batched cache 빌드 ───────────────────────────────────────
    with torch.no_grad():
        batched_caches = agent_enc.prepare_inference_cache_anchor_batched(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_indices=anchor_indices,
            step_current_2hz=step_current_2hz,
            seq_keys=seq_keys,
        )
    print(f"[parity] batched cache built ({len(batched_caches)} anchors).")
    assert len(batched_caches) == len(per_anchor_caches), "anchor count mismatch"

    # ── 6. 비교 ───────────────────────────────────────────────────────────
    print(f"\n[parity] per-anchor vs batched cache deltas (target: < {args.rtol}):")
    all_pass = True
    for i, anchor_idx in enumerate(anchor_indices):
        deltas = _compare_cache_dicts(per_anchor_caches[i], batched_caches[i], anchor_idx)
        worst_key = max(deltas, key=lambda k: deltas[k]) if deltas else "(none)"
        worst = deltas.get(worst_key, 0.0)
        status = "OK" if worst <= args.rtol else "FAIL"
        print(f"  anchor {anchor_idx}: max delta = {worst:.3e} on '{worst_key}' [{status}]")
        if worst > args.rtol:
            all_pass = False
            for k, d in sorted(deltas.items(), key=lambda x: -x[1])[:8]:
                print(f"    {k}: {d:.3e}")

    print(f"\n[parity] {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
