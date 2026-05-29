"""l2_to_gt proxy loss frame alignment 검증.

학습 step 안에서 비교되는 두 텐서:
  - committed_path_norm  : closed-loop rollout 의 anchor k 정규화 path
  - flow_eval_clean_norm : GT trajectory 의 같은 anchor 정규화 path

이 둘이
  (1) 같은 dim (3 control-space vs 4 pose-space)
  (2) 같은 (local) frame
  (3) 같은 stride / channel 의미
  (4) identity inject 시 proxy_loss == 0
를 만족하는지 dump.

CUDA_VISIBLE_DEVICES=3 python tools/debug_proxy_l2_to_gt_frame_alignment.py
"""

from __future__ import annotations
import os
import sys

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = "/home2/pnc2/repos_python/kinematic_flow"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> None:
    config_dir = os.path.join(REPO_ROOT, "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="run",
            overrides=[
                "experiment=self_forced_npfm_pareto",
                "action=finetune",
                "ckpt_path=logs/pretrained/pretrained.ckpt",
                "task_name=debug_proxy_l2_to_gt_alignment",
                "paths.cache_root=/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1",
                "data.train_batch_size=2",
                "data.val_batch_size=2",
                "data.num_workers=0",
                "data.persistent_workers=false",
                "data.train_epoch_sample_fraction=1.0",
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "trainer.precision=32-true",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=0",
                "trainer.check_val_every_n_epoch=null",
                "trainer.val_check_interval=null",
                "model.model_config.self_forced.enabled=true",
                "model.model_config.self_forced.debug_proxy_loss=l2_to_gt_nearest",
                "model.model_config.self_forced.n_rollouts=1",
                "model.model_config.self_forced.n_anchors=1",
                "model.model_config.self_forced.estimator_warmup_steps=0",
                "model.model_config.self_forced.estimator_warmup_epochs=0",
                "model.model_config.self_forced.estimator_updates_per_step=1",
                "model.model_config.self_forced.initialize_aux_from_generator_on_fit_start=true",
            ],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")
    print(f"[init] use_kinematic_control_flow="
          f"{cfg.model.model_config.token_processor.use_kinematic_control_flow}")

    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")

    # SMARTFlow.__init__ reads HydraConfig().runtime.output_dir for video_dir;
    # compose() doesn't populate it.  Stub the call.
    import hydra.core.hydra_config as _hc
    from types import SimpleNamespace
    _hc.HydraConfig.get = staticmethod(  # type: ignore[assignment]
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir="/tmp/debug_proxy_l2_to_gt")),
    )

    from src.smart.model.smart_flow import SMARTFlow
    model: SMARTFlow = hydra.utils.instantiate(cfg.model)

    ckpt_path = cfg.ckpt_path
    print(f"[load] ckpt={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    # teacher/estimator 동기화 (training_start 가 하는 일).
    try:
        model.on_train_start()
    except Exception as e:
        print(f"[warn] on_train_start raised: {e}")

    batch = next(iter(dm.train_dataloader()))
    batch = batch.to(device)
    print(f"[batch] num_graphs={getattr(batch, 'num_graphs', 'n/a')}")

    # tokenize 만으로 GT 텐서 얻기 (eval-mode 토큰 = flow_eval_* 포함).
    with torch.no_grad():
        model.token_processor.eval()
        tokenized_map, tokenized_agent = model.token_processor(batch)

    fem = tokenized_agent["flow_eval_mask"]
    fec_ctrl = tokenized_agent.get("flow_eval_clean_norm", None)
    fec_pose = tokenized_agent.get("flow_eval_clean_metric_norm", None)
    print("=" * 70)
    if fec_ctrl is not None:
        print("[GT-ctrl] flow_eval_clean_norm.shape       =", tuple(fec_ctrl.shape),
              "dtype=", fec_ctrl.dtype)
    if fec_pose is not None:
        print("[GT-pose] flow_eval_clean_metric_norm.shape=", tuple(fec_pose.shape),
              "dtype=", fec_pose.dtype)
    print("[GT] flow_eval_mask.shape       =", tuple(fem.shape),
          "n_anchor=", fem.shape[1], "anchor0_count=", int(fem[:, 0].sum().item()))
    if fec_pose is not None:
        print("[GT-pose] per-channel stats:")
        for c in range(fec_pose.shape[-1]):
            x = fec_pose[..., c].float()
            print(f"      ch{c}: mean={x.mean():+8.4f} std={x.std():+8.4f}"
                  f" min={x.min():+8.4f} max={x.max():+8.4f}")

    # closed-loop rollout 으로 committed_path_norm.
    print("=" * 70)
    print("[rollout] running closed-loop self-forced rollout (1 anchor) ...")
    with torch.no_grad():
        rollout = model._run_self_forced_rollout(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
        )

    committed, committed_pose, anchor_mask = model._pack_self_forced_committed_rollout(
        rollout=rollout, tokenized_agent=tokenized_agent, anchor_idx=0,
    )
    print("[CMT-default] shape =", tuple(committed.shape),
          "(control 3-dim if use_kinematic_control_flow else pose 4-dim)")
    print("[CMT-pose]    shape =", tuple(committed_pose.shape), "(always 4-dim)")
    print("[CMT] anchor_mask.shape =", tuple(anchor_mask.shape),
          "sum=", int(anchor_mask.sum().item()))
    print("[CMT-pose] per-channel stats:")
    for c in range(committed_pose.shape[-1]):
        x = committed_pose[..., c].float()
        print(f"      ch{c}: mean={x.mean():+8.4f} std={x.std():+8.4f}"
              f" min={x.min():+8.4f} max={x.max():+8.4f}")

    # GT anchor-0 chunk (proxy_loss 함수와 동일 로직).
    n_per_anchor = fem.sum(dim=0)
    start, end = 0, int(n_per_anchor[0].item())
    gt0 = fec_pose[start:end]
    print("=" * 70)
    print("[GT-anchor0] gt0.shape =", tuple(gt0.shape))

    print(f"[CHECK] pose dim: CMT[-1]={committed_pose.shape[-1]} GT[-1]={gt0.shape[-1]}"
          f"  (expected both = 4)")

    if gt0.shape != committed_pose.shape:
        print(f"[CHECK] !!! shape mismatch !!!  GT={tuple(gt0.shape)}"
              f" committed_pose={tuple(committed_pose.shape)}")
    else:
        diff = (committed_pose.float() - gt0.float())
        print(f"[CHECK] element-wise L2 mean over all (pose 4-dim) = {diff.square().mean():.6e}")
        for c in range(diff.shape[-1]):
            print(f"        ch{c}: L2 mean={diff[..., c].square().mean():.6e}"
                  f"  abs_mean={diff[..., c].abs().mean():.6e}")

    # IDENTITY: GT 자체를 committed 자리에 박아넣으면 proxy_loss == 0 ?
    print("=" * 70)
    print("[IDENTITY] inject GT as committed, recompute proxy_loss (expected 0)")
    shift = int(model.encoder.agent_encoder.shift)
    if gt0.numel() == 0:
        print("[IDENTITY] gt0 is empty — skip")
    else:
        fake_committed = gt0
        if fake_committed.shape[1] >= shift:
            cc = fake_committed[:, shift - 1::shift, :].float()
            gc = gt0[:, shift - 1::shift, :].float()
        else:
            cc = fake_committed.float()
            gc = gt0.float()
        pos_loss = (cc[..., :2] - gc[..., :2]).square().mean()
        print(f"[IDENTITY] pos_loss = {pos_loss:.6e}   (== 0 means pos frame matches)")
        if cc.shape[-1] >= 4 and gc.shape[-1] >= 4:
            head_loss = (cc[..., 2:4] - gc[..., 2:4]).square().mean()
            print(f"[IDENTITY] head_loss = {head_loss:.6e}")
        else:
            print("[IDENTITY] head_loss DIM CHECK FAIL (>=4 required) — head_loss NOT computed")

    # 실제 proxy_loss 값 (모델 함수 그대로) — pose-space 4-dim 입력.
    print("=" * 70)
    print("[ACTUAL] proxy_loss returned by model._compute_self_forced_debug_proxy_loss:")
    with torch.no_grad():
        actual_loss = model._compute_self_forced_debug_proxy_loss(
            tokenized_agent=tokenized_agent,
            committed_path_norm=committed_pose,   # pose 4-dim
            anchor_idx=0,
        )
    print(f"[ACTUAL] = {float(actual_loss):.6e}")

    # ------------------------------------------------------------------
    # INFERENCE INTERNALS — closed-loop rollout 의 의심 지점 직접 print.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[INTERNALS] rollout dict keys + shape:")
    for k, v in rollout.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:24s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:24s} type={type(v).__name__}")

    pred_traj = rollout["pred_traj_10hz"]   # [N, T, 2] global
    pred_head = rollout["pred_head_10hz"]   # [N, T]    global rad
    ctx_pos = tokenized_agent["ctx_sampled_pos"][:, 1]      # [N, 2] anchor 0 origin global
    ctx_head = tokenized_agent["ctx_sampled_heading"][:, 1] # [N]    anchor 0 origin heading

    print("=" * 70)
    print("[INTERNALS] origin sanity — first rollout step vs anchor-0 origin")
    delta0 = (pred_traj[:, 0] - ctx_pos)        # [N, 2]
    dist0 = delta0.norm(dim=-1)                 # [N]  meters traversed in 0.1s
    print(f"  pred_traj[:,0] - ctx_pos[:,1] norm:"
          f" mean={dist0.mean():.4f} m  median={dist0.median():.4f}"
          f" max={dist0.max():.4f}  min={dist0.min():.4f}")
    print(f"  → 0.1s 이동량.  10 m/s (36 km/h) 면 ~1.0 m 예상")

    print("=" * 70)
    print("[INTERNALS] step-by-step rollout progress (mean over agents)")
    print("  step | pos_x_local mean | pos_y_local mean | yaw_local mean | yaw_local std")
    from src.smart.utils.rollout import transform_to_local
    pos_local, head_local = transform_to_local(
        pos_global=pred_traj, head_global=pred_head,
        pos_now=ctx_pos, head_now=ctx_head,
    )
    for t in range(0, pred_traj.shape[1]):
        x = pos_local[:, t, 0].float()
        y = pos_local[:, t, 1].float()
        h = head_local[:, t].float()
        if t % 2 == 0 or t == pred_traj.shape[1] - 1:
            print(f"  t={t:2d} | x={x.mean():+7.3f} | y={y.mean():+7.3f}"
                  f" | yaw={h.mean():+7.4f} | yaw_std={h.std():+7.4f}")

    print("=" * 70)
    print("[INTERNALS] GT step-by-step (for comparison; anchor-0 mask only)")
    # GT pose-space anchor 0 chunk: gt0[..., 0]=x/20, gt0[..., 1]=y/20, [2]=cos, [3]=sin
    gt_x = gt0[..., 0] * 20.0
    gt_y = gt0[..., 1] * 20.0
    gt_yaw = torch.atan2(gt0[..., 3].float(), gt0[..., 2].float())
    for t in range(0, gt0.shape[1]):
        if t % 2 == 0 or t == gt0.shape[1] - 1:
            print(f"  t={t:2d} | x={gt_x[:, t].mean():+7.3f}"
                  f" | y={gt_y[:, t].mean():+7.3f}"
                  f" | yaw={gt_yaw[:, t].mean():+7.4f}"
                  f" | yaw_std={gt_yaw[:, t].std():+7.4f}")

    print("=" * 70)
    print("[INTERNALS] yaw drift CMT vs GT (last step, anchor 0 agents only)")
    # CMT anchor 0 agents only.  committed_pose 는 이미 anchor 0 mask 적용된 packed pose 4-dim.
    cmt_yaw_last = torch.atan2(committed_pose[..., 3].float(),
                                committed_pose[..., 2].float())[:, -1]
    gt_yaw_last = gt_yaw[:, -1]
    yaw_err_last = cmt_yaw_last - gt_yaw_last
    yaw_err_last = torch.atan2(yaw_err_last.sin(), yaw_err_last.cos())
    print(f"  CMT yaw last:  mean={cmt_yaw_last.mean():+7.4f} std={cmt_yaw_last.std():+7.4f}")
    print(f"  GT  yaw last:  mean={gt_yaw_last.mean():+7.4f} std={gt_yaw_last.std():+7.4f}")
    print(f"  |Δyaw| last:   mean={yaw_err_last.abs().mean():+7.4f} max={yaw_err_last.abs().max():+7.4f}")

    print("=" * 70)
    print("[INTERNALS] velocity_head (model.encoder.agent_encoder.flow_decoder.velocity_head):")
    vh = model.encoder.agent_encoder.flow_decoder.velocity_head
    print(f"  module: {type(vh).__name__}")
    total = sum(p.numel() for p in vh.parameters())
    trainable = sum(p.numel() for p in vh.parameters() if p.requires_grad)
    print(f"  params: total={total}  trainable={trainable}")
    for n, p in vh.named_parameters():
        print(f"    {n:20s} shape={tuple(p.shape)} mean={p.data.float().mean():+.4e}"
              f" std={p.data.float().std():+.4e}")


if __name__ == "__main__":
    main()
