# OCSC Experiment Setting List

Last updated: 2026-06-01 KST.

This is a compact memory log of the OCSC / DMD fine-tuning experiments that were used for decisions. It is not a complete dump of every smoke run.

## Common Baseline

- Pretrained reference validation
  - W&B: `hdbfyfn2`
  - Name: `pareto_pretrained_val_dmdmatch_clsft_pareto_clsft_v100x4_0529_162458_valb4`
  - RMM: `0.7792712450`
  - Open ADE/FDE: `0.1194286123 / 0.2794527709`

- OCSC clean reference signal
  - W&B: `dk3njfnf`
  - Name: `ocsc-clean-v2-pos1-h001-lastcoarseF-pred4-b8`
  - Used mainly as a trend reference: Open ADE/FDE kept decreasing.
  - Logged RMM series around `0.7669 -> 0.7673 -> 0.7665 -> 0.7657`
  - Open ADE: `0.13435 -> 0.12862`
  - Open FDE: `0.30822 -> 0.29902`

## Common OCSC Settings

- Mode: `finetune.mode=ocsc_ft`
- Target: open-loop samples from frozen pretrained ref decoder unless noted
- Matching: nearest open-loop target unless noted
- Loss: L2 in pose-normalized space, `position_weight=1.0`, `heading_weight=0.01`
- Active mask: strict active mask enabled in the main guarded runs
- Anchor: anchor 0 / history-end
- Closed-loop validation: enabled
- Open-loop validation: enabled
- Closed-loop metric rollouts: `n_rollout_closed_val=16`
- Validation frequency: `val_check_interval=200`, `limit_val_batches=0.1`
- LQR: off (`decoder.use_lqr=false`)
- Stop motion: off
- Kinematic control flow: on
- First-validation guard: stop if first RMM `< 0.77917`
- No-signal guard: after 3 validations, stop unless RMM or Open ADE/FDE gives a useful signal

## Early Debug / Setup Runs

- `ocsc_gtbc_*`, `ocsc_gtbc_xy2hz_*`
  - Purpose: initial OCSC/GT-BC path checks, XY 2 Hz variants, full vs velocity-only branches.
  - Main value: implementation and smoke/debug signal, not final comparison.

- `ocsc_ft_velhead_g4_m4_2gpu_v1`
  - Purpose: early 2-GPU OCSC fine-tune smoke.
  - G/M: `G=4`, `M=4`
  - Trainable range: velocity-head style path.

- `ocsc_pose_2hz_g4_m8_velhead_lr2e6_*`
  - Purpose: pose 2 Hz OCSC with `G=4`, `M=8`, velocity-head range.
  - LR: `2e-6`
  - Follow-up included batch/eval-worker tweaks and ref-sync/global nearest matching check.

## Weekend Search Runs

- `ocsc_cleanmatch_lr1e6_noshuffle_trainselect_20260529_191741`
  - Purpose: reproduce/compare clean-style matching with training agent selection and no shuffle.
  - LR: `1e-6`

- `ocsc_lr1e6_evalselect_noshuffle_20260529_192938`
  - Purpose: switch to eval-agent selection, no shuffle.
  - LR: `1e-6`

- `ocsc_lr1e6_evalselect_shuffle_20260529_194542`
  - Purpose: eval-agent selection with shuffle.
  - LR: `1e-6`

- `ocsc_m16_lr1e6_evalselect_20260529_200148`
  - Purpose: increase open-loop candidate count.
  - M: `16`
  - LR: `1e-6`

- `ocsc_m12_lr1e6_evalselect_b8_20260529_204856`
  - Purpose: M=12, batch-size change.
  - M: `12`
  - LR: `1e-6`

- `ocsc_steprefiner_lr1e6_20260529_212750`
  - Purpose: restrict trainable range to step refiner.
  - LR: `1e-6`

- `ocsc_gt_target_lr1e6_20260529_235800`
  - Purpose: GT target ablation instead of open-loop ref target.
  - LR: `1e-6`

- `ocsc_sharednoise_*`
  - Purpose: align OL and CL noise tapes / stochastic branch.
  - Variants: `lr2e-6_b16`, `lr1e-6_b8`, step-refiner `lr5e-7`

- `ocsc_steprefiner_lr1e6_wd1e4_20260530_044034`
  - Purpose: step-refiner with lower weight decay.
  - LR/WD: `1e-6 / 1e-4`

- `ocsc_steprefiner_lr5e7_wd1e2_retry_20260530_052201`
  - Purpose: lower LR retry.
  - LR/WD: `5e-7 / 1e-2`

- `ocsc_steprefiner_lr2e7_wd1e2_guarded_20260530_060531`
  - Purpose: lower LR guarded run.
  - LR/WD: `2e-7 / 1e-2`

- `ocsc_steprefiner_m12_lr5e7_wd1e2_20260530_064434`
  - Purpose: M=12 with step-refiner.
  - LR/WD: `5e-7 / 1e-2`

- `ocsc_steprefiner_m12_lr2e7_wd1e2_from_best_20260530_100739`
  - Purpose: continue from best checkpoint with lower LR.
  - LR/WD: `2e-7 / 1e-2`

- `ocsc_steprefiner_m12_lr1e7_wd1e2_from_best_20260530_194330`
  - Purpose: long/better base checkpoint for later experiments.
  - LR/WD: `1e-7 / 1e-2`
  - Later experiments resumed from its `epoch_000.ckpt`.

## Long-Best Fallback Runs

- `ocsc_m24_lr1e6_b4_wd1e4_20260531_110837`
  - W&B: `wue42h20`
  - Purpose: larger M search.
  - M: `24`
  - LR/WD: `1e-6 / 1e-4`
  - Result: failed first-validation guard.
  - First RMM: `0.77859795`
  - First Open ADE/FDE: `0.12018611 / 0.28164828`

- `ocsc_steprefiner_m12_lr8e8_wd1e2_from_long_best_20260531_114504`
  - W&B: `6ardo3fi`
  - Fallback index: 15
  - Trainable range: step refiner
  - M/G: `M=12`, `G=4`
  - LR/WD: `8e-8 / 1e-2`
  - Result: Open ADE/FDE consistently decreased; RMM mostly hovered around baseline.
  - Example early RMM: `0.77931565 -> 0.77918243 -> 0.77915639`
  - Example Open ADE/FDE: `0.11583083 / 0.27283010 -> 0.11549975 / 0.27215496`

- `ocsc_steprefiner_m12_lr5e8_wd1e2_from_long_best_20260531_214133`
  - Fallback index: 16
  - Trainable range: step refiner
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-2`
  - Purpose: lower LR from fallback 15.

- `ocsc_steprefiner_m12_lr5e8_wd1e3_from_long_best_20260531_222030`
  - W&B: `ujrpkg5d`
  - Fallback index: 17
  - Trainable range: step refiner only, about `103K` trainable params
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Best remembered RMM: `0.7794274092`
  - Open ADE/FDE decreased steadily.
  - This became the best RMM setting among the guarded search runs.

## Trainable-Range Search

- `ocsc_velhead_m12_lr5e8_wd1e3_from_long_best_20260601_100830`
  - W&B: `xycqwp4g`
  - Trainable range: velocity head only
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Result: stopped for no learning signal after 3 validations.
  - RMM: `0.77933377 -> 0.77929932 -> 0.77927923`
  - Open ADE/FDE: `0.11599075 / 0.27315548 -> 0.11595444 / 0.27308139`

- `ocsc_fullflow_m12_lr5e8_wd1e3_from_long_best_20260601_114316`
  - W&B: `kq8oxu2b`
  - Trainable range: full `agent_encoder.flow_decoder`, about `490K` trainable params
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Loss horizon/stride: 2 seconds, default 2 Hz endpoints
  - Result: Open ADE/FDE clearly decreased, RMM did not keep improving.
  - RMM: `0.77935719 -> 0.77923101 -> 0.77925366`
  - Open ADE/FDE: `0.11558266 / 0.27224544 -> 0.11510749 / 0.27117306`
  - Observation: kinematic likelihood dropped despite Open ADE/FDE improving, mainly suspected from closed-loop distribution/acceleration likelihood rather than open displacement error alone.

## Horizon / Temporal-Striding Runs

- `ocsc_fullflow_m12_lr5e8_wd1e3_1s_from_long_best_20260601_134235`
  - W&B: `n07ojb6e`
  - Trainable range: full flow decoder, about `490K`
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Loss horizon: `ocsc_loss_window_steps=10` (first 1 second)
  - Temporal stride: default 2 Hz endpoint stride
  - Result: stopped by operator decision to relaunch 2s/10Hz.
  - RMM: `0.77929348 -> 0.77932727 -> 0.77912647`
  - Open ADE/FDE: `0.11567034 / 0.27240992 -> 0.11499688 / 0.27087402`

- `ocsc_fullflow_m12_lr5e8_wd1e3_2s10hz_from_long_best_20260601_151700`
  - W&B: `ulg0xuqg`
  - Current active run as of 2026-06-01 16:06 KST
  - Trainable range: full flow decoder, about `490K`
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Loss horizon: `ocsc_loss_window_steps=20` (2 seconds)
  - Temporal stride: `ocsc_loss_temporal_stride=1` (all 10 Hz steps)
  - Accumulation: `trainer.accumulate_grad_batches=2`
  - First validation: RMM `0.77936894`, Open ADE/FDE `0.11563430 / 0.27236068`
  - Runtime note: first compile was slow, then training started; no separate `ocsc_log` tmux window.
  - Monitor note: if a later validation drops more than `1e-5` below the best RMM after at least 2 validations, stop and launch the control-space matching fallback below.

- `ocsc_fullflow_control_m12_lr5e8_wd1e3_2s10hz_from_long_best`
  - Fallback index: 20
  - Status: armed in monitor, not launched at time of note
  - Trainable range: full flow decoder, about `490K`
  - M/G: `M=12`, `G=4`
  - LR/WD: `5e-8 / 1e-3`
  - Loss horizon/stride: 2 seconds, 10 Hz all steps
  - Matching: `ocsc_match_space=control`, raw normalized 3D control vectors `[delta_s, delta_n, delta_yaw]`
  - Intended diagnostic: test whether pose-space integration/matching is hurting RMM while Open ADE/FDE improves.

## Search Axes Already Tried

- Trainable range
  - velocity head only
  - step refiner only
  - full flow decoder

- Open-loop candidate count
  - `M=8`, `M=12`, `M=16`, `M=24`
  - `G` mostly fixed at `4`

- Learning rate
  - Coarse range included `2e-6`, `1e-6`, `5e-7`, `2e-7`, `1e-7`, `8e-8`, `5e-8`

- Weight decay
  - Tried mainly `1e-2`, `1e-3`, `1e-4`
  - Later judged less important because many runs terminate after only a few validations.

- Target/matching
  - Open-loop ref target
  - GT target ablation
  - nearest matching
  - paired / shared-noise branch alignment

- Data/validation control
  - train-agent selection vs eval-agent selection
  - shuffle vs no shuffle
  - validation spawn/worker crash fixes

- Loss temporal coverage
  - 2 seconds at default 2 Hz endpoints
  - 1 second window
  - 2 seconds at 10 Hz all steps

## Current Best Known Setting by RMM

- `ocsc_steprefiner_m12_lr5e8_wd1e3_from_long_best_20260531_222030`
- W&B: `ujrpkg5d`
- Fallback index: 17
- Best remembered RMM: `0.7794274092`
- Setting: step-refiner-only, `M=12`, `G=4`, `lr=5e-8`, `wd=1e-3`, frozen OL ref, nearest matching.
