# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project conventions

### GPU usage

**This machine is shared.** Only use GPUs **2 and 3** for any background job, training, or evaluation. Always set `CUDA_VISIBLE_DEVICES=2,3` (or `=2` / `=3` for single-GPU runs) when launching commands. Do not touch GPU 0 or 1 — they belong to other users / jobs.

When using `torchrun --nproc_per_node=2` for DDP, set `CUDA_VISIBLE_DEVICES=2,3`. For single-process scripts, either device works once the env var is set; the first visible device becomes `cuda:0` inside the process.

### Time display

Always convert times to **KST (UTC+9)** when reporting to the user. The server runs in UTC; tools like `date` return UTC. Convert before displaying:

- Internal/log timestamps may stay UTC.
- All user-facing time references (current time, ETAs, deadlines, schedules, "X minutes ago", etc.) must be KST.
- When showing both, label explicitly: `01:30 UTC (10:30 KST)`.

## Environment

- Conda env name: `catk` (Python 3.11.9, torch 2.4.1, lightning 2.4.0, hydra 1.3.2, waymo-open-dataset-tf-2-12-0).
- Conda activation differs by host: scripts probe `${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}` and fall back to `/home2/pnc2/miniforge3/etc/profile.d/conda.sh` (the active path on this machine). Override `CONDA_SH` if neither exists.
- WandB is the default logger (`logger=wandb`); set `WANDB_MODE=offline` to disable network upload.
- Cached dataset root is configured in `configs/paths/default.yaml` (`paths.cache_root`); override with `paths.cache_root=...` rather than editing the YAML.

## Entry point

All training/validation/test runs go through Hydra:

```sh
python -m src.run experiment=<NAME> action=<fit|finetune|validate|test> task_name=<TAG>
```

`configs/run.yaml` composes `data` + `model` + `callbacks` + `logger` + `trainer` + `paths` + `hydra` + `experiment`. Experiment files in `configs/experiment/` use `# @package _global_` and are the canonical place to override model/trainer/data settings — prefer adding/editing an experiment YAML over per-flag CLI overrides.

Hydra output dirs land under `logs/<task_name>/runs/<timestamp>/`. `cfg.paths.output_dir` is `${hydra:runtime.output_dir}`; `VisWaymo` videos and submission tarballs are written there.

DDP runs use `torchrun --nproc_per_node=N -m src.run ...` (see `scripts/train_flow_bptt_ft.sh` for the canonical multi-GPU pattern with `get_free_port` and `--rdzv_endpoint`).

## Two model families

This repo contains two parallel model stacks driven by separate Lightning modules. They share the encoder/map/agent token plumbing but diverge at the decoder and finetuning loop. Picking the wrong family for a script is the most common point of confusion.

**1. CAT-K / token AR (`SMART`, `EgoGMMSMART`)** — the published CVPR'25 work.
- `src/smart/model/smart.py`, `src/smart/model/ego_gmm_smart.py`.
- Tokenized AR over discrete agent + map vocabularies (`src/smart/tokens/*.pkl`, `TokenProcessor`).
- Closed-loop fine-tuning is **CAT-K rollout sampling** (`training_rollout_sampling.criterium=topk_prob_sampled_with_dist, temp=1e-5`) on top of `CrossEntropy` loss.
- Validation/submission metric: WOSAC 2024 (`WOSACMetrics`, `WOSACSubmission`).
- Experiments: `pre_bc.yaml`, `clsft.yaml`, `local_val.yaml`, `wosac_sub.yaml`, plus the `ego_gmm_*` variants.
- Scripts: `scripts/train.sh`, `scripts/local_val.sh`, `scripts/wosac_sub.sh`.

**2. SMART-Flow (`SMARTFlow`)** — a continuous flow-based extension layered onto SMART.
- `src/smart/model/smart_flow.py`, `src/smart/modules/smart_flow_decoder.py`, `src/smart/modules/flow_*`, `src/smart/tokens/flow_token_processor.py`.
- Flow ODE generates continuous trajectories; coarse rollouts feed Sim Agents 2025 (`SimAgentsMetrics`, `SimAgentsSubmission`, `HardSimAgentsMetrics`).
- `automatic_optimization = False` — the LightningModule drives the optimizer manually inside `_run_*_step` helpers (one per finetune mode). Do NOT add `optimizer.step()` calls expecting Lightning to handle them.
- Finetune mode dispatch lives on `model.model_config.finetune.mode` and is read by `set_model_for_finetuning` in `src/smart/utils/finetune.py`. Modes wired today: `adjoint_matching`, `rmm_bptt_ft`, `ref_nll_ft`, `kinematic_proj_ft`, `kinematic_reward_ft`, `flow_epg_ft`, `flow_rwr_ft`, `flow_dpo_ft`, `dice_ft`, plain `flow_bptt_ft` plus `terminal_cost_final_step`. Each has a matching `configs/experiment/*.yaml` and `scripts/train_flow_*.sh` wrapper.
- Experiments: `pre_bc_flow.yaml`, `flow_bptt_ft.yaml`, `flow_ref_nll.yaml`, `am_finetune_flow*.yaml`, `kinematic_*_ft.yaml`, `local_val_flow*.yaml`, `sim_agents_sub_flow.yaml`, etc.

When asked to add a feature, first identify which family the request targets — flow scripts/configs almost always have `flow` in the name; CAT-K scripts do not.

## Sim Agents metric variants (flow stack)

`model.model_config.validation_metric` selects the validation-side scorer for `SMARTFlow`:

- `real` → `SimAgentsMetrics` (official Waymo TF metrics, runs in a forkserver `mp.Pool`; pool size from `WOSAC_REAL_POOL_WORKERS`). Slow but authoritative.
- `hard` → `HardSimAgentsMetrics` (pure-PyTorch in-process; numerically equivalent within tolerance, much faster). Pool size from `WOSAC_HARD_POOL_WORKERS`. Set `WOSAC_TORCH_COMPILE=1` to compile the dno/ttc/d_road kernels.
- Set `WOSAC_VERIFY=1` to cross-check Hard vs Real per submetric (logs absolute deltas to stdout).

The **soft RMM** in `wosac_metametric_pytorch_differentiable.py` is the *training* objective for `rmm_bptt_ft` (differentiable surrogate). Validation never uses soft. See `docs/flow_bptt_finetuning_guide.md` for the full math (Korean).

The training loop caches scenario protos and GT log feature dicts in module-level dicts (`_SCENARIO_PROTO_CACHE`, `_LOG_FEAT_DICT_CACHE`) at the top of `smart_flow.py` — `rmm_bptt_ft` requires the per-scenario TFRecord at `${cache_root}/<split>_tfrecords_splitted/<scenario_id>.tfrecords`.

## Data preprocessing

`src/data_preprocess.py` converts raw WOMD `scenario` shards into per-scenario pickles (and optionally per-scenario `tfrecords_splitted/`). The flow BPTT path requires the splitted tfrecords to be present.

```sh
# Single split, single process:
sh scripts/cache_womd.sh                     # vars: DATA_SPLIT, INPUT_DIR, OUTPUT_DIR, NUM_WORKERS

# Parallelised (multi-job, multi-worker) with tfrecord shards — needed for rmm_bptt_ft training data:
NJ=8 NW=4 sh scripts/run_preprocess_train_with_tfrecords_parallel.sh
```

Use `--output_split closed_loop_train` (with `--write_tfrecords always`) when you need tfrecords for the *training* split without colliding with the plain `training/` cache.

## Common script entrypoints

| Task | Script | Notes |
|------|--------|-------|
| BC pre-train (CAT-K family) | `scripts/train.sh` | Edit `MY_EXPERIMENT` to switch experiment. |
| CAT-K fine-tune | `scripts/train.sh` with `MY_EXPERIMENT=clsft` | Loads BC ckpt via `ckpt_path` in YAML. |
| Local validation (CAT-K) | `scripts/local_val.sh` | Single GPU; `VAL_K` controls rollout `num_k`. |
| WOSAC submission packing | `scripts/wosac_sub.sh` | Writes `wosac_submission.tar.gz` under the run's output dir. |
| BC pre-train (Flow) | `scripts/train_flow.sh` |  |
| Flow BPTT fine-tune (RMM) | `scripts/train_flow_bptt_ft.sh` | Heavily parameterised by env vars; read header comment before changing. |
| Flow Ref-NLL fine-tune | `scripts/train_flow_ref_nll.sh` | Closed-loop BPTT + open-loop ref likelihood. |
| Adjoint-matching finetune | `scripts/train_flow_feasibility_full_grad.sh` |  |
| Local validation (Flow) | `scripts/local_val_flow.sh` |  |
| Sim Agents 2025 submission | `scripts/sim_agents_sub_flow.sh` | Set `ACTION=test` for the test split. |
| Parity / verification | `scripts/verify_wosac_*.py`, `scripts/parity_check_hard_rmm.py` | One-off checks for torch-vs-TF metric parity. |

The flow training scripts read most knobs from environment variables (LR, batch size, BPTT options, …). Override at the call site rather than editing the script:

```sh
LR=1e-6 TRAIN_B=8 BPTT_MAX_COARSE_STEPS=4 sh scripts/train_flow_bptt_ft.sh
```

The table lists canonical entrypoints; `scripts/` also contains close variants (`train_flow_consistency_bptt*.sh`, `train_kinematic_*_ft.sh`, `val_*.sh`, single-scenario / no-val variants, `monitor_exp.sh`). Skim `scripts/` before duplicating one — there is usually already a wrapper for the case you want.

## Things that bite

- `SMARTFlow.automatic_optimization = False` — when adding a new finetune mode, you own the `optimizer.zero_grad/backward/step` calls. Mirror an existing `_run_*_step` helper.
- `bptt_sequential_rollouts=true` is incompatible with DDP (multiple `backward()` calls confuse the bucket reducer → silent zero-grad). Single-GPU only.
- Hard vs Real RMM are equivalent in expectation but each maintains its own forkserver pool; both must be `close_pool()`-ed at trainer teardown to avoid orphaned TF processes.
- `data.train_raw_dir` defaults to the `validation/` split for `flow_bptt_ft` (because rmm_bptt_ft needs tfrecords and the legacy cache only had them on val). For real training runs, point at a `training/` cache that was preprocessed with `--write_tfrecords always`, otherwise you will leak val into train.
- WandB watches the model with `log="all"` in `src/run.py` — large grad/param histograms can dominate run time on big models; comment it out for profiling.
