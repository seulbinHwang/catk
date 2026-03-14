# SMART-flow 7M on `brand_new`

This patch adds a **Flow matching based 2-second trajectory decoder** on top of the existing SMART scene-context trunk.

## What is included

The patch keeps the original `brand_new` map/context trunk intact and adds a new flow path:

- `src/smart/model/smart_flow.py`
- `src/smart/modules/smart_flow_decoder.py`
- `src/smart/modules/flow_agent_decoder.py`
- `src/smart/modules/flow_local_decoder.py`
- `src/smart/tokens/flow_token_processor.py`
- `src/smart/metrics/flow_metrics.py`
- `configs/model/smart_flow.yaml`
- `configs/experiment/pre_bc_flow.yaml`
- `configs/experiment/local_val_flow.yaml`
- `configs/experiment/wosac_sub_flow.yaml`
- `configs/logger/wandb.yaml`
- `scripts/train_flow.sh`
- `scripts/local_val_flow.sh`
- `scripts/wosac_sub_flow.sh`

## Core design

### Training

1. Run the original tokenization first.
2. Build a 14-slot causal context pack `{5,10,...,70}`.
3. Build 13 valid supervision anchors `{10,15,...,70}`.
4. Build normalized future targets `[x/20, y/20, cos(dyaw), sin(dyaw)]` for 20 future 10 Hz steps.
5. Use the original SMART trunk to encode scene context.
6. Use the new hierarchical flow decoder to predict velocity targets.
7. Train with a single flow-matching MSE loss.

### Inference

1. Start from the observed coarse history.
2. Run the SMART trunk once per 0.5-second rollout step.
3. Run the local flow decoder inside a 4-step ODE sampler to generate 2 seconds.
4. Commit only the first 0.5 seconds.
5. Keep geometry as continuous state.
6. Re-tokenize only to pick the next semantic token embedding.
7. Repeat for 16 coarse steps to fill 80 future 10 Hz steps.

## Important implementation choices

- The ODE loop never re-runs the full scene trunk.
- The new agent-to-agent relation adds two **relative coarse-motion** channels instead of raw m/s velocity. This keeps the added relation channels on a meter-scale comparable to the existing distance feature and avoids introducing a separate normalization rule into the main trunk.
- `pred_traj_10hz`, `pred_head_10hz`, and `pred_z_10hz` stay unchanged so the existing WOSAC submission path remains usable.

## Training

Use the new flow config:

```bash
bash scripts/train_flow.sh
```

This launches:

```bash
python -m src.run experiment=pre_bc_flow
```

Recommended starting point in `pre_bc_flow.yaml`:

- `precision=bf16-mixed`
- `max_epochs=64`
- `train_batch_size=12` per GPU on 6x H100
- `val_batch_size=4`
- `test_batch_size=4`
- `lr=5e-4`
- `lr_warmup_steps=2`

## Local validation

```bash
bash scripts/local_val_flow.sh
```

This uses `local_val_flow.yaml` with 32 closed-loop rollouts.

## WOSAC submission

1. Put your checkpoint path in `configs/experiment/wosac_sub_flow.yaml`.
2. Fill the submission metadata fields.
3. Run:

```bash
bash scripts/wosac_sub_flow.sh
```

The output interface matches the original repository:

- `pred_traj_10hz`
- `pred_head_10hz`
- `pred_z_10hz`

so the existing WOSAC packing path stays intact.

## Dataset and preprocessing

This patch assumes the original `brand_new` dataset preparation flow remains unchanged:

1. Download WOMD v1.2.1.
2. Run the original cache script:

```bash
bash scripts/cache_womd.sh
```

3. Prepare `training`, `validation`, and `testing` caches exactly as in the base repository.

## WandB

`configs/logger/wandb.yaml` is switched to:

- `project=SMART-FLOW`
- `entity=jksg01019-naver-labs`

Adjust these if needed.