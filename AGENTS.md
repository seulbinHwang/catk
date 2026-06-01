# Repository Guidelines

## Project Structure & Module Organization

This repository trains and evaluates CAT-K flow-matching models for Waymo Open Sim Agents. Core Python code lives in `src/`: `src/run.py` is the Hydra entry point, `src/smart/model/` contains the Lightning model, `src/smart/modules/` contains decoder, control, and self-forced components, and `src/smart/metrics/` contains WOSAC/RMM/CPD evaluation code. Configuration is under `configs/`, with experiment recipes in `configs/experiment/`. Tests live in `tests/`. Operational scripts are in `scripts/`, analysis utilities in `tools/`, dependencies in `install/`, and documentation/assets in `docs/`.

## Build, Test, and Development Commands

Create the recommended environment from `README.md`:

```bash
conda create -n catk python=3.11.9 -y
conda activate catk
python -m pip install -r install/requirements.txt
```

Run training and evaluation through Hydra:

```bash
python -m src.run experiment=pre_bc_flow task_name=<name>
python -m src.run experiment=local_val_flow action=validate ckpt_path=<ckpt>
bash scripts/train_flow.sh
bash scripts/local_val_flow.sh
```

For quick checks, use focused pytest runs such as:

```bash
pytest tests/test_kinematic_control.py -q
pytest tests/test_fast_wosac_metric.py -q
```

## Coding Style & Naming Conventions

Use Python 3.11 syntax, 4-space indentation, type hints where they clarify tensor/config contracts, and explicit imports. Follow existing naming: modules and functions use `snake_case`, classes use `PascalCase`, constants use `UPPER_SNAKE_CASE`, and Hydra config files use descriptive lowercase names such as `finetune_flow_prefix_valid_a100_4x2.yaml`. Keep changes scoped; prefer existing helpers in `src/smart/utils/`, `src/utils/`, and current module patterns over new abstractions.

## Testing Guidelines

Tests use `pytest` and are named `tests/test_*.py`. Add or update focused regression tests when changing metric logic, rollout behavior, token processing, kinematic control, samplers, or config synchronization. Prefer small CPU tests when possible; GPU or Waymo-dependent checks should be clearly documented in the test or PR. Before larger training changes, run a targeted pytest plus an import smoke test for touched modules.

## Commit & Pull Request Guidelines

Recent history uses concise, imperative or scope-prefixed summaries, sometimes in Korean, for example `smart_flow: ...` or `multi-anchor: ...`. Keep commits focused on one behavior or fix. PRs should explain the motivation, list key config changes, include commands/tests run, link related issues or experiments, and attach W&B/log paths or metric screenshots when training behavior changes.

## Security & Configuration Tips

Do not commit secrets, W&B credentials, Waymo browser storage state, checkpoints, or generated submission archives. Keep local paths and credentials in environment variables or ignored files under `secrets/`. For shared GPU machines, set `CUDA_VISIBLE_DEVICES` explicitly before smoke training or validation.
