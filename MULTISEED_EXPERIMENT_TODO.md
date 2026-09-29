# Five-seed training and comparison TODO

## Training audit

- [x] Validate Chemprop Phase 1/2/3 commands against the installed Chemprop CLI.
- [x] Validate the full-data SMILES Transformer dry-run and device configuration.
- [x] Confirm README contains runnable commands for every model.
- [x] Ensure every training path produces a comparable `delta_e` metrics artifact.

## Five-seed execution

- [x] Define one YAML matrix with five seeds and all model configurations.
- [x] Generate seed-specific preprocessing and training configs without editing source YAML.
- [x] Guarantee every model for a seed reads the same processed split directory.
- [x] Add configurable single-GPU concurrency (`parallel_jobs` / `--jobs`).
- [x] Capture per-run logs, return codes, generated configs, and batch status.
- [x] Support optional split preparation, dry-run, overwrite, and model filtering.

## Metrics and comparison

- [x] Add MAE, RMSE, R², and zero-safe MAPE to the common regression metrics.
- [x] Read both Chemprop nested metrics and Transformer flat metrics.
- [x] Preserve per-seed records and calculate per-model mean/std/min/max.
- [x] Validate expected seed completeness and reject duplicate model/seed records.
- [x] Export JSON, CSV, Markdown, a metric bar chart, and a table image.
- [x] Support both one-model summaries and multi-model comparisons.

## Verification and documentation

- [x] Add tests for matrix expansion, command construction, MAPE, aggregation, and plots.
- [x] Run tiny CPU smoke tests and all existing tests.
- [x] Run Ruff, formatting, config dry-runs, CLI help, and lock checks.
- [x] Document data preparation, parallel training, summary, and comparison commands.
