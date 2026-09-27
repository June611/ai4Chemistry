# Training pipeline TODO

## Shared foundations

- [x] Audit the current data, configuration, training, prediction, and evaluation interfaces.
- [x] Make the processed Chemprop splits reusable by Phase 1 and Phase 2 with identical rows.
- [x] Define and validate one YAML schema for experiment, data, model, training, loss, evaluation, and output settings.
- [x] Record the resolved configuration, command, environment, data lineage identity, and random seeds per run.
- [x] Align predictions and truth by `sample_id`; report MAE, RMSE, and R² per target.

## Phase 1 — single-target Chemprop baseline

- [x] Configure `SMILES -> delta_e` with the official Chemprop CLI and no Chemprop source changes.
- [x] Build train/predict/evaluate commands entirely from YAML.
- [x] Validate the Phase 1 CLI with dry-run and tests.

## Phase 2 — native Chemprop multitask baseline

- [x] Configure `SMILES -> homo, lumo, delta_e` with equal task weights.
- [x] Reuse exactly the Phase 1 train/val/test molecule assignments.
- [x] Report per-target metrics and prediction consistency error.
- [x] Validate the Phase 2 CLI with dry-run and tests.

## Phase 3 — physics-consistent multitask model

- [x] Implement the consistency loss in `src/molgap/models/` without editing Chemprop.
- [x] Build a Chemprop D-MPNN through its public Python API from YAML settings.
- [x] Keep HOMO, LUMO, gap, and consistency weights configurable for ablation.
- [x] Add Phase 3 train/predict support and artifact recording.
- [x] Validate loss mathematics, gradients, model construction, and a tiny CPU smoke run.

## Final checks

- [x] Run Ruff lint and format checks.
- [x] Run the full unit-test suite.
- [x] Verify all three configurations load and dry-run successfully.
- [x] Update README commands, experiment matrix, outputs, and limitations.
