# RDKit-29 single-run ML implementation

## Agreed contract

Use the existing qm9_full/random/seed3407 split. Compute the fixed 29 descriptors from SMILES only; never use homo/lumo/delta_e as features. No feature or target normalization; predictions are in Hartree. Each model fits once with random_state=3407, without Optuna, resplitting, or a multi-seed batch. Keep source CSVs unchanged.

## Tasks

- [x] Extract the ordered RDKit-29 feature definition into a shared module; validate molecules and finite values.
- [x] Add XGBoost and Random Forest configurations with fixed reference-notebook parameters and configurable CPU threads.
- [x] Implement split validation and single-fit training; preserve IDs and reject leakage/invalid values.
- [x] Save model, config, environment/input hashes, features, normalization, timing, metrics, and test predictions under outputs/<model>/seed3407.
- [x] Adapt summary validation for explicitly declared single-run models, with null standard deviation and no error bars for n=1.
- [x] Test training, reload equivalence, original-unit predictions, artifact parsing, invalid inputs, overwrite protection, and mixed run counts.
- [x] Run project tests, Ruff, configuration checks, and document local/server commands.

## Deferred experiments

Full-data XGBoost/Random Forest training and the final scientific comparison remain follow-up tasks.

## Mean baseline follow-up

- [x] Add a deterministic train-only mean model without descriptor calculation or normalization.
- [x] Reuse the single-run config, model, metric and prediction artifact interface.
- [x] Verify train-only fitting with deliberately different held-out labels and constant-label edge cases.
- [x] Verify standardization/inverse transformation preserves the training mean.
- [x] Execute on qm9_full seed3407; validate all 13380 test IDs, predictions, reloaded model, and recomputed metrics.
- [x] Export summary tables/plots using the existing comparison command; document seven-model usage.
- [x] Verify all 49 project tests and Ruff checks after adding the mean baseline.

## Verification findings

- XGBoost 2.0.3 fitting passed with scikit-learn 1.9.1, but save_model failed because the estimator metadata interface changed. Pinning scikit-learn to the reference project's 1.5.1 fixes save/reload and preserves the existing deep-learning tests.
- Both ML configs pass dry-run against the full 107038/13380/13380 split without creating formal run artifacts.
- All 47 project tests passed, including both ML save/reload paths and the existing CPU deep-learning smoke tests.
- Ruff lint/format checks and uv sync --locked passed. Formal training directories were not created.
- Run commands and the full artifact contract are in ML_TRAINING.md.
