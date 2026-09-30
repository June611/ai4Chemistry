# seed3407 traditional-model and mean baselines

## Scope and execution order

Current stage: establish dependencies, the exact descriptor list, and the data/evaluation contract. Next stage: implement and run XGBoost, Random Forest, and training-mean baselines locally; compare with the four existing Chemprop/Transformer runs.

- [x] Read both projects' declared dependencies.
- [x] Extract the final 29 RDKit descriptors from the reference notebook.
- [x] Identify the existing split files, target column, and physical unit.
- [x] Inspect installed environments and descriptor availability.
- [x] Verify split integrity and record the existing evaluation conventions.
- [x] Inspect online result layouts read-only: four model families, five training seeds each.
- [x] Define the prediction CSV interface for later result import.
- [x] Verify RDKit-29 calculation in this project's uv environment on qm9_full.
- [x] Finalize the environment recommendation and first-stage findings.
- [x] Install XGBoost 2.0.3 locally with uv and update pyproject.toml/uv.lock.
- [x] Fix explicit prediction-column precedence for Transformer evaluation; add regression tests.
- [x] Prepare Git-delivered dependency locks, checks, parsing fixes, and server pull/sync instructions.
- [x] Implement descriptor extraction with fixed column order, sample IDs, and explicit invalid/nonfinite-value handling.
- [x] Implement CPU XGBoost regression and Random Forest regression (single fit, raw Hartree labels).
- [ ] Next stage: implement the training-target mean baseline and verify normalization equivalence.
- [ ] Next stage: run all three baselines on this split; save configuration, versions, predictions, and metrics.
- [ ] Next stage: populate one comparison table with all seven runs and assess improvements over both traditional models and the mean baseline.

## Sources

- Reference notebook: `/home/june/deepLear_zone/Co_Elcd/ML4CoAdditives/2.feature_engineering/feature_engineering.ipynb`.
- Reference dependencies: `/home/june/deepLear_zone/Co_Elcd/ML4CoAdditives/pyproject.toml`.
- Data: `data/processed/qm9_full/random/seed3407` in this repository (corrected by the user).

## Dependencies

Reference pins: Python >=3.10, xgboost==2.0.3, rdkit==2022.9.5, scikit-learn==1.5.1, numpy==1.26.4, pandas==2.2.2, scipy==1.13.1. RandomForestRegressor is provided by scikit-learn; no separate random-forest dependency is needed.

This repository requires Python >=3.11,<3.15 and rdkit>=2024.3. The reference RDKit pin conflicts with that requirement.

Verified on 2026-09-30: the reference project's existing `.venv/bin/python` runs Python 3.11.9 and has exactly all six reference package versions listed above. This is reference provenance, not the selected runtime. Per the user's updated requirement, use this project's uv-managed `.venv` for RDKit and subsequent baseline work.

The project environment uses Python 3.11.9, RDKit 2026.3.6, scikit-learn 1.5.1, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, and XGBoost 2.0.3. scikit-learn was pinned to the reference's 1.5.1 after model-saving tests exposed an incompatibility between XGBoost 2.0.3 and scikit-learn 1.9.1. Joblib is now explicit. uv.lock records the resolved versions. Keep the project's compatible RDKit version; do not force the conflicting 2022.9.5 pin. Descriptor names match the reference, but numerical equivalence across RDKit versions has not been established. All installations are local; server installation is via Git pull followed by uv sync --locked.

## Exact feature contract

Notebook code cell 12 defines 30 selected_columns; cell 16 drops BCUT2D_LOGPHI and saves gap_29.pkl. Preserve the resulting order:

```text
SMR_VSA7, FractionCSP3, SMR_VSA10, BertzCT, SlogP_VSA6,
BCUT2D_MRHI, HallKierAlpha, MolLogP, BCUT2D_MWHI, SMR_VSA5,
BalabanJ, PEOE_VSA11, BCUT2D_CHGHI, SMR_VSA9, PEOE_VSA2,
SlogP_VSA2, BCUT2D_LOGPLOW, VSA_EState2, BCUT2D_MWLOW,
VSA_EState4, VSA_EState5, BCUT2D_MRLOW, Kappa3, fr_piperzine,
fr_piperdine, fr_Ar_NH, fr_aniline, fr_imidazole, fr_pyridine
```

Reuse these fixed descriptors. Do not rerun the notebook's feature selection or its random_state=42 split on the new dataset.

Validation: extracted selected_columns using Python AST, removed BCUT2D_LOGPHI, and confirmed 29 remaining names. The earlier 60-row check used the subset and reference environment; it is not validation of the full dataset. The project-local `scripts/check_rdkit.py` now defaults to qm9_full and checks descriptor availability, molecule parsing, finite values, and agreement between calculator and direct descriptor calls. Full-data extraction remains a next-stage task.

Project uv verification passed on 2026-09-30: `uv run --locked python scripts/check_rdkit.py` checked 100 molecules per full-data split (300 total), all 29 descriptors finite and matching direct calls. RDKit and Chemprop 2.3.1 imported together successfully. After adding XGBoost, uv sync --locked and uv pip check passed for 115 installed packages. `--check-models` also exercises CPU XGBoost and Random Forest fitting/prediction on the sampled data; this is not a formal training run.

Validation after the dependency and parser changes: all 30 project tests passed, including CPU Chemprop and Transformer smoke tests. Ruff passed for the changed Python files. The RDKit-29/XGBoost/Random Forest combined smoke check passed in the local uv environment. Server sync commands are documented in UV_GUIDE.md; no server environment changes were made.

## Data and evaluation contract

- Existing random split seed: 3407. Train: 107,038; validation: 13,380; test: 13,380.
- All three CSVs contain sample_id, smiles, homo, lumo, delta_e. Labels are in Hartree.
- Train on train.csv; use val.csv for any tuning; reserve test.csv for final evaluation.
- Mean baseline predicts mean(train.delta_e) for every test sample, without including validation or test labels in the mean.
- Fit any preprocessing on training data only; retain sample_id alignment when scoring predictions.
- Compare in original Hartree units using the project's common regression metrics.
- Verified unique sample IDs within each split and zero pairwise overlaps of both IDs and stored SMILES between train, validation, and test.
- `src/molgap/metrics/regression.py` reports n, MAE, RMSE, R2, and percentage MAPE with a 1e-12 zero-target exclusion threshold and count. It masks nonfinite pairs, so the runner should explicitly reject nonfinite labels/predictions and require n=13380 for a complete test evaluation. Include MAPE and its valid count in the final exported comparison.
- For an affine normalization z=(y-a)/b, inverse_transform(mean(z_train)) equals mean(y_train), up to floating-point error. This does not apply generally to nonlinear transforms such as log. MAE/RMSE measured in normalized units differ in scale; compare after inverse transformation.
- Training-mean prediction is a constant baseline, not literal random guessing. Its test R2 need not equal zero, because test R2 uses the test mean as its reference.

## Comparison table to populate

The four existing run identities and artifacts must be verified before filling their metrics. Pending values are not experimental results.

| Run | Test MAE (Hartree) | Test RMSE (Hartree) | Test R2 | Status |
| --- | --- | --- | --- | --- |
| chemprop_single_gap | pending | pending | pending | Remote artifact identified |
| chemprop_multitask | pending | pending | pending | Remote artifact identified |
| chemprop_consistency | pending | pending | pending | Remote artifact identified |
| smiles_transformer | pending | pending | pending | Remote artifact identified |
| RDKit-29 + XGBoost | pending | pending | pending | Next stage |
| RDKit-29 + Random Forest | pending | pending | pending | Next stage |
| Training delta_e mean | pending | pending | pending | Next stage |

Before comparison, verify dataset, split identity, target definition, sample IDs, and units for each existing run. Report conclusions for this seed/split without treating one split as a multi-seed robustness result.

## Existing-result audit and unresolved comparison input

- Read-only inspection confirmed results for chemprop_single_gap, chemprop_multitask, chemprop_consistency, and smiles_transformer under `/root/lanyun-tmp/evan/ai4Chemistry/outputs/<model>/seed<training_seed>/`. `outputs/batches` contains orchestration metadata, not the model metrics themselves.
- That matrix uses `configs/data/qm9_full.yaml`, fixed split_seed=3407, and separate training seeds [3407, 42, 2026, 7, 123]. Distinguish split seed from training seed in every report.
- The local `outputs/exp001_single_gap/seed3407/config.yaml` also points to qm9_full, consistent with the corrected data requirement.
- All 20 metrics.json files were located. The four training-seed3407 configs, metrics, and prediction headers were inspected; no server installation or modification was performed. No result files have been downloaded. Keep the comparison table pending until the formal comparison step.
- On import, verify exact test membership, training seed, target units, and dataset provenance before populating the four result rows.

The current remote and local split CSV SHA256 hashes match exactly:

```text
train 1e14487f4607f8f0b71443447ce683da195df476b4f8a982a2aa18fd4c44023c
val   b4bafe0cd8ba08cc6d81202d1f9ce313c17db3628dc945a67223ad617becc579
test  a2bf1945045c83a8e4b31daff16834b018298cf458df3f7226f4d648226bce45
```

## Reserved result interface

The actual Chemprop CSV contains sample_id, smiles, homo, lumo, delta_e; delta_e is its prediction. Transformer CSV contains sample_id, smiles, delta_e, delta_e_pred; delta_e is its truth and delta_e_pred its prediction. New baseline predictions should use sample_id, smiles, delta_e_pred in original Hartree units and cover all 13,380 test IDs exactly once. `molgap.training.evaluate.evaluate_files` now prioritizes explicit prediction columns, aligns by sample_id, and rejects nonfinite values. Check the full evaluated count in the future importer.

Keep metadata alongside each CSV: model name, training seed, split seed (3407), dataset (qm9_full), unit (hartree), and SHA256 of the matching test.csv. Existing metrics.json files may also be retained, but recomputation from predictions provides a common evaluation convention. This is the reserved file interface; automated import and comparison remain future tasks.

## Next-stage implementation sequence

1. Add a baseline runner using the now-installed XGBoost, RDKit and scikit-learn in this project's uv environment. Persist the ordered descriptor names and actual dependency versions with the experiment configuration.
2. Calculate all split descriptors, checking molecule parsing and finite values; preserve sample IDs and split membership. Log file hashes and package versions.
3. Train CPU XGBoost and Random Forest with recorded training seed and hyperparameters. If tuning is used, select only on validation data and retain the original training set definition.
4. Predict the training-label mean on test and verify that affine normalization followed by inverse transformation gives the same predictions and original-unit metrics.
5. Save per-sample predictions and project-compatible metrics for each baseline, then fill the comparison table only with matched existing runs.

Implementation is tracked in ML_IMPLEMENTATION_TODO.md, with commands in ML_TRAINING.md. The single-run ML training code and mixed-count summary are implemented; full-data model execution, mean baseline, and final result comparison remain separate follow-up tasks.
