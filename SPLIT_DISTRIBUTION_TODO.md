# Delta-E split distribution TODO

## Audit current implementation and outputs

- [x] Inspect `src/molgap/data/split.py` and locate every caller.
- [x] Confirm the current `random` method only permutes rows and does not use `delta_e`.
- [x] Save before-change statistics and overlaid train/val/test density plots for both datasets.
- [x] Quantify distribution differences with descriptive statistics and two-sample distances.

## Implement regression-aware splitting

- [x] Add deterministic quantile-bin stratification to `split.py` while retaining plain random mode.
- [x] Make stratification parameters explicit in each data YAML.
- [x] Record the target, requested/effective bin counts, and row bin in `split_manifest.csv`.
- [x] Preserve exact split sizes, unique-SMILES isolation, scaffold isolation, and deterministic order.
- [x] Add validation and tests for distribution balance, determinism, and invalid settings.

## Rebuild and compare

- [x] Regenerate `qm9_full` and `qm9_cn_subset` artifacts without modifying raw CSV files.
- [x] Verify row counts, SHA-256 lineage, no leakage, and repeat-run determinism.
- [x] Generate after-change statistics and overlaid density plots for both datasets.
- [x] Write a before/after comparison report and document commands and limitations.

## Final verification

- [x] Run Ruff lint and format checks.
- [x] Run the full test suite.
- [x] Confirm both data configurations complete with no pending confirmations.
