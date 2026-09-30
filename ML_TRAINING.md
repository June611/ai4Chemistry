# Single-run RDKit-29 baselines

## Environment

Run commands from the project root, locally or after pulling the repository on the server:

```bash
uv sync --locked
uv run --locked python scripts/check_rdkit.py --check-models
```

XGBoost is pinned to 2.0.3 and scikit-learn to the reference project's 1.5.1. XGBoost 2.0.3 could fit with scikit-learn 1.9.1 but failed to save its sklearn estimator metadata; the 1.5.1 pin is verified by save/reload tests. RDKit remains at the project's locked compatible version, not the reference's conflicting 2022.9.5. Joblib is an explicit dependency for Random Forest serialization. Python 3.11 is the tested runtime (`.python-version`).

## Data and normalization

Both configs read `data/processed/qm9_full/random/seed3407/{train,val,test}.csv`. The input files remain unchanged, and no new split is generated. Only SMILES enter the feature extractor. `homo`, `lumo`, and `delta_e` are never model inputs; `delta_e` is the training target. Features follow the reference notebook's fixed 29-column order, with BCUT2D_LOGPHI excluded.

Feature and target normalization are both `none`. Models predict original Hartree units. The loader rejects missing/duplicate IDs, overlapping split IDs or SMILES, and nonfinite labels. Feature extraction additionally rejects invalid/empty molecules, nonfinite descriptors, and cross-split canonical molecule overlap. Rows are never silently dropped. Descriptors are calculated independently per split without fitting a scaler or selecting features.

## Validate and train

Validate configs and all split tables without fitting or writing artifacts:

```bash
uv run --locked molgap-train-ml --config configs/xgboost_rdkit29.yaml --dry-run
uv run --locked molgap-train-ml --config configs/random_forest_rdkit29.yaml --dry-run
```

Dry-run checks tabular structure and split overlap; full molecular parsing and descriptor checks happen during training. Train each model once when ready:

```bash
uv run --locked molgap-train-ml --config configs/xgboost_rdkit29.yaml
uv run --locked molgap-train-ml --config configs/random_forest_rdkit29.yaml
```

Each command makes one `fit` call on train only, then one validation prediction and one test prediction. There is no Optuna, cross-validation, early stopping, train+validation refit, or multi-seed loop. Validation metrics are diagnostic only. `training.seed=3407` and `training.run_mode=single` are enforced. The CPU thread count is configurable through `training.n_jobs` (default 4).

Initial hyperparameters are fixed from the reference final notebooks: XGBoost uses 293 trees, depth 8, learning rate 0.032845274, gamma 0.000123129, and min_child_weight 2; Random Forest uses 170 trees, depth 35, min_samples_split 2, and min_samples_leaf 1. XGBoost uses CPU hist. These are starting configurations, not parameters optimized for this full dataset.

Existing output directories are rejected, including partially completed runs. To deliberately run another configuration, use a new `experiment.name`; the runner never overwrites or deletes prior results. These commands are independent of `molgap-run-multiseed`.

## Artifacts

Each model writes to its standard run directory:

```text
outputs/xgboost_rdkit29/seed3407/
outputs/random_forest_rdkit29/seed3407/
    config.yaml
    environment.json
    feature_names.json
    normalization.json
    model.json                 # XGBoost only
    model.joblib               # Random Forest only
    training_timing.json
    validation_metrics.json
    metrics.json               # test metrics; written last on success
    predictions/test_predictions.csv
```

`environment.json` records actual package versions and input hashes. `normalization.json` explicitly records no scaler and Hartree labels. Test CSV columns are `sample_id,smiles,delta_e,delta_e_pred`, preserving the test-file row order. `metrics.json` contains `rows`, `unit`, and `targets.delta_e` with MAE, RMSE, R2, MAPE (%) and evaluated counts; only test metrics use the name `metrics.json` so recursive summaries cannot mistake validation metrics for another run.

`molgap.training.ml.load_estimator(run_dir)` reloads the saved estimator. For inference, compute features in the saved `feature_names.json` order with the recorded RDKit version. The test suite verifies predictions before/after reload and recomputes metrics from exported CSVs. Only load joblib files from trusted runs.

## Compare with deep learning

Once all formal runs exist:

```bash
uv run --locked molgap-summarize \
  outputs/chemprop_single_gap outputs/chemprop_multitask \
  outputs/chemprop_consistency outputs/smiles_transformer \
  outputs/xgboost_rdkit29 outputs/random_forest_rdkit29 \
  --seeds 3407 42 2026 7 123 \
  --output-dir outputs/comparisons/qm9_full_dl_ml
```

The summary reads `training.run_mode` from each saved config. Explicit `single` runs must contain only seed3407; other models must contain all requested seeds. Missing deep-learning runs still fail validation, and duplicate ML runs are rejected. No ML result is replicated to fill the five-seed set.

Single-run results have `runs=1`, JSON `std=null`, an empty CSV SD field, and `SD N/A` in tables. Plots label the number of runs and omit error bars for single runs. Repeated runs retain their mean and sample SD. This summarizes variation across training seeds on one fixed split, not variation across independent dataset splits. For a direct one-run comparison, pass the six `seed3407` directories and `--seeds 3407`.

Full-data XGBoost and Random Forest training remain separate follow-up tasks. Their implementation verification uses small synthetic splits and does not populate their formal output directories.

## Training-mean baseline

```bash
uv run --locked molgap-train-ml --config configs/training_mean.yaml
```

This baseline uses sklearn DummyRegressor(strategy="mean"), fitted exclusively to train.delta_e. Validation and test labels never contribute to the constant prediction. No RDKit descriptors are computed; feature_names.json is an empty list. A dummy input column supplies only the number of rows, not molecular information. The run uses the same single-run artifact layout under `outputs/training_mean/seed3407/`, including model.joblib and predictions/test_predictions.csv. mean_baseline.json additionally records the training count, mean in Hartree, and a train-only standardization/inverse-transformation equivalence check. Constant training labels are handled using a unit scale for that check. Actual predictions remain unnormalized.

The local full-data run produced a training mean of 0.2511385199648723 Hartree from 107038 training rows. All 13380 test rows receive this constant. Test MAE=0.03962738219940107 Hartree, RMSE=0.047468370054752476 Hartree, MAPE=16.873272567045355%, R2=-2.151474736145076e-7. The small negative R2 reflects the difference between training and test means; R2 uses the test mean as its reference. The normalization check's inverse-transformed mean matched the original mean exactly in this run.

For the seven-model comparison, add `outputs/training_mean` to the six-model summary command above. It will appear as one run, with no SD or error bar. For a standalone export:

```bash
uv run --locked molgap-summarize outputs/training_mean \
  --output-dir outputs/comparisons/training_mean
```

The baseline code/config are versioned; generated outputs remain local under the repository's existing outputs ignore rule. After pulling on another machine, run the baseline command once to reproduce its prediction files using that machine's copy of the same split. Existing output directories remain protected against overwrites.
