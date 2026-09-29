# Training throughput and progress Todo

- [x] 1. Audit the four training paths and Chemprop 2.3.1 capabilities.
  - Chemprop CLI already enables Lightning tqdm and exposes train/validation loss.
  - Chemprop CLI does not expose Lightning `precision`; do not add an ineffective AMP key.
  - Phase 3 and Transformer use project-owned Lightning trainers.
- [x] 2. Archive the current experiment YAML files before changing hyperparameters.
- [x] 3. Set all four primary models to batch size 256 and early-stopping patience 20.
- [x] 4. Scale Chemprop learning rates with the 4x batch increase; keep Transformer LR 3e-4.
- [x] 5. Set data-loader workers to 8 and verify pin-memory behavior.
  - Transformer and project-owned Phase 3 use pinned memory and persistent workers.
  - Chemprop 2.3.1 CLI exposes workers but does not expose a pin-memory option.
- [x] 6. Add explicit tqdm progress and auditable elapsed-time reporting.
  - Direct trainers show epoch metrics with Lightning tqdm and save `training_timing.json`.
  - The multi-seed runner shows run-level elapsed time/ETA and records per-command durations.
- [x] 7. Validate target scaling, fixed split identity, metrics, and five-training-seed behavior.
  - All models use `data/processed/qm9_full/random/seed3407`; the Transformer verified the
    Chemprop manifest SHA-256 and exact 107038/13380/13380 split rows.
  - Target scalers are fit on train only. Phase 3 now applies that scaler to validation, matching
    Chemprop CLI behavior, while final test evaluation remains in Hartree.
  - The matrix dry-run generated seeds `3407, 42, 2026, 7, 123` against the same processed path.
- [x] 8. Run config dry-runs, Ruff, formatting, tests, and document server commands.
  - Four model dry-runs passed.
  - `uv lock --check`, Ruff lint, Ruff formatting, and all 26 tests passed.
  - README documents throughput settings, progress visibility, worker concurrency, and commands.

## Fixed experimental controls

- Data split: `data/processed/qm9_full/random/seed3407`
- Training seeds: `3407, 42, 2026, 7, 123`
- Epoch budget: 100
- Batch size: 256
- Early-stopping patience: 20
- Evaluation unit: Hartree

## Precision decision

Chemprop 2.3.1 constructs its CLI Lightning trainer without a `precision` argument and its parser has
no `--precision` option. The comparable four-model run therefore remains full precision. Phase 3 and
Transformer will not silently enable AMP while Phase 1/2 cannot receive the same setting.
