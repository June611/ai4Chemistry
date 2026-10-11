# Project instructions

Read the root README and `docs/features/paired-seed-100epoch-benchmark/` before changing training commands. Organize documents by feature, not conversation.

## Current protocol

- Exactly three matched (split seed, training seed) pairs: (3407,3407), (42,42), (2026,2026). No five-seed or Cartesian-product runs.
- Neural models including 3D Infomax pretraining: batch 256, at most 100 epochs, early-stopping patience 20. Do not silently lower batch on OOM.
- Traditional baselines fit train only once per split; batch/epoch/patience are N/A.
- Formal outputs: `outputs/paired_gap_mae_v2/`. Direct legacy Chemprop/sequence configs can still write under `outputs/`.
- Exclude archive configs, exp003_consistency_0, exp004_consistency_01, egnn_test, spe_transformer_10epoch from formal runs.
- Preserve common split identity and Hartree labels. Fit scalers and tokenizers on train only; select checkpoints on validation only; use test for final evaluation.
- Budgets, seed pairs and raw Hartree gap MAE selection are unified. Loss and label scaling remain model-specific by explicit decision; do not describe the optimization recipes as identical.
- Report 3D inputs, extra HOMO/LUMO supervision and pretraining as method differences. DeMol is a proposal, not an implemented adapter.

## Environment and entry points

Python 3.11, Chemprop 2.3.1, PyTorch 2.6.0 CUDA 12.4, locked uv dependencies. Do not change installed Chemprop or replace the pinned CUDA wheel.

```bash
uv sync --locked --extra qm9
uv run python -m molgap.qm9_models.bootstrap
uv run python scripts/audit_paired_seed_protocol.py
uv run python scripts/audit_qm9_split_seeds.py
uv run ruff check .
uv run pytest -q
```

Formal matrices: `configs/multiseed*.yaml` and `configs/spe_affinity_ablation_multiseed*.yaml`; each contains one matched seed. QM9/ML launchers: `scripts/train_qm9_models.sh`, `scripts/train_ml_paired.sh`. First pretrain with `scripts/pretrain_3dinfomax_paired.sh` and provide a split-specific checkpoint template for the tenth QM9 model. Full commands are in README.

## Evidence and handoff

Record requirements, decisions, verification and pending work in each feature's README.md, tasks.md, decisions.md. Mark unexecuted preflights and server training as pending. Smoke tests do not prove convergence or paper-level reproduction. Three-run SD includes changes in split and training randomness.

Data, outputs, upstream checkouts and archives are separate artifacts. Ignore rules do not untrack files already in Git. Do not rewrite published history without authorization. See `docs/guides/data-transfer.md`.
