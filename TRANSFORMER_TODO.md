# SMILES Transformer experiment TODO

## Shared data contract

- [x] Audit the unfinished split changes with `git diff`.
- [x] Confirm both QM9 split manifests are deterministic and have no pending confirmations.
- [x] Validate Transformer split membership and row order against Chemprop's existing manifest.
- [x] Record split-manifest and input CSV SHA-256 values in every Transformer run.

## Tokenization and dataset

- [x] Implement the specified standard SMILES regex tokenizer with full-string coverage checks.
- [x] Build vocabulary from the training split only with `[PAD]`, `[UNK]`, and `[CLS]`.
- [x] Enforce configured `max_length: 40` and report actual token-length/unknown-token statistics.
- [x] Keep `delta_e` in Hartree and fit any target scaler on training labels only.

## Model and training

- [x] Implement a compact Transformer encoder and two-layer regression head.
- [x] Support `[CLS]` and mean pooling through YAML.
- [x] Configure AdamW, warmup/cosine decay, dropout, batch size, clipping, and early stopping in YAML.
- [x] Monitor validation MAE in Hartree and report test MAE and R².
- [x] Save best checkpoint, portable model, tokenizer vocabulary, predictions, metrics, and metadata.

## Interfaces and verification

- [x] Add `molgap-train-transformer` with `--dry-run` and overwrite protection.
- [x] Document the experiment configuration and commands in README.
- [x] Test tokenizer examples, malformed input, split identity, model shape, and parameter budget.
- [x] Run a tiny CPU train/predict smoke test.
- [x] Run Ruff, formatting, full tests, config dry-run, and lock consistency checks.
