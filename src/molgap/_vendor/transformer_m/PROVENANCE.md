# Transformer-M source provenance

Source: https://github.com/lsj2408/Transformer-M
Commit: `45647e143a5282e0e97117969396446084bcf1ab` (MIT; LICENSE retained).

The five modules here come from `Transformer-M/modules/` at that commit.
Original and adapted SHA256 values are in `upstream_files.json`.

Changes: redirect Fairseq imports to a local compatibility module implementing
the configured PyTorch-equivalent operations; add optional non-reentrant layer
activation checkpointing; initialize the unused QM9 denoising return value to
None (the upstream QM9 forward references it without assigning it).
The project's identity-checked geometry is always present: do not disable the
3D channel for an all-zero coordinate tensor. A single heavy atom at the origin
is valid. The upstream sentinel caused that molecule's result to depend on batch
composition; an explicit regression test reproduces and fixes this case.

Attention, Gaussian distance encoding, embeddings, normalization ordering and
QM9 encoder computations otherwise remain upstream code. The adapter implements
the upstream CLS classification head and the default QM9 `no_2d=True` path.
External pretraining weights, Fairseq orchestration, Cython graph processing and
the pretraining denoising task are not used. Dataset handling and experimental
budget follow this project's existing protocol, not the original 600k updates.
