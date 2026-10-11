#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=pytorch
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/chempro-mpl}"

paired_seeds=( ${QM9_PAIRED_SEEDS:-3407 42 2026} )
for seed in "${paired_seeds[@]}"; do
  uv run python -m molgap.qm9_models.pretrain_infomax \
    --config configs/qm9_models/3dinfomax_pretrain.yaml \
    --split-seed "$seed" --seed "$seed"
done
