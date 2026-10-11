#!/usr/bin/env bash
set -euo pipefail

paired_seeds=( ${ML_PAIRED_SEEDS:-3407 42 2026} )
models=(xgboost_rdkit29 random_forest_rdkit29 training_mean)

for seed in "${paired_seeds[@]}"; do
  for model in "${models[@]}"; do
    suffix=""
    if [[ "$seed" != "3407" ]]; then
      suffix="_seed${seed}"
    fi
    uv run python -m molgap.training.ml --config "configs/${model}${suffix}.yaml"
  done
done
