#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=pytorch
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/chempro-mpl}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/molgap-warp-cache}"

paired_seeds=( ${QM9_PAIRED_SEEDS:-3407 42 2026} )
models=(egnn schnet painn dimenet_pp esa torchmd_net cormorant lie_transformer alignn)

for split_seed in "${paired_seeds[@]}"; do
  seed="$split_seed"
  checkpoint="${INFOMAX_CHECKPOINT:-}"
  if [[ -n "${INFOMAX_CHECKPOINT_TEMPLATE:-}" ]]; then
    checkpoint="${INFOMAX_CHECKPOINT_TEMPLATE//\{split_seed\}/$split_seed}"
    checkpoint="${checkpoint//\{seed\}/$seed}"
  fi
  for model in "${models[@]}"; do
    uv run python -m molgap.qm9_models.train \
      --config "configs/qm9_models/${model}.yaml" \
      --split-seed "$split_seed" --seed "$seed"
  done
  if [[ -n "$checkpoint" ]]; then
    uv run python -m molgap.qm9_models.train \
      --config configs/qm9_models/3dinfomax.yaml \
      --split-seed "$split_seed" --seed "$seed" \
      --pretrain-checkpoint "$checkpoint"
  else
    echo "Skipping 3D Infomax split ${split_seed}, training seed ${seed}: checkpoint is unset."
  fi
done
