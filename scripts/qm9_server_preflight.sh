#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export DGLBACKEND=pytorch
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/chempro-mpl}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/molgap-warp-cache}"

uv sync --locked --extra qm9
uv run python -m molgap.qm9_models.bootstrap
uv run python scripts/check_qm9_device.py --require-cuda

geometry=data/geometry/qm9_full_seed3407.npz
if [[ ! -f "$geometry" ]]; then
  uv run python scripts/download_qm9_xyz.py --output-dir data/raw/qm9_xyz
  uv run python -m molgap.qm9_models.prepare \
    --config configs/qm9_models/egnn.yaml \
    --xyz-dir data/raw/qm9_xyz \
    --output "$geometry"
fi

split_seeds=( ${QM9_PREFLIGHT_SPLIT_SEEDS:-3407 42 2026} )
infomax_skipped=0
for split_seed in "${split_seeds[@]}"; do
  split_geometry="data/geometry/qm9_full_seed${split_seed}.npz"
  if [[ ! -f "$split_geometry" ]]; then
    uv run python scripts/reindex_qm9_geometry.py \
      --source "$geometry" \
      --processed-dir "data/processed/qm9_full/random/seed${split_seed}" \
      --output "$split_geometry"
  fi
done
uv run python scripts/audit_qm9_split_seeds.py \
  --seeds "${split_seeds[@]}" \
  --output reports/qm9_split_seed_audit.json

uv run pytest -q \
  tests/test_qm9_adapters.py \
  tests/test_egnn_adapter.py \
  tests/test_schnetpack_adapters.py \
  tests/test_dimenet_adapter.py \
  tests/test_esa_adapter.py \
  tests/test_torchmd_net_adapter.py \
  tests/test_cormorant_adapter.py \
  tests/test_lie_transformer_adapter.py \
  tests/test_alignn_adapter.py \
  tests/test_3dinfomax_adapter.py

for split_seed in "${split_seeds[@]}"; do
  for model in egnn schnet painn dimenet_pp esa torchmd_net cormorant lie_transformer alignn; do
    uv run python -m molgap.qm9_models.train \
      --config "configs/qm9_models/${model}.yaml" \
      --split-seed "$split_seed" --seed "$split_seed" --dry-run
  done

  checkpoint="${INFOMAX_CHECKPOINT:-}"
  if [[ -n "${INFOMAX_CHECKPOINT_TEMPLATE:-}" ]]; then
    checkpoint="${INFOMAX_CHECKPOINT_TEMPLATE//\{split_seed\}/$split_seed}"
    checkpoint="${checkpoint//\{seed\}/$split_seed}"
  fi
  if [[ -n "$checkpoint" ]]; then
    uv run python -m molgap.qm9_models.train \
      --config configs/qm9_models/3dinfomax.yaml \
      --split-seed "$split_seed" --seed "$split_seed" \
      --pretrain-checkpoint "$checkpoint" --dry-run
  else
    infomax_skipped=$((infomax_skipped + 1))
    echo "3D Infomax split ${split_seed} dry-run skipped: checkpoint is unset."
  fi
done

if [[ "$infomax_skipped" -gt 0 ]]; then
  echo "QM9 nine-model preflight passed; 3D Infomax remains pending for ${infomax_skipped} split(s)."
else
  echo "QM9 ten-model preflight passed."
fi
