#!/usr/bin/env bash
set -uo pipefail
# One task per GPU; set CUDA_VISIBLE_DEVICES=1 to select the second card.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# Explicit stage reference; failures do not cancel later tasks.
failed=0
run_job() {
  if "$@"; then return 0; else failed=$((failed+1)); return 0; fi
}
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/demol/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/single_task_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/single_task_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/single_task_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/painn/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/single_task_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/single_task_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/single_task_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/egnn/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/transformer_m/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/single_task_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/single_task_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/single_task_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/3dinfomax/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/single_task_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/single_task_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/single_task_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/schnet/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/esa/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/spe_transformer/physics_thermo10_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/smiles_transformer/physics_thermo10_seed3407.yaml
((failed == 0))
