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
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/homo_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/homo_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/homo_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/lumo_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/lumo_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/lumo_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/delta_e_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/delta_e_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/delta_e_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/alpha_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/alpha_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/alpha_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/mu_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/mu_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/mu_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/r2_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/r2_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/r2_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/zpve_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/zpve_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/zpve_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D1_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D1_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D1_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D2_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D2_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop_variants/D2_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/task_attention7_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/task_attention7_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/task_attention7_seed3407.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/physics_thermo10_seed42.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/physics_thermo10_seed2026.yaml
run_job uv run python -m molgap.modular.train --config configs/qm9_epoch200/fresh/chemprop/physics_thermo10_seed3407.yaml
((failed == 0))
