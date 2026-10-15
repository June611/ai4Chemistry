#!/usr/bin/env bash
# Explicit prepare/audit/status actions never start training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
QM9_PYTHON="${QM9_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$QM9_PYTHON" ]]; then
  printf '%s\n' "Python environment missing: run uv sync --extra qm9, or set QM9_PYTHON." >&2
  exit 2
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$QM9_PYTHON" -m molgap.experiments.neural_groups "$@"
