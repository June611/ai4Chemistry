#!/usr/bin/env bash
# Run inside the repository. Existing server uv environment needs no dependency download.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f artifacts/server_deployment/server-env.sh ]]; then
  source artifacts/server_deployment/server-env.sh
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "--remote" ]]; then
  shift
  exec python -m molgap.scheduling.remote "$@"
fi
if [[ "${1:-}" == "--plan" ]]; then
  shift
  exec python -m molgap.scheduling plan "$@"
fi
if [[ "${1:-}" == "--status" || "${1:-}" == "--stop" ]]; then
  action="${1#--}"
  shift
  exec python -m molgap.scheduling "$action" "$@"
fi
if [[ "${1:-}" == "--sync" ]]; then
  shift
  exec python -m molgap.tracking.bridge "$@"
fi
exec python -m molgap.scheduling run "$@"
