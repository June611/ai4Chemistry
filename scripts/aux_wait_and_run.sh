#!/usr/bin/env bash
# Wait for a trigger job's COMPLETED.json and a free GPU, then run aux jobs.
# Usage: aux_wait_and_run.sh <trigger_job_id> <gpu_index> <jobs_csv> <log_file>
set -euo pipefail
TRIGGER_JOB="$1"
GPU="$2"
JOBS="$3"
LOG="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TRIG="artifacts/qm9_epoch200/formal_4gpu/runs/${TRIGGER_JOB}/COMPLETED.json"
echo "[$(date '+%F %T')] waiting for trigger ${TRIGGER_JOB} on GPU${GPU}"
while [ ! -f "$TRIG" ]; do sleep 60; done
# GPU must also be empty (previous lane fully exited).
while true; do
  m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
  [ "$m" -lt 1000 ] && break
  sleep 60
done
sleep 30
echo "[$(date '+%F %T')] GPU${GPU} free; starting aux jobs: ${JOBS}"
uv run --no-sync python scripts/run_qm9_formal_4gpu.py \
  --gpus "$GPU" --jobs "$JOBS" \
  --output-root artifacts/qm9_epoch200/formal_4gpu >> "$LOG" 2>&1
echo "[$(date '+%F %T')] aux on GPU${GPU} finished"
