#!/bin/bash
# Submit all 5 configurations using fixed evolution settings.
# Enforces: NUM_GENERATIONS=15, START_POPULATION_SIZE=8, POPULATION_SIZE=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

configs=(hf vllm constrained speculative all)

RESULT_ROOT="$SCRIPT_DIR/switch_config_results"
CSV_FILE="$RESULT_ROOT/jobs_8x15.csv"
mkdir -p "$RESULT_ROOT"

echo "config,job_id,submit_time,num_generations,start_population_size,population_size" > "$CSV_FILE"

for cfg in "${configs[@]}"; do
  echo "=== Submitting: $cfg ==="

  # Switch .env and set config-specific run output directories.
  bash switch_config.sh "$cfg"

  # Create the run directory before submitting the job
  RUN_ID="${cfg}_8x15"
  RUN_DIR="runs/${RUN_ID}"
  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/results" "${RUN_DIR}/logs" "${RUN_DIR}/metrics"

  # Submit run with explicit overrides to guarantee 15/8/8 regardless of .env defaults.
  jid=$(RUN_ID="$RUN_ID" sbatch \
    --export=ALL,NUM_GENERATIONS=15,START_POPULATION_SIZE=8,POPULATION_SIZE=8 \
    run.sh | awk '{print $4}')

  echo "$cfg,$jid,$(date --iso-8601=seconds),15,8,8" >> "$CSV_FILE"
  echo "Submitted $cfg -> job $jid"
  sleep 2
done

echo ""
echo "All configurations submitted."
echo "Job list: $CSV_FILE"
