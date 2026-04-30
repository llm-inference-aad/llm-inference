#!/bin/bash
# Batch-submit draft model speculative decoding sweep
# Tests 1B (small same-family), 3B (medium same-family), and cross-family drafts
# All runs use fixed parameters: 15 generations, 8 start pop, 8 pop size

DRAFT_CONFIGS=("draft_1b" "draft_3b" "draft_cross")
JOB_IDS=()
TIMESTAMP=$(date +%s)
RESULTS_DIR="switch_config_results"

echo "=========================================="
echo "Draft Model Speculative Decoding Sweep"
echo "=========================================="
echo "Config: 15 generations, 8 pop size (start+main)"
echo "Timestamp: $TIMESTAMP"
echo ""

for config in "${DRAFT_CONFIGS[@]}"; do
  echo "Switching to config: $config"
  bash switch_config.sh "$config" > /dev/null 2>&1
  
  # Pre-create the run directory to avoid job failures
  RUN_ID="${config}_8x15_draft"
  RUN_DIR="runs/${RUN_ID}"
  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/results" "${RUN_DIR}/logs" "${RUN_DIR}/metrics"
  
  echo "Submitting to Slurm..."
  JOB_OUTPUT=$(RUN_ID="$RUN_ID" sbatch --export=ALL,NUM_GENERATIONS=15,START_POPULATION_SIZE=8,POPULATION_SIZE=8 run.sh 2>&1)
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
  
  JOB_IDS+=("$JOB_ID")
  echo "✅ Submitted $config: Job $JOB_ID"
  echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Job Summary:"
for i in "${!DRAFT_CONFIGS[@]}"; do
  echo "  ${DRAFT_CONFIGS[$i]}: ${JOB_IDS[$i]}"
done

echo ""
echo "Monitor jobs with:"
echo "  watch squeue -u \$USER"
echo "  sacct -j ${JOB_IDS[0]} -j ${JOB_IDS[1]} -j ${JOB_IDS[2]} --format=JobID,State,Elapsed"
echo ""
echo "View metrics:"
echo "  find $RESULTS_DIR -name 'latency-*.json' | head -20"
echo ""
echo "Compare results:"
echo "  python compare_constrained_runs.py"
