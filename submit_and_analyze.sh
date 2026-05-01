#!/bin/bash
# Submit RAG benchmark jobs and automatically analyze tradespace upon completion

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RAG 4-Way Benchmark + Tradespace Analysis Pipeline"
echo "==================================================="
echo

# Step 1: Submit server job (if needed)
echo "[1] Checking server job status..."
RUNNING_SERVERS=$(squeue -h -o "%j" | grep -c "LLMGE01_Server" || true)

if [ "$RUNNING_SERVERS" -gt 0 ]; then
    echo "    Server already running. Skipping submission."
else
    echo "    Submitting server job..."
    SBATCH_OUT=$(sbatch "${ROOT_DIR}/server_container.sh")
    SERVER_JOB_ID=$(echo "$SBATCH_OUT" | awk '{print $4}')
    echo "    Server job submitted: ${SERVER_JOB_ID}"
    sleep 3
fi

# Step 2: Submit benchmark jobs
echo
echo "[2] Submitting RAG benchmark jobs..."
cd "${ROOT_DIR}"
bash submit_rag_100_jobs.sh 2>&1 | grep "Submitted"

# Step 3: Monitor and auto-analyze
echo
echo "[3] Starting monitor + auto-analysis loop..."
echo "    Will poll for benchmark_summary.json and auto-run tradespace analysis when ready."
echo

bash "${ROOT_DIR}/monitor_and_analyze_rag_benchmark.sh"
