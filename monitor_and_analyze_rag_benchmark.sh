#!/bin/bash
# Monitor RAG benchmark jobs and auto-run tradespace analysis upon completion

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${ROOT_DIR}/runs/rag_100_jobs/metrics"
BENCHMARK_SUMMARY="${BENCHMARK_DIR}/benchmark_summary.json"

echo "RAG + Decoding Benchmark Monitor"
echo "================================="
echo "Monitoring for benchmark completion..."
echo "Benchmark summary path: ${BENCHMARK_SUMMARY}"
echo

# Submit jobs (or assume they're already submitted)
JOB_IDS="${1:-}"
if [ ! -z "${JOB_IDS}" ]; then
    echo "Job IDs to monitor: ${JOB_IDS}"
else
    echo "No job IDs provided. Assuming benchmark jobs are already queued."
    echo "You can pass job IDs as: $0 <job_id_1> <job_id_2> ..."
fi

# Poll for benchmark completion
POLL_INTERVAL=30  # seconds
MAX_WAIT=$((3600 * 8))  # 8 hours
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "${BENCHMARK_SUMMARY}" ]; then
        echo "[$(date)] Benchmark summary found! Starting tradespace analysis..."
        
        # Run the tradespace analysis
        python "${ROOT_DIR}/analyze_rag_decoding_tradespace.py" \
            --benchmark-summary "${BENCHMARK_SUMMARY}" \
            --output-dir "${BENCHMARK_DIR}"
        
        ANALYSIS_SUMMARY="${BENCHMARK_DIR}/tradespace_summary.json"
        ANALYSIS_REPORT="${BENCHMARK_DIR}/tradespace_report.md"
        
        if [ -f "${ANALYSIS_SUMMARY}" ]; then
            echo "[$(date)] Tradespace analysis complete!"
            echo "  Summary: ${ANALYSIS_SUMMARY}"
            echo "  Report:  ${ANALYSIS_REPORT}"
            echo
            echo "Report snippet:"
            head -n 30 "${ANALYSIS_REPORT}"
            echo
            echo "=== Analysis finished ==="
            exit 0
        else
            echo "[$(date)] ERROR: Analysis failed (no output summary)"
            exit 1
        fi
    fi
    
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
    WAIT_MINS=$((ELAPSED / 60))
    echo "[$(date)] Benchmark not ready yet. Waited ${WAIT_MINS}m. Sleeping ${POLL_INTERVAL}s..."
    sleep ${POLL_INTERVAL}
done

echo "[$(date)] ERROR: Timeout waiting for benchmark completion (${MAX_WAIT}s)"
exit 1
