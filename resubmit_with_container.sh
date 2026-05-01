#!/bin/bash
set -euo pipefail
echo "Resubmission helper: cancel existing server jobs, submit containerized server, then submit benchmarks"

# Cancel existing server jobs named LLMGE01_Server
EXISTING=$(squeue -h -o "%A %j" | awk '$2=="LLMGE01_Server" {print $1}') || true
if [ ! -z "${EXISTING}" ]; then
    echo "Cancelling existing server jobs: ${EXISTING}"
    echo ${EXISTING} | xargs -r scancel
else
    echo "No running server jobs found."
fi

echo "Submitting server via sbatch server_container.sh"
SBATCH_OUT=$(sbatch server_container.sh)
echo "$SBATCH_OUT"

# Parse job id
JOBID=$(echo "$SBATCH_OUT" | awk '{print $4}')
if [ -z "$JOBID" ]; then
    echo "Failed to get server job id from sbatch output. Aborting benchmark submission."
    exit 1
fi

echo "Server job submitted: $JOBID"

echo "Sleeping 3s to allow server job to register files"
sleep 3

# Submit benchmark jobs using existing helper script if present
if [ -x submit_rag_100_jobs.sh ]; then
    echo "Running submit_rag_100_jobs.sh to submit benchmark jobs (they may already include dependency logic)."
    bash submit_rag_100_jobs.sh
else
    echo "submit_rag_100_jobs.sh not found or not executable. Please run your benchmark submission helper and ensure it depends on server job $JOBID (afterok:$JOBID)."
fi

echo "Resubmission helper finished. Monitor with squeue or sacct."
