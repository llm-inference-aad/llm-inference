#!/bin/bash
# migrate_slurm_logs.sh - Manually migrate SLURM logs to a specific run directory
#
# Usage: ./scripts/migrate_slurm_logs.sh <run_id> [job_id]
#   run_id: The run directory name (e.g., auto_20251015_180811)
#   job_id: Optional. Main SLURM job ID. If not provided, will be auto-detected.
#
# Example: ./scripts/migrate_slurm_logs.sh auto_20251015_180811 3403227

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_id> [job_id]"
    echo "Example: $0 auto_20251015_180811 3403227"
    exit 1
fi

RUN_ID="$1"
JOB_ID="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
SLURM_RESULTS="${REPO_ROOT}/slurm-results"
LOGS_DIR="${RUN_DIR}/logs"

# Validate run directory exists
if [ ! -d "${RUN_DIR}" ]; then
    echo "Error: Run directory not found: ${RUN_DIR}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Get run start time from metadata
RUN_START_TIME=""
if [ -f "${RUN_DIR}/run_metadata.json" ]; then
    RUN_START_TIME=$(python3 -c "
import json
try:
    with open('${RUN_DIR}/run_metadata.json', 'r') as f:
        data = json.load(f)
        print(data.get('started_at', ''))
except:
    pass
" 2>/dev/null)
fi

echo "=== Migrating SLURM logs for run: ${RUN_ID} ==="
echo "Run directory: ${RUN_DIR}"
echo "Logs directory: ${LOGS_DIR}"

# If job ID provided, move main job logs
if [ -n "${JOB_ID}" ]; then
    echo ""
    echo "Moving main job logs (Job ID: ${JOB_ID})..."
    
    MAIN_OUT="${SLURM_RESULTS}/slurm-main-${JOB_ID}.out"
    MAIN_ERR="${SLURM_RESULTS}/slurm-main-${JOB_ID}.err"
    
    if [ -f "${MAIN_OUT}" ]; then
        mv "${MAIN_OUT}" "${LOGS_DIR}/"
        echo "  ✅ Moved slurm-main-${JOB_ID}.out"
    else
        echo "  ⚠ Not found: ${MAIN_OUT}"
    fi
    
    if [ -f "${MAIN_ERR}" ]; then
        mv "${MAIN_ERR}" "${LOGS_DIR}/"
        echo "  ✅ Moved slurm-main-${JOB_ID}.err"
    else
        echo "  ⚠ Not found: ${MAIN_ERR}"
    fi
fi

# Move evaluation and LLM job logs created during this run
echo ""
echo "Moving evaluation and LLM job logs..."

MOVED_COUNT=0

if [ -n "${RUN_START_TIME}" ]; then
    echo "Filtering by logs created after: ${RUN_START_TIME}"
    
    # Find logs newer than the run start time
    while IFS= read -r -d '' log_file; do
        if [ -f "${log_file}" ]; then
            filename=$(basename "${log_file}")
            mv "${log_file}" "${LOGS_DIR}/"
            echo "  ✅ Moved ${filename}"
            ((MOVED_COUNT++))
        fi
    done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.out" -o -name "eval-*.err" -o -name "llm-*.out" -o -name "llm-*.err" \) -newer "${RUN_DIR}/run_metadata.json" -print0 2>/dev/null)
else
    echo "⚠ Warning: Could not determine run start time. Moving all eval/llm logs..."
    echo "This may include logs from other runs. Proceed? (y/N)"
    read -r response
    if [[ "${response}" =~ ^[Yy]$ ]]; then
        while IFS= read -r -d '' log_file; do
            if [ -f "${log_file}" ]; then
                filename=$(basename "${log_file}")
                mv "${log_file}" "${LOGS_DIR}/"
                echo "  ✅ Moved ${filename}"
                ((MOVED_COUNT++))
            fi
        done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.out" -o -name "eval-*.err" -o -name "llm-*.out" -o -name "llm-*.err" \) -print0)
    fi
fi

echo ""
echo "=== Migration Complete ==="
echo "Total files moved: $((MOVED_COUNT + ([ -n \"${JOB_ID}\" ] && echo 2 || echo 0)))"
echo "Logs location: ${LOGS_DIR}"
echo ""
echo "You can view the logs with:"
echo "  ls -lh ${LOGS_DIR}"
