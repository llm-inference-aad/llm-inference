#!/bin/bash
# migrate_slurm_logs.sh - Manually migrate SLURM logs to a specific run directory
#
# Usage: ./scripts/migrate_slurm_logs.sh <run_id> [job_id] [OPTIONS]
#   run_id: The run directory name (e.g., auto_20251015_180811)
#   job_id: Optional. Main SLURM job ID. If not provided, will be auto-detected.
#
# Options:
#   --cancel-server   Read ${RUN_LOG_DIR}/hostname_server_job.txt and scancel the
#                     server job if it is still running.  Covers the SIGKILL case
#                     where run.sh's EXIT trap could not fire.
#   --update-status   Rewrite run_metadata.json setting status="cancelled".  Use
#                     together with --cancel-server after a hard kill of the main
#                     job to leave the metadata in a consistent state.
#
# Example: ./scripts/migrate_slurm_logs.sh auto_20251015_180811 3403227
# Example: ./scripts/migrate_slurm_logs.sh auto_20251015_180811 --cancel-server --update-status

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_id> [job_id] [--cancel-server] [--update-status]"
    echo "Example: $0 auto_20251015_180811 3403227"
    exit 1
fi

RUN_ID="$1"
shift

JOB_ID=""
DO_CANCEL_SERVER=false
DO_UPDATE_STATUS=false

# Parse remaining positional and flag arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cancel-server)
            DO_CANCEL_SERVER=true
            shift
            ;;
        --update-status)
            DO_UPDATE_STATUS=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            # First non-flag positional is the job_id
            if [[ -z "${JOB_ID}" ]]; then
                JOB_ID="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
SLURM_RESULTS="${REPO_ROOT}/slurm-results"
LOGS_DIR="${RUN_DIR}/logs"
ERRORS_DIR="${RUN_DIR}/errors"

# Validate run directory exists
if [ ! -d "${RUN_DIR}" ]; then
    echo "Error: Run directory not found: ${RUN_DIR}"
    exit 1
fi

# Create logs/errors directories if they don't exist
mkdir -p "${LOGS_DIR}"
mkdir -p "${ERRORS_DIR}"

# =============================================================================
# --cancel-server: scancel the nested server job if still running
# =============================================================================
if [[ "${DO_CANCEL_SERVER}" == "true" ]]; then
    echo ""
    echo "=== --cancel-server: scanning for orphaned server job ==="

    # The tracking file is written by server.sh as
    # "${HOSTNAME_LOG_FILE%.log}_server_job.txt" which resolves to
    # "${RUN_LOG_DIR}/hostname_server_job.txt".
    SERVER_JOB_FILE="${LOGS_DIR}/hostname_server_job.txt"

    if [[ -f "${SERVER_JOB_FILE}" ]]; then
        SERVER_JOB_ID=$(cat "${SERVER_JOB_FILE}" 2>/dev/null || true)
        if [[ -n "${SERVER_JOB_ID}" && "${SERVER_JOB_ID}" != "null" ]]; then
            echo "  Cancelling server job: ${SERVER_JOB_ID}"
            scancel "${SERVER_JOB_ID}" 2>/dev/null || \
                echo "  Warning: scancel ${SERVER_JOB_ID} failed (may have already finished)"
            sleep 2
            rm -f "${SERVER_JOB_FILE}"
            echo "  Removed tracking file: ${SERVER_JOB_FILE}"
        else
            echo "  No valid server job ID found in ${SERVER_JOB_FILE}"
        fi
    else
        echo "  Tracking file not found: ${SERVER_JOB_FILE}"
        echo "  Nothing to cancel."
    fi
fi

# =============================================================================
# --update-status: flip run_metadata.json.status = "cancelled"
# =============================================================================
if [[ "${DO_UPDATE_STATUS}" == "true" ]]; then
    echo ""
    echo "=== --update-status: setting run_metadata.json status=cancelled ==="

    METADATA_FILE="${RUN_DIR}/run_metadata.json"
    if [[ -f "${METADATA_FILE}" ]]; then
        python3 - <<PYEOF
import json, sys, datetime

try:
    with open('${METADATA_FILE}', 'r') as fh:
        metadata = json.load(fh)
    metadata['status'] = 'cancelled'
    metadata['cancelled_at'] = datetime.datetime.now().isoformat()
    with open('${METADATA_FILE}', 'w') as fh:
        json.dump(metadata, fh, indent=2)
    print("  Updated: status=cancelled")
except Exception as exc:
    print(f"  WARNING: could not update run_metadata.json: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
    else
        echo "  run_metadata.json not found at ${METADATA_FILE}"
        echo "  Skipping status update."
    fi
fi

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
        echo "  Moved slurm-main-${JOB_ID}.out"
    else
        echo "  Not found: ${MAIN_OUT}"
    fi

    if [ -f "${MAIN_ERR}" ]; then
        mv "${MAIN_ERR}" "${ERRORS_DIR}/"
        echo "  Moved slurm-main-${JOB_ID}.err"
    else
        echo "  Not found: ${MAIN_ERR}"
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
            echo "  Moved ${filename}"
            MOVED_COUNT=$((MOVED_COUNT + 1))
        fi
    done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.out" -o -name "llm-*.out" \) -newer "${RUN_DIR}/run_metadata.json" -print0 2>/dev/null)
    while IFS= read -r -d '' err_file; do
        if [ -f "${err_file}" ]; then
            filename=$(basename "${err_file}")
            mv "${err_file}" "${ERRORS_DIR}/"
            echo "  Moved ${filename}"
            MOVED_COUNT=$((MOVED_COUNT + 1))
        fi
    done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.err" -o -name "llm-*.err" \) -newer "${RUN_DIR}/run_metadata.json" -print0 2>/dev/null)
else
    echo "Warning: Could not determine run start time. Moving all eval/llm logs..."
    echo "This may include logs from other runs. Proceed? (y/N)"
    read -r response
    if [[ "${response}" =~ ^[Yy]$ ]]; then
        while IFS= read -r -d '' log_file; do
            if [ -f "${log_file}" ]; then
                filename=$(basename "${log_file}")
                mv "${log_file}" "${LOGS_DIR}/"
                echo "  Moved ${filename}"
                MOVED_COUNT=$((MOVED_COUNT + 1))
            fi
        done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.out" -o -name "llm-*.out" \) -print0)
        while IFS= read -r -d '' err_file; do
            if [ -f "${err_file}" ]; then
                filename=$(basename "${err_file}")
                mv "${err_file}" "${ERRORS_DIR}/"
                echo "  Moved ${filename}"
                MOVED_COUNT=$((MOVED_COUNT + 1))
            fi
        done < <(find "${SLURM_RESULTS}" -type f \( -name "eval-*.err" -o -name "llm-*.err" \) -print0)
    fi
fi

JOB_LOG_COUNT=0
[[ -n "${JOB_ID}" ]] && JOB_LOG_COUNT=2

echo ""
echo "=== Migration Complete ==="
echo "Total files moved: $((MOVED_COUNT + JOB_LOG_COUNT))"
echo "Logs location: ${LOGS_DIR}"
echo "Errors location: ${ERRORS_DIR}"
echo ""
echo "You can view the logs with:"
echo "  ls -lh ${LOGS_DIR}"
