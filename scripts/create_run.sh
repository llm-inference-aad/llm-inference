#!/bin/bash
# ==============================================================================
# Create Run Directory Script
# ==============================================================================
# Purpose: Generate a timestamped run directory and metadata for tracking
# Usage: ./scripts/create_run.sh [optional_run_name]
# Output: Creates run directory and prints the RUN_ID for use in run.sh
# ==============================================================================

set -Eeuo pipefail

# Generate timestamp-based run ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Optional: Allow user to provide a descriptive name
if [ $# -eq 0 ]; then
    RUN_ID="run_${TIMESTAMP}"
else
    RUN_NAME="$1"
    RUN_ID="${RUN_NAME}_${TIMESTAMP}"
fi

# Create run directory structure
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/results"
mkdir -p "${RUN_DIR}/logs"

# Create run metadata file
cat > "${RUN_DIR}/run_metadata.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "created_at": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "user": "$(whoami)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "status": "initialized"
}
EOF

# Create a symlink to the latest run for easy access
ln -sfn "${RUN_ID}" "runs/latest"

# When called from run.sh, only output the RUN_ID (for script capture)
# When called manually, show helpful information
if [[ "${AUTOMATED_CALL:-}" == "true" ]]; then
  echo "${RUN_ID}"
else
  echo "✅ Run directory created: ${RUN_DIR}"
  echo ""
  echo "To start this run, execute:"
  echo "  export RUN_ID=${RUN_ID}"
  echo "  sbatch scripts/run.sh"
  echo ""
  echo "Or modify run.sh to use this run directory directly."
  echo ""
  echo "RUN_ID: ${RUN_ID}"
fi
