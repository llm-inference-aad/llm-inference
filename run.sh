#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 15:00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1                      # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Removed GPU constraint - ice-gpu partition uses different naming (a100, v100, etc.)
#SBATCH --output=metrics/slurm-results/slurm-main-%j.out
#SBATCH --error=metrics/slurm-results/slurm-main-%j.err

set -Eeuo pipefail

# ==============================================================================
# Determine Repository Root
# ==============================================================================
# Use SLURM_SUBMIT_DIR if running under SLURM, otherwise use script location
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Change to repository root to ensure all paths work correctly
cd "${REPO_ROOT}"

# ==============================================================================
# Automatic Run Directory Setup
# ==============================================================================
# If RUN_ID is not set, create a new run directory automatically
if [[ -z "${RUN_ID:-}" ]]; then
  echo "No RUN_ID provided. Creating new run directory..."
  RUN_ID=$(AUTOMATED_CALL=true bash scripts/create_run.sh "auto")
  echo "Created RUN_ID: ${RUN_ID}"
fi

# Set the run directory path (absolute path to avoid issues with subprocess working directories)
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
RUN_LOG_DIR="${RUN_DIR}/logs"
RUN_METRICS_DIR="${RUN_DIR}/metrics"

# Validate run directory exists
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: Run directory ${RUN_DIR} does not exist!"
  echo "Expected path: ${RUN_DIR}"
  echo "REPO_ROOT: ${REPO_ROOT}"
  echo "RUN_ID: ${RUN_ID}"
  echo "Create it first with: bash scripts/create_run.sh [optional_name]"
  exit 1
fi

export RUN_ID
export RUN_DIR
export RUN_LOG_DIR
export RUN_METRICS_DIR
mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/run-main-${SLURM_JOB_ID:-manual}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/run-main-${SLURM_JOB_ID:-manual}.err" >&2)
echo "Using run directory: ${RUN_DIR}"
# ==============================================================================

echo "=== Launching LLM Guided Evolution ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" metrics/slurm-results

# ----------------------------
# Load modules / CUDA / Python
# ----------------------------
module load cuda
module load anaconda3 || true     # Load Python 3.12.5 which is compatible with our requirements
export CUDA_VISIBLE_DEVICES=0

# ----------------------------
# Ensure uv on PATH
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Install with: pipx install uv  (or per your cluster setup)"
  exit 1
fi

echo "UV version: $(uv --version)"
echo "Python via uv: $(uv run python --version)"

# ----------------------------
# Load environment variables
# ----------------------------
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  exit 1
fi

# Auto-export all keys defined in .env into this shell's environment
set -a
source .env
set +a

# Quick masked sanity checks (show only prefixes)


: "${LLM_INFERENCE_ROOT_DIR:=/home/hice1/satmuri6/scratch/llm-inference}"
export LLM_INFERENCE_ROOT_DIR
echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"

# ------------------------------------------------------------------------------
# Enforce run-scoped paths for all runtime logs and metrics
# ------------------------------------------------------------------------------
export RUN_LOG_DIR="${RUN_LOG_DIR}"
export RUN_METRICS_DIR="${RUN_METRICS_DIR}"
export SLURM_LOG_DIR="${RUN_LOG_DIR}"
export METRICS_PATH="${RUN_METRICS_DIR}"
export HOSTNAME_LOG_FILE="${RUN_LOG_DIR}/hostname.log"
export LOADBALANCER_LOG_FILE="${RUN_LOG_DIR}/loadbalancer.log"
export SERVER_REGISTRY_FILE="${RUN_LOG_DIR}/servers.json"

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}"
echo "RUN_LOG_DIR: ${RUN_LOG_DIR}"
echo "RUN_METRICS_DIR: ${RUN_METRICS_DIR}"

# ----------------------------
# CUDA libs for uv environment (best-effort)
# ----------------------------
# Some clusters need nvjitlink visible to the Python env used by uv.
UV_SITE_PKGS="$(uv run python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$UV_SITE_PKGS/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH updated for nvjitlink (best-effort)."

# ----------------------------
# Final info dump
# ----------------------------
echo "UV Python path: $(uv run which python)"
nvidia-smi || true

# ----------------------------
# Run your job
# ----------------------------
echo "=== Running: uv run python run_improved.py ${RUN_DIR}/checkpoints ==="
uv run python run_improved.py "${RUN_DIR}/checkpoints"

# Move ALL SLURM logs from this run to run directory for organization
echo "=== Moving SLURM logs to run directory ==="
if [[ -d "${REPO_ROOT}/metrics/slurm-results" ]]; then
  mkdir -p "${RUN_LOG_DIR}"

  # Move main job logs
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUT="${REPO_ROOT}/metrics/slurm-results/slurm-main-${SLURM_JOB_ID}.out"
    SLURM_ERR="${REPO_ROOT}/metrics/slurm-results/slurm-main-${SLURM_JOB_ID}.err"
    
    [[ -f "${SLURM_OUT}" ]] && mv "${SLURM_OUT}" "${RUN_LOG_DIR}/" && echo "Moved main job .out log"
    [[ -f "${SLURM_ERR}" ]] && mv "${SLURM_ERR}" "${RUN_LOG_DIR}/" && echo "Moved main job .err log"
  fi
  
  # Move any legacy helper/server logs generated under metrics/slurm-results.
  find "${REPO_ROOT}/metrics/slurm-results" -type f \( -name "eval-*.out" -o -name "eval-*.err" -o -name "llm-*.out" -o -name "llm-*.err" -o -name "slurm-server-*.out" -o -name "slurm-server-*.err" \) -newer "${RUN_DIR}/run_metadata.json" -exec mv {} "${RUN_LOG_DIR}/" \;
  
  LOG_COUNT=$(find "${RUN_LOG_DIR}/" -type f | wc -l)
  echo "Run log files available in ${RUN_LOG_DIR}: ${LOG_COUNT}"
fi

# Update run metadata on completion
if [[ -f "${RUN_DIR}/run_metadata.json" ]]; then
  python3 -c "
import json
import sys
with open('${RUN_DIR}/run_metadata.json', 'r') as f:
    metadata = json.load(f)
metadata['status'] = 'completed'
metadata['completed_at'] = '$(date -Iseconds)'
metadata['slurm_job_id'] = '${SLURM_JOB_ID:-}'
with open('${RUN_DIR}/run_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"
fi

# =============================================================================
# Automatic LLM Server Shutdown and Log Collection
# =============================================================================
echo "=== Shutting down LLM server ==="
SERVER_JOB_FILE="${HOSTNAME_LOG_FILE%.log}_server_job.txt"

if [[ -f "${SERVER_JOB_FILE}" ]]; then
  SERVER_JOB_ID=$(cat "${SERVER_JOB_FILE}" 2>/dev/null || echo "")
  
  if [[ -n "${SERVER_JOB_ID}" && "${SERVER_JOB_ID}" != "null" ]]; then
    echo "Canceling server job: ${SERVER_JOB_ID}"
    scancel "${SERVER_JOB_ID}" || echo "Warning: Could not cancel server job ${SERVER_JOB_ID} (may have already finished)"
    
    # Wait for server to shut down and logs to flush
    echo "Waiting for server shutdown and log flush..."
    sleep 15
    
    # Move server logs to run directory
    echo "Moving server logs to run directory..."
    SERVER_OUT_LOG_PRIMARY="${RUN_LOG_DIR}/slurm-server-${SERVER_JOB_ID}.out"
    SERVER_ERR_LOG_PRIMARY="${RUN_LOG_DIR}/slurm-server-${SERVER_JOB_ID}.err"
    SERVER_OUT_LOG_LEGACY="${REPO_ROOT}/metrics/slurm-results/slurm-server-${SERVER_JOB_ID}.out"
    SERVER_ERR_LOG_LEGACY="${REPO_ROOT}/metrics/slurm-results/slurm-server-${SERVER_JOB_ID}.err"
    
    if [[ -f "${SERVER_OUT_LOG_LEGACY}" ]]; then
      mv "${SERVER_OUT_LOG_LEGACY}" "${RUN_LOG_DIR}/" && echo "✅ Moved legacy server .out log"
    fi
    if [[ -f "${SERVER_OUT_LOG_PRIMARY}" ]]; then
      echo "✅ Server .out log found in run log directory"
    else
      echo "⚠️  Server .out log not found in expected locations"
    fi
    
    if [[ -f "${SERVER_ERR_LOG_LEGACY}" ]]; then
      mv "${SERVER_ERR_LOG_LEGACY}" "${RUN_LOG_DIR}/" && echo "✅ Moved legacy server .err log"
    fi
    if [[ -f "${SERVER_ERR_LOG_PRIMARY}" ]]; then
      echo "✅ Server .err log found in run log directory"
    else
      echo "⚠️  Server .err log not found in expected locations"
    fi
    
    # Clean up tracking files
    rm -f "${SERVER_JOB_FILE}"
    rm -f "${HOSTNAME_LOG_FILE}"
    
    echo "✅ Server shutdown and cleanup complete"
  else
    echo "⚠️  No valid server job ID found in ${SERVER_JOB_FILE}"
  fi
else
  echo "⚠️  Server job tracking file not found: ${SERVER_JOB_FILE}"
  echo "   Server may need to be shut down manually"
fi

echo "=== Job complete ==="
date
