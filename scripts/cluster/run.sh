#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1                      # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-results/slurm-main-%j.out
#SBATCH --error=slurm-results/slurm-main-%j.err

set -Eeuo pipefail

# ==============================================================================
# Determine Repository Root
# ==============================================================================
# Use SLURM_SUBMIT_DIR if running under SLURM, otherwise use script location
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
echo "Using run directory: ${RUN_DIR}"
# ==============================================================================

echo "=== Launching LLM Guided Evolution ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

mkdir -p slurm-results

# ----------------------------
# Load modules / CUDA / Python
# ----------------------------
module load cuda
module load anaconda3 || true     # Load Python 3.12.5 which is compatible with our requirements
export CUDA_VISIBLE_DEVICES=0

# ----------------------------
# Ensure Python tooling on PATH
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  echo "UV version: $(uv --version)"
else
  echo "uv not found on PATH; continuing with the activated Python environment."
fi

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

if [[ -d "${VENV_PATH:-$REPO_ROOT/.venv}/bin" ]]; then
  echo "Activating virtual environment at: ${VENV_PATH:-$REPO_ROOT/.venv}"
  source "${VENV_PATH:-$REPO_ROOT/.venv}/bin/activate"
fi
echo "Python: $(python --version)"

# Quick masked sanity checks (show only prefixes)


: "${LLM_INFERENCE_ROOT_DIR:=/home/hice1/rmanimaran8/scratch/llm-inference/llm-inference}"
export LLM_INFERENCE_ROOT_DIR
echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"

# ----------------------------
# CUDA libs for uv environment (best-effort)
# ----------------------------
# Some clusters need nvjitlink visible to the Python env used by uv.
UV_SITE_PKGS="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$UV_SITE_PKGS/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH updated for nvjitlink (best-effort)."

# ----------------------------
# Final info dump
# ----------------------------
echo "Python path: $(which python)"
nvidia-smi || true

# ----------------------------
# Wait for gateway workers when using load balancer mode
# ----------------------------
if [[ "${USE_LOAD_BALANCER:-false}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
  echo "=== Waiting for load balancer healthy workers ==="
  python - <<'PY'
import json
import os
import time
from pathlib import Path

import requests

root = Path(os.environ.get("LLM_INFERENCE_ROOT_DIR", "."))
host_file = Path(os.environ.get("LOADBALANCER_LOG_FILE", root / "loadbalancer.log"))
port = int(os.environ.get("LOAD_BALANCER_PORT", "9000"))
timeout = int(os.environ.get("LLMGE_WAIT_FOR_HEALTHY_SERVERS", "3600"))
interval = int(os.environ.get("LLMGE_WAIT_INTERVAL", "30"))

if not host_file.exists():
    raise SystemExit(f"Load balancer host file not found: {host_file}")

host = host_file.read_text().strip()
url = f"http://{host}:{port}/servers"
deadline = time.time() + timeout
last_status = None

while time.time() < deadline:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        status = response.json()
        last_status = status
        print(f"Gateway worker status: {json.dumps(status)}", flush=True)
        if status.get("healthy_servers", 0) > 0:
            print("At least one healthy worker is available.", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        print(f"Waiting for healthy workers: {exc}", flush=True)
    time.sleep(interval)

raise SystemExit(f"Timed out waiting for healthy workers. Last status: {last_status}")
PY
fi

# ----------------------------
# Run your job (unbuffered so logs appear immediately)
# ----------------------------
echo "=== Running: python run_improved.py ${RUN_DIR}/checkpoints ==="
export PYTHONUNBUFFERED=1
python run_improved.py "${RUN_DIR}/checkpoints"

# Move ALL SLURM logs from this run to run directory for organization
echo "=== Moving SLURM logs to run directory ==="
if [[ -d "${REPO_ROOT}/slurm-results" ]]; then
  # Move main job logs
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUT="${REPO_ROOT}/slurm-results/slurm-main-${SLURM_JOB_ID}.out"
    SLURM_ERR="${REPO_ROOT}/slurm-results/slurm-main-${SLURM_JOB_ID}.err"
    
    [[ -f "${SLURM_OUT}" ]] && mv "${SLURM_OUT}" "${RUN_DIR}/logs/" && echo "Moved main job .out log"
    [[ -f "${SLURM_ERR}" ]] && mv "${SLURM_ERR}" "${RUN_DIR}/logs/" && echo "Moved main job .err log"
  fi
  
  # Move all evaluation job logs (eval-*.out, eval-*.err, llm-*.out, llm-*.err)
  # These are created during the run and should be associated with this run
  find "${REPO_ROOT}/slurm-results" -type f \( -name "eval-*.out" -o -name "eval-*.err" -o -name "llm-*.out" -o -name "llm-*.err" \) -newer "${RUN_DIR}/run_metadata.json" -exec mv {} "${RUN_DIR}/logs/" \;
  
  MOVED_COUNT=$(find "${RUN_DIR}/logs/" -type f -name "*.out" -o -name "*.err" | wc -l)
  echo "Moved ${MOVED_COUNT} SLURM log files to ${RUN_DIR}/logs/"
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
SERVER_JOB_FILE="${REPO_ROOT}/hostname_server_job.txt"

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
    SERVER_OUT_LOG="${REPO_ROOT}/slurm-results/slurm-server-${SERVER_JOB_ID}.out"
    SERVER_ERR_LOG="${REPO_ROOT}/slurm-results/slurm-server-${SERVER_JOB_ID}.err"
    
    if [[ -f "${SERVER_OUT_LOG}" ]]; then
      mv "${SERVER_OUT_LOG}" "${RUN_DIR}/logs/" && echo "✅ Moved server .out log"
    else
      echo "⚠️  Server .out log not found: ${SERVER_OUT_LOG}"
    fi
    
    if [[ -f "${SERVER_ERR_LOG}" ]]; then
      mv "${SERVER_ERR_LOG}" "${RUN_DIR}/logs/" && echo "✅ Moved server .err log"
    else
      echo "⚠️  Server .err log not found: ${SERVER_ERR_LOG}"
    fi
    
    # Clean up tracking files
    rm -f "${SERVER_JOB_FILE}"
    rm -f "${REPO_ROOT}/hostname.log"
    
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
