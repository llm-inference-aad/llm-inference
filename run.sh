#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1                      # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
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

# Update run metadata on completion
if [[ -f "${RUN_DIR}/run_metadata.json" ]]; then
  python3 -c "
import json
import sys
with open('${RUN_DIR}/run_metadata.json', 'r') as f:
    metadata = json.load(f)
metadata['status'] = 'completed'
metadata['completed_at'] = '$(date -Iseconds)'
with open('${RUN_DIR}/run_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"
fi

echo "=== Job complete ==="
date
