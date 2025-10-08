#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1                      # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=results/slurm-main-%j.out
#SBATCH --error=results/slurm-main-%j.err

set -Eeuo pipefail

echo "=== Launching LLM Guided Evolution ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

mkdir -p results/slurm

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
echo "=== Running: uv run python run_improved.py first_test ==="
uv run python run_improved.py first_test

echo "=== Job complete ==="
date
