#!/bin/bash
#SBATCH --job-name=llm_call_test
#SBATCH -t 00:15:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --output=metrics/slurm-results/slurm-test-%j.out
#SBATCH --error=metrics/slurm-results/slurm-test-%j.err

set -Eeuo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p metrics/slurm-results

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH."
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
else
  echo "Warning: .env not found, using defaults"
fi

echo "=== LLM test call job ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Time: $(date)"

uv run python scripts/test_llm_call.py "$@"
