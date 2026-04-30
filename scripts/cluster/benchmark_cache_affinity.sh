#!/bin/bash
#SBATCH --job-name=cache_affinity_bench
#SBATCH -t 6:00:00
#SBATCH --mem=4G
#SBATCH -c 2
#SBATCH --output=slurm-results/slurm-cache-bench-%j.out
#SBATCH --error=slurm-results/slurm-cache-bench-%j.err

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p slurm-results

module load anaconda3 || true
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $REPO_ROOT"
  exit 1
fi

set -a
source .env
set +a

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$REPO_ROOT}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${LLM_INFERENCE_ROOT_DIR}/loadbalancer.log}"

if [[ -d "${VENV_PATH:-$REPO_ROOT/.venv}/bin" ]]; then
  echo "Activating virtual environment at: ${VENV_PATH:-$REPO_ROOT/.venv}"
  source "${VENV_PATH:-$REPO_ROOT/.venv}/bin/activate"
fi
echo "Python: $(python --version)"

echo "===== Running cache affinity benchmark ====="
echo "Working dir: $(pwd)"
echo "LOADBALANCER_LOG_FILE: $LOADBALANCER_LOG_FILE"
echo "LOAD_BALANCER_PORT: $LOAD_BALANCER_PORT"
date

python scripts/benchmark_cache_affinity.py "$@"

echo "===== Cache affinity benchmark complete ====="
date
