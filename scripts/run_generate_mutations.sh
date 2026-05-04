#!/bin/bash
#SBATCH --job-name=gen_mutations
#SBATCH -t 12:00:00
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --output=slurm-results/gen-mutations-%j.out
#SBATCH --error=slurm-results/gen-mutations-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

set -a; source .env; set +a
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Generating Synthetic Mutations (local Llama server) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

uv run python scripts/generate_mutations.py \
  --run-id "${QLORA_RUN_ID:-my_run_20260428_011918}" \
  --max-mutations "${MAX_MUTATIONS:-200}" \
  --temperatures 0.3 0.7 1.0

echo "=== Done: $(date) ==="
