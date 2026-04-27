#!/bin/bash
# Slurm submission for the RAG isolation harness.
# Requires a running LLM server (see server.sh) reachable at the host in
# $HOSTNAME_LOG_FILE (or pass --server-url to run_paired_eval.py).
#
# Usage:
#   sbatch --export=ALL,DATASET=scripts/rag_isolation/datasets/small_validation.json,\
#OUTPUT=experiments/rag_isolation/$(date +%Y%m%d_%H%M),\
#SERVER_URL=<host>:8000 \
#     scripts/rag_isolation/run_harness.sbatch
#
#SBATCH --job-name=rag_isolation
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -p ice-gpu
#SBATCH --output=logs/rag_isolation-%j.out
#SBATCH --error=logs/rag_isolation-%j.err

set -euo pipefail

mkdir -p logs

# Load env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

module load cuda
source "${VENV_PATH}/bin/activate"

DATASET=${DATASET:-scripts/rag_isolation/datasets/small_validation.json}
OUTPUT=${OUTPUT:-experiments/rag_isolation/$(date +%Y%m%d_%H%M%S)}
SERVER_URL=${SERVER_URL:-}

ARGS=(--dataset "$DATASET" --output "$OUTPUT")
if [ -n "$SERVER_URL" ]; then
    ARGS+=(--server-url "$SERVER_URL")
fi

echo "[rag_isolation] dataset=$DATASET output=$OUTPUT server_url=${SERVER_URL:-from-hostname.log}"

uv run python scripts/rag_isolation/run_paired_eval.py "${ARGS[@]}"

echo "[rag_isolation] running analysis"
uv run python scripts/rag_isolation/analyze.py "$OUTPUT"

echo "[rag_isolation] done. See $OUTPUT/report.md"
