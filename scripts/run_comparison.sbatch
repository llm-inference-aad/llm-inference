#!/bin/bash
#SBATCH --job-name=rag-compare
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rag_compare_%j.out
#SBATCH --error=logs/rag_compare_%j.err

set -euo pipefail

cd /storage/ice1/6/6/bcharest3/llm-inference/llm-inference
mkdir -p logs results

# Activate venv
source .venv/bin/activate

# Load .env for API keys (handles spaces around = that bash can't)
while IFS= read -r line; do
    # Skip comments and blanks
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    # Strip spaces around the first =
    key="${line%%=*}"
    val="${line#*=}"
    key="$(echo "$key" | xargs)"
    val="$(echo "$val" | xargs)"
    [ -n "$key" ] && export "$key=$val"
done < .env

echo "=== RAG Comparison: $(date) ==="
echo "Python: $(which python) ($(python --version))"
echo ""

# Run full comparison with relevance eval, golden recall, and JSON output
python scripts/compare_rag_mvp.py \
    --top-k 3 \
    --eval-relevance \
    --golden scripts/comparison_golden.json \
    --judge-max-words 1500 \
    --output-json results/comparison.json

echo ""
echo "=== Done: $(date) ==="
