#!/bin/bash
#SBATCH -J deepseek-finetune
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coe-ice
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -e

echo "=== DeepSeek Fine-tuning Job ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

# Load modules
module load cuda
module load anaconda3 || true

# Set cache directory to /storage to avoid disk quota issues
export UV_CACHE_DIR="/storage/ice1/8/6/rmanimaran8/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# Navigate to project directory (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_DIR="/storage/ice1/8/6/rmanimaran8/deepseek-mutation-finetune"

# Run finetuning with uv
echo "Starting fine-tuning..."
uv run --active python -m finetune.finetune_mutation --output-dir "$OUTPUT_DIR"

echo "=== Fine-tuning Complete ==="
date


