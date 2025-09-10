#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"

echo "launching LLM Guided Evolution"
hostname

# Load required modules
module load cuda
module load anaconda3
export CUDA_VISIBLE_DEVICES=0

# Setup UV environment
export PATH="$HOME/.local/bin:$PATH"

# Verify UV is available
echo "UV version: $(uv --version)"
echo "Python version: $(uv run python --version)"

# Set up CUDA libraries for the UV environment
export LD_LIBRARY_PATH="$(uv run python -c 'import site; print(site.getsitepackages()[0])')/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# Show environment info
echo "Current working directory: $(pwd)"
echo "UV Python location: $(uv run which python)"

# Run the script with UV
uv run python run_improved.py first_test
