#!/bin/bash
#SBATCH -J deepseek-finetune
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coe-ice
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Load any necessary modules (if applicable)
# module load cuda/12.1  # Example, adjust if needed

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if not already present (safety check)
pip install peft bitsandbytes trl scipy datasets

# Run finetuning
python finetune_mutation.py


