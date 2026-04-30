#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=/storage/ice1/5/5/jgil37/llm-inference/slurm-results/eval-%j.out
#SBATCH --error=/storage/ice1/5/5/jgil37/llm-inference/slurm-results/eval-%j.err
echo "Launching Python Evaluation"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda
# module load anaconda3
# Activate Conda environment
# conda activate llm_guided_env
# export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.13/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
source "$VENV_PATH/bin/activate"

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Change to repository root directory to ensure consistent paths
cd "${LLM_INFERENCE_ROOT_DIR:-/storage/ice1/5/5/jgil37/llm-inference}"

# Run Python script
python ./sota/ExquisiteNetV2/train.py -bs 216 -network "models.network_xXxuus6WqrOKigWnDDWRl1zxIFL" -data ./cifar10 -end_lr 0.001 -seed 21 -val_r 0.2 -amp
