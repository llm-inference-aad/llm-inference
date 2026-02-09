#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=/home/hice1/rkansal8/scratch/llm-inference/metrics/slurm-results/eval-%j.out
#SBATCH --error=/home/hice1/rkansal8/scratch/llm-inference/metrics/slurm-results/eval-%j.err
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
cd "${LLM_INFERENCE_ROOT_DIR:-/home/hice1/rkansal8/scratch/llm-inference/}"

# Time the evaluation
EVAL_START=$(date +%s)

# Run Python script
python ./sota/ExquisiteNetV2/train.py -bs 216 -network "models.network_xXxVy6pimgnPHTzIPcOr64DVEsp" -data ./cifar10 -end_lr 0.001 -seed 21 -val_r 0.2 -amp -epoch 24
EXIT_CODE=$?

EVAL_END=$(date +%s)
EVAL_ELAPSED=$((EVAL_END - EVAL_START))
echo "EVAL_TIME_SECONDS=$EVAL_ELAPSED" >> /home/hice1/rkansal8/scratch/llm-inference/metrics/slurm-results/eval-$SLURM_JOB_ID.time
echo "EXIT_CODE=$EXIT_CODE" >> /home/hice1/rkansal8/scratch/llm-inference/metrics/slurm-results/eval-$SLURM_JOB_ID.time

exit $EXIT_CODE
