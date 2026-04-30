#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --partition=coc-gpu
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH -C A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=/home/hice1/jgil37/scratch/llm-inference/slurm-results/llm-%j.out
#SBATCH --error=/home/hice1/jgil37/scratch/llm-inference/slurm-results/llm-%j.err
echo "Launching AIsurBL"
hostname

# Load modules
module load cuda
module load python/3.12.5


# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate virtual environment and set library paths
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script with uv
python src/llm_mutation.py /home/hice1/jgil37/scratch/llm-inference/sota/ExquisiteNetV2/models/network_xXxNqgXJASs1hapXjyCFdf7QPHR.py /home/hice1/jgil37/scratch/llm-inference/sota/ExquisiteNetV2/models/network_xXxEJC9GMa27Xl3vgbgw8tHdApF.py 0/xXxEJC9GMa27Xl3vgbgw8tHdApF_model.txt --top_p 0.1 --temperature 0.17 --apply_quality_control 'False' --inference_submission False --gene_id xXxEJC9GMa27Xl3vgbgw8tHdApF
