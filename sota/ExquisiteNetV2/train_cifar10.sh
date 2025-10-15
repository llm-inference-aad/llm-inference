#!/bin/bash
#SBATCH --job-name=exquisite_cifar10
#SBATCH -t 12:00:00                # Runtime 12 hours (training takes several hours)
#SBATCH --mem-per-gpu=16G
#SBATCH -n 4                       # 4 CPU cores for data loading
#SBATCH -N 1
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-exquisite-%j.out

set -Eeuo pipefail

echo "=== ExquisiteNetV2 CIFAR-10 Training ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

# ----------------------------
# Load modules / CUDA / Python
# ----------------------------
module load cuda
module load anaconda3 || true
export CUDA_VISIBLE_DEVICES=0

# ----------------------------
# Show GPU info
# ----------------------------
echo "=== GPU Information ==="
nvidia-smi

# ----------------------------
# Navigate to project root
# ----------------------------
cd /home/hice1/rmanimaran8/scratch/llm-inference/llm-inference

# ----------------------------
# Run training
# ----------------------------
echo "=== Starting ExquisiteNetV2 Training ==="
echo "Command: python sota/ExquisiteNetV2/train.py -data cifar10 -end_lr 0.001 -seed 21 -val_r 0.2 -amp"

python sota/ExquisiteNetV2/train.py -data cifar10 -end_lr 0.001 -seed 21 -val_r 0.2 -amp

echo "=== Training Complete ==="
date

# Show final results
echo "=== Training outputs ==="
ls -lh sota/ExquisiteNetV2/weight/seed/


