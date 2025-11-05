#!/bin/bash
#SBATCH --job-name=finetune_llm
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100"
#SBATCH --mem-per-gpu 40G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=slurm-results/slurm-finetune-%j.out
#SBATCH --error=slurm-results/slurm-finetune-%j.err

echo "---------------------------------------"
echo "Begin Slurm Prolog: $(date)"
echo "Job ID:    $SLURM_JOB_ID"
echo "User ID:   $USER"
echo "Job name:  $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "---------------------------------------"

echo "=== Starting Model Finetuning ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"

# Activate virtual environment
source .venv/bin/activate

nvidia-smi

# Default values:
MODEL_NAME=${MODEL_NAME:-"google/gemma-2-9b-it"}
DATASET=${DATASET:-"finetuning_dataset_all.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"finetuned_model"}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "---------------------------------------"
echo "Finetuning Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "---------------------------------------"

# Run finetuning script
python scripts/finetune_model.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --use-lora

EXIT_CODE=$?

echo "---------------------------------------"
echo "Finetuning completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "---------------------------------------"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Finetuned model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Update .env: MODEL_PATH=$(pwd)/$OUTPUT_DIR"
    echo "2. Update src/cfg/constants.py: LLM_MODEL='local_server'"
    echo "3. Start server: sbatch server.sh"
fi

exit $EXIT_CODE

