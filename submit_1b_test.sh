#!/bin/bash
#SBATCH --job-name=llm_1b_cpu_test
#SBATCH --output=/home/hice1/jgil37/scratch/llm-inference/slurm_logs/1b_test_%j.log
#SBATCH --error=/home/hice1/jgil37/scratch/llm-inference/slurm_logs/1b_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1

# Comprehensive 1B model CPU test on cluster compute node

# Change to script directory first, before set -e
cd /home/hice1/jgil37/scratch/llm-inference
mkdir -p slurm_logs runs/server-only/metrics/smoke_tests

# Activate virtual environment
source venv/bin/activate

set -e
echo "[INFO] Starting 1B Model CPU Testing on Compute Node..."
date
echo "[INFO] Python: $(which python)"
echo "[INFO] Python version: $(python --version)"

# Setup variables
MODEL_CACHE_ROOT="/home/hice1/jgil37/scratch/llm_models"
MODEL_PATH="/home/hice1/jgil37/scratch/llm_models/meta-llama/Llama-3.2-1B-Instruct"
MODEL_REPO="meta-llama/Llama-3.2-1B-Instruct"
PORT=8003
PYTHONPATH=".:$(pwd)"

# Prefer the existing Hugging Face snapshot if it already contains model.safetensors
SNAPSHOT_CANDIDATE=$(find "$MODEL_CACHE_ROOT/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 2 -name model.safetensors 2>/dev/null | head -1 | xargs -r dirname)
if [ -n "$SNAPSHOT_CANDIDATE" ] && [ -f "$SNAPSHOT_CANDIDATE/model.safetensors" ]; then
    MODEL_PATH="$SNAPSHOT_CANDIDATE"
    echo "[INFO] Using existing snapshot path: $MODEL_PATH"
fi

# ============================================================================
# Step 1: Ensure Model Weights are Downloaded
# ============================================================================
echo "[STEP 1] Verifying model download..."

if [ -f "$MODEL_PATH/model.safetensors" ] || [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    WEIGHTS_SIZE=$(ls -lh "$MODEL_PATH"/*.safetensors "$MODEL_PATH"/*.bin 2>/dev/null | awk '{sum+=$5} END {print sum}')
    echo "[OK] Model weights already present"
else
    echo "[WARN] Model weights not found. Resuming download..."
    python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
os.makedirs('$MODEL_CACHE_ROOT', exist_ok=True)
print('Downloading model: $MODEL_REPO')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_REPO', cache_dir='$MODEL_CACHE_ROOT', trust_remote_code=True)
print('Downloading model weights (this may take 5-10 minutes)...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_REPO',
    cache_dir='$MODEL_CACHE_ROOT',
    torch_dtype='auto',
    device_map='cpu',
    load_in_4bit=False,
    trust_remote_code=True
)
print('Model download complete!')
    " 2>&1 | tee slurm_logs/download_log.txt
fi

# Resolve actual Hugging Face snapshot path if direct path doesn't contain weights.
# The snapshot uses symlinks, so search by filename rather than only regular files.
if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
    SNAPSHOT_PATH=$(find "$MODEL_CACHE_ROOT/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 2 \( -name "model.safetensors" -o -name "pytorch_model.bin" \) 2>/dev/null | head -1 | xargs -r dirname)
    if [ -n "$SNAPSHOT_PATH" ] && [ -f "$SNAPSHOT_PATH/model.safetensors" -o -f "$SNAPSHOT_PATH/pytorch_model.bin" ]; then
        MODEL_PATH="$SNAPSHOT_PATH"
        echo "[INFO] Resolved model snapshot path: $MODEL_PATH"
    fi
fi

if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
    echo "[ERROR] No model weight files found after download attempt under $MODEL_CACHE_ROOT"
    exit 1
fi

echo "[OK] Model ready at $MODEL_PATH"
echo

# ============================================================================
# Step 2: Start server.py on CPU with 4-bit quantization
# ============================================================================
echo "[STEP 2] Starting server.py on CPU..."

# Create environment config for server
cat > .env.test_cpu << 'ENVEOF'
MODEL_PATH=__MODEL_PATH_PLACEHOLDER__
DEVICE_MAP=auto
ENABLE_QUANTIZATION=false
QUANTIZATION_BITS=0
BATCH_SIZE=1
MAX_CONCURRENT_REQUESTS=2
SERVER_PORT=8003
LOG_LEVEL=INFO
ENVEOF

sed -i "s|__MODEL_PATH_PLACEHOLDER__|$MODEL_PATH|g" .env.test_cpu

# Start server in background
export $(cat .env.test_cpu | xargs)
uvicorn server:app \
    --host 127.0.0.1 \
    --port $PORT \
    --log-level info \
    > slurm_logs/server_output.log 2>&1 &

SERVER_PID=$!
echo "[INFO] Server started with PID $SERVER_PID on port $PORT"

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..."
for i in {1..1800}; do
    if curl -s http://127.0.0.1:$PORT/ > /dev/null 2>&1; then
        echo "[OK] Server is ready!"
        break
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "[INFO] Still waiting... (${i}s elapsed)"
        echo "[INFO] Current server log tail:"
        tail -5 slurm_logs/server_output.log || true
    fi
    if [ $i -eq 1800 ]; then
        echo "[ERROR] Server failed to start within 1800 seconds"
        echo "[ERROR] Last 50 lines of server log:"
        tail -50 slurm_logs/server_output.log || true
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo

# ============================================================================
# Step 3: Run 5-Configuration Test Suite
# ============================================================================
echo "[STEP 3] Running 5-configuration test suite..."

# Run Python test harness with port 8003 (real model server)
SERVER_PORT=8003 python run_five_config_tests.py --port 8003

echo

# ============================================================================
# Step 4: Cleanup and Summary
# ============================================================================
echo "[STEP 4] Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
sleep 2

echo
echo "========================================"
echo "Test Execution Complete"
echo "========================================"
echo "Results directory: runs/server-only/metrics/smoke_tests"
echo "Server log: slurm_logs/server_output.log"
echo "Job log: slurm_logs/1b_test_${SLURM_JOB_ID}.log"
echo

# Display summary
echo "Test Results Summary:"
ls -1 runs/server-only/metrics/smoke_tests/config_*.json 2>/dev/null | while read f; do
    name=$(basename "$f" .json)
    if [ -f "$f" ]; then
        echo "  ✓ $name: $(wc -l < "$f") lines"
    fi
done

date
echo "[INFO] Test job complete!"
