#!/bin/bash
# Comprehensive testing workflow for 1B model CPU inference
# This script provides a guided walkthrough of all testing options

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  1B Model CPU Inference - 5 Configuration Test Suite          ║"
echo "║  Comprehensive Testing Workflow                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

echo "This script will guide you through testing constrained and speculative"
echo "decoding on the Llama-3.2-1B model on CPU."
echo
echo "Testing Configurations:"
echo "  1. Baseline - no constraints, no speculative decoding"
echo "  2. Constrained JSON - JSON schema validation only"
echo "  3. Constrained Regex - regex constraint matching only"
echo "  4. Speculative Suffix - suffix-based draft prediction"
echo "  5. Combined - constrained JSON + speculative draft models"
echo
echo "Available Execution Methods:"
echo "  [A] Local testing with mock server (instant, no model needed)"
echo "  [B] Local testing with real 1B model on CPU (requires ~8GB RAM)"
echo "  [C] Cluster testing via sbatch (recommended for CPU, handles large RAM)"
echo

read -p "Choose method [A/B/C]: " method

if [ "$method" = "A" ] || [ "$method" = "a" ]; then
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "Method A: Mock Server Testing"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    
    if ! pgrep -f "server_mock.py" > /dev/null; then
        echo "[INFO] Starting mock server on port 8002..."
        python server_mock.py > /dev/null 2>&1 &
        MOCK_PID=$!
        sleep 2
        echo "[OK] Mock server started (PID: $MOCK_PID)"
    fi
    
    echo "[INFO] Running 5-configuration tests against mock server..."
    bash test_five_configs.sh
    
    echo
    echo "Results saved to: runs/server-only/metrics/smoke_tests/"
    echo

elif [ "$method" = "B" ] || [ "$method" = "b" ]; then
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "Method B: Local Real Model Testing on CPU"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    
    # Check model weights
    echo "[STEP 1] Verifying model download..."
    MODEL_PATH="/home/hice1/jgil37/scratch/llm_models/meta-llama/Llama-3.2-1B-Instruct"
    
    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "[INFO] Model weights not found. Downloading..."
        echo "(This may take 10-15 minutes and requires ~8GB disk space)"
        echo
        python download_and_verify_model.py
        if [ $? -ne 0 ]; then
            echo "[ERROR] Model download failed. See error messages above."
            exit 1
        fi
    else
        echo "[OK] Model weights found"
    fi
    
    echo
    echo "[STEP 2] Starting server.py on CPU with 4-bit quantization..."
    
    export MODEL_PATH="$MODEL_PATH"
    export DEVICE_MAP="auto"
    export ENABLE_QUANTIZATION="true"
    export QUANTIZATION_BITS="4"
    export BATCH_SIZE="1"
    
    python server.py \
        --model_path "$MODEL_PATH" \
        --device_map auto \
        --enable_quantization \
        --quantization_bits 4 \
        --batch_size 1 \
        --port 8003 \
        > /tmp/server.log 2>&1 &
    
    SERVER_PID=$!
    echo "[INFO] Server started (PID: $SERVER_PID)"
    
    # Wait for ready
    echo "[INFO] Waiting for server to be ready (checking health endpoint)..."
    for i in {1..60}; do
        if curl -s http://127.0.0.1:8003/health > /dev/null 2>&1; then
            echo "[OK] Server is ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "[ERROR] Server startup timeout"
            kill $SERVER_PID 2>/dev/null || true
            cat /tmp/server.log
            exit 1
        fi
        sleep 1
    done
    
    echo
    echo "[STEP 3] Running 5-configuration tests..."
    sed 's/PORT=8003/PORT=8003/g' test_five_configs.sh > /tmp/test_5_configs_local.sh
    bash /tmp/test_5_configs_local.sh
    
    echo
    echo "[CLEANUP] Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    sleep 2
    
    echo "[OK] Local testing complete!"
    echo "Results saved to: runs/server-only/metrics/smoke_tests/"
    echo

elif [ "$method" = "C" ] || [ "$method" = "c" ]; then
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "Method C: Cluster Testing via sbatch"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    
    echo "[INFO] Submitting job to cluster..."
    echo "Job script: submit_1b_test.sh"
    echo
    
    if [ ! -d slurm_logs ]; then
        mkdir -p slurm_logs
    fi
    
    # Submit the sbatch job
    sbatch submit_1b_test.sh
    
    JOB_ID=$(sbatch submit_1b_test.sh | grep -oP '(?<=Submitted batch job )\d+')
    
    echo "[OK] Job submitted with ID: $JOB_ID"
    echo
    echo "Monitor job progress with:"
    echo "  squeue -j $JOB_ID                    # Check job status"
    echo "  tail -f slurm_logs/1b_test_${JOB_ID}.log   # View live log output"
    echo "  ls -lh runs/server-only/metrics/smoke_tests  # Check results"
    echo

else
    echo "[ERROR] Invalid choice. Please run again and select A, B, or C."
    exit 1
fi

echo
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Testing workflow complete!                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "Next Steps:"
echo "  1. Review test results in: runs/server-only/metrics/smoke_tests/"
echo "  2. Analyze performance across 5 configurations"
echo "  3. Compare constraint validation times"
echo "  4. Evaluate speculative decoding acceptance rates"
echo
