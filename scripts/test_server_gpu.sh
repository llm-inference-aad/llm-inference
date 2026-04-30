#!/bin/bash
#SBATCH -J test-server
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0:20:00
#SBATCH -o slurm-results/test-server-%j.out
#SBATCH -e slurm-results/test-server-%j.err

# Run on GPU node to verify inference server works with model
# Usage: sbatch scripts/test_server_gpu.sh
# Quick test (default): small HF model gpt2 — needs network on first download or cached weights.
# Full .env model: USE_ENV_MODEL=1 sbatch scripts/test_server_gpu.sh

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p slurm-results

module load cuda 2>/dev/null || true

[[ -f .env ]] && set -a && source .env && set +a
# Default to small model for a fast sanity check (gpt2). .env often sets a huge MODEL_PATH;
# to use .env's model instead: USE_ENV_MODEL=1 sbatch scripts/test_server_gpu.sh
if [[ "${USE_ENV_MODEL:-0}" != "1" ]]; then
  export MODEL_PATH="${QUICK_MODEL_PATH:-gpt2}"
fi

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$REPO_ROOT}"
export VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
export PYTHONUNBUFFERED=1
# Prefer project-local HF cache (avoids slow home FS on some clusters)
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

if [[ -d "${VENV_PATH}/bin" ]]; then
  echo "Activating venv: $VENV_PATH"
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: No venv at $VENV_PATH (create with uv sync or scripts/setup_uv.sh)"
  exit 1
fi

SERVER_LOG="$REPO_ROOT/slurm-results/server-test-${SLURM_JOB_ID:-local}.log"

echo "=== Inference Server Test ==="
echo "Host: $(hostname)"
echo "MODEL_PATH: $MODEL_PATH"
echo "Server log: $SERVER_LOG"
echo ""

# Start server in background (same pattern as scripts/cluster/server.sh — avoid uv run NFS stalls)
echo "Starting server (cold start - model loading)..."
python -m uvicorn app.server:app --host 0.0.0.0 --port 8060 >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() { kill $SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

# Wait for server to respond (startup loads the model before accepting traffic)
echo "Waiting for server ready..."
for i in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8060/ 2>/dev/null | grep -q 200; then
    echo "Server ready after ${i}0 seconds"
    break
  fi
  sleep 10
  echo "  ... ${i}0s (see $SERVER_LOG)"
  if [[ $i -eq 120 ]]; then
    echo "ERROR: Server did not start within 20 min"
    echo "--- tail server log ---"
    tail -80 "$SERVER_LOG" 2>/dev/null || true
    exit 1
  fi
done

# Test inference
echo ""
echo "Sending inference request..."
RESP=$(curl -s -X POST http://127.0.0.1:8060/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a one-line Python function to add two numbers.","max_new_tokens":100,"temperature":0.3}')

if echo "$RESP" | grep -q "generated_text"; then
  echo "✅ SUCCESS"
  echo "Response preview: $(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('generated_text','')[:300])" 2>/dev/null)"
  echo "Latency: $(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('e2e_latency_sec','?'))" 2>/dev/null)s"
else
  echo "❌ FAILED"
  echo "$RESP"
  exit 1
fi
