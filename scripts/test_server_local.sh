#!/bin/bash
# Local test of inference server (app/server.py)
# Usage: ./scripts/test_server_local.sh [--quick]
#   --quick: Use gpt2 for fast cold-start test (~1-2 min on GPU, longer on CPU)
#   default: Use MODEL_PATH from .env (Llama 3.3 70B - needs GPU node, ~5-10 min cold start)
#
# NOTE: Run on a GPU node for large models. Login nodes may lack GPU or have strict limits.
#   srun --gres=gpu:1 --mem=32G -t 0:30:00 --pty bash -c './scripts/test_server_local.sh --quick'

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=${TEST_SERVER_PORT:-8050}
QUICK=false

for arg in "$@"; do
  [[ "$arg" == "--quick" ]] && QUICK=true
done

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ "$QUICK" == "true" ]]; then
  export MODEL_PATH="gpt2"
  echo "=== Quick mode: Using gpt2 (small model, ~30s cold start) ==="
else
  echo "=== Full mode: Using MODEL_PATH from .env: $MODEL_PATH ==="
  echo "    (Llama 70B cold start: ~5-10 min, needs multi-GPU)"
fi

echo ""
echo "Starting server on port $PORT ..."
echo "Cold start: Model will load on first request. This can take several minutes for large models."
echo ""

# Start server in background
uv run python -m uvicorn app.server:app --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

# Cleanup on exit
cleanup() {
  echo ""
  echo "Stopping server (PID $SERVER_PID)..."
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT

# Wait for server to respond (health check at /)
echo "Waiting for server to start..."
MAX_WAIT=600  # 10 min for large model
ELAPSED=0
while ! curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q 200; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  echo "  ... still waiting (${ELAPSED}s, server loading model...)"
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
    exit 1
  fi
done

echo "Server is up (took ${ELAPSED}s)."
echo ""

# Make test inference request
echo "Sending test inference request..."
RESPONSE=$(curl -s -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that returns the sum of two numbers.",
    "max_new_tokens": 256,
    "temperature": 0.3,
    "top_p": 0.9
  }')

if echo "$RESPONSE" | grep -q "generated_text"; then
  echo "✅ SUCCESS: Inference request completed."
  echo ""
  echo "Response (generated_text):"
  echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('generated_text','')[:500])"
  echo ""
  echo "Latency: $(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('e2e_latency_sec', d.get('_latency_sec','?')))" 2>/dev/null)s"
  echo "Evaluation score: $(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('evaluationScore','?'))" 2>/dev/null)"
  exit 0
else
  echo "❌ FAILED: Invalid response"
  echo "$RESPONSE"
  exit 1
fi
