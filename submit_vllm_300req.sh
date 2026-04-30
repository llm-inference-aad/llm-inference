#!/bin/bash
#SBATCH --job-name=vllm_300req
#SBATCH --output=/home/hice1/jgil37/scratch/llm-inference/slurm_logs/vllm_300req_%j.out
#SBATCH --error=/home/hice1/jgil37/scratch/llm-inference/slurm_logs/vllm_300req_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
# Prefer a stronger GPU for this run (H100 or A100-80GB); avoid V100.
#SBATCH -C "H100|A100-80GB"

set -euo pipefail

ROOT_DIR="/home/hice1/jgil37/scratch/llm-inference"
cd "$ROOT_DIR"
mkdir -p slurm_logs runs/vllm_300request/metrics runs/vllm_300request/logs runs/vllm_300request/errors

# Use the adaptive vLLM config as the base
cp .env.vllm_all .env

# Disable server-wide defaults to ensure mixed workload works
if grep -q '^CONSTRAINED_DECODING_ENABLED=' .env; then
  sed -i 's/^CONSTRAINED_DECODING_ENABLED=.*/CONSTRAINED_DECODING_ENABLED=false/' .env
else
  echo 'CONSTRAINED_DECODING_ENABLED=false' >> .env
fi

if grep -q '^DEFAULT_CONSTRAINT_TYPE=' .env; then
  sed -i 's/^DEFAULT_CONSTRAINT_TYPE=.*/DEFAULT_CONSTRAINT_TYPE=/' .env
else
  echo 'DEFAULT_CONSTRAINT_TYPE=' >> .env
fi

if grep -q '^ENABLE_JSON_CONSTRAINTS=' .env; then
  sed -i 's/^ENABLE_JSON_CONSTRAINTS=.*/ENABLE_JSON_CONSTRAINTS=false/' .env
else
  echo 'ENABLE_JSON_CONSTRAINTS=false' >> .env
fi

# Use the faster native precision path on stronger GPUs.
if grep -q '^VLLM_DTYPE=' .env; then
  sed -i 's/^VLLM_DTYPE=.*/VLLM_DTYPE=bfloat16/' .env
else
  echo 'VLLM_DTYPE=bfloat16' >> .env
fi
export VLLM_DTYPE="bfloat16"

# Allow vLLM compilation/cudagraphs on stronger GPUs.
if grep -q '^ENFORCE_EAGER=' .env; then
  sed -i 's/^ENFORCE_EAGER=.*/ENFORCE_EAGER=false/' .env
else
  echo 'ENFORCE_EAGER=false' >> .env
fi
export ENFORCE_EAGER="false"

# Restore the standard context window on stronger GPUs.
if grep -q '^MAX_MODEL_LEN=' .env; then
  sed -i 's/^MAX_MODEL_LEN=.*/MAX_MODEL_LEN=32768/' .env
else
  echo 'MAX_MODEL_LEN=32768' >> .env
fi
export MAX_MODEL_LEN="32768"

export RUN_ID="vllm_300request"
export RUN_LOG_DIR="$ROOT_DIR/runs/$RUN_ID/logs"
export RUN_METRICS_DIR="$ROOT_DIR/runs/$RUN_ID/metrics"
export RUN_ERRORS_DIR="$ROOT_DIR/runs/$RUN_ID/errors"
export LLM_INFERENCE_ROOT_DIR="$ROOT_DIR"
export SERVER_PORT="8001"
export CONSTRAINT_LOGGING_ENABLED="true"

source venv/bin/activate

SERVER_LOG="$ROOT_DIR/runs/$RUN_ID/logs/server.log"
SERVER_PID_FILE="$ROOT_DIR/runs/$RUN_ID/logs/server.pid"

cleanup() {
  if [[ -f "$SERVER_PID_FILE" ]]; then
    kill "$(cat "$SERVER_PID_FILE")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting server..."
bash server.sh > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID_FILE"

for i in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:${SERVER_PORT}/" >/dev/null 2>&1; then
    echo "[OK] Server ready after ${i}s"
    break
  fi
  if [[ $((i % 30)) -eq 0 ]]; then
    echo "[INFO] Waiting for server... (${i}s elapsed)"
    tail -n 5 "$SERVER_LOG" || true
  fi
  if [[ $i -eq 900 ]]; then
    echo "[ERROR] Server failed to become ready"
    tail -n 80 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 1
done

# Run 300-request benchmark workload
echo "[INFO] Starting 300-request benchmark..."
SERVER_PORT=8001 python run_500_config_tests.py --port 8001 --output-dir "$ROOT_DIR/runs/$RUN_ID/metrics" --num-requests 300

echo "[INFO] Benchmark complete"
