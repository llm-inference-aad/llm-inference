#!/bin/bash
#SBATCH --job-name=LLM_LoadBalancer
#SBATCH -t 8:00:00
#SBATCH --mem=8G
#SBATCH -c 4

echo "===== Launching LLM Load Balancer ====="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

# ----------------------------
# Load modules / Python
# ----------------------------
module load cuda
module load anaconda3 || true

# ----------------------------
# Ensure uv on PATH
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Install with: pipx install uv"
  exit 1
fi

echo "UV version: $(uv --version)"
echo "Python via uv: $(uv run python --version)"

# ----------------------------
# Load environment variables
# ----------------------------
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  exit 1
fi

# Auto-export all keys defined in .env into this shell's environment
set -a
source .env
set +a

# ----------------------------
# Set defaults
# ----------------------------
export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$(pwd)}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${LLM_INFERENCE_ROOT_DIR}/loadbalancer.log}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${LLM_INFERENCE_ROOT_DIR}/servers.json}"

echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"
echo "LOAD_BALANCER_PORT: $LOAD_BALANCER_PORT"
echo "LOADBALANCER_LOG_FILE: $LOADBALANCER_LOG_FILE"
echo "SERVER_REGISTRY_FILE: $SERVER_REGISTRY_FILE"

# ----------------------------
# Initialize server registry file
# ----------------------------
if [[ ! -f "$SERVER_REGISTRY_FILE" ]]; then
    echo "Creating empty server registry file: $SERVER_REGISTRY_FILE"
    echo '{"servers": []}' > "$SERVER_REGISTRY_FILE"
fi

# ----------------------------
# Write load balancer hostname
# ----------------------------
export LOADBALANCER_HOSTNAME=$(hostname)
echo "Writing load balancer hostname '$LOADBALANCER_HOSTNAME' to file: $LOADBALANCER_LOG_FILE"
echo "$LOADBALANCER_HOSTNAME" > "$LOADBALANCER_LOG_FILE"

# ----------------------------
# Start load balancer
# ----------------------------
echo "Starting load balancer on $LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT"
echo "====================================="

uv run python -m uvicorn load_balancer:app --host 0.0.0.0 --port $LOAD_BALANCER_PORT

echo "===== Load balancer stopped ====="
date


