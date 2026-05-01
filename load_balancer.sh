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
export RUN_ID="${RUN_ID:-server-only}"
export RUN_DIR="${RUN_DIR:-${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}}"
export RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_DIR}/logs}"
export RUN_METRICS_DIR="${RUN_METRICS_DIR:-${RUN_DIR}/metrics}"
export RUN_ERRORS_DIR="${RUN_ERRORS_DIR:-${RUN_DIR}/errors}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RUN_LOG_DIR}}"
export SLURM_ERROR_DIR="${SLURM_ERROR_DIR:-${RUN_ERRORS_DIR}}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${RUN_LOG_DIR}/loadbalancer.log}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${RUN_LOG_DIR}/servers.json}"

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}" "${SLURM_LOG_DIR}" "${SLURM_ERROR_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/load-balancer-runtime-${SLURM_JOB_ID:-manual}.out") \
     2> >(tee -a "${RUN_ERRORS_DIR}/load-balancer-runtime-${SLURM_JOB_ID:-manual}.err" >&2)

echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"
echo "RUN_ID: $RUN_ID"
echo "RUN_LOG_DIR: $RUN_LOG_DIR"
echo "RUN_ERRORS_DIR: $RUN_ERRORS_DIR"
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
