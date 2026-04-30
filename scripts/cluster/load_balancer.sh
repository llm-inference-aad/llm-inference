#!/bin/bash
#SBATCH --job-name=LLM_LoadBalancer
#SBATCH --partition=pace-cpu
#SBATCH --nodelist=atl1-1-02-010-6-1
#SBATCH -t 00:45:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --output=slurm-results/slurm-loadbalancer-%j.out
#SBATCH --error=slurm-results/slurm-loadbalancer-%j.err

# Ensure we run from repo root (in case sbatch was called from elsewhere)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

echo "===== Launching LLM Load Balancer ====="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

mkdir -p slurm-results

# ----------------------------
# Load modules / Python
# ----------------------------
module load cuda
module load anaconda3 || true

# ----------------------------
# Ensure Python environment is available
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"

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
export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$REPO_ROOT}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${LLM_INFERENCE_ROOT_DIR}/loadbalancer.log}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${LLM_INFERENCE_ROOT_DIR}/servers.json}"
export PREFIX_HASH_CHARS="${PREFIX_HASH_CHARS:-2048}"

if [[ -d "${VENV_PATH:-$REPO_ROOT/.venv}/bin" ]]; then
  echo "Activating virtual environment at: ${VENV_PATH:-$REPO_ROOT/.venv}"
  source "${VENV_PATH:-$REPO_ROOT/.venv}/bin/activate"
fi

echo "Python: $(python --version)"

echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"
echo "LOAD_BALANCER_PORT: $LOAD_BALANCER_PORT"
echo "LOADBALANCER_LOG_FILE: $LOADBALANCER_LOG_FILE"
echo "SERVER_REGISTRY_FILE: $SERVER_REGISTRY_FILE"
echo "PREFIX_HASH_CHARS: $PREFIX_HASH_CHARS"

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
echo "Starting prefix/KV-cache-aware gateway on $LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT"
echo "====================================="

python -m uvicorn app.load_balancer:app --host 0.0.0.0 --port $LOAD_BALANCER_PORT

echo "===== Load balancer stopped ====="
date


