#!/bin/bash
# NOTE: SBATCH directives are evaluated at submission time.
# To make this "modular" per-model, override these on the `sbatch` command line.
# Example overrides:
# - `sbatch -p coe-gpu -C H100 --gpus-per-node=1 server.sh`
# - `sbatch -p ice-gpu -C gpu-a100 --gpus-per-node=1 server.sh`
#
# Defaults below are tuned for the current exploratory GLM baseline:
# - `zai-org/GLM-4.7-Flash`
# - tentative 1x80GB-class NVIDIA placement with `TENSOR_PARALLEL_SIZE=1`
#
# IMPORTANT: GGUF models (e.g. `*GGUF` repos with `Q5_K_M`) are not loadable by
# `server_vllm.py`'s Hugging Face loader path. Stage GGUF files for llama.cpp,
# or use a non-GGUF HF checkpoint for vLLM.
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 08:00:00
#SBATCH --gpus-per-node=2
#SBATCH -C "nvidia-gpu"        # Keep CUDA-capable placement broad unless a submission overrides the constraint.
#SBATCH -p coe-gpu             # Default to the same queue family typically used by the main run job.
#SBATCH --mem 80G
#SBATCH -c 16
#SBATCH --output=metrics/slurm-results/slurm-server-%j.out
#SBATCH --error=metrics/slurm-results/slurm-server-%j.err

echo "launching LLM Server"

hostname

mkdir -p metrics/slurm-results

module load cuda
module load anaconda3 || true

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found. Using default values."
    # Set defaults if .env doesn't exist
    export LLM_INFERENCE_ROOT_DIR="$(pwd)"
    export RUN_ID="server-only"
    export RUN_DIR="${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}"
    export RUN_LOG_DIR="${RUN_DIR}/logs"
    export RUN_METRICS_DIR="${RUN_DIR}/metrics"
    export SLURM_LOG_DIR="${RUN_LOG_DIR}"
    export METRICS_PATH="${RUN_METRICS_DIR}"
    export VENV_PATH="$(pwd)/.venv"
    export SERVER_HOST="0.0.0.0"
    export SERVER_PORT="8000"
    export SERVER_WORKERS="1"
    export HOSTNAME_LOG_FILE="${RUN_LOG_DIR}/hostname.log"
    export LOADBALANCER_LOG_FILE="${RUN_LOG_DIR}/loadbalancer.log"
    export SERVER_REGISTRY_FILE="${RUN_LOG_DIR}/servers.json"
    export CUDA_VISIBLE_DEVICES="0"
    export MKL_THREADING_LAYER="GNU"
fi

# IMPORTANT: Command-line/sbatch exports override .env values
# This allows parallel jobs to use different ports
echo "Checking for environment overrides..."
if [ ! -z "$SERVER_PORT" ]; then
    echo "  SERVER_PORT override detected: $SERVER_PORT"
fi
if [ ! -z "$HOSTNAME_LOG_FILE" ]; then
    echo "  HOSTNAME_LOG_FILE override detected: $HOSTNAME_LOG_FILE"
fi

# Make sure CUDA can see all GPUs
# NOTE: CUDA_VISIBLE_DEVICES export is intentionally disabled to allow Slurm
#       to manage GPU visibility/indexing for multi-GPU allocations.
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# Set root directory if not already set
export LLM_INFERENCE_ROOT_DIR=${LLM_INFERENCE_ROOT_DIR:-$(pwd)}
export RUN_ID=${RUN_ID:-server-only}
export RUN_DIR=${RUN_DIR:-${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}}
export RUN_LOG_DIR=${RUN_LOG_DIR:-${RUN_DIR}/logs}
export RUN_METRICS_DIR=${RUN_METRICS_DIR:-${RUN_DIR}/metrics}
export SLURM_LOG_DIR=${SLURM_LOG_DIR:-${RUN_LOG_DIR}}
export METRICS_PATH=${METRICS_PATH:-${RUN_METRICS_DIR}}
export HOSTNAME_LOG_FILE=${HOSTNAME_LOG_FILE:-"${RUN_LOG_DIR}/hostname.log"}
export LOADBALANCER_LOG_FILE=${LOADBALANCER_LOG_FILE:-"${RUN_LOG_DIR}/loadbalancer.log"}
export SERVER_REGISTRY_FILE=${SERVER_REGISTRY_FILE:-"${RUN_LOG_DIR}/servers.json"}

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${SLURM_LOG_DIR}" "${RUN_METRICS_DIR}/gpu"
echo "RUN_ID: ${RUN_ID}"
echo "RUN_LOG_DIR: ${RUN_LOG_DIR}"
echo "RUN_METRICS_DIR: ${RUN_METRICS_DIR}"

# Force uv to use Python 3.12 to avoid Python 3.13 source-build failures
# in transitive deps (e.g., outlines-core via vllm).
export UV_PYTHON=${UV_PYTHON:-3.12}
echo "UV requested Python: ${UV_PYTHON}"
echo "Python via uv: $(uv run python --version)"

echo "Syncing project dependencies with uv (Python ${UV_PYTHON})..."
uv sync --python "${UV_PYTHON}"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=${HOSTNAME_LOG_FILE}

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"

# Write server job ID for automatic shutdown tracking
SERVER_JOB_FILE="${HOSTNAME_FILE%.log}_server_job.txt"
echo "Writing server job ID '${SLURM_JOB_ID}' to file: $SERVER_JOB_FILE"
echo "${SLURM_JOB_ID}" > "$SERVER_JOB_FILE"

echo "Starting LLM server on host: $SERVER_HOSTNAME"

# Use environment variables for server configuration
SERVER_HOST=${SERVER_HOST:-$SERVER_HOSTNAME}
SERVER_WORKERS=${SERVER_WORKERS:-1}

# ----------------------------
# Auto-assign port if not specified
# ----------------------------
if [ -z "$SERVER_PORT" ]; then
    echo "SERVER_PORT not specified, auto-assigning next available port..."
    
    # Set base port from environment or default
    SERVER_BASE_PORT=${SERVER_BASE_PORT:-8000}
    
    # Find next available port by checking existing servers
    if [ ! -z "$SERVER_REGISTRY_FILE" ] && [ -f "$SERVER_REGISTRY_FILE" ]; then
        # Read existing servers and find highest port
        HIGHEST_PORT=$(uv run python << EOF
import json
import sys

try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    
    servers = registry.get('servers', [])
    if servers:
        ports = [s.get('port', $SERVER_BASE_PORT) for s in servers]
        highest = max(ports)
        print(highest + 1)
    else:
        print($SERVER_BASE_PORT)
except Exception as e:
    print($SERVER_BASE_PORT)
EOF
        )
        SERVER_PORT=$HIGHEST_PORT
        echo "  Auto-assigned port: $SERVER_PORT"
    else
        # No registry file, use base port
        SERVER_PORT=$SERVER_BASE_PORT
        echo "  Using base port: $SERVER_PORT"
    fi
else
    echo "Using specified port: $SERVER_PORT"
fi

# ----------------------------
# Register with load balancer (if SERVER_REGISTRY_FILE is set)
# ----------------------------
if [ ! -z "$SERVER_REGISTRY_FILE" ]; then
    echo "Registering server with load balancer..."
    echo "  Server: $SERVER_HOSTNAME:$SERVER_PORT"
    echo "  Registry: $SERVER_REGISTRY_FILE"
    
    # Create a temporary file for the new server entry
    TEMP_ENTRY=$(mktemp)
    TIMESTAMP=$(date -Iseconds)
    echo "{\"hostname\": \"$SERVER_HOSTNAME\", \"port\": $SERVER_PORT, \"registered_at\": \"$TIMESTAMP\"}" > "$TEMP_ENTRY"
    
    # Use flock for file locking to prevent race conditions
    (
        flock -x 200
        
        # Read existing registry or create new one
        if [ -f "$SERVER_REGISTRY_FILE" ]; then
            EXISTING=$(cat "$SERVER_REGISTRY_FILE")
        else
            EXISTING='{"servers": []}'
        fi
        
        # Add new server to registry using Python
        uv run python << EOF
import json
import sys

# Read existing registry
registry = json.loads('''$EXISTING''')

# Read new server entry
with open('$TEMP_ENTRY', 'r') as f:
    new_server = json.load(f)

# Check if server already exists (same hostname and port)
existing_servers = registry.get('servers', [])
server_key = (new_server['hostname'], new_server['port'])
updated = False

for i, server in enumerate(existing_servers):
    if (server['hostname'], server['port']) == server_key:
        # Update existing entry
        existing_servers[i] = new_server
        updated = True
        break

if not updated:
    # Add new server
    existing_servers.append(new_server)

registry['servers'] = existing_servers

# Write back to registry
with open('$SERVER_REGISTRY_FILE', 'w') as f:
    json.dump(registry, f, indent=2)

print(f"Registered: {new_server['hostname']}:{new_server['port']}")
EOF
        
    ) 200>"$SERVER_REGISTRY_FILE.lock"
    
    # Clean up temp file
    rm -f "$TEMP_ENTRY"
    
    echo "  Registration complete"
else
    echo "SERVER_REGISTRY_FILE not set, skipping load balancer registration"
fi

# Start GPU Monitoring
echo "Starting GPU monitoring..."
mkdir -p "${RUN_METRICS_DIR}/gpu"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${RUN_METRICS_DIR}/gpu/server-${SLURM_JOB_ID}.csv" &
MONITOR_PID=$!

# UVICORN_PID is set after the uvicorn launch below; declare it here so the trap
# can reference it even if the process hasn't started yet.
UVICORN_PID=""

# On EXIT/SIGTERM: gracefully stop uvicorn so vLLM flushes buffered stdout to
# the SLURM .out file before the job terminates. Uvicorn's default graceful
# shutdown window is 30 s, but 5 s is enough for log flushing in practice.
_server_cleanup() {
  if [[ -n "${UVICORN_PID}" ]]; then
    echo "Sending SIGTERM to uvicorn (pid=${UVICORN_PID}) for graceful shutdown..."
    kill -TERM "${UVICORN_PID}" 2>/dev/null || true
    sleep 5
  fi
  kill "${MONITOR_PID}" 2>/dev/null || true
}
trap _server_cleanup EXIT INT TERM

# Select backend: vllm (default, PagedAttention) or hf (legacy HuggingFace pipeline)
VLLM_BACKEND=${VLLM_BACKEND:-true}

if [ "$VLLM_BACKEND" = "true" ]; then
    echo "Starting vLLM backend (PagedAttention + prefix caching)"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # vLLM manages multi-GPU internally via tensor_parallel_size — use 1 worker.
    # Background the process and capture its PID so the cleanup trap can SIGTERM
    # it gracefully before the job exits, flushing vLLM's buffered stdout.
    uv run python -m uvicorn server_vllm:app --host $SERVER_HOST --port $SERVER_PORT --workers 1 &
    UVICORN_PID=$!
    wait "${UVICORN_PID}"
else
    echo "Starting legacy HuggingFace backend"
    uv run python -m uvicorn server:app --host $SERVER_HOST --port $SERVER_PORT --workers $SERVER_WORKERS &
    UVICORN_PID=$!
    wait "${UVICORN_PID}"
fi
