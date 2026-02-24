#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 08:00:00
#SBATCH -C "H100"
#SBATCH --gpus-per-node=2
#SBATCH -p ice-gpu
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=metrics/slurm-results/slurm-server-%j.out
#SBATCH --error=metrics/slurm-results/slurm-server-%j.err

echo "launching LLM Server"

hostname

mkdir -p metrics/slurm-results

module load cuda

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    echo "Warning: .env file not found. Using default values."
    # Set defaults if .env doesn't exist
    export LLM_INFERENCE_ROOT_DIR="$(pwd)"
    export VENV_PATH="$(pwd)/.venv"
    export SERVER_HOST="0.0.0.0"
    export SERVER_PORT="8000"
    export SERVER_WORKERS="1"
    export HOSTNAME_LOG_FILE="$(pwd)/runs/server-only/logs/hostname.log"
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
exec > >(tee -a "${RUN_LOG_DIR}/server-runtime-${SLURM_JOB_ID:-manual}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/server-runtime-${SLURM_JOB_ID:-manual}.err" >&2)
echo "RUN_ID: ${RUN_ID}"
echo "RUN_LOG_DIR: ${RUN_LOG_DIR}"
echo "RUN_METRICS_DIR: ${RUN_METRICS_DIR}"

# Activate virtual environment
if [ -d "${VENV_PATH}/bin" ]; then
    echo "Activating virtual environment at: $VENV_PATH"
    source "${VENV_PATH}/bin/activate"
else
    echo "Warning: Virtual environment not found at: $VENV_PATH"
fi

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

# Ensure monitor is killed when script exits
trap "kill $MONITOR_PID" EXIT

# Select backend: vllm (default, PagedAttention) or hf (legacy HuggingFace pipeline)
VLLM_BACKEND=${VLLM_BACKEND:-true}

if [ "$VLLM_BACKEND" = "true" ]; then
    echo "Starting vLLM backend (PagedAttention + prefix caching)"
    # vLLM manages multi-GPU internally via tensor_parallel_size — use 1 worker
    python -m uvicorn server_vllm:app --host $SERVER_HOST --port $SERVER_PORT --workers 1
else
    echo "Starting legacy HuggingFace backend"
    python -m uvicorn server:app --host $SERVER_HOST --port $SERVER_PORT --workers $SERVER_WORKERS
fi
